# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys

import datasets
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import (
    ModelConfig,
    SFTConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from tqdm import tqdm
import json
import random

from time_r1.utils.qwen_vl_utils import process_vision_info
from time_r1.dataset.lazy_dataset_sft import LazyVLDatasetSFT
logger = logging.getLogger(__name__)
import re
from copy import deepcopy

@dataclass
class SFTScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """
    tool_name_list: list[str] = field(
        default_factory=lambda: ["seek_video_frames"],
        metadata={"help": "List of video tool names."},
    )
    train_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data JSON file."},
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    video_folder: str = field(
        default="workdir/datasets",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )

    prompt_template: str = field(
        default="v3",
        metadata={"help": "Prompt template to use."},
    )


def compute_tool_response_mask(seq, id_im_start=151644, id_tool=14172):
    """
    向量化计算掩码：从 seq (1, T) 形状的 LongTensor 中屏蔽所有 `<|im_start|>tool…<|im_end|>` 区域。
    返回长度 T 的 IntTensor，其中 1 表示保留，0 表示屏蔽。
    这里 id_im_start 和 id_tool 与模型相关，需要从 tokenizer 中获取。
    id_im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    id_tool     = tokenizer.convert_tokens_to_ids("tool")
    """
    # 1) 标记所有 "<|im_start|>"
    is_im_start = seq == id_im_start
    # 2) 计算 region_id
    region_id = is_im_start.int().cumsum(dim=1)
    # 3) 标记段开头是否为工具段
    next_is_tool = torch.zeros_like(seq, dtype=torch.bool)
    next_is_tool[:, :-1] = is_im_start[:, :-1] & (seq[:, 1:] == id_tool)
    # 4) 累加得到每段是否为工具段
    region_flag = torch.zeros_like(region_id)
    region_flag = region_flag.scatter_add(
        dim=1,
        index=region_id,
        src=next_is_tool.int().to(region_flag.dtype)
    )
    # 5) 映射回每个 token 是否在工具段
    tool_region_mask = region_flag.gather(dim=1, index=region_id)
    # 6) 最终掩码：非工具段为 1，工具段为 0
    completion_mask = (~tool_region_mask.bool()).int()
    return completion_mask

processor = None

def clear_output(text, reserve_pad=True):
    if reserve_pad:
        text = replace_n_pattern(text, "<|endoftext|>", "{N} * <|endoftext|>")
    else:
        text = replace_n_pattern(text, "<|endoftext|>", "")
    text = replace_n_pattern(text, "<|video_pad|>", "{N} * <|video_pad|>")
    text = replace_n_pattern(text, "<|image_pad|>", "{N} * <|image_pad|>")
    return text.strip("\n")

def replace_n_pattern(text, pattern, replace_pattern):
    """
    替换字符串中连续出现的指定模式。

    参数:
        text (str): 原始字符串。
        pattern (str): 需要匹配的模式。
        replace_pattern (str): 替换模式，其中 `{N}` 会被替换为连续匹配的数量。

    返回:
        str: 替换后的字符串。
    """
    regex = re.compile(f'({re.escape(pattern)})+')
    
    def replace_match(match):
        matched_text = match.group()
        count = matched_text.count(pattern)
        return replace_pattern.replace('{N}', str(count))
    
    result = regex.sub(replace_match, text)
    return result

def collate_fn(examples):
    prompts_text = [
        processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        for example in examples
    ]
    messages = [deepcopy(example["messages"]) for example in examples]
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    batch = processor(
        text=deepcopy(prompts_text),    # bug fix: inplace operation will change the original prompts_text
        images=images,
        videos=videos,
        fps=video_kwargs["fps"],
        padding=True, 
        return_tensors="pt", 
    )

    labels = batch["input_ids"].clone()
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
    non_assistant_response_mask = compute_tool_response_mask(labels, im_start_id, assistant_id)
    labels[non_assistant_response_mask == 1] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == video_token_id] = -100
    labels[labels == image_token_id] = -100
    # 屏蔽所有数字token
    # for i in range(10):
        # labels[labels == processor.tokenizer.convert_tokens_to_ids(str(i))] = -100
    batch["labels"] = labels
    # debug 打印解码后的labels != -100的token
    # labels_debug = labels.clone()
    # labels_debug[labels_debug == -100] = processor.tokenizer.pad_token_id
    # print("=====>", clear_output(processor.decode(labels_debug[0])))
    return batch


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    train_dataset = LazyVLDatasetSFT(script_args.train_data_path,
                                    script_args.video_folder,
                                    prompt_template=script_args.prompt_template,
                                    tool_name_list=script_args.tool_name_list)
    eval_dataset = LazyVLDatasetSFT(script_args.eval_data_path,
                                    script_args.video_folder,
                                    prompt_template=script_args.prompt_template,
                                    tool_name_list=script_args.tool_name_list) if script_args.eval_data_path else None

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_init_kwargs = training_args.model_init_kwargs or {}
    model_init_kwargs["use_cache"] = (
        False if training_args.gradient_checkpointing else model_init_kwargs.get("use_cache")
    )
    model_init_kwargs["attn_implementation"] = "flash_attention_2"
    model_init_kwargs['torch_dtype'] = torch.bfloat16
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, 
        **model_init_kwargs
    )
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=None
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["R1-V"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
