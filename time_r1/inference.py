import copy
import torch
import json
import os
import sys
from PIL import Image
try:
    from peft import PeftModel
except:
    print('fail to load peft')
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import glob
import datetime
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from time_r1.dataset import LazyVLDataset, LazyVLDatasetBaseline, LazyVLDatasetSFT, get_dataset_class
from time_r1.utils import setup_ddp, cleanup_ddp, merge_results
from time_r1.utils.qwen_vl_utils import process_vision_info, replace_vision_info_with_placeholder, replace_vision_info_with_base64
from time_r1.environment.video_env import VideoInteraction


@torch.no_grad()
def forward_model(data_batch, model, processor, max_new_tokens=2048):
    rank = dist.get_rank()
    device = f"cuda:{rank}"
    messages_batch = [item["messages"] for item in data_batch]
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages_batch, return_video_kwargs=True)
    text_inputs = processor.apply_chat_template(messages_batch, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text_inputs, 
        images=image_inputs,
        videos=video_inputs, 
        fps=video_kwargs["fps"],
        padding=True, 
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = inputs.to(device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text


def forward_model_with_env(data_batch, model, processor, use_vllm=False, save_vision_info=False, **kwargs):
    env = VideoInteraction(processor, model, max_turns=10, max_new_tokens_per_turn=512, use_vllm=use_vllm)
    messages_batch = [item["messages"] for item in data_batch]
    multimodal_cache_batch = [item["multimodal_cache"] for item in data_batch]
    output_msgs = env.generate(messages_batch, multimodal_cache=multimodal_cache_batch, **kwargs)
    if save_vision_info:
        output_msgs = replace_vision_info_with_base64(output_msgs)
    else:
        output_msgs = replace_vision_info_with_placeholder(output_msgs)
    return output_msgs


def setup_model(model_base, lora_checkpoint=None):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        use_sliding_window=True,
        attn_implementation="flash_attention_2",
        device_map=device
    )
    if lora_checkpoint is not None:
        print("Model loaded, type:", type(model))
        model = PeftModel.from_pretrained(model, lora_checkpoint)
        print("LORA loaded, type:", type(model))
        model = model.merge_and_unload()
        print("LORA merged, type:", type(model))
    model.eval()
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_base)
    return model, processor


def setup_vllm(model_base, limit_image_per_prompt=1024, limit_video_per_prompt=10):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    print("Setting up VLLM model on rank:", local_rank)
    from vllm import LLM, SamplingParams
    model = LLM(
        model=model_base,
        revision=None,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=None,
        distributed_executor_backend="external_launcher",
        seed=local_rank,
        limit_mm_per_prompt={"image": limit_image_per_prompt, "video": limit_video_per_prompt},
    )
    processor = AutoProcessor.from_pretrained(model_base)
    sampling_params = SamplingParams(
        n=1,
        repetition_penalty=1.0,
        max_tokens=512,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=42
    )
    return model, processor, sampling_params


def main(input_path, save_path,
         data_root="datasets", 
         model_base="Qwen/Qwen2.5-VL-7B-Instruct",
         prompt_template="v4",
         tool_name_list=["seek_video_frames"],
         use_env=True,
         use_vllm=False,
         batch_size=1,
         lora_checkpoint=None,
         dataset_type="lazy_dataset",
         num_data_workers=4,
         total_video_tokens=15360,
         max_frames=768,
         min_tokens=16,
         max_tokens=192,
         save_vision_info=False,
         append_time_instruction=False,
         ):
    setup_ddp()

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    dataset_class = get_dataset_class(dataset_type)
    video_kwargs = {
        "total_pixels": total_video_tokens * 28 * 28,
        "min_pixels": min_tokens * 28 * 28,
        "max_pixels": max_tokens * 28 * 28,
        "max_frames": max_frames,
    }
    dataset = dataset_class(input_path, data_root, prompt_template=prompt_template, tool_name_list=tool_name_list, video_kwargs=video_kwargs, append_time_instruction=append_time_instruction)
    if use_vllm:
        model, processor, sampling_params = setup_vllm(model_base)
    else:
        model, processor = setup_model(model_base, lora_checkpoint)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_data_workers, collate_fn=lambda x: x)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    local_save_path = os.path.join(save_path, f"rank{rank}.jsonl")

    with open(local_save_path, "w") as writer:
        for data_batch in tqdm.tqdm(dataloader):
            if use_env:
                pred_batch = forward_model_with_env(data_batch, model, processor, use_vllm=use_vllm, save_vision_info=save_vision_info, sampling_params=sampling_params)
            else:
                pred_batch = forward_model(data_batch, model, processor)
            for data, pred in zip(data_batch, pred_batch):
                data['prediction'] = pred
                for k in ["image_inputs", "video_inputs", "video_kwargs", "messages", "multimodal_cache"]:
                    if k in data:
                        data.pop(k)
                writer.write(json.dumps(data) + "\n")
                writer.flush()

    dist.barrier()
    if rank == 0:
        merge_results(save_path)
    cleanup_ddp()
    print("Inference done.")



if __name__ == "__main__":
    import fire
    fire.Fire(main)