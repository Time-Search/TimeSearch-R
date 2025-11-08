# Training script for Video-Language GRPO
# @Junwen Pan

import numpy as np
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from math_verify import parse, verify
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from time_r1.trainer.grpo_trainer_env import GRPOTrainer
from tqdm import tqdm
import torch
import json
import random
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from time_r1.dataset.lazy_dataset import LazyVLDataset
from time_r1.environment.video_env import VideoInteraction
from time_r1.utils.utils import rank0_print
from time_r1.reward import get_reward_functions
from time_r1.utils.qwen_vl_utils import process_vision_info

@dataclass
class GRPOScriptArguments(ScriptArguments):
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
    
    max_interaction_turns: int = field(
        default=4,
        metadata={"help": "Maximum number of video interaction turns."},
    )
    
    max_completion_length_per_turn: int = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion per turn."},
    )

    reward_func: str = field(
        default="v4",
        metadata={"help": "Reward version."},
    )

    train_data_path: str = field(
        default="scripts/toy_dataset.yaml",
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
    
    total_video_tokens: int = field(
        default=15360,
        metadata={"help": "Maximum number of video tokens."},
    )
    
    max_per_frame_tokens: int = field(
        default=192,
        metadata={"help": "Maximum number of frame tokens."},
    )
    
    min_per_frame_tokens: int = field(
        default=16,
        metadata={"help": "Minimum number of frame tokens."},
    )
    
    max_frames: int = field(
        default=768,
        metadata={"help": "Maximum number of frames to preview."},
    )

    append_time_instruction: bool = field(
        default=False,
        metadata={"help": "Whether to append time instruction."},
    )


@dataclass
class GRPOConfigEnv(GRPOConfig):
    replay_buffer_type: str = field(
        default="none",
        metadata={"help": "Type of replay buffer."},
    )

    replay_buffer_capacity: int = field(
        default=16,
        metadata={"help": "Capacity of the replay buffer."},
    )
    replay_buffer_alpha: float = field(
        default=1.0,
        metadata={"help": "Alpha of the replay buffer."},
    )

    use_counterfactual_reasoning: bool = field(
        default=False,
        metadata={"help": "Whether to use counterfactual reasoning."},
    )


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs, reward_weights = get_reward_functions(script_args.reward_func)
    if training_args.reward_weights is None:
        # Use default reward weights
        training_args.reward_weights = reward_weights
        print(f"[DEBUG] Using default reward weights: {reward_weights}")
    video_kwargs = {
        "total_pixels": script_args.total_video_tokens * 28 * 28,
        "min_pixels": script_args.min_per_frame_tokens * 28 * 28,
        "max_pixels": script_args.max_per_frame_tokens * 28 * 28,
        "max_frames": script_args.max_frames,
    }
    train_dataset = LazyVLDataset(script_args.train_data_path,
                                  script_args.video_folder,
                                  prompt_template=script_args.prompt_template,
                                  tool_name_list=script_args.tool_name_list,
                                  video_kwargs=video_kwargs,
                                  append_time_instruction=script_args.append_time_instruction)
    eval_dataset = LazyVLDataset(script_args.eval_data_path,
                                 script_args.video_folder,
                                 model_name_or_path=model_args.model_name_or_path,
                                 prompt_template=script_args.prompt_template,
                                 tool_name_list=script_args.tool_name_list,
                                 video_kwargs=video_kwargs,
                                 append_time_instruction=script_args.append_time_instruction) if script_args.eval_data_path else None
    if script_args.tool_name_list:
        video_interaction = VideoInteraction(model=None,
                                             processor=None,
                                             max_turns=script_args.max_interaction_turns,
                                             max_new_tokens_per_turn=script_args.max_completion_length_per_turn,
                                             use_vllm=True,
                                             avoid_mm_missing=True)
    else:
        video_interaction = None
    # model, processor = setup_model(model_args.model_name_or_path, training_args)
    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        environment=video_interaction,
        process_vision_fn=process_vision_info,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfigEnv, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
