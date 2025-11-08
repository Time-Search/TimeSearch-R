import json
import os
import re
import math
import random
import time
import copy
import torch
import yaml
from collections import defaultdict
from typing import Dict
from time_r1.utils.qwen_vl_utils import process_vision_info, fetch_video, smart_resize, replace_vision_info_with_placeholder, IMAGE_FACTOR, FRAME_FACTOR
from time_r1.utils import rank0_print, rank_print, parse_dataset_yaml, load_jsonl, load_json
from time_r1.prompt import get_prompt_fn
from time_r1.utils.visualize_frames import tensor_to_pil
from time_r1.prompt.tool_use import get_tool_use_prompt
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from time_r1.utils.video_tools import video_tool_call
import numpy as np
from time_r1.utils.clip_service import SiglipClient
from time_r1.utils.tafr import construct_temporal_augmented_frames

TOTAL_PIXELS = 256 * 60 * 28 * 28
MIN_PIXELS = 16 * 28 * 28
MAX_PIXELS = 192 * 28 * 28
MAX_CACHE_FRAMES = 2048

DEFAULT_VIDEO_KWARGS = {
    "total_pixels": TOTAL_PIXELS,
    "min_pixels": MIN_PIXELS,
    "max_pixels": MAX_PIXELS,
}

MIN_CACHE_PIXELS = 32 * 28 * 28
MAX_CACHE_PIXELS = 256 * 28 * 28
TOTAL_CACHE_PIXELS = MAX_CACHE_FRAMES * MAX_CACHE_PIXELS
DEFAULT_CACHE_VIDEO_KWARGS = {
    "max_frames": MAX_CACHE_FRAMES,
    "fps": 2,
    "min_pixels": MIN_CACHE_PIXELS,
    "max_pixels": MAX_CACHE_PIXELS,
    "total_pixels": TOTAL_CACHE_PIXELS,
}

clip_model = SiglipClient()
DEFAULT_SYSTEM_PROMPT = "You are a helpful video assistant."


def load_video_with_decord(video_path, max_frames_num=7200, fps=1, force_sample=False):
    import decord
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = decord.VideoReader(video_path)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    actual_fps = fps
    if len(frame_idx) > max_frames_num or force_sample:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        actual_fps = max_frames_num / video_time
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    spare_frames = torch.from_numpy(spare_frames).permute(0, 3, 1, 2)
    return spare_frames, frame_time, actual_fps


class LazyVLDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str,
                 data_root: str,
                 prompt_template: str, 
                 tool_name_list: list[str]=[],
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 video_kwargs: dict = {},
                 cache_video_kwargs: dict = {},
                 append_time_instruction: bool = False,
                ):
        super(LazyVLDataset, self).__init__()
        self.data_root = data_root
        self.prompt_template = get_prompt_fn(prompt_template)
        if tool_name_list:
            self.tool_use_prompt = get_tool_use_prompt(tool_name_list)
        else:
            self.tool_use_prompt = ""
        self.system_prompt = f"{system_prompt}\n{self.tool_use_prompt}"
        rank0_print(f"System prompt: {self.system_prompt}")
        self.video_kwargs = copy.deepcopy(DEFAULT_VIDEO_KWARGS)
        self.video_kwargs.update(video_kwargs)
        self.cache_video_kwargs = copy.deepcopy(DEFAULT_CACHE_VIDEO_KWARGS)
        self.cache_video_kwargs.update(cache_video_kwargs)
        self.append_time_instruction = append_time_instruction
        rank0_print(f"Append time instruction: {self.append_time_instruction}")
        rank0_print(f"Video kwargs: {self.video_kwargs}")
        rank0_print(f"Cache video kwargs: {self.cache_video_kwargs}")
        if data_path.endswith(".yaml"):
            dataset_info = parse_dataset_yaml(data_path)
            self.list_data_dict = dataset_info["data_dict_list"]
            json_file_list = dataset_info["json_file_list"]
            rank0_print(f"Loaded {len(self.list_data_dict)} samples from {json_file_list}")
        elif data_path.endswith(".jsonl"):
            rank0_print(f"Loading {data_path}")
            self.list_data_dict = load_jsonl(data_path)
        elif data_path.endswith(".json"):
            self.list_data_dict = load_json(data_path)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        status = defaultdict(int)
        for data in self.list_data_dict:
            if "image" in data:
                status["image"] += 1
            elif "multi_video" in data:
                status["multi_video"] += 1
            elif "video" in data:
                status["video"] += 1
            else:
                status["unknow"] += 1
        rank0_print(f"Dataset modalities status: {status}")
        rank0_print(f"Dataset template: {self.prompt_template}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = copy.deepcopy(self.list_data_dict[idx])
        if 'video' in item:
            video_abs_path = os.path.join(self.data_root, item["video"])
            item["video"] = video_abs_path
        model_inputs = self.qwen_vl_preprocess_fn(item)
        item.update(model_inputs)
        return item

    def qwen_vl_preprocess_fn(self, item):
        """
        【任务无关】Qwen2.5-VL预处理, apply chat template, process_vision_info
        item: {
            "id": str,
            "video": str,
            "question": str,
            "target": list[float],
        }
        outputs:
            messages: list[dict],
            multimodal_cache: dict, for video interaction
        """
        cache_video_ele = {
            "type": "video", 
            "video": item["video"],
        }
        cache_video_ele.update(self.cache_video_kwargs)
        if os.path.exists(item["video"] + ".frame_cache"):
            # Limit max frames and per-frame tokens;
            # Maybe you should use fetch_video to do this
            try:
                cache_data = torch.load(item["video"] + ".frame_cache")
                cache_video_frames = cache_data["frame_tensor"]
                cache_video_sample_fps = cache_data["fps"]
                num_frames = len(cache_video_frames)
                if num_frames > MAX_CACHE_FRAMES:
                    sample_idx = torch.linspace(0, num_frames - 1, MAX_CACHE_FRAMES).round().long()
                    cache_video_frames = cache_video_frames[sample_idx]
                    cache_video_sample_fps = MAX_CACHE_FRAMES / num_frames * cache_video_sample_fps
                # smart resize
                nframes, _, height, width = cache_video_frames.shape
                min_pixels = cache_video_ele.get("min_pixels", MIN_CACHE_PIXELS)
                total_pixels = cache_video_ele.get("total_pixels", TOTAL_CACHE_PIXELS)
                max_pixels = max(min(MAX_CACHE_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
                max_pixels_supposed = cache_video_ele.get("max_pixels", max_pixels)
                if max_pixels_supposed > max_pixels:
                    print(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
                max_pixels = min(max_pixels_supposed, max_pixels)
                resized_height, resized_width = smart_resize(height,
                    width,
                    factor=IMAGE_FACTOR,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                cache_video_frames = transforms.functional.resize(
                    cache_video_frames,
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).float()
                
                # print(f"cache_video_frames: {cache_video_frames.shape}")
                # print(f"Loaded frame cache from {item['video'] + '.frame_cache'}")
            except Exception as e:
                print(f"Error loading frame cache: {e}")
                cache_video_frames, cache_video_sample_fps = fetch_video(cache_video_ele, return_video_sample_fps=True)
        else:
            cache_video_frames, cache_video_sample_fps = fetch_video(cache_video_ele, return_video_sample_fps=True)
        if os.path.exists(item["video"] + ".feature_cache"):
            try:
                cache_video_features = torch.load(item["video"] + ".feature_cache")
                num_frames = len(cache_video_features)
                if num_frames > MAX_CACHE_FRAMES:
                    sample_idx = torch.linspace(0, num_frames - 1, MAX_CACHE_FRAMES).round().long()
                    cache_video_features = cache_video_features[sample_idx]
                # print(f"cache_video_features: {cache_video_features.shape}")
                # print(f"Loaded feature cache from {item['video'] + '.feature_cache'}")
            except Exception as e:
                print(f"Error loading feature cache: {e}")
                cache_video_features = clip_model.encode_images(cache_video_frames)
                torch.save(cache_video_features, item["video"] + ".feature_cache")
        else:
            cache_video_features = clip_model.encode_images(cache_video_frames)
            torch.save(cache_video_features, item["video"] + ".feature_cache")
        multimodal_cache = {
            "video": cache_video_frames,
            "embedding": cache_video_features,
            "fps": cache_video_sample_fps,
        }
        preview_video_ele = {
            "type": "video", 
            "video": item["video"],
        }
        preview_video_ele.update(self.video_kwargs)
        preview_video_frames, preview_video_sample_fps = fetch_video(preview_video_ele, return_video_sample_fps=True)
        # print(f"====>preview_video_frames: {preview_video_frames.shape}", f"cache_video_frames: {cache_video_frames.shape}")
        if "duration" not in item:
            item["duration"] = len(cache_video_frames) / cache_video_sample_fps
        multimodal_cache["duration"] = item["duration"]
        multimodal_cache["question"] = item["question"]

        if self.append_time_instruction:
            timestamps = [frame_id / preview_video_sample_fps for frame_id in range(len(preview_video_frames))]
            prompt_text = self.prompt_template({
                "question": item["question"],
                "timestamps": [frame_id / preview_video_sample_fps for frame_id in range(len(preview_video_frames))],
                "duration": item["duration"],
            })

            frames_and_prompt = construct_temporal_augmented_frames(timestamps, preview_video_frames)
            frames_and_prompt.append({"type": "text", "text": prompt_text})

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt}
                    ],
                },
                {
                    "role": "user", 
                    "content": frames_and_prompt,
                },
            ]
        else:
            prompt_text = self.prompt_template(item)
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt}
                    ],
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "video", 
                            "video": preview_video_frames,
                            "fps": preview_video_sample_fps,
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                    ]
                },
            ]
        return {
            "messages": messages,
            "multimodal_cache": multimodal_cache,
        }

if __name__ == "__main__":
    data_path = "workdir/datasets/videomme/bes_videomme_withouttitle_test.json"
    data_root = "dataset/evaluation/llava_next/videomme/data/"
    from tqdm import tqdm
    from time_r1.prompt import get_prompt_fn
    dataset = LazyVLDataset(data_path, data_root, "v6", tool_name_list=["seek_video_frames"])
    item = dataset[0]
    print(item)
    print(replace_vision_info_with_placeholder(item["messages"]))
    # for k in item:
    #     if k == "messages":
    #         print(replace_vision_info_with_placeholder(item["messages"]))
    #     else:
    #         print(f"{k}: {item[k]}")
