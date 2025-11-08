import math
import mimetypes
from PIL import Image
from typing import Union, Tuple, List, Dict
from qwen_agent.tools.base import BaseTool, register_tool, TOOL_REGISTRY
from time_r1.utils.key_frame_selector_v2 import DppTimeSelector
from time_r1.utils.utils import rank_debug
from time_r1.prompt import get_prompt_fn
import numpy as np
import torch
import os


MAX_NUM_KEY_FRAMES = int(os.environ.get("MAX_NUM_KEY_FRAMES", 8))
FORCE_NUM_KEY_FRAMES = int(os.environ.get("FORCE_NUM_KEY_FRAMES", 0))
FORCE_UNIFORM_SAMPLING = int(os.environ.get("FORCE_UNIFORM_SAMPLING", 0))
PROMPT_FN = get_prompt_fn("tool_response")


def construct_temporal_augmented_frames(
        timestamps: List[float],
        frames: torch.Tensor,
    ):
    """
    Return the content of video frames.
    """
    content = []
    for t, frame in zip(timestamps, frames):
        content.append({"type": "text", "text": f"{t:.1f}s"})
        content.append({"type": "image", "image": frame})
    return content


@register_tool("seek_video_frames", allow_overwrite=True)
class VideoFrameSeeker(BaseTool):
    description = (
        "Search and select video frames according to textual query and temporal window. "
        "Time is in seconds."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The query is used to describe the object, scene, or event of interest in the video thoroughly and clearly. "
                ),
            },
            "start_time": {
                "type": "number",
                "description": (
                    "Start time of the segment of interest. "
                ),
            },
            "end_time": {
                "type": "number",
                "description": (
                    "End time of the segment of interest. "
                ),
            },
            "num_frames": {
                "type": "integer",
                "description": (
                    f"Number of frames to sample (maximum {MAX_NUM_KEY_FRAMES}). Default is {MAX_NUM_KEY_FRAMES}."
                ),
            },
        },
        "required": ["query"],
    }
    def __init__(self, cfg=None):
        super().__init__(cfg)
    
    def cast(self, params):
        if "start_time" in params and params["start_time"] is not None:
            try:
                params["start_time"] = float(params["start_time"])
            except Exception:
                print(f"Failed to cast start_time: {params['start_time']}, set to 0.")
                params["start_time"] = 0
        else:
            params["start_time"] = 0
        if "end_time" in params and params["end_time"] is not None:
            try:
                params["end_time"] = float(params["end_time"])
            except Exception:
                print(f"Failed to cast end_time: {params['end_time']}, set to 0xffff.")
                params["end_time"] = 0xffff
        else:
            params["end_time"] = 0xffff
        if "num_frames" in params and params["num_frames"] is not None:
            try:
                params["num_frames"] = int(params["num_frames"])
            except Exception:
                print(f"Failed to cast num_frames: {params['num_frames']}, set to {MAX_NUM_KEY_FRAMES}.")
                params["num_frames"] = MAX_NUM_KEY_FRAMES
        else:
            params["num_frames"] = MAX_NUM_KEY_FRAMES
        if params["num_frames"] < 1:
            params["num_frames"] = 1
        elif params["num_frames"] > MAX_NUM_KEY_FRAMES:
            params["num_frames"] = MAX_NUM_KEY_FRAMES
        if "query" in params and params["query"] is not None:
            params["query"] = str(params["query"])
            # 模型可能会在query中给出本地路径，需要处理防止hack搜索引擎
            if mimetypes.guess_type(params["query"])[0]:
                params["query"] = params["query"] + "  \t."
        else:
            params["query"] = ""     # 默认查询任何内容
        return params

    def call(self, params: Union[str, dict], multimodal_cache: Dict, **kwargs):
        params = self.cast(params)
        params = self._verify_json_format_args(params)
        return self.seek_video_frames(params["query"], params["start_time"], params["end_time"], params["num_frames"], multimodal_cache=multimodal_cache)

    def seek_video_frames(self, query: str, start_time: int, end_time: int, num_frames: int, multimodal_cache: Dict) -> List[Dict[str, int]]:
        fps = multimodal_cache["fps"]
        frames = multimodal_cache["video"]
        frame_embeddings = multimodal_cache["embedding"]
        start_frame_idx = max(math.floor(start_time * fps), 0)
        end_frame_idx = min(math.ceil(end_time * fps), len(frames))
        frames_to_select = frames[start_frame_idx:end_frame_idx+1]
        assert len(frames_to_select) > 0, f"The number of frames to select could not be 0. query: {query}, start_time: {start_time}, end_time: {end_time}, num_frames: {num_frames}"
        frame_embeddings_to_select = frame_embeddings[start_frame_idx:end_frame_idx+1]
        # 1) 确保 num_frames 不超过切片长度，且至少取 1 帧
        num_frames = max(min(num_frames, len(frames_to_select), MAX_NUM_KEY_FRAMES), 1)
        if FORCE_NUM_KEY_FRAMES > 0:
            num_frames = FORCE_NUM_KEY_FRAMES
        if query and not FORCE_UNIFORM_SAMPLING:
            frame_selector = DppTimeSelector(k_selection=num_frames)
            # 2) 先得到局部索引，再转成全局索引
            local_idx = frame_selector.__call__(frames_to_select, [query], frame_embeddings=frame_embeddings_to_select)
            if len(local_idx) == 0:
                # dump
                print(f"Degraded case: frames selected: {query}, start_time: {start_time}, end_time: {end_time}, num_frames: {num_frames}")
                if len(frames_to_select) <= num_frames:
                    local_idx = list(range(len(frames_to_select)))
                else:
                    local_idx = np.round(np.linspace(0, len(frames_to_select) - 1, num_frames)).astype(int).tolist()
        else:
            # 从frames_to_select中均匀采样num_frames帧
            if len(frames_to_select) <= num_frames:
                local_idx = list(range(len(frames_to_select)))
            else:
                local_idx = np.round(np.linspace(0, len(frames_to_select) - 1, num_frames)).astype(int).tolist()
        # assert len(local_idx) > 0, f"The number of selected frames could not be 0. query: {query}, start_time: {start_time}, end_time: {end_time}, num_frames: {num_frames}"
        # assert max(local_idx) < len(frames_to_select), f"Index out of bounds: max(local_idx)={max(local_idx)}, len(frames_to_select)={len(frames_to_select)}"
        # assert min(local_idx) >= 0, f"Negative index detected: min(local_idx)={min(local_idx)}"
        global_idx = [i + start_frame_idx for i in local_idx]
        # 3) 用局部索引取图像，用全局索引输出 id 和时间戳
        # selected_frames = frames_to_select[local_idx]
        local_idx = torch.tensor(local_idx, device=frames_to_select.device)
        # 防止local idx超出范围
        local_idx = torch.clamp(local_idx, 0, len(frames_to_select) - 1)
        selected_frames = frames_to_select.index_select(0, local_idx)
        timestamps = [gid / fps for gid in global_idx]
        question = multimodal_cache["question"]
        duration = multimodal_cache["duration"]
        response_content = construct_temporal_augmented_frames(timestamps, selected_frames)
        timestamps_str = ",".join([f"{t:.1f}s" for t in timestamps])
        response_content.append({"type": "text", "text": PROMPT_FN({"timestamps": timestamps_str, "question": question, "duration": duration})})
        return response_content


def get_video_tool_by_name(fn_name: str):
    tool_cls = TOOL_REGISTRY.get(fn_name)
    if tool_cls is None:
        raise ValueError(f"Unknown function name: {fn_name}")
    return tool_cls()


def video_tool_call(
        params: Dict, 
        multimodal_cache: Dict,
    ):
    """
    params: LLM predicted tool call parameters
    multimodal_cache: cache
    """
    func = params.get("function", {})
    fn_name = func.get("name", "unknown")
    fn_args = func.get("arguments", {})
    try:
        tool_response = get_video_tool_by_name(fn_name).call(
            fn_args, multimodal_cache
        )
        message = {
            "role": "tool",
            "name": fn_name,
            "arguments": fn_args,
            "content": tool_response
        }
        return message
    except Exception as e:
        print(f"Failed to call tool function: {fn_name=}, {fn_args=}, got err {e}, duration: {multimodal_cache.get('duration', 0)}")
        return None


if __name__ == "__main__":
    from time_r1.utils.qwen_vl_utils import fetch_video
    
    frames = fetch_video({"video": "workdir/datasets/Charades/videos/0A8CF.mp4"})
    print(frames.shape)

    multimodal_cache = {
        "video": frames,
        "embedding": torch.randn(frames.shape[0], 1024),
        "fps": 1,
        "question": "What is the man doing?",
        "duration": 100,
    }

    resp = video_tool_call({
        "function": {
            "name": "seek_video_frames",
            "arguments": {
                "query": "",
            },
        },
    }, multimodal_cache)
    print(resp)
