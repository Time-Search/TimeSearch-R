from typing import List
import torch


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
