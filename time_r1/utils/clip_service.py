import os
import copy
import torch
import numpy as np
from torch import nn
import time
from typing import List
from PIL import Image
from time_r1.utils.io import encode_base64
import clip_client

from rich.progress import Progress
def stop(self) -> None:
    """Stop the progress display."""
    self.live.stop()
    if not self.console.is_interactive and not self.console.is_jupyter:
        # self.console.print()
        pass
Progress.stop = stop

# TODO: Fix this
SIGLIP_URL = os.environ.get("SIGLIP_URL", "grpc://127.0.0.1:51000")


class SiglipClient():
    def __init__(self, base_url: str = SIGLIP_URL, num_workers: int = 16):
        self.client = clip_client.Client(base_url)

    def encode_images(self, frames):
        if isinstance(frames[0], torch.Tensor):
            frames = [tensor_to_pil(frame) for frame in frames]
        if isinstance(frames[0], np.ndarray):
            frames = [Image.fromarray(frame.transpose(1, 2, 0)) for frame in frames]
        frames = [encode_base64(frame) for frame in frames]
        image_embeds = self.client.encode(frames, batch_size=32, show_progress=False)
        image_embeds = torch.tensor(image_embeds)
        # normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds
    
    def encode_texts(self, prompts):
        text_embeds = self.client.encode(prompts)
        text_embeds = torch.tensor(text_embeds)
        # normalize
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds.mean(dim=0, keepdim=True)
        return text_embeds


def tensor_to_pil(frame_tensor: torch.Tensor) -> Image.Image:
    """(C, H, W) float32 0–255  →  PIL.Image(RGB)."""
    frame_uint8 = frame_tensor.clamp(0, 255).byte().cpu().numpy()
    if frame_uint8.shape[0] == 3:          # C,H,W → H,W,C
        frame_uint8 = frame_uint8.transpose(1, 2, 0)
    return Image.fromarray(frame_uint8, mode="RGB")