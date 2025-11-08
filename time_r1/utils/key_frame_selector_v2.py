import os
import copy
import torch
import numpy as np
from torch import nn
from transformers import SiglipProcessor, SiglipModel
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List
from PIL import Image
import warnings

from time_r1.utils.clip_service import SiglipClient
from time_r1.utils.utils import timer


DEFAULT_THETA = float(os.environ.get("DPP_THETA", -1))  # default use relevance scores without exponential
DEFAULT_K = int(os.environ.get("DPP_K", 8))


class DppTimeSelector(object):
    def __init__(self, k_selection=DEFAULT_K, theta=DEFAULT_THETA, clip_service_url: str = None):
        super().__init__()
        self.k_selection = k_selection
        self.vlm = SiglipClient(base_url=clip_service_url) if clip_service_url is not None else SiglipClient()
        self.theta = theta

    def dpp_fast_inference(self, kernel):
        N = kernel.shape[0]
        device = kernel.device
        cis = torch.zeros((self.k_selection, N), device=device)
        di2s = torch.diag(kernel)
        select_idx = torch.empty(self.k_selection, dtype=torch.long, device=device)
        for i in range(self.k_selection):
            j = torch.argmax(di2s)
            select_idx[i] = j
            eis = (kernel[j, :] - cis[:i, j] @ cis[:i, :]) / torch.sqrt(di2s[j])
            cis[i, :] = eis
            di2s -= torch.square(eis)
            di2s[j] = -float('inf')
        return select_idx

    def __call__(self, frames: List[Image], prompts: List[str], frame_embeddings: List[torch.Tensor] = None):
        try:
            if isinstance(prompts, str):
                prompts = [prompts]
            with timer("Embedding"):
                if frame_embeddings is None:
                    image_embeds = self.vlm.encode_images(frames)
                    text_embeds = self.vlm.encode_texts(prompts)
                else:
                    image_embeds = frame_embeddings.clone().cpu()
                    text_embeds = self.vlm.encode_texts(prompts)
            with timer("Select Frames"):
                with torch.no_grad():
                    if len(image_embeds) < self.k_selection:
                        print(f"Warning: Available frames ({len(image_embeds)}) less than requested frames ({self.k_selection})")
                        selected_idx = list(range(min(len(image_embeds), self.k_selection)))
                    else:
                        relevance_scores = text_embeds @ image_embeds.T # (1, N)
                        relevance_scores = relevance_scores.squeeze(0)
                        relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())
                        if self.theta >= 0:
                            alpha = self.theta / (2 * (1 - self.theta))     # https://arxiv.org/abs/1709.05135
                            relevance_scores = torch.exp(alpha * relevance_scores)
                        relevance_scores_diag = torch.diag(relevance_scores)
                        similarity_scores = image_embeds @ image_embeds.T # (N, N)
                        conditional_similarity_scores = relevance_scores_diag @ similarity_scores @ relevance_scores_diag
                        selected_idx = self.dpp_fast_inference(conditional_similarity_scores)
                        selected_idx = sorted(selected_idx.tolist())
            return selected_idx
        except ValueError as e:
            print(f"Frame Selection Error: {e}")
            return []


if __name__ == "__main__":
    from PIL import Image
    im1 = Image.open("tests/both_motion_blur.png")
    im2 = Image.open("tests/both.png")
    selector = DppTimeSelector(k_selection=1)
    selected_idx = selector([im1, im2], "blur dog image")
    print(selected_idx)
    