import math
import numpy as np
from pathlib import Path
from typing import List, Union, Optional

import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch
from PIL import Image


def tensor_to_pil(frame_tensor: torch.Tensor) -> Image.Image:
    """(C, H, W) float32 0–255  →  PIL.Image(RGB)."""
    frame_uint8 = frame_tensor.clamp(0, 255).byte().cpu().numpy()
    if frame_uint8.shape[0] == 3:          # C,H,W → H,W,C
        frame_uint8 = frame_uint8.transpose(1, 2, 0)
    return Image.fromarray(frame_uint8, mode="RGB")


def visualize_video_frames(
    video_frames: Union[torch.Tensor, List[Image.Image]],
    max_frames: int = 16,
    cols: int = 4,
    figsize: tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    save_grid: Optional[str] = None,
    save_gif: Optional[str] = None,
    gif_fps: int = 4,
) -> None:
    """
    可视化或保存视频帧网格/动图.

    Parameters
    ----------
    video_frames : torch.Tensor | list[PIL.Image]
        - Tensor: shape (T, C, H, W), float32, 0–255.
        - List : 每帧 PIL.Image (RGB) 对象.
    max_frames : int
        最多展示/保存多少帧（自动等间隔采样）。
    cols : int
        网格列数。
    figsize : tuple
        matplotlib 图像大小。
    title : str | None
        网格标题。
    save_grid : str | None
        若给出，则把网格保存为 PNG/JPEG 等。（文件后缀决定格式）
    save_gif : str | None
        若给出，则把所选帧顺序保存为 GIF。
    gif_fps : int
        保存 GIF 时的帧率。
    """
    # ---------- 1. 标准化为 PIL 列表 ----------
    if isinstance(video_frames, torch.Tensor):
        pil_frames = [tensor_to_pil(f) for f in video_frames]
    else:
        pil_frames = list(video_frames)

    if len(pil_frames) == 0:
        raise ValueError("video_frames 不能为空。")

    # 采样
    step = max(1, len(pil_frames) // max_frames)
    pil_frames = pil_frames[::step][:max_frames]

    # ---------- 2. 绘制网格 ----------
    rows = math.ceil(len(pil_frames) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for i, frame in enumerate(pil_frames):
        r, c = divmod(i, cols)
        axes[r][c].imshow(frame)
        axes[r][c].set_title(f"Frame {i * step}")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    if save_grid:
        Path(save_grid).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_grid, dpi=200)
        print(f"[INFO] Grid image saved to {save_grid}")
    plt.show()

    # ---------- 3. 可选保存 GIF ----------
    if save_gif:
        Path(save_gif).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(
            save_gif,
            [np.asarray(f) for f in pil_frames],
            fps=gif_fps,
        )
        print(f"[INFO] GIF saved to {save_gif}")
