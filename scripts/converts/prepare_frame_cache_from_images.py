from time_r1.utils.qwen_vl_utils import floor_by_factor, FRAME_FACTOR, smart_resize
import decord
import torch
import os
import tqdm
import glob
import multiprocessing
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from functools import partial
from time_r1.utils.io import load_jsonl
from PIL import Image
import numpy as np


def load_images_from_pathlist(filenames):
    images = []
    for filename in filenames:
        img = Image.open(filename)
        img_array = np.array(img)   # shape: (H, W, 3)
        images.append(img_array)
    return np.array(images)


def load_video_frames(video_path, frame_fps=1):
    filenames = sorted(os.listdir(video_path))
    image_paths = [os.path.join(video_path, filename) for filename in filenames]
    images = load_images_from_pathlist(image_paths)
    return images


def get_video_tensor(video_path, target_fps=1, image_factor = 28, min_pixels = 28 * 28 * 128, max_pixels = 28 * 28 * 256):
    """
    将视频以固定帧率提前抽帧、解码保存为tensor，用于后续训练
    """
    images = load_video_frames(video_path)
    frame_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)  # Convert to TCHW format
    height, width = frame_tensor.shape[2], frame_tensor.shape[3]
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    frame_tensor = transforms.functional.resize(
        frame_tensor,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    frame_cache = {
        "frame_tensor": frame_tensor,
        "fps": target_fps,
    }
    return frame_cache


def process_single_video(video_path, target_fps=1, image_factor = 28, min_pixels = 28 * 28 * 128, max_pixels = 28 * 28 * 256):
    """Helper function to process and save frame cache for a single video."""
    print(f"Processing {video_path}...")
    try:
        frame_cache = get_video_tensor(video_path, target_fps, image_factor, min_pixels, max_pixels)
        torch.save(frame_cache, video_path + ".frame_cache")
        print(f"Successfully saved frame cache for {video_path}")
    except Exception as e:
        print(f"Error processing {video_path}: {e}")


def prepare_frame_cache(video_root, dataset_path=None, num_workers=8, target_fps=1, overwrite=False, image_factor = 28, min_pixels = 28 * 28 * 128, max_pixels = 28 * 28 * 256):
    if dataset_path is not None:
        video_list = load_jsonl(dataset_path)
        video_list = [os.path.join(video_root, v["video"]) for v in video_list]
    else:
        video_list = glob.glob(os.path.join(video_root, "*"))
    if not video_list:
        print(f"No MP4 videos found in {video_root}")
        return
    # remove videos that already have frame cache
    if not overwrite:
        print("skipping videos that already have frame cache")
        num_total = len(video_list)
        video_list = [v for v in video_list if not os.path.exists(v + ".frame_cache")]
        num_skipped = num_total - len(video_list)
        print(f"skipped {num_skipped} videos")

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()  # Default to using all available CPU cores
    
    print(f"Found {len(video_list)} videos. Starting processing with {num_workers} workers...")

    # Use a multiprocessing Pool to process videos in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Using tqdm with pool.imap_unordered for progress bar and efficient iteration
        # We wrap process_single_video if it needs more arguments or if we want to handle results
        # For this case, process_single_video only takes video_path
        func = partial(process_single_video, target_fps=target_fps, image_factor = image_factor, min_pixels = min_pixels, max_pixels = max_pixels)
        list(tqdm.tqdm(pool.imap_unordered(func, video_list), total=len(video_list)))

    print("All videos processed.")


if __name__ == "__main__":
    import fire
    fire.Fire(prepare_frame_cache)
