import torch
import os
import tqdm
import glob
import multiprocessing
from functools import partial

from time_r1.utils.clip_service import SiglipClient
from time_r1.utils.qwen_vl_utils import fetch_video
from time_r1.utils.io import load_jsonl
import os

SIGLIP_URL = os.environ.get("SIGLIP_URL", "grpc://127.0.0.1:51000")
clip_model = SiglipClient(base_url=SIGLIP_URL)


def process_single_video(video_path):
    # ele = {
    #     "video": video_path,
    #     "fps": fps,
    #     "max_frames": max_frames,
    #     "total_pixels": max_frames * 1024 * 28 * 28,
    # }
    # video, sample_fps = fetch_video(ele, return_video_sample_fps=True)
    # print(video_path)
    try:
        video = torch.load(video_path + ".frame_cache")["frame_tensor"]
        features = clip_model.encode_images(video)
        print(features.shape, video.shape)
        # save features
        torch.save(features, video_path + ".feature_cache")
    except Exception as e:
        print(f'{e}, {video_path}')


def prepare_feature_cache(video_root, dataset_path=None, num_workers=8, overwrite=False):
    if dataset_path is not None:
        video_list = load_jsonl(dataset_path)
        video_list = [os.path.join(video_root, v["video"]) for v in video_list]
    else:
        video_list = glob.glob(os.path.join(video_root, "*.mp4"))

    if not video_list:
        print(f"No MP4 videos found in {video_root}")
        return
    if not overwrite:
        print("skipping videos that already have feature cache")
        num_total = len(video_list)
        video_list = [v for v in video_list if not os.path.exists(v + ".feature_cache")]
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
        list(tqdm.tqdm(pool.imap_unordered(process_single_video, video_list), total=len(video_list)))

    print("All videos processed.")


if __name__ == "__main__":
    import fire
    fire.Fire(prepare_feature_cache)
