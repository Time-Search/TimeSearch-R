#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Calculation Script

This script reads a JSON file containing video analysis results, extracts ground truth and predicted frame timestamps,
and calculates Temporal PRF (Precision, Recall, F1) and SSIM (Structural Similarity Index) metrics.

Usage:
    python calculate_metrics.py --search_result_path <path_to_json> --fps <video_fps>

Dependencies:
    - numpy
    - opencv-python
    - scikit-image
    - tqdm
    - torch
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Tuple, Any, Dict

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utility Functions (SSIM Calculation)
# -----------------------------------------------------------------------------
def gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """Creates a 1D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g

def create_window(window_size: int, channel: int) -> torch.Tensor:
    """Creates a 2D Gaussian kernel window."""
    kernel_1d = gaussian_kernel(window_size, sigma=1.5).unsqueeze(1)
    window_2d = kernel_1d @ kernel_1d.T
    window = window_2d.expand(channel, 1, window_size, window_size)
    return window

def ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
               C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """Calculates the SSIM between two images using PyTorch."""
    channel = img1.size(0)
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1.unsqueeze(0), window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2.unsqueeze(0), window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1.unsqueeze(0) * img1.unsqueeze(0), window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def pairwise_ssim(gt_frames: List[np.ndarray], pred_frames: List[np.ndarray]) -> np.ndarray:
    """
    Calculates pairwise SSIM between two lists of images.
    Returns a numpy array of shape (num_gt, num_pred) containing SSIM scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert images to torch tensors and scale to [0, 1]
    gt_tensors = [torch.tensor(frame, dtype=torch.float32, device=device) / 255.0 for frame in gt_frames]
    pred_tensors = [torch.tensor(frame, dtype=torch.float32, device=device) / 255.0 for frame in pred_frames]

    ssim_results = np.zeros((len(gt_tensors), len(pred_tensors)))
    for i in range(len(gt_tensors)):
        for j in range(len(pred_tensors)):
            ssim_score = ssim_torch(gt_tensors[i], pred_tensors[j])
            ssim_results[i, j] = ssim_score.item()
    return ssim_results

# -----------------------------------------------------------------------------
# Video I/O Functions
# -----------------------------------------------------------------------------
def load_video_fps(video_path: str) -> float:
    """
    Get the frames per second (FPS) of a video.
    
    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        raise ValueError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    logger.debug(f"Video FPS for {video_path}: {fps}")
    return fps

def extract_frames(video_path: str, frame_indices: List[int]) -> List[np.ndarray]:
    """
    Extract specified frames from a video.
    
    Args:
        video_path: Path to the video file.
        frame_indices: List of frame indices to extract.
    
    Returns:
        List of extracted frames (RGB format). If a frame is not read successfully,
        an empty numpy array is returned in its place.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        raise ValueError(f"Cannot open video file: {video_path}")

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            logger.debug(f"Extracted frame {idx} from {video_path}")
        else:
            frames.append(np.array([]))
            logger.warning(f"Failed to extract frame {idx} from {video_path}")
    cap.release()
    return frames

# -----------------------------------------------------------------------------
# JSON I/O Functions
# -----------------------------------------------------------------------------
def load_json_file(file_path: str) -> Any:
    """
    Load a JSON file.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not os.path.exists(file_path):
        logger.error(f"JSON file does not exist: {file_path}")
        raise FileNotFoundError(f"JSON file does not exist: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded JSON file: {file_path}")
            return data
    except json.JSONDecodeError as e:
        with open(file_path, 'r', encoding='utf-8') as f:
            # maybe this is a jsonl file, load like jsonl
            data = [json.loads(line) for line in f]
            logger.info(f"Successfully loaded JSONL file: {file_path}")
            return data
    except Exception as e:
        logger.error(f"Error decoding JSON file: {file_path}")
        raise e

def save_json_file(data: Any, file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"Metrics results saved to: {file_path}")

# -----------------------------------------------------------------------------
# Metrics Calculation Functions
# -----------------------------------------------------------------------------
def calculate_prf(list_gt: List[np.ndarray], list_pred: List[np.ndarray], threshold: int = 5) -> Tuple[float, float, float]:
    """
    Calculate average Temporal Precision, Recall and F1 Score based on frame distances.
    """
    precision_list, recall_list, f1_list = [], [], []
    for gt_array, pred_array in zip(list_gt, list_pred):
        if gt_array.size == 0 or pred_array.size == 0:
            continue
        # Compute the minimum absolute differences between predicted and ground truth frame numbers.
        distances_gt_to_pred = np.min(np.abs(gt_array[:, np.newaxis] - pred_array), axis=1)
        distances_pred_to_gt = np.min(np.abs(pred_array[:, np.newaxis] - gt_array), axis=1)

        covered_gt = np.sum(distances_gt_to_pred <= threshold)
        covered_pred = np.sum(distances_pred_to_gt <= threshold)
        total_gt_frames = len(gt_array)
        total_pred_frames = len(pred_array)

        precision = covered_pred / total_pred_frames if total_pred_frames > 0 else 0.0
        recall = covered_gt / total_gt_frames if total_gt_frames > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    avg_precision = np.mean(precision_list) if precision_list else 0.0
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    avg_f1 = np.mean(f1_list) if f1_list else 0.0
    return avg_precision, avg_recall, avg_f1

def calculate_ssim_scores(list_gt_images: List[List[np.ndarray]], list_pred_images: List[List[np.ndarray]]) -> List[Tuple[float, float]]:
    """
    Calculate SSIM Precision and Recall for each video entry.
    
    For each video:
      - SSIM Precision is computed as the mean of the maximum SSIM values for each predicted frame (across all ground truth frames).
      - SSIM Recall is computed as the mean of the maximum SSIM values for each ground truth frame (across all predicted frames).
    """
    ssim_list = []
    for gt_images, pred_images in zip(list_gt_images, list_pred_images):
        if not gt_images or not pred_images:
            continue

        # Filter out empty images
        paired_gt = [img for img in gt_images if img.size > 0]
        paired_pred = [img for img in pred_images if img.size > 0]
        if not paired_gt or not paired_pred:
            continue

        ssim_matrix = pairwise_ssim(paired_gt, paired_pred)
        ssim_precision = np.mean(np.max(ssim_matrix, axis=0))
        ssim_recall = np.mean(np.max(ssim_matrix, axis=1))
        ssim_list.append((ssim_precision, ssim_recall))
    return ssim_list

def calculate_annd(list_gt: List[np.ndarray], list_pred: List[np.ndarray]) -> List[Tuple[float, float]]:
    """
    Calculate the Average Nearest Neighbor Distance (ANND) for each video entry.
    Returns a list of (precision, recall) tuples.
    """
    annd_list = []
    for gt_array, pred_array in zip(list_gt, list_pred):
        if gt_array.size == 0 or pred_array.size == 0:
            continue

        distances_gt_to_pred = np.min(np.abs(gt_array[:, np.newaxis] - pred_array), axis=1)
        distances_pred_to_gt = np.min(np.abs(pred_array[:, np.newaxis] - gt_array), axis=1)
        annd_precision = np.mean(distances_pred_to_gt)
        annd_recall = np.mean(distances_gt_to_pred)
        annd_list.append((annd_precision, annd_recall))
    return annd_list

def extract_frame_timestamps(item: dict) -> List[float]:
    """
    Extract the frame timestamps from the multi-turn dialogs
    """
    timestamps_multi_turn = []
    for turn in item['prediction']:
        timestamps_single_turn = []
        if turn['role'] == 'tool':
            for content in turn['content']:
                if content['type'] == 'text' and not content['text'].startswith('Here are selected frames. They are located at'):
                    try:
                        timestamps_single_turn.append(float(content['text'].split('s')[0]))
                    except ValueError:
                        logger.warning(f"Invalid timestamp: {content['text']}")
                        continue
            timestamps_multi_turn.extend(timestamps_single_turn)
    return list(set(timestamps_multi_turn))

def extract_metrics_data(result_data: List[Dict[str, Any]]) -> Tuple[List[str], List[List[float]], List[List[float]]]:
    """
    Process the raw JSON data:
      - Extract video paths.
      - Convert ground truth frame indices to seconds.
      - Extract predicted keyframe timestamps (assumed to be in seconds).
    """
    video_paths = []
    searching_sec = []
    gt_seconds = []
    searching_indexes = []
    gt_indexes = []

    for idx, item in enumerate(result_data):
        try:
            if "video_path" in item:
                video_path = item['video_path']
            else:
                video_path = item['video']
            if "keyframe_timestamps" in item:
                frame_timestamps = item['keyframe_timestamps']  # predicted timestamps (seconds)
            else:
                frame_timestamps = extract_frame_timestamps(item)
            gt_frame_indexes = item['gt_frame_index']
        except KeyError as e:
            logger.error(f"Entry {idx} missing required field: {e}")
            continue

        try:
            fps_video = load_video_fps(video_path)
        except ValueError as e:
            logger.error(f"Failed to get FPS for video {video_path}: {e}")
            continue

        gt_sec = [position / fps_video for position in gt_frame_indexes]
        searching_sec.append(frame_timestamps)
        searching_indexes.append([f * fps_video for f in frame_timestamps])
        gt_indexes.append(gt_frame_indexes)

        gt_seconds.append(gt_sec)
        video_paths.append(video_path)

    return video_paths, searching_sec, gt_seconds, searching_indexes, gt_indexes



def calculate_metrics(result_data: List[Dict[str, Any]], 
                      fps: float = 30, threshold: int = 5,
                      max_workers: int = 4) -> Dict[str, Any]:
    """
    Calculate Temporal PRF and SSIM metrics for all video entries.
    """
    video_paths, searching_sec, gt_seconds, searching_indexes, gt_indexes = extract_metrics_data(result_data)
    list_gt = []
    list_pred = []
    list_gt_images = []
    list_pred_images = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, (video_path, pred_timestamps, gt_positions, pred_indexes_one, gt_indexes_one) in enumerate(zip(video_paths, searching_sec, gt_seconds, searching_indexes, gt_indexes)):
            try:
                list_gt.append(np.array(gt_positions)) # temporal-based gt, in seconds
                list_pred.append(np.array(pred_timestamps)) # temporal-based pred, in seconds
                gt_frame_nums = np.array(gt_indexes_one, dtype=int)
                pred_frame_nums = np.array([int(ts * fps) for ts in pred_indexes_one], dtype=int)
                combined_frames = gt_frame_nums.tolist() + pred_frame_nums.tolist()
                future = executor.submit(extract_frames, video_path, combined_frames)
                future_to_idx[future] = idx
            except Exception as e:
                logger.error(f"Error processing video {video_path}: {e}")
                list_gt.append(np.array([]))
                list_pred.append(np.array([]))
                list_gt_images.append([])
                list_pred_images.append([])
        logger.info("Calculating Temporal PRF Scores...")
        avg_precision, avg_recall, avg_f1 = calculate_prf(list_gt, list_pred, threshold=threshold)
        logger.info(f"Average Precision: {avg_precision:.4f}")
        logger.info(f"Average Recall: {avg_recall:.4f}")
        logger.info(f"Average F1 Score: {avg_f1:.4f}")
        logger.info(f"Average Number of GT Frames: {np.mean([len(gt) for gt in list_gt])}")
        logger.info(f"Average Number of Pred Frames: {np.mean([len(pred) for pred in list_pred])}")
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Extracting Frames"):
            idx = future_to_idx[future]
            try:
                frames = future.result()
                if not frames:
                    logger.warning(f"No frames extracted for video index {idx}")
                    list_gt_images.append([])
                    list_pred_images.append([])
                    continue

                item = result_data[idx]
                gt_num = len(item['gt_frame_index'])
                # pred_num = len(item[frame_index_key])
                gt_images = frames[:gt_num]
                pred_images = frames[gt_num:]

                list_gt_images.append(gt_images)
                list_pred_images.append(pred_images)
            except Exception as e:
                logger.error(f"Error extracting frames for video index {idx}: {e}")
                list_gt_images.append([])
                list_pred_images.append([])

    logger.info("Calculating SSIM Scores...")
    ssim_scores = calculate_ssim_scores(list_gt_images, list_pred_images)
    if ssim_scores:
        avg_ssim_precision = np.mean([s[0] for s in ssim_scores])
        avg_ssim_recall = np.mean([s[1] for s in ssim_scores])
        logger.info(f"Average SSIM Precision: {avg_ssim_precision:.4f}")
        logger.info(f"Average SSIM Recall: {avg_ssim_recall:.4f}")
        if avg_ssim_precision + avg_ssim_recall > 0:
            ssim_f1 = 2 * avg_ssim_precision * avg_ssim_recall / (avg_ssim_precision + avg_ssim_recall)
        else:
            ssim_f1 = 0.0
        logger.info(f"Average SSIM F1 Score: {ssim_f1:.4f}")
    else:
        avg_ssim_precision = avg_ssim_recall = ssim_f1 = 0.0
        logger.warning("No SSIM scores were calculated.")

    metrics = {
        "Average Temporal Precision": avg_precision,
        "Average Temporal Recall": avg_recall,
        "Average Temporal F1 Score": avg_f1,
        "Average SSIM Precision": avg_ssim_precision,
        "Average SSIM Recall": avg_ssim_recall,
        "Average SSIM F1 Score": ssim_f1
    }
    return metrics

# -----------------------------------------------------------------------------
# Argument Parsing and Main Function
# -----------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate PRF and SSIM metrics from video analysis results."
    )
    parser.add_argument('--search_result_path', type=str,
                        default="./results/frame_search/yolo-World_TStar_LVHaystack_tiny.json",
                        help='Path to the input JSON file containing video analysis results.') 
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frames per second (FPS) of the raw sampling.')
    parser.add_argument('--threshold', type=int, default=5,
                        help='Distance threshold for PRF calculation.')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of threads for parallel processing.')

    return parser.parse_args()


def main():
    """Main function to execute metrics calculation."""
    args = parse_arguments()

    # Load JSON data.
    try:
        result_data = load_json_file(args.search_result_path)
    except Exception as e:
        logger.error(f"Failed to load JSON data: {e}")
        sys.exit(1)


    # Validate JSON structure.
    required_fields = {'gt_frame_index'}
    
    valid_data = [item for item in result_data if required_fields.issubset(item.keys())]
    if not valid_data:
        logger.error("No valid entries found in JSON data.")
        sys.exit(1)

    # Calculate metrics.
    metrics = calculate_metrics(
        result_data=valid_data,
        fps=args.fps,
        threshold=args.threshold,
        max_workers=args.max_workers,
    )

    # Save metrics to output file.
    output_root = "./results/lvhaystack_score"
    search_result_file_name = os.path.basename(args.search_result_path)
    os.makedirs(output_root, exist_ok=True)
    output_file = os.path.join(
        output_root,
        search_result_file_name.replace(".json", "lvhaystack_score.json")
    )
    save_json_file(metrics, output_file)
    logger.info("Metrics calculation completed successfully.")

if __name__ == "__main__":
    main()
