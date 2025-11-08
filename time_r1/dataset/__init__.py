from .lazy_dataset import LazyVLDataset
from .lazy_dataset_baseline import LazyVLDatasetBaseline
from .lazy_dataset_sft import LazyVLDatasetSFT
from .lazy_dataset_post_frame_selection import LazyVLDatasetPostFrameSelection

def get_dataset_class(dataset_type):
    if dataset_type == "lazy_dataset":
        return LazyVLDataset
    elif dataset_type == "lazy_dataset_baseline":
        return LazyVLDatasetBaseline
    elif dataset_type == "lazy_dataset_sft":
        return LazyVLDatasetSFT
    elif dataset_type == "lazy_dataset_post_frame_selection":
        return LazyVLDatasetPostFrameSelection
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
