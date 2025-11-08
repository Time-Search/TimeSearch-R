#! /bin/bash
# 参数
VIDEO_ROOT=$1
DATASET=$2
python3 scripts/converts/prepare_frame_cache.py $VIDEO_ROOT $DATASET --num_workers 16 --target_fps 2
python3 scripts/converts/prepare_feature_cache.py $VIDEO_ROOT $DATASET --num_workers 16