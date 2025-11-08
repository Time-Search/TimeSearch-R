#! /bin/bash
export CUDA_VISIBLE_DEVICES=RR
export GRPC_VERBOSITY=debug
export HF_HUB_OFFLINE=1
# 将当前目录添加到PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m clip_server