import torch.distributed as dist
import os
import torch
import datetime
from contextlib import contextmanager
import time
import json

def json_loads(text: str) -> dict:
    text = text.strip('\n')
    if text.startswith('```') and text.endswith('\n```'):
        text = '\n'.join(text.split('\n')[1:-1])
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError as json_err:
        try:
            return json.loads(text + "}")
        except Exception:
            raise json_err

@contextmanager
def timer(hint=""):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    # rank0_debug(f"🕙 {hint} runtime: {end - start:.2f} s")


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank0_debug(*args):
    if os.getenv("DEBUG") != "true":
        return
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_debug(*args):
    if os.getenv("DEBUG") != "true":
        return
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def setup_ddp():
    """初始化分布式环境。"""
    # 使用环境变量获取多机多卡的信息
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(days=365))
    torch.cuda.set_device(local_rank)
    
    print(f"初始化进程: rank={rank}, local_rank={local_rank}, world_size={world_size}")


def cleanup_ddp():
    """销毁分布式进程组。"""
    dist.destroy_process_group()

