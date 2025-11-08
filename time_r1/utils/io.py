import os
import time
import json
import pandas as pd
from contextlib import contextmanager

import torch
import torch.distributed as dist
import base64
import io
from PIL import Image

def encode_base64(img):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    b64_str =  base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    b64_str = "data:image/jpeg;base64," + b64_str
    return b64_str


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')


def load_file_to_df(file_path: str, **kwargs) -> pd.DataFrame:
    """
    根据文件后缀类型自动选择适当的 Pandas 数据读入方法，
    将文件读入为 DataFrame 对象。

    支持类型：
    - CSV (.csv)
    - Excel (.xls, .xlsx)
    - Parquet (.parquet)
    - JSON (.json)

    参数：
        file_path: 文件路径 (字符串)
        **kwargs: 可传递给相应 pd.read_XXX 方法的其它参数。

    返回：
        pd.DataFrame 对象
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension == '.csv':
        df = pd.read_csv(file_path, **kwargs)
    elif extension in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path, **kwargs)
    elif extension == '.parquet':
        df = pd.read_parquet(file_path, **kwargs)
    elif extension == '.jsonl':
        df = pd.read_json(file_path, lines=True, **kwargs)
    elif extension == '.json':
        df = pd.read_json(file_path, lines=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

    return df


def save_df_to_file(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    根据文件后缀类型自动选择适当的 Pandas 数据写入方法，
    将 DataFrame 保存至指定文件。

    支持类型：
    - CSV (.csv) -> df.to_csv
    - Excel (.xls, .xlsx) -> df.to_excel
    - Parquet (.parquet) -> df.to_parquet
    - JSON (.json) -> df.to_json

    参数：
        df: 待保存的 pd.DataFrame 对象
        file_path: 文件路径 (字符串)
        **kwargs: 可传递给相应 df.to_XXX 方法的其它参数。

    返回：
        None
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension == '.csv':
        index = kwargs.pop('index', False)
        df.to_csv(file_path, index=index, **kwargs)
    elif extension in ['.xls', '.xlsx']:
        index = kwargs.pop('index', False)
        df.to_excel(file_path, index=index, **kwargs)
    elif extension == '.parquet':
        df.to_parquet(file_path, **kwargs)
    elif extension == '.json':
        kwargs['force_ascii'] = False
        df.to_json(file_path, lines=False, **kwargs)
    elif extension == '.jsonl':
        kwargs['force_ascii'] = False
        kwargs['orient'] = "records"
        df.to_json(file_path, lines=True, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")


@contextmanager
def timer(hint=""):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    rank0_debug(f"🕙 {hint} runtime: {end - start:.2f} s")


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
    if os.getenv("VERL_DEBUG") != "true":
        return
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"[DEBUG][Rank{dist.get_rank()}]", *args)
    else:
        print("[DEBUG]", *args)


def rank_debug(*args):
    if os.getenv("VERL_DEBUG") != "true":
        return
    if dist.is_initialized():
        print(f"[DEBUG][Rank{dist.get_rank()}]", *args)
    else:
        print("[DEBUG]", *args)
