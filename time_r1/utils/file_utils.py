import json
import math
import random
import glob
import torch.distributed as dist
from time_r1.utils.utils import rank0_print
import yaml

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def parse_dataset_yaml(data_path):
    """
    # file should be in the format of:
    # datasets:
    #   - json_path: xxxx1.json
    #     sampling_strategy: first:1000
    #   - json_path: xxxx2.json
    #     sampling_strategy: end:3000
    #   - json_path: xxxx3.json
    #     sampling_strategy: random:999
    """
    
    json_file_list = []
    data_dict_list = []
    with open(data_path, "r") as file:
        yaml_data = yaml.safe_load(file)
        datasets = yaml_data.get("datasets")

    
    for dataset in datasets:
        json_path = dataset.get("json_path")
        json_file_list.append(json_path)
        sampling_strategy = dataset.get("sampling_strategy", "all")
        sampling_number = None
        rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

        if json_path.endswith(".jsonl"):
            cur_data_dict = []
            with open(json_path, "r") as json_file:
                for line in json_file:
                    cur_data_dict.append(json.loads(line.strip()))
        elif json_path.endswith(".json"):
            with open(json_path, "r") as json_file:
                cur_data_dict = json.load(json_file)
        else:
            raise ValueError(f"Unsupported file type: {json_path}")

        if ":" in sampling_strategy:
            sampling_strategy, sampling_number = sampling_strategy.split(":")
            if "%" in sampling_number:
                sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
            else:
                sampling_number = int(sampling_number)

        # Apply the sampling strategy
        if sampling_strategy == "first" and sampling_number is not None:
            cur_data_dict = cur_data_dict[:sampling_number]
        elif sampling_strategy == "end" and sampling_number is not None:
            cur_data_dict = cur_data_dict[-sampling_number:]
        elif sampling_strategy == "random" and sampling_number is not None:
            random.shuffle(cur_data_dict)
            cur_data_dict = cur_data_dict[:sampling_number]

        rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
        data_dict_list.extend(cur_data_dict)
    return {
        "json_file_list": json_file_list,
        "data_dict_list": data_dict_list
    }



def merge_results(save_path):
    """合并所有 rank 的结果文件，仅主进程负责执行。"""
    print("Merging result files...")
    result_files = glob.glob(f"{save_path}/rank*.jsonl")
    
    # 确保所有文件都存在
    world_size = dist.get_world_size()
    expected_files = [f"{save_path}/rank{i}.jsonl" for i in range(world_size)]
    missing_files = [f for f in expected_files if f not in result_files]
    
    if missing_files:
        print(f"警告：缺少以下结果文件: {missing_files}")
    
    with open(f"{save_path}.jsonl", "w") as writer:
        for file in sorted(result_files):
            print(f"合并文件: {file}")
            with open(file, "r") as f:
                for line in f:
                    writer.write(line)
    print("结果合并成功。")
