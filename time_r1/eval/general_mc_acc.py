import numpy as np
import json
from tqdm import tqdm
import os
import re
import pickle
import random
import ast
import os
import json
from math import ceil
from time_r1.utils.io import load_jsonl


def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union


def parse_timestamp_output(output_string):
    matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", output_string)
    if not matches:
        answer_match = re.search(r"<answer>(.*?)</answer>", output_string)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            answer_matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", answer_content)
            if answer_matches:
                last_match = answer_matches[-1]
                return float(last_match[0]), float(last_match[2])
        return None, None

    last_match = matches[-1]
    start_time_str = last_match[0]
    end_time_str = last_match[2]
    
    try:
        start_time = float(start_time_str)
        end_time = float(end_time_str)
        return start_time, end_time
    except ValueError:
        return None, None


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEFG]", s):
        return ""

    matches = re.search(r"[ABCDEFG]", s)
    if matches is None:
        return ""
    return matches[0]


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = [list(i) for i in intervals] # tuple to list
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0][:]]
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1][1] = max(last[1], current[1])
        else:
            merged.append(current[:])
    
    return merged

def compute_iou(list_a, list_b):
    merged_a = merge_intervals(list_a)
    merged_b = merge_intervals(list_b)
    
    len_a = sum(end - start for start, end in merged_a)
    len_b = sum(end - start for start, end in merged_b)
    
    intersection = 0
    i = j = 0
    while i < len(merged_a) and j < len(merged_b):
        a_start, a_end = merged_a[i]
        b_start, b_end = merged_b[j]
        
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start < end:
            intersection += end - start
        
        if a_end < b_end:
            i += 1
        else:
            j += 1
    
    union = len_a + len_b - intersection
    if union == 0:
        return 1.0
    
    return intersection / union

def is_valid_two_d_list_format(s):
    pattern = r'^\[(\(\d+(\.\d+)?,\s*\d+(\.\d+)?\)(,\s*\(\d+(\.\d+)?,\s*\d+(\.\d+)?\))*(,)?|)\]$'
    if not re.match(pattern, s):
        return False
    try:
        lst = ast.literal_eval(s)
        if not isinstance(lst, list):
            return False
        for item in lst:
            if not isinstance(item, tuple):
                return False
            if len(item) != 2:
                return False
            for num in item:
                if not isinstance(num, (int, float)):
                    return False
            if item[0] > item[1]: # 保证符合时序区间
                return False
        return True
    except:
        return False
    

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = []
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception as e:
                print(f"Error: {e}, line: {line}")
                continue
    return data


def main(prediction_file, use_iou=False, save_path=None):
    datas = load_jsonl(prediction_file)
    accs = []
    valid_accs = []     # 排除格式错误的数据
    ious = []
    for idx, item in enumerate(datas):
        if 'target' not in item:
            # 兼容TimeSearch数据结构
            target = item['meta']
            target = json.loads(target)
        else:
            target = item['target']
        try:
            ans = item['prediction'][-1]['content'][-1]['text']
            pattern_answer = r'<answer>(.*?)</answer>'
            match_answer = re.search(pattern_answer, ans, re.DOTALL)
        except Exception as e:
            print(f"Error: {e}, idx: {idx}, {item}")
            continue
        acc = 0.0
        if match_answer:
            answer = match_answer.group(1)
            print(f"idx: {idx}, pred: {answer}, target: {target['answer']}")
            if extract_characters_regex(answer) == extract_characters_regex(target['answer']):
                # print(f"idx: {idx}, pred: {answer}, target: {target['answer']}")
                acc = 1.0
            if extract_characters_regex(answer):
                valid_accs.append(acc)
        else:
            if ans == target['answer']:
                acc = 1.0
                valid_accs.append(acc)
            elif extract_characters_regex(ans) == extract_characters_regex(target['answer']):
                acc = 1.0
                valid_accs.append(acc)
            else:
                print(f"--idx: {idx}, pred: {ans}, target: {target['answer']}")
            # print(f"idx: {idx}, pred: {ans}, target: {target['answer']}")
        accs.append(acc)
        item['accuracy'] = acc
        if not use_iou:
            continue

        # IoU
        pattern_time = r'<time>(.*?)</time>'
        match_time = re.search(pattern_time, ans, re.DOTALL)

        if match_time:
            time = match_time.group(1)
            if is_valid_two_d_list_format(time):
                pred_time = ast.literal_eval(time)
                iou = compute_iou(pred_time, target['time'])
        else:
            iou = 0.0
        ious.append(iou)
    print(f"num: {len(datas)}, num_acc: {len(accs)}, num_valid_acc: {len(valid_accs)}, num_iou: {len(ious)}")
    if use_iou:
        print('mIoU:', sum(ious) / len(ious))
    print("Accuacy:", sum(accs)/len(accs))
    if save_path:
        with open(save_path, 'w') as f:
            for item in datas:
                f.write(json.dumps(item) + '\n')
    print('Accuacy without format error:', sum(valid_accs)/(len(valid_accs)+1))
    # return ious, accs


if __name__=='__main__':
    import fire
    fire.Fire(main)
