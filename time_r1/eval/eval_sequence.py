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
# from time_r1.reward.llm_judge import llm_judge_score


def extract_sequence_index(answer):
    # input: The sequence of the topics introduced in this video is (a) Men are setting up a tent in the dark, (c) Women do their beauty routine in the bathroom, (b) A baby is eating from a large platter of french fries on a black tray.
    # 输出: (a)(c)(b)
    pattern = r'(\([a-g,1-6]\))'
    matches = re.findall(pattern, answer)
    return ''.join(matches)


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
        target_sequence = extract_sequence_index(target['answer'])
        sequence_length = len(target_sequence.split(")("))
        if match_answer:
            answer = match_answer.group(1).strip()
            pred_sequence = extract_sequence_index(answer)
            
            acc = 1.0 if pred_sequence == target_sequence else 0.0
            print(f"idx: {idx}, acc: {acc}, pred_sequence: {pred_sequence}, target_sequence: {target_sequence}, pred: {answer}, sequence_length: {sequence_length}")
            valid_accs.append(acc)
        accs.append(acc)
        item['accuracy'] = acc
        item['sequence_length'] = sequence_length
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
    print("Accuacy:", sum(accs)/len(accs))
    if save_path:
        with open(save_path, 'w') as f:
            for item in datas:
                f.write(json.dumps(item) + '\n')
    print('Accuacy without format error:', sum(valid_accs)/len(valid_accs))
    # return ious, accs
    # 计算每个sequence length的准确率
    sequence_length_accs = {}
    for item in datas:
        sequence_length = item['sequence_length']
        # print(f"sequence_length: {sequence_length}")
        if sequence_length not in sequence_length_accs:
            sequence_length_accs[sequence_length] = []
        sequence_length_accs[sequence_length].append(item['accuracy'])
    for sequence_length, accs in sorted(sequence_length_accs.items()):
        print(f"sequence_length: {sequence_length}, acc: {sum(accs)/len(accs)}")

if __name__=='__main__':
    import fire
    fire.Fire(main)
