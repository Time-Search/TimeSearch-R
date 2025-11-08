import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
from loguru import logger as eval_logger

VIDEO_TYPE = ["short", "medium", "long"]
CATEGORIES = ["Knowledge", "Film & Television", "Sports Competition", "Artistic Performance", "Life Record", "Multilingual"]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual",
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


matrices = []

for i in VIDEO_TYPE:
    for j in CATEGORIES:
        for k in SUB_CATEGORIES:
            for l in TASK_CATEGORIES:
                matrices.append(f"{i}_{j}_{k}_{l}")


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]

def videomme_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for video_type in VIDEO_TYPE:
        for category in CATEGORIES:
            for sub_category in SUB_CATEGORIES:
                for task_category in TASK_CATEGORIES:
                    key = f"{video_type}_{category}_{sub_category}_{task_category}"
                    category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        video_type = result["duration"]
        category = result["domain"]
        sub_category = result["sub_category"]
        task_category = result["task_type"]
        pred = result['prediction']
        pred = extract_characters_regex(pred)
        key = f"{video_type}_{category}_{sub_category}_{task_category}"
        category2score[key]["answered"] += 1
        category2score[key]["correct"] += pred == result["answer"]

    for video_type in VIDEO_TYPE:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if video_type in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"video Type: {video_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Categories: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for sub_cate in SUB_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if sub_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Video Sub Categories: {sub_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for task_cate in TASK_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    print(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0


def extract_prediction_from_message(messages):
    """
    从消息列表中提取最后一个 assistant 消息中的 answer 标签内容
    
    Args:
        messages: 消息列表，包含系统、用户和助手消息
        
    Returns:
        str: 最后一个匹配到的答案，如果没有找到则返回空字符串
    """
    answers = []
    # 尝试更宽松的正则表达式模式
    patterns = [
        r'<answer>(.*?)</answer>',  # 原始模式
        # r'<answer>([\s\S]*?)</answer>',  # 匹配包括换行符在内的所有字符
        # r'<answer>\s*(.*?)\s*</answer>',  # 处理可能的空白字符
    ]

    for message in messages:
        if message['role'] == 'assistant':
            for content in message['content']:
                if content['type'] == 'text':
                    text = content['text']                    
                    # 尝试所有正则表达式模式
                    for pattern in patterns:
                        match = re.search(pattern, text)
                        all_answers = re.findall(pattern, text, re.DOTALL)
                        # print(f"使用模式 '{pattern}' 的匹配结果: {all_answers}")
                        if all_answers:
                            answer = all_answers[-1]
                            answers.append(answer)
                            break
    if len(answers) > 0:
        result = answers[-1]
    else:
        result = ""
        for message in messages:
            if message['role'] == 'assistant':
                for content in message['content']:
                    if content['type'] == 'text':
                        text = content['text']                        
                        # 尝试所有正则表达式模式
                        
                        for pattern in patterns:
                            match = re.search(pattern, text)
                            if match:
                                answer = match.group(1).strip()
                        else: # 如果没有任何匹配，则返回文本内容
                            result = text
    # print(f"\n最终返回结果: {result}")
    return result

def eval_main(input_path):
    results = []
    with open(input_path) as f:
        for data in f:
            # print(data)
            try:
                dct = json.loads(data.strip())
                pred = dct['prediction']
                if isinstance(pred, list):
                    pred = extract_prediction_from_message(pred)
                    # print(pred)
                doc = json.loads(dct['meta'])
                doc['prediction'] = pred
                results.append(doc)
            except ValueError as e:
                eval_logger.error(f"Error loading data: {e}")
                eval_logger.error(f"Data: {data}")
                continue
    
    videomme_aggregate_results(results)

if __name__ == '__main__':
    import fire
    fire.Fire(eval_main)