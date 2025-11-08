import ast
import re
from datetime import datetime
import os


def is_valid_two_d_list_format(s):
    """检查时间区间格式是否正确
    期望格式：[[0, 10], [2, 4], [22, 25]]
    """
    pattern = r'^\[(\(\d+(\.\d+)?,\s*\d+(\.\d+)?\)(,\s*\(\d+(\.\d+)?,\s*\d+(\.\d+)?\))*(,)?|)\]$'
    if not re.match(pattern, s):
        return False
    try:
        # 尝试将字符串转换为 Python 对象
        lst = ast.literal_eval(s)
        # 检查对象是否为列表
        if not isinstance(lst, list):
            return False
        # 检查列表中的每个元素是否为元组
        for item in lst:
            if not isinstance(item, tuple):
                return False
            # 检查元组是否包含两个元素
            if len(item) != 2:
                return False
            # 检查元组中的元素是否为数字
            for num in item:
                if not isinstance(num, (int, float)):
                    return False
            if item[0] > item[1]: # 保证符合时序区间
                return False
        return True
    except:
        return False


def merge_intervals(intervals):
    """合并重叠或相邻的时间区间"""
    if not intervals:
        return []
    intervals = [list(i) for i in intervals] # tuple to list
    # 按起始时间排序
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0][:]]  # 复制第一个区间
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            # 合并区间
            merged[-1][1] = max(last[1], current[1])
        else:
            merged.append(current[:])
    return merged


def compute_iou(list_a, list_b):
    # # 示例用法
    # list_a = [[0, 3], [2, 4], [22, 25]]
    # list_b = [[1, 5], [2, 2], [2, 4]]
    # iou = compute_iou(list_a, list_b)
    # 合并两个列表的区间
    merged_a = merge_intervals(list_a)
    merged_b = merge_intervals(list_b)
    
    # 计算各自的总长度
    len_a = sum(end - start for start, end in merged_a)
    len_b = sum(end - start for start, end in merged_b)
    
    # 计算交集的总长度
    intersection = 0
    i = j = 0
    while i < len(merged_a) and j < len(merged_b):
        a_start, a_end = merged_a[i]
        b_start, b_end = merged_b[j]
        
        # 计算当前两个区间的重叠部分
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start < end:
            intersection += end - start
        
        # 移动指针
        if a_end < b_end:
            i += 1
        else:
            j += 1
    
    # 计算并集总长度
    union = len_a + len_b - intersection
    if union == 0:
        return 1.0    
    return intersection / union


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


def choice_answer_reward(completions, target, **kwargs):
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    rewards = []

    for content, sol in zip(completions, target): 
        reward = 0.0
        
        pattern_answer = r'<answer>(.*?)</answer>'
        # 使用 search 方法查找首个匹配项
        match_answer = re.search(pattern_answer, content, re.DOTALL)
        if match_answer:
            # 获取捕获组中的内容
            answer = match_answer.group(1)
            if extract_characters_regex(answer) == extract_characters_regex(sol['answer']):
                reward = 1.0
        rewards.append(reward)
        if os.getenv("DEBUG") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- answer reward: {reward}, match_answer: {match_answer}, gt_answer: {sol['answer']} -------------\n")
                f.flush()
    return rewards


def gqa_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>\s*<time>.*?</time>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    # print('matches:', matches)
    reward_list = []
    for i, match in enumerate(matches):
        if match:
            pattern_time = r'<time>(.*?)</time>'
            match_time = re.search(pattern_time, completions[i], re.DOTALL)
            if match_time:
                # 获取捕获组中的内容
                time = match_time.group(1)
            else:
                raise ValueError(completions[i])

            if is_valid_two_d_list_format(time):
                r = 1.0
            else:
                r = 0.0
        else:
            r = 0.0
        reward_list.append(r)
        if os.getenv("DEBUG") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- format reward: {r}, match_time: {match} -------------\n")
                f.flush()
    return reward_list


def iou_time_reward(completions, target, **kwargs):
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    rewards = []
    for content, sol in zip(completions, target): # Added video_durations
        reward = 0.0

        gt_time = sol['time']

        pattern_time = r'<time>(.*?)</time>'
        match_time = re.search(pattern_time, content, re.DOTALL)

        if match_time:
            time = match_time.group(1)
            if is_valid_two_d_list_format(time):
                pred_time = ast.literal_eval(time)
                reward = compute_iou(pred_time, gt_time)
        else:
            reward = 0.0
        rewards.append(reward)
        if os.getenv("DEBUG") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- iou reward: {reward}, match_time: {match_time}, gt_time: {gt_time}, content: {content} -------------\n")
                f.flush()
    return rewards


def gqa_mmgen_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r'<think>.*?</think>.*?<answer>.*?</answer>.*?<time>.*?</time>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    # print('matches:', matches)
    reward_list = []
    for i, match in enumerate(matches):
        if match:
            pattern_time = r'<time>(.*?)</time>'
            match_time = re.search(pattern_time, completions[i], re.DOTALL)
            if match_time:
                # 获取捕获组中的内容
                time = match_time.group(1)
            else:
                raise ValueError(completions[i])

            if is_valid_two_d_list_format(time):
                r = 1.0
            else:
                r = 0.0
        else:
            r = 0.0
        reward_list.append(r)
        if os.getenv("DEBUG") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- format reward: {r}, match_time: {match} -------------\n")
                f.flush()
    return reward_list
