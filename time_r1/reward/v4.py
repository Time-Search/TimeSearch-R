import ast
import json
import re
from datetime import datetime
import os
from time_r1.reward.llm_judge import llm_judge_score


MAX_TOOL_USE_NUM=10


def is_valid_two_d_list_format(s):
    """检查时间区间格式是否正确
    期望格式：[[0, 10], [2, 4], [22, 25]]
    """
    pattern = r'^\[\s*(?:\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\](?:\s*,\s*\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\])*)?\s*\]$'
    if not re.match(pattern, s):
        return False
    try:
        # 尝试将字符串转换为 Python 对象
        lst = ast.literal_eval(s)
        # 检查对象是否为列表
        if not isinstance(lst, list):
            return False
        # 检查列表中的每个元素是否为列表
        for item in lst:
            if not isinstance(item, list):
                return False
            # 检查子列表是否包含两个元素
            if len(item) != 2:
                return False
            # 检查子列表中的元素是否为数字
            for num in item:
                if not isinstance(num, (int, float)):
                    return False
            if item[0] > item[1]: # 保证符合时序区间
                return False
        return True
    except:
        return False


def tool_argument_check(text: str):
    """
    Check if the tool call arguments is correct.
    """
    try:
        tool_call_text = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if tool_call_text:
            func = json.loads(tool_call_text.group(1))
            if "name" in func and "arguments" in func:
                return 1.0
            else:
                return 0.0
    except Exception as e:
        return 0.0


def has_tag(text: str, tag: str) -> bool:
    return re.search(fr"<{tag}>", text)


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


def answer_format_check(text):
    pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>', re.DOTALL)
    match = re.fullmatch(pattern, text.strip())
    return 1.0 if match else 0.0


def tool_call_format_check(text):
    pattern = re.compile(r'<think>.*?</think>\s*<tool_call>.*?</tool_call>', re.DOTALL)
    match = re.fullmatch(pattern, text.strip())
    return 1.0 if match else 0.0


def multiturn_format_check(messages, **kwargs):
    """
    检查多轮对话中每条 assistant 消息的格式是否严格符合要求：
    1. 如果包含 answer，必须符合 answer_format_check
    2. 如果包含 tool_call，必须符合 tool_call_format_check
    3. answer 和 tool_call 不能同时出现在同一条消息中
    """
    answer_format_error = False
    tool_call_format_error = False
    
    for message in messages:
        if message["role"] == "assistant":
            for content in message["content"]:
                if isinstance(content, dict) and content["type"] == "text":
                    text = content["text"]
                    # 检查 answer 和 tool_call 不能同时出现
                    if has_tag(text, "answer") and has_tag(text, "tool_call"):
                        answer_format_error = True
                        tool_call_format_error = True
                    if has_tag(text, "answer") and answer_format_check(text) == 0.0:
                        answer_format_error = True
                    if has_tag(text, "tool_call") and tool_call_format_check(text) == 0.0:
                        tool_call_format_error = True
    if answer_format_error and tool_call_format_error:
        return -2.0
    elif answer_format_error:
        return -1.0
    elif tool_call_format_error:
        return -1.0
    else:
        return 0.0


def compute_llm_judge_score(messages, ground_truth, question, completion):
    """
    Compute the llm judge score for a single message.
    """
    answer_text_list = []
    answer_score_list = []
    
    for message in messages:
        if message["role"] == "assistant":
            for content in message["content"]:
                if isinstance(content, dict) and content["type"] == "text":
                    text = content["text"]
                    # 检查 answer 和 tool_call 不能同时出现
                    if has_tag(text, "answer") and has_tag(text, "tool_call"):
                        return 0.0
                    pattern_answer = r'<answer>(.*?)</answer>'
                    # 使用 search 方法查找首个匹配项
                    match_answer = re.search(pattern_answer, text, re.DOTALL)
                    if match_answer:
                        # 获取捕获组中的内容
                        answer = match_answer.group(1)
                        answer_text_list.append(answer)
                        answer_score_list.append(llm_judge_score(answer, ground_truth, question))
    if len(answer_score_list) == 1:
        score = answer_score_list[0]
    else:
        score = 0.0
    if os.getenv("DEBUG") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"-------------\nquestion: {question}, \nanswer: {answer_text_list}, \nreward: {answer_score_list}, \ngt: {ground_truth} \ncompletion: {completion} \n-------------\n")
            f.flush()
    return score


def compute_tool_score(messages, ground_truth, question, completion):
    """
    Compute the llm judge score for a single message.
    NOTE: 答对且格式正确、调用工具正确，额外奖励
    """
    answer_score_list = []
    
    for message in messages:
        if message["role"] == "assistant":
            for content in message["content"]:
                if isinstance(content, dict) and content["type"] == "text":
                    text = content["text"]
                    if has_tag(text, "answer") and has_tag(text, "tool_call"):
                        return 0.0
                    pattern_answer = r'<answer>(.*?)</answer>'
                    match_answer = re.search(pattern_answer, text, re.DOTALL)
                    if match_answer:
                        # 获取捕获组中的内容
                        answer = match_answer.group(1)
                        answer_score_list.append(llm_judge_score(answer, ground_truth, question))
    if len(answer_score_list) == 1:
        answer_score = answer_score_list[0]
    else:
        answer_score = 0.0
    tool_score = tool_success_check(messages)
    return answer_score * tool_score


def accuracy_reward(completions, messages, target, question, **kwargs):
    """
    Calculate the llm judge reward.
    """
    reward_list = []
    for msg, sol, q, comp in zip(messages, target, question, completions):
        score = compute_llm_judge_score(msg, sol["answer"], q, comp)
        reward_list.append(score)
    return reward_list


def tool_reward(completions, messages, target, question, **kwargs):
    """
    Calculate the llm judge reward.
    """
    reward_list = []
    for msg, sol, q, comp in zip(messages, target, question, completions):
        score = compute_tool_score(msg, sol["answer"], q, comp)
        reward_list.append(score)
    return reward_list


def tool_success_check(messages):
    """
    single case
    检查工具调用是否成功
    遍历所有工具：
    如果至少有一个工具调用成功，则返回 1.0
    """
    if not messages:
        return 0.0
    stats = []
    for message in messages:
        if message.get("role") == "tool":
            content = message.get("content", [])
            if not isinstance(content, list):
                return 0.0
            tool_success = False
            for item in content:
                if isinstance(item, dict) and item.get("type") in ["video", "image"]:
                    tool_success = True
                    break
                elif not isinstance(item, dict):
                    print(f"Error in tool_success_check: {item}, content: {content}")
            stats.append(tool_success)
    return 1.0 if any(stats) else 0.0


def multiturn_format_reward(messages, **kwargs):
    """
    Calculate the multiturn format reward.
    """
    reward_list = []
    for msg in messages:
        reward_list.append(multiturn_format_check(msg))
    return reward_list


def time_iou_reward(completions, target, **kwargs):
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    rewards = []
    for content, sol in zip(completions, target): # Added video_durations
        reward = 0.0

        gt_time = sol['time']
        pattern_answer = r'<answer>(.*?)</answer>'
        match_answer = re.search(pattern_answer, content, re.DOTALL)

        if match_answer:
            answer = match_answer.group(1)
            if is_valid_two_d_list_format(answer):
                pred_time = ast.literal_eval(answer)
                reward = compute_iou(pred_time, gt_time)
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards


reward_functions = [
    multiturn_format_reward,
    accuracy_reward,
    tool_reward,
]

reward_weights = [
    1.0,
    1.0,
    1.0,
]
