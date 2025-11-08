import ast
import json
import re
from datetime import datetime
import os
# import json_repair

MAX_TOOL_USE_NUM=6

def extract_characters_regex(s, strict=True):
    s = s.strip()
    if not strict:
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option is",
            "The correct option is",
            "Best answer:",
            "Best option:",
        ]
        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, "")
        s = s.strip()

    # 首先尝试匹配标准选项格式：单独的字母，后面可能跟着点号或空格
    matches = re.search(r'^[ABCDEFG][\.\s]|^[ABCDEFG]$', s)
    if matches:
        return matches[0][0]  # 只返回字母部分

    # 如果找不到标准格式，再尝试在文本中查找选项字母
    # 但要求这个字母必须是独立的（前后有空格或标点）
    if not strict:
        matches = re.search(r'(?<![a-zA-Z])[ABCDEFG](?![a-zA-Z])', s)
        if matches:
            return matches[0]

    return ""


def choice_answer_reward(completions, target, **kwargs):
    """
    Choice answer reward
    """
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
                f.write(f"------------- answer reward: {reward}, match_answer: {match_answer}, gt_answer: {sol} -------------\n")
                f.write(f"------------- content: {content} -------------\n")
                f.flush()
    return rewards


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

def extract_time_windows(messages):
    """
    Extract time windows from the tool call arguments
    Return:
        List[Tuple[float, float]], the list of time windows
    """
    time_windows = []
    pattern_tool_call = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    for message in messages:
        if message["role"] != "assistant":
            continue
        for content in message["content"]:
            if content["type"] != "text":
                continue
            text = content["text"]
            match_tool_call = pattern_tool_call.search(text)
            if not match_tool_call:
                continue
            try:
                tool_call_text = match_tool_call.group(1)
                func = json.loads(tool_call_text)
                if not all(key in func.get("arguments", {}) for key in ["start_time", "end_time"]):
                    continue
                start_time = float(func["arguments"]["start_time"])
                end_time = float(func["arguments"]["end_time"])
                if start_time < 0 or end_time < 0:
                    continue
                if start_time > end_time:
                    continue
                time_windows.append((start_time, end_time))
            except Exception as e:
                print(f"Error in extract_time_windows: {e}, text: {text}")
                continue
    return time_windows


def has_tag(text: str, tag: str) -> bool:
    return re.search(fr"<{tag}>", text)


def order_check(conversation: str) -> float:
    """
    会话必须满足：
    ① 最后一条 assistant 消息 **只包含** <answer> … </answer>，不得再出现 <tool_call>
    ② 在该 <answer> 之前，必须至少有一条 tool 消息
    ③ 任何 assistant 消息中，<tool_call>   和  <answer>  不能同时出现
    ④ 任何 assistant 消息中，<think> … </think> 和 <tool_call> … </tool_call> 必须成对出现
    满足全部条件返回 1.0，否则 0.0
    """
    try:
        # 把会话分成单条消息
        turns = conversation.split("assistant\n")
        turns = [t.strip() for t in turns if t.strip()]
        last_asst = turns[-1]
        if not has_tag(last_asst,"answer"):             # 必须给答案
            return 0.0
        for t in turns:
            if has_tag(t,"tool_call") and has_tag(t,"answer"):
                return 0.0
        return 1.0
    except Exception as e:
        print(f"Error in order_check: {e}, conversation: {conversation}")
        return 0.0


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
    answer_format_stats = []
    tool_call_format_stats = []
    
    for message in messages:
        if message["role"] == "assistant":
            for content in message["content"]:
                if isinstance(content, dict) and content["type"] == "text":
                    text = content["text"]
                    # 检查 answer 和 tool_call 不能同时出现
                    if has_tag(text, "answer") and has_tag(text, "tool_call"):
                        return 0.0
                    if has_tag(text, "answer"):
                        answer_format_stats.append(answer_format_check(text))
                    elif has_tag(text, "tool_call"):
                        tool_call_format_stats.append(tool_call_format_check(text))
        
    answer_format_reward = 1.0 if len(answer_format_stats) > 0 and all(answer_format_stats) else 0.0
    tool_call_format_reward = 1.0 if len(tool_call_format_stats) > 0 and all(tool_call_format_stats) else 0.0
    return answer_format_reward + tool_call_format_reward


def multiturn_tool_argument_check(messages, **kwargs):
    """
    Deprecated!
    检查多轮对话中每条 assistant 消息的格式是否严格符合要求：
    1. 如果包含 tool_call，必须符合 tool_call_format_check
    """
    stats = []
    
    for message in messages:
        if message["role"] == "assistant":
            for content in message["content"]:
                if isinstance(content, dict) and content["type"] == "text":
                    text = content["text"]
                    if has_tag(text, "tool_call"):
                        stats.append(tool_argument_check(text))

    if len(stats) == 0:
        return 0.0
    return 1.0 if all(stats) else 0.0


def tool_success_check(messages):
    """
    single case
    检查工具调用是否成功
    遍历所有工具：
    1. 如果所有工具调用都成功，则返回 1.0
    2. 如果至少有一个工具调用失败，则返回 0.0
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
    return 1.0 if len(stats) > 0 and all(stats) else 0.0


def tool_call_process_reward(target, messages, **kwargs):
    """
    过程奖励：
    1. IoU得分
    2. 工具调用成功得分
    """
    reward_list = []
    for sol, msg in zip(target, messages):
        predicted_time_windows = extract_time_windows(msg)
        target_time_windows = sol["time"]
        iou_score = compute_iou(predicted_time_windows, target_time_windows)
        is_tool_success = tool_success_check(msg)
        reward = iou_score * is_tool_success
        reward_list.append(reward)
    return reward_list


def multiturn_format_reward(messages, **kwargs):
    """
    Calculate the multiturn format reward.
    """
    reward_list = []
    for msg in messages:
        reward_list.append(multiturn_format_check(msg))
    return reward_list


def tool_argument_reward(messages, **kwargs):
    """
    Calculate the tool argument reward.
    """
    reward_list = []
    for message in messages:
        reward_list.append(multiturn_tool_argument_check(message))
    return reward_list


reward_functions = [
    multiturn_format_reward,
    choice_answer_reward,
]

reward_weights = [
    1.0,
    1.0,
]
