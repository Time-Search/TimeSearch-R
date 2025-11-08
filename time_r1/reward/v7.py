import json
import re
from datetime import datetime
import os
import math
from time_r1.reward.llm_judge import llm_judge_score
import ast

MAX_TOOL_USE_NUM=10


def general_answer_score(answer, ground_truth, question, type):
    """
    General answer score function.
    """
    if type == 'open_ended':
        s = llm_judge_score(answer, ground_truth, question)
    elif type == 'grounding':
        if answer.startswith("```json"):
            answer = answer.replace("```json", "").replace("```", "").strip()
        if is_valid_json_time_format(answer):
            pred_time = json.loads(answer)
            gt_time = json.loads(ground_truth)
            pred_start_time = pred_time.get("start_time")
            pred_end_time = pred_time.get("end_time")
            gt_start_time = gt_time.get("start_time")
            gt_end_time = gt_time.get("end_time")
            s = compute_iou([[pred_start_time, pred_end_time]], [[gt_start_time, gt_end_time]])
        else:
            s = 0.0
    elif type == 'sequence':
        s = 1.0 if extract_sequence_index(answer) == extract_sequence_index(ground_truth) else 0.0
    elif type == 'multiple_choice':
        s = 1.0 if extract_characters_regex(answer) == extract_characters_regex(ground_truth) else 0.0
    else:
        print(f"Error type in general_answer_score: {type}, question: {question}, answer: {answer}, ground_truth: {ground_truth}")
        s = llm_judge_score(answer, ground_truth, question)
    return s


def is_valid_json_time_format(s):
    """检查JSON格式是否正确"""
    try:
        item = json.loads(s)
        start_time = item.get("start_time")
        end_time = item.get("end_time")
        if start_time is None or end_time is None:
            return False
        if start_time < 0 or end_time < 0:
            return False
        if start_time > end_time:
            return False
        if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
            return False
        return True
    except Exception as e:
        print(f"Error in is_valid_json_time_format: {e}, s: {s}")
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


def extract_sequence_index(answer):
    # input: The sequence of the topics introduced in this video is (a) Men are setting up a tent in the dark, (c) Women do their beauty routine in the bathroom, (b) A baby is eating from a large platter of french fries on a black tray.
    # 输出: (a)(c)(b)
    pattern = r'(\([a-g,1-6]\))'
    matches = re.findall(pattern, answer)
    return ''.join(matches)


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
    matches = re.search(r'^[ABCDEFGHIJ][\.\s]|^[ABCDEFGHIJ]$', s)
    if matches:
        return matches[0][0]  # 只返回字母部分

    # 如果找不到标准格式，再尝试在文本中查找选项字母
    # 但要求这个字母必须是独立的（前后有空格或标点）
    if not strict:
        matches = re.search(r'(?<![a-zA-Z])[ABCDEFG](?![a-zA-Z])', s)
        if matches:
            return matches[0]

    return ""


def has_tag(text: str, tag: str) -> bool:
    return re.search(fr"<{tag}>", text)


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
    0. 必须有answer，且answer/tool_call都符合格式
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
    if len(answer_format_stats) > 0 and all(answer_format_stats) and all(tool_call_format_stats):
        return 1.0
    else:
        return 0.0


def multiturn_format_reward(messages, **kwargs):
    """
    Calculate the multiturn format reward.
    """
    reward_list = []
    for msg in messages:
        reward_list.append(multiturn_format_check(msg))
    return reward_list


def compute_answer_score(messages, ground_truth, question, completion, type):
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
                        answer_score_list.append(general_answer_score(answer, ground_truth, question, type))
    if len(answer_score_list) == 1:
        score = answer_score_list[0]
    else:
        score = 0.0
    # if os.getenv("DEBUG") == "true":
    #     log_path = os.getenv("LOG_PATH")
    #     with open(log_path, "a", encoding="utf-8") as f:
    #         f.write(f"-------------\nquestion: {question}, \nanswer: {answer_text_list}, \nreward: {answer_score_list}, \ngt: {ground_truth} \ncompletion: {completion} \n-------------\n")
    #         f.flush()
    return score


def advanced_tool_success_check(messages):
    """
    综合评估工具调用成功情况，包括：
    1. 基础工具调用成功检查
    2. 工具多样性和数量评估 
    3. 重复调用惩罚
    4. 调用失败惩罚
    NOTE:  VideoInteraction.avoid_mm_missing=True时，这项永远为1；当使用counterfactual reasoning时，这项不再重要

    """
    if not messages:
        return 0.0
    
    # 基础工具调用成功检查
    successful_tools = 0
    total_tool_calls = 0
    response_signitures_count = dict()
    tool_failure_count = 0

    for message in messages:
        if message.get("role") == "tool" and message.get("name") == "parse_error":
            tool_failure_count += 1
        if message.get("role") == "tool" and message.get("name") != "parse_error":
            total_tool_calls += 1
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") in ["video", "image"]:
                    successful_tools += 1
                    break
                elif not isinstance(item, dict):
                    print(f"Error in tool_success_check: {item}, content: {content}")
    tool_score = 1.0 / (1.0 + math.exp(-(successful_tools - 2)))
    if successful_tools == 0:
        tool_score = 0.0
    return tool_score


def compute_counterfactual_reasoning_score(messages, ground_truth, question, completion, visual_trace_completion, type):
    """
    Compute the llm judge score for a single message.
    NOTE: 答对且格式正确、调用工具正确，额外奖励
    """
    answer_score_list = []
    answer_text_list = []

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
                        answer_score_list.append(general_answer_score(answer, ground_truth, question, type))
                        answer_text_list.append(answer)
    if len(answer_score_list) == 1:
        answer_score = answer_score_list[0]
    else:
        answer_score = 0.0
    counterfactual_score = general_answer_score(visual_trace_completion, ground_truth, question, type)
    # tool_score = advanced_tool_success_check(messages)
    if os.getenv("DEBUG") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"-------------\nquestion: {question}, \nanswer: {answer_text_list}, \nreward: {answer_score_list}, \ngt: {ground_truth} \ncompletion: {completion} \n counterfactual: {visual_trace_completion} \n type: {type} \n counterfactual_score: {counterfactual_score} \n-------------\n")
            f.flush()
    if answer_score > 0.5:
        answer_score = 1.0
    if counterfactual_score > 0.5:
        counterfactual_score = 1.0
    return answer_score * counterfactual_score


def accuracy_reward(completions, messages, target, question, type, **kwargs):
    """
    Calculate the llm judge reward.
    """
    reward_list = []
    for msg, sol, q, comp, t in zip(messages, target, question, completions, type):
        score = compute_answer_score(msg, sol["answer"], q, comp, t)
        reward_list.append(score)
    return reward_list


def counterfactual_reasoning_reward(completions, messages, target, question, visual_trace_completions, type, **kwargs):
    """
    Calculate the llm judge reward.
    """
    reward_list = []
    for msg, sol, q, comp, visual_trace_comp, t in zip(messages, target, question, completions, visual_trace_completions, type):
        score = compute_counterfactual_reasoning_score(msg, sol["answer"], q, comp, visual_trace_comp, t)
        reward_list.append(score)
    return reward_list


reward_functions = [
    multiturn_format_reward,
    accuracy_reward,
    counterfactual_reasoning_reward,
]

reward_weights = [
    1.0,
    0.5,
    0.5,
]
