import ast
import json
import re
from datetime import datetime
import os
from time_r1.utils.utils import json_loads


def extract_characters_regex(s):
    s = s.strip()
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


def tool_format_check(content: str):
    """
    调用工具&每次调用格式都正确
    """
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    function_call_parse_status = []
    for m in pattern.finditer(content):
        try:
            text = m.group(1)
            func = json_loads(text)
            if "name" in func and "arguments" in func:
                function_call_parse_status.append(True)
            else:
                # print(f"tool_format_check: {text}")
                function_call_parse_status.append(False)
        except Exception as e:
            # print(f"tool_format_check: {e}, {m.group(1)}")
            function_call_parse_status.append(False)
    if len(function_call_parse_status) > 0 and all(function_call_parse_status):
        # print(len(function_call_parse_status), all(function_call_parse_status))
        return 1.0
    else:
        return 0.0

def answer_format_check(content: str):
    """
    答案格式检查
    """
    pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    match = pattern.search(content)
    if match:
        return 1.0
    else:
        return 0.0

def think_format_check(content: str):
    """
    思考格式检查
    """
    pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    match = pattern.search(content)
    return 1.0 if match else 0.0

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    reward_list = []
    for content in completions:
        answer_r = answer_format_check(content)
        think_r = think_format_check(content)
        tool_r = tool_format_check(content)
        if answer_r == 1.0 and think_r == 1.0 and tool_r == 1.0:
            r = 1.0
        else:
            r = 0.0
        reward_list.append(r)
        if os.getenv("DEBUG") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- format reward: {r}, content: {content} -------------\n")
                f.flush()
    return reward_list

def tool_use_reward(completions, **kwargs):
    """Reward function that checks if the completion has tool use."""
    reward_list = []
    for content in completions:
        tool_r = tool_format_check(content)
        reward_list.append(tool_r)
        if os.getenv("DEBUG") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- tool use reward: {tool_r}, content: {content} -------------\n")
                f.flush()
    return reward_list


reward_functions = [
    choice_answer_reward,
    format_reward,
    tool_use_reward,
]

reward_weights = [
    1.0,
    0.5,
    0.25,
]
