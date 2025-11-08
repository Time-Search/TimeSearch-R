import ast
import json
import re
from datetime import datetime
import os
import json_repair

MAX_TOOL_USE_NUM=6

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
                f.write(f"------------- answer reward: {reward}, match_answer: {match_answer}, gt_answer: {sol['answer']} -------------\n")
                f.write(f"------------- content: {content} -------------\n")
                f.flush()
    return rewards


def tool_format_check(content: str):
    """
    Check if the tool call format is correct
    """
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    function_call_parse_status = []
    for m in pattern.finditer(content):
        try:
            text = m.group(1)
            func = json_repair.loads(text)
            if "name" in func and "arguments" in func:
                function_call_parse_status.append(True)
            else:
                function_call_parse_status.append(False)
        except Exception as e:
            function_call_parse_status.append(False)
    if len(function_call_parse_status) > 0 and all(function_call_parse_status) and len(function_call_parse_status) <= MAX_TOOL_USE_NUM:
        return 1.0
    else:
        return 0.0


def answer_format_check(response):
    # we check the last turn
    pattern = re.compile(r'<answer>.*?</answer>', re.DOTALL)
    last_response = response.split("assistant\n")[-1]
    match = re.fullmatch(pattern, last_response.strip())
    reward = 1.0 if match else 0.0
    return reward


def has_tag(text: str, tag: str) -> bool:
    return re.search(fr"<{tag}>", text)


def order_check(conversation: str) -> float:
    """
    会话必须满足：
    ① 最后一条 assistant 消息 **只包含** <answer> … </answer>，不得再出现 <tool_call>
    ② 在该 <answer> 之前，必须至少有一条 tool 消息
    ③ 任何 assistant 消息中，<tool_call>   和  <answer>  不能同时出现
    满足全部条件返回 1.0，否则 0.0
    """
    try:
        # 把会话分成单条消息
        turns = conversation.split("assistant\n")
        turns = [t.strip() for t in turns if t.strip()][1:] # 去掉第一条，因为第一条是 system / user
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


def answer_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    reward_list = []
    for content in completions:
        reward_list.append(answer_format_check(content))
    return reward_list


def order_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    reward_list = []
    for content in completions:
        reward_list.append(order_check(content))
    return reward_list


def tool_use_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has tool use."""
    reward_list = []
    for content in completions:
        reward_list.append(tool_format_check(content))
    return reward_list


reward_functions = [
    choice_answer_reward,
    order_format_reward,
    tool_use_format_reward,
    answer_format_reward,
]

reward_weights = [
    1.0,
    0.5,
    0.25,
    0.25,
]
