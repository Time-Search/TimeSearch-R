import ast
import json
import re
from datetime import datetime
import os
from time_r1.reward.llm_judge import llm_judge_score


MAX_TOOL_USE_NUM=10


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


reward_functions = [
    multiturn_format_reward,
    accuracy_reward,
    tool_reward,
]

reward_weights = [
    1.0,
    0.8,
    1.2,
]
