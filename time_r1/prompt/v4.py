TEMPLATE_V4 = (
    "You must ALWAYS conduct thorough reasoning inside <think> and </think> tags BEFORE calling any tool or answering the question.\n"
    "You must invoke tools to explore any video content you are interested in within <tool_call> </tool_call> tags.\n"
    "You are allowed to use <tool_call></tool_call> tags for a maximum of 8 rounds.\n"
    "When you have enough information to answer the question, provide your answer within <answer> </answer> tags. Your answer should be supported by evidence from the video.\n"
    "Your output must follow the format: <think>Your reasoning process</think><tool_call>Parameters</tool_call> or <think>Your reasoning process</think><answer>Your answer</answer>"
    "Question: {question}\n"
    "The video duration: {duration} seconds.\n"
)
TEMPLATE_V4_MULTIPLE_CHOICE = (
    "You must ALWAYS conduct thorough reasoning inside <think> and </think> tags BEFORE calling any tool or answering the question.\n"
    "You must invoke tools to explore any video content you are interested in within <tool_call> </tool_call> tags.\n"
    "You are allowed to use <tool_call></tool_call> tags for a maximum of 8 rounds.\n"
    "When you have enough information to answer the question, provide your answer within <answer> </answer> tags. Your answer should be supported by evidence from the video.\n"
    "Your output must follow the format: <think>Your reasoning process</think><tool_call>Parameters</tool_call> or <think>Your reasoning process</think><answer>Your answer</answer>"
    "Question: {question}\nAnswer with the option's letter from the given choices within <answer> </answer> tags.\n"
    "The video duration: {duration} seconds.\n"
)

def make_prompt(example):
    if example.get('type', 'multiple_choice') == 'multiple_choice':
        # 默认都是多选
        return TEMPLATE_V4_MULTIPLE_CHOICE.format(question=example['question'], duration=example['duration'])
    else:
        return TEMPLATE_V4.format(question=example['question'], duration=example['duration'])
