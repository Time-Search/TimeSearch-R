TEMPLATE_V3_1 = (
    "You must ALWAYS conduct thorough reasoning inside <think> and </think> tags BEFORE calling any tool or answering the question.\n"
    "At the very beginning of your response, please identify the relevant time segments related to the question based on the provided video frames and their corresponding timestamps, and then start your reasoning.\n"
    "You must invoke tools to explore any video content you are interested in within <tool_call> </tool_call> tags.\n"
    "You will then receive the tool response along with the corresponding video frames. You must wait for the response from the tool before answering or invoking the tool again.\n"
    "You are allowed to use <tool_call></tool_call> tags for a maximum of 8 rounds. Each <tool_call> must differ in either the function or its parameters.\n"
    "When you have enough information to answer the question, provide your answer within <answer> </answer> tags. Your answer should be supported by evidence from the video.\n"
    "Your output must follow the format: <think>Your reasoning process</think><tool_call>Parameters</tool_call> or <think>Your reasoning process</think><answer>Your answer</answer></answer>"
    "Question: {question}\n"
    "The video duration: {duration} seconds.\n"
)
TEMPLATE_V3_1_MULTIPLE_CHOICE = (
    "You must ALWAYS conduct thorough reasoning inside <think> and </think> tags BEFORE calling any tool or answering the question.\n"
    "At the very beginning of your response, please identify the relevant time segments related to the question based on the provided video frames and their corresponding timestamps, and then start your reasoning.\n"
    "You must invoke tools to explore any video content you are interested in within <tool_call> </tool_call> tags.\n"
    "You will then receive the tool response along with the corresponding video frames. You must wait for the response from the tool before answering or invoking the tool again.\n"
    "You are allowed to use <tool_call></tool_call> tags for a maximum of 8 rounds. Each <tool_call> must differ in either the function or its parameters.\n"
    "When you have enough information to answer the question, provide your answer within <answer> </answer> tags. Your answer should be supported by evidence from the video.\n"
    "Your output must follow the format: <think>Your reasoning process</think><tool_call>Parameters</tool_call> or <think>Your reasoning process</think><answer>Your answer</answer></answer>"
    "Question: {question}\nAnswer with the option's letter from the given choices within <answer> </answer> tags.\n"
    "The video duration: {duration} seconds.\n"
)

def make_prompt(example):
    if example['type'] == 'multiple_choice':
        return TEMPLATE_V3_1_MULTIPLE_CHOICE.format(question=example['question'], duration=example['duration'])
    else:
        return TEMPLATE_V3_1.format(question=example['question'], duration=example['duration'])
