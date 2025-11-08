TEMPLATE_V3 = (
    "You must conduct reasoning inside <think> and </think> first every time you call the tool or answer the question. "
    "You must invoke tools to explore any video content you are interested in within <tool_call> </tool_call> tags.\n"
    "You will then receive the tool response along with the corresponding video frames. You must wait for the response from the tool before answering or invoking the tool again.\n"
    "If you are not sure, invoke again before answering."
    "When you have enough information to answer the question, provide your answer within <answer> </answer> tags. Your answer should be supported by evidence from the video.\n"
    "Question: {question}\n"
    "The video lasts for {duration} seconds.\n"
)

def make_prompt(example):
    return TEMPLATE_V3.format(question=example['question'], duration=example['duration'])
