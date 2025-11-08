TEMPLATE_V2 = (
    "You may attempt to invoke tools to explore any video content you are interested in — when doing so, wrap your tool invocations in <tool_call> </tool_call> tags.\n"
    "You will then receive the tool response along with the corresponding video frames. You must wait for the response from the tool before answering.\n"
    "After the tool response, provide your answer enclosed within <answer> </answer> tags. \n"
    "Question: {question}\n"
)

def make_prompt(example):
    return TEMPLATE_V2.format(question=example['question'])
