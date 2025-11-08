TOOL_RESPONSE_PROMPT = "Here are selected frames. They are located at {timestamps}.\nIf the frames provided above are sufficient to answer the user's question, please put your final answer within <answer></answer>. Otherwise invoke the tool again with different parameters in JSON format.\n"


def get_tool_response_prompt(item: dict):
    return TOOL_RESPONSE_PROMPT.format(timestamps=item["timestamps"])
