TEMPLATE_V5_MULTIPLE_CHOICE = (
    "Question: {question}\nAnswer with the option's letter from the given choices within <answer> </answer> tags.\n"
    "The video duration: {duration} seconds.\n"
    "Think first, call **seek_video_frames** if needed, then answer. Format strictly as:  <think>...</think><tool_call>...</tool_call> (if tools needed) OR  <think>...</think><answer>...</answer> (if you can answer the question)"
)

TEMPLATE_V5_OPEN_ENDED = (
    "Question: {question}\n"
    "The video duration: {duration} seconds.\n"
    "Think first, call **seek_video_frames** if needed, then answer. Format strictly as:  <think>...</think><tool_call>...</tool_call> (if tools needed) OR  <think>...</think><answer>...</answer> (if you can answer the question)"
)

def make_prompt(example):
    if example.get('type', 'multiple_choice') == 'multiple_choice':
        return TEMPLATE_V5_MULTIPLE_CHOICE.format(question=example['question'], duration=example['duration'])
    else:
        return TEMPLATE_V5_OPEN_ENDED.format(question=example['question'], duration=example['duration'])
