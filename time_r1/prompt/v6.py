TEMPLATE_V6 = "Here are sampled frames. They are located at {timestamps}.\nThe overall duration of the video is {duration} seconds.\nQuestion: {question}\nThink first about the adequacy of the current visual information, then invoke **seek_video_frames** or answer the question. Format strictly as: <think>...</think><tool_call>...</tool_call> (if more frames needed) OR <think>...</think><answer>...</answer> (if you can answer the question)."


def make_prompt(example):
    timestamps = example['timestamps']
    timestamps_str = ",".join([f"{t:.1f}" for t in timestamps])
    return TEMPLATE_V6.format(question=example['question'], timestamps=timestamps_str, duration=example['duration'])
