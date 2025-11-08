QUESTION_TEMPLATE = (
    "Question: {question}\n"
    "Please reason about the video question step by step."
    "You interact with video frame images to search for visual cues."
    "The video lasts for {duration} seconds.\n"
    "You can break down complex questions or long video contents to better understand them, using expressions such as 'Let's break it down'.\n"
    "If some visual information is missing, you can call the tools. Verify the tool outputs using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', etc, or other natural language reflection expressions. You can call the tools and verify the outputs multiple times. Finally, give the answer within the <answer> </answer> tags. For example <answer>A</answer>."
)

QUESTION_TEMPLATE = (
    "Answer the given question about a video. "
    "You must conduct reasoning inside <think> and </think> first every time you call the tool or answer the question. "
    "After reasoning, if you find you lack some video information, you can call a video frame selector. You can call the selector as many times as you want. "
    "If you find no further video frames needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> xxx </answer>. \n"
    "Question: {question}\n"
    "Video information: The video lasts for {duration} seconds.\n"
)
# print(QUESTION_TEMPLATE)
def make_prompt(example):
    return QUESTION_TEMPLATE.format(question=example['question'], duration=example['duration'])
