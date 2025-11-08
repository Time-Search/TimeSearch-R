PROMPT_TEMPLATE = """Please find the visual event described by a sentence in the video, determining its starting and ending times. 
Now I will give you the textual sentence: '{sentence}'

Provide your thought process within the <think> </think> tags, including analysis with either specific events or time ranges.
Then, provide the start and end times (in seconds, precise to two decimal places) in the format 'start time to end time' within the <answer> </answer> tags. For example: '12.54 to 17.83'."""


def make_prompt(example):
    return PROMPT_TEMPLATE.format(sentence=example['question'])
