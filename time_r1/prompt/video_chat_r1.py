QUESTION_TEMPLATE_GQA = """Question: {question}

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx) or time ranges (xx to xx).

Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. At the same time, in the <time> </time> tags, present the precise time period in seconds of the video clips on which you base your answer to this question in the format of [(s1, e1), (s2, e2), ...]. For example: <answer>A</answer><time>[(5.2, 10.4)]</time>.
"""


def make_prompt(example):
    return QUESTION_TEMPLATE_GQA.format(question=example['question'])
