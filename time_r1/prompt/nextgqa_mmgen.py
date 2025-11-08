QUESTION_TEMPLATE_GQA = """Question: {question}

You need to follow these steps:
1. First, output your thought process within the <think> </think> tags.
2. Next, analysis with specific time ranges within the <cuts> </cuts>\n. Provide relevant video clips based on the thinking process, in the format of [(s1, e1), (s2, e2), ...]. For example: <cuts>[(25, 31), (35, 38)]</cuts>\n
3. Then, The video clip corresponding to the time ranges you provided will be displayed within <infomation> </information> tags.
4. Continue output your thought process within the <think> </think> tags. Based on the results of previous thinking and corresponding video clips
5. Final, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. present the precise time period in seconds of the video clips on which you base your answer to this question in the format of [(s1, e1), (s2, e2), ...]. For example: <answer>A</answer><time>[(5.2, 10.4)]</time>.
"""


def make_prompt(example):
    return QUESTION_TEMPLATE_GQA.format(question=example['question'])
