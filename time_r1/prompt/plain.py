DEFAULT_TEMPLATE = """{question}"""


def make_prompt(example):
    return DEFAULT_TEMPLATE.format(question=example['question'])
