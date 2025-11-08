TEMPLATE_BASELINE = (
    "Question: {question}\nAnswer with the option's letter from the given choices directly."
)

def make_prompt(example):
    return TEMPLATE_BASELINE.format(question=example['question'])
