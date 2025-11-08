import re
from transformers import StoppingCriteria


class StopOnKeywords(StoppingCriteria):
    def __init__(self, keywords, tokenizer, prompt_length=-1):
        if not isinstance(keywords, list):
            keywords = [keywords]
        self.keywords = keywords  # 指定的停止关键词，例如 "结束"
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length  # 输入的 prompt 的长度

    def check_keywords(self, input_id):
        if self.prompt_length > 0 and input_id.shape[0] < self.prompt_length + 10:
            return False
        text = self.tokenizer.decode(input_id.cpu())
        if len(text) > 0 and any(keyword in text.strip()[-(len(keyword) + 3):] for keyword in self.keywords):
            return True
        return False

    def __call__(self, input_ids, scores=None, **kwargs):
        if all(self.check_keywords(input_id) for input_id in input_ids):
            return True
        return False


def replace_n_pattern(text, pattern, replace_pattern):
    """
    替换字符串中连续出现的指定模式。

    参数:
        text (str): 原始字符串。
        pattern (str): 需要匹配的模式。
        replace_pattern (str): 替换模式，其中 `{N}` 会被替换为连续匹配的数量。

    返回:
        str: 替换后的字符串。
    """
    regex = re.compile(f'({re.escape(pattern)})+')
    
    def replace_match(match):
        matched_text = match.group()
        count = matched_text.count(pattern)
        return replace_pattern.replace('{N}', str(count))
    
    result = regex.sub(replace_match, text)
    return result


def clean_prompt(text, dedup_patterns, delete_patterns):
    """
    清洗文本，包括去重和删除指定模式。
    参数:
        text (str): 原始文本。
        dedup_patterns (list): 去重模式列表。
        delete_patterns (list): 删除模式列表。
    返回:
        str: 清洗后的文本。
    """
    if not isinstance(dedup_patterns, list):
        dedup_patterns = [dedup_patterns]
    if not isinstance(delete_patterns, list):
        delete_patterns = [delete_patterns]

    for pattern in dedup_patterns:
        text = replace_n_pattern(text, pattern, "[{N}" + pattern + "]")

    for pattern in delete_patterns:
        text = text.replace(pattern, '')

    return text.strip()
