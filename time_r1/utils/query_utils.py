from transformers import AutoProcessor
import torch


def clear_query(query):
    """
    clear query from extra information
    """
    heads = [
        "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.",
        "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    ]
    tails = [
        "Answer with the option's letter from the given choices directly.",
        "The best answer is:",
        "Answer the question using a single word or phrase.",
        "Only give the best option.\n",
        "Best option: ("
    ]
    for head in heads:
        query = query.split(head)[-1]
    for tail in tails:
        query = query.split(tail)[0]
    query = query.strip()
    return query


def split_query(input_text_list, processor):
    """
    [Batch operation]
    split text into 64 tokens
    """
    inputs = processor(text=input_text_list, padding="max_length", return_tensors="pt", truncation=False)
    stride_num = (int(inputs["input_ids"].shape[-1]) + 63) // 64
    stride = (inputs["input_ids"].shape[-1] + stride_num - 1) // stride_num

    input_id_heads, input_id_tails = [], []
    l, r = 0, inputs["input_ids"].shape[-1]
    while l < r:
        input_id_heads.append(inputs["input_ids"][:, l:l + stride])
        l += stride
        if l < r:
            input_id_tails.append(inputs["input_ids"][:, r - stride:r])
            r -= stride

    input_ids = input_id_heads + input_id_tails[::-1]
    input_ids = torch.cat(input_ids)
    resume_texts = processor.batch_decode(input_ids, skip_special_tokens=True)
    return resume_texts
