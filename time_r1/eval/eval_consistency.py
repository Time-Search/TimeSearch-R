import os
import openai
import time
import re
import json
import multiprocessing
from tqdm import tqdm

api_key = os.getenv("OPENAI_API_KEY")
client = openai.AzureOpenAI(
    azure_endpoint="/your/azure/endpoint",
    api_version="2023-07-01-preview",
    api_key=api_key
)

def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1)
    return None

def extract_think(text):
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1)
    return None

def gpt4o_infer(messages):
    while True:
        try:
            completion = client.chat.completions.create(
                        timeout=120,
                        model="gpt-4o-2024-11-20", 
                        temperature=0,
                        max_tokens=4000,
                        messages=messages
                    )
            # print(completion)
            gpt4o_result = json.loads(completion.model_dump_json())['choices'][0]['message']['content']
            return gpt4o_result
        except Exception as e:
            error_message = str(e.args[0])
            if error_message.startswith('Error code: 429'):
                time.sleep(2)
                continue
            if error_message.startswith('Error code: 400'):
                return None
            print(error_message)

PROMPT_TEMPLATE = """You are a careful and logical reviewer. Your task is to verify whether the given reasoning process and the final answer are consistent in addressing the given question.

Please carefully read the following:

Question:
\"\"\"
{question}
\"\"\"

Reasoning Process:
\"\"\"
{reasoning}
\"\"\"

Final Answer:
\"\"\"
{answer}
\"\"\"

Please follow this format strictly:
<think>
Your analysis here
</think>
<answer>
Yes/No
</answer>
"""

def evaluate_consistency(data):
    pred = data['pred']
    reasoning = extract_think(pred)
    answer = extract_answer(pred)
    if reasoning is None or answer is None:
        judge = 'error'
        gpt4o_judge = ''
        data['judge'] = judge
        data['gpt4o_judge'] = gpt4o_judge
        return data
    messages = [
        {"role": "user", "content": PROMPT_TEMPLATE.format(question=data['question'],reasoning=reasoning, answer=answer)}
    ]
    res = gpt4o_infer(messages)
    judge = extract_answer(res)
    if judge is None:
        judge = 'error'
    gpt4o_judge = res
    data['judge'] = judge
    data['gpt4o_judge'] = gpt4o_judge
    return data

def evaluate(file_path):
    file_out = file_path.replace('.jsonl', '_consistency.jsonl')
    if os.path.exists(file_out):
        with open(file_out) as f:
            data_list = [json.loads(line) for line in f]
            return data_list

    data_list = []
    with open(file_path) as f:
        for data in f:
            data = json.loads(data)
            if 'prediction' in data:
                prediction = data['prediction']
            
                pred = prediction[-1]['content'][-1]['text']
            
                data['pred'] = pred
            data_list.append(data)
    pool = multiprocessing.Pool(5)
    result = []

    with tqdm(total=len(data_list)) as pbar:
        for i, res in tqdm(enumerate(pool.imap_unordered(evaluate_consistency, data_list))):
            pbar.update()
            result.append(res)
    pbar.close()
    pool.close()
    pool.join()
    
    
    writer = open(file_out, 'w')
    for res in result:
        writer.write(json.dumps(res, ensure_ascii=False) + '\n')
    return result


def main(input_path):
    error_cnt = 0
    right_cnt = 0
    all_cnt = 0
    result = evaluate(input_path)
    for res in result:
        if res['judge'] is None:
            res['judge'] = 'error'
            print(res['judge'], res['gpt4o_judge'])
        all_cnt += 1
        if res['judge'] == 'error':
            error_cnt += 1
        elif 'yes' in res['judge'].lower():
            right_cnt += 1

    print(f'error_cnt={error_cnt}, right_cnt={right_cnt}, all_cnt={all_cnt}, acc={right_cnt/all_cnt}')

if __name__ == '__main__':
    import fire
    fire.Fire(main)