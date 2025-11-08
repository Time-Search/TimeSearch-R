import json
import random
import numpy as np
from collections import defaultdict
import re
from tabulate import tabulate

def calculate_ins_level_acc(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def eval_multi_choice(gold_i, pred_i):
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def evaluate_longvideobench(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        correct = eval_multi_choice(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def longvideobench_aggregate_results(results, verbose=True):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["duration_group"]].append(result)
        subset_to_eval_samples[result["question_category"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_longvideobench(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}

    for cat_name, cat_results in evaluation_result.items():
        printable_results[cat_name] = {
            "num": int(cat_results["num_example"]),
            "acc": round(cat_results["acc"], 5),
        }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }
    if verbose:
        # 准备表格数据
        table_data = []
        headers = ["指标类别", "样本数 (num)", "准确率 (acc)"]
        
        # 定义输出顺序
        order = [
            # 时长类别
            15, 60, 600, 3600,
            # 第二组
            "S2E", "S2O", "S2A", "E2O", "O2E", "T2E", "T2O", "T2A",
            # 第三组
            "E3E", "O3O", "SSS", "SOS", "SAA", "T3E", "T3O", "TOS", "TAA"
        ]
        
        # 按照指定顺序添加数据
        for cat_name in order:
            if cat_name in printable_results:
                cat_results = printable_results[cat_name]
                table_data.append([
                    cat_name,
                    cat_results["num"],
                    f"{cat_results['acc']:.3f}"
                ])
        
        # 添加Overall行
        if "Overall" in printable_results:
            table_data.append([
                "Overall",
                printable_results["Overall"]["num"],
                f"{printable_results['Overall']['acc']:.3f}"
            ])
        
        # 打印表格
        print(tabulate(table_data, headers=headers, tablefmt="pipe"))
    return printable_results["Overall"]["acc"]


def longvideobench_process_results(doc, pred):
    all_choices = []
    index2ans = {}
    for i, option in enumerate(doc['candidates']):
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))

    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["video_id"]
    lvb_acc = {"id": id, "duration_group": doc["duration_group"], "question_category": doc["question_category"], "answer": chr(ord("A") + doc["correct_choice"]), "parsed_pred": parsed_pred}
    return lvb_acc



def extract_prediction_from_message(messages):
    """
    从消息列表中提取最后一个 assistant 消息中的 answer 标签内容
    
    Args:
        messages: 消息列表，包含系统、用户和助手消息
        
    Returns:
        str: 最后一个匹配到的答案，如果没有找到则返回空字符串
    """
    answers = []
    # 尝试更宽松的正则表达式模式
    patterns = [
        r'<answer>(.*?)</answer>',  # 原始模式
        # r'<answer>([\s\S]*?)</answer>',  # 匹配包括换行符在内的所有字符
        # r'<answer>\s*(.*?)\s*</answer>',  # 处理可能的空白字符
    ]

    for message in messages:
        if message['role'] == 'assistant':
            for content in message['content']:
                if content['type'] == 'text':
                    text = content['text']                    
                    # 尝试所有正则表达式模式
                    for pattern in patterns:
                        match = re.search(pattern, text)
                        all_answers = re.findall(pattern, text, re.DOTALL)
                        # print(f"使用模式 '{pattern}' 的匹配结果: {all_answers}")
                        if all_answers:
                            answer = all_answers[-1]
                            answers.append(answer)
                            break
    if len(answers) > 0:
        result = answers[-1]
    else:
        result = ""
        for message in messages:
            if message['role'] == 'assistant':
                for content in message['content']:
                    if content['type'] == 'text':
                        text = content['text']                        
                        # 尝试所有正则表达式模式
                        
                        for pattern in patterns:
                            match = re.search(pattern, text)
                            if match:
                                answer = match.group(1).strip()
                        else: # 如果没有任何匹配，则返回文本内容
                            result = text
    # print(f"\n最终返回结果: {result}")
    return result


def eval_main(path_in):    
    results = dict()
    with open(path_in) as f:
        for data in f:
            dct = json.loads(data.strip())
            try:
                pred_messages = dct['prediction']
                id = dct['id']
                doc = json.loads(dct['meta'])
                pred_text = extract_prediction_from_message(pred_messages)
            except:
                print(dct)
            process_res = longvideobench_process_results(doc, pred_text)
            results[id] = process_res
    results = list(results.values())
    longvideobench_aggregate_results(results)


if __name__ == '__main__':
    import fire
    fire.Fire(eval_main)