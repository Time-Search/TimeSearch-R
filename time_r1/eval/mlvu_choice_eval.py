import argparse
import json
from collections import defaultdict
import re

def compute_acc(results, data_name):
    score = [0, 0]
    invalid_count = 0
    for result in results:
        candidates = result['candidates']
        answer = result['answer']
        gt = chr(ord('A') + candidates.index(answer))
        pred = result['pred']
        # if pred[0] not in ['A','B','C','D','E','F','G']:
        #     print(pred)
        if pred is None:
            invalid_count += 1
            score[1] += 1
        elif pred and pred[0] == gt:
            score[0] += 1
            score[1] += 1
        elif pred.strip() == answer.strip():
            score[0] += 1
            score[1] += 1
        else:
            score[1] += 1

    print(f"{data_name} ACC: {score[0]/score[1]}, invalid_count: {invalid_count}")
    return score[0]/score[1]


def eval_main(prediction_file):
    results_dct = defaultdict(list)
    with open(prediction_file) as f:
        for data in f:
            item = json.loads(data.strip())
            target = item['target']
            try:
                ans = item['prediction'][-1]['content'][-1]['text']
                pattern_answer = r'<answer>(.*?)</answer>'
                match_answer = re.search(pattern_answer, ans, re.DOTALL)
            except Exception as e:
                print(f"Error: {e}, {item}")
                continue
            answer = None   # "A"
            if match_answer:
                answer = match_answer.group(1)
                # print(f"pred: {answer}, target: {target['answer']}")
            else:
                print(f"No answer found, {ans}")
            doc = json.loads(item['meta'])
            doc['pred'] = answer
            results_dct[doc['question_type']].append(doc)
    
    data_names = ['topic_reasoning', 'anomaly_reco', 'findNeedle', 'ego', 'plotQA', 'order', 'count']
    data_names_trans = {
        'topic_reasoning': 'TR',
        'anomaly_reco': 'AR',
        'findNeedle': 'NQA',
        'ego': 'ER',
        'plotQA': 'PQA',
        'order': 'AO',
        'count': 'AC'
    }
    scores = []
    for key, value in results_dct.items():
        scores.append(compute_acc(value, data_names_trans[key]))
    
    print(f"M-Avg: {sum(scores)/len(scores)}")

if __name__ == '__main__':
    import fire
    fire.Fire(eval_main)