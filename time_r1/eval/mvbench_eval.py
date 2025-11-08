import argparse
import json
from collections import defaultdict


def compute_acc(results, data_name):
    score = [0, 0]
    for result in results:
        candidates = result['candidates']
        answer = result['answer']
        gt = chr(ord('A') + candidates.index(answer))
        pred = result['pred'][0]
        if pred not in ['A','B','C','D','E','F','G']:
            print(pred)
        if pred == gt:
            score[0] += 1
            score[1] += 1
        else:
            score[1] += 1
    
    print(f"{data_name} ACC: {score[0]/score[1]}")
    return score[0], score[1]

def eval_main():
    import argparse
    parser = argparse.ArgumentParser(description="Moments and Highlights Evaluation Script")
    parser.add_argument("--path_in", type=str, help="path to generated prediction file")
    args = parser.parse_args()
    
    results_dct = defaultdict(list)
    with open(args.path_in) as f:
        for data in f:
            dct = json.loads(data.strip())
            pred = dct['pred']
            doc = json.loads(json.loads(dct['meta'][0])['meta'])
            doc['pred'] = pred
            results_dct[doc['type']].append(doc)
    
    scores_right = []
    scores_all = []

    for key in sorted(results_dct):
        value = results_dct[key]
        score_r, score_a = compute_acc(value, key)
        scores_right.append(score_r)
        scores_all.append(score_a)
    print(f"M-Avg: {sum(scores_right)/sum(scores_all)}")

if __name__ == '__main__':
    eval_main()