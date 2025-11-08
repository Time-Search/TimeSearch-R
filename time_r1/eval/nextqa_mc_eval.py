import argparse
import json
from collections import defaultdict


def compute_acc(results):
    score = [0, 0]
    for result in results:
        answer = int(result['answer'])
        gt = chr(ord('A') + answer)
        pred = result['pred'][0]
        if pred not in ['A','B','C','D','E','F','G']:
            print(pred)
        if pred == gt:
            score[0] += 1
            score[1] += 1
        else:
            score[1] += 1
    
    print(f"ACC: {score[0]/score[1]}")
    return score[0]/score[1]

def eval_main():
    import argparse
    parser = argparse.ArgumentParser(description="Moments and Highlights Evaluation Script")
    parser.add_argument("--path_in", type=str, help="path to generated prediction file")
    args = parser.parse_args()
    
    results_list = []
    with open(args.path_in) as f:
        for data in f:
            dct = json.loads(data.strip())
            pred = dct['pred']
            doc = json.loads(json.loads(dct['meta'][0])['meta'])
            doc['pred'] = pred
            results_list.append(doc)
    compute_acc(results_list)

if __name__ == '__main__':
    eval_main()