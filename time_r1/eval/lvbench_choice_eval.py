import json
from collections import defaultdict
import pdb

def compute_accuracy(file_path):
    total_qa_num = 0
    right_num = 0
    category_right = defaultdict(int)
    category_total = defaultdict(int)
    category_acc = defaultdict(int)

    with open(file_path) as f:
        for data in f:
            dct = json.loads(data.strip())
            model_answer = dct['pred']
            # pdb.set_trace()
            qa = json.loads(json.loads(dct['meta'][0])['meta'])['qa']
            for category in qa['question_type']:
                category_total[category] += 1
                if model_answer == qa["answer"]:
                    category_right[category] += 1
            if model_answer == qa["answer"]:
                right_num += 1
            total_qa_num += 1

    for key in category_total:
        category_acc[key] = category_right[key] / category_total[key]

    acc = float(right_num) / total_qa_num
    category_acc.update({"acc": acc})
    print(category_acc, total_qa_num)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Moments and Highlights Evaluation Script")
    parser.add_argument("--path_in", type=str, help="path to generated prediction file")
    args = parser.parse_args()
    compute_accuracy(args.path_in)