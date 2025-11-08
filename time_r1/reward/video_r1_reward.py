import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from time_r1.utils.reward_utils import extract_completion


def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        print(f"Error converting '{num_str}' to float: {e}")
        return None

def wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    m = len(ref_words)
    n = len(hyp_words)
    d = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        d[i][0] = i
    for j in range(n+1):
        d[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[m][n] / max(1, m)


def compute_rouge_score(reference, hypothesis, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
    scores = scorer.score(reference, hypothesis)
    average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
    return average_fmeasure


def parse_timestamp_output(output_string):
    """Parses timestamp output, similar to the example code."""
    # 1. Find all <answer>...</answer> blocks.
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)
    if not answer_matches:
        return None  # No <answer> tags found.

    # 2. Use the content of the *last* <answer> block.
    last_answer_content = answer_matches[-1]
    matches = re.findall(r"(\d+\.?\d*) (to|and|-) (\d+\.?\d*)", last_answer_content, re.IGNORECASE)
    if not matches:
        return None
    last_match = matches[-1]
    start_time = float(last_match[0])
    end_time = float(last_match[2])
    return start_time, end_time


def iou_timestamp_reward(predict, target, **kwargs):
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    reward = 0.0
    parsed_times = parse_timestamp_output(predict)
    start_time, end_time = 0, 0
    gt = eval(target)
    s, e = gt[0][0], gt[0][1]
    if parsed_times:
        start_time, end_time = parsed_times
        from_number = start_time
        to_number = end_time

        intersection = max(0, min(to_number, e) - max(from_number, s))
        union = max(to_number, e) - min(from_number, s)
        if union > 0:
            reward = intersection / union

    print(f"------------- gt window: [{s}, {e}] pred window: [{start_time}, {end_time}] \n IoU reward: {reward} \n {predict} -------------\n")
    return reward


def multiple_choice_reward(predict, target, **kwargs):
    reward = 1.0 if predict.strip() == target.strip() else 0.0
    return reward


def multiple_choice_with_moment_retrieval_reward(predict, target, **kwargs):
    if extract_answer(predict) == extract_answer(target):
        reward1 = 1.0
    else:
        reward1 = 0.0
    
    
        
        
def accuracy_reward(completions, target, task_type, **kwargs):
    contents = [extract_completion(completion) for completion in completions]
    rewards = []

    for content, sol, task_type in zip(contents, target, task_type):
        try:
            output_ans = extract_answer(content)
            gt_ans = sol    # extract_answer(sol)
            print(f"------------- gt_ans: {gt_ans}, output_ans: {output_ans}, task_type: {task_type} -------------\n")
            if task_type == "multiple choice":
                reward = multiple_choice_reward(output_ans, gt_ans)
            elif task_type == "moment retrieval":
                reward = iou_timestamp_reward(content, sol)
            elif task_type == "multiple choice with moment retrieval":
                reward1 = multiple_choice_reward(output_ans, gt_ans)
                reward2 = iou_timestamp_reward(output_ans, gt_ans)
                reward = reward1 + reward2
            elif task_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif task_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif task_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif task_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for task_type '{task_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        # print(f"------------- Accuracy reward: {reward}, content: {content}, target: {sol} -------------\n")
            
    return rewards