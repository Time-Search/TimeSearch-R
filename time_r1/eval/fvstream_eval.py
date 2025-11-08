# Based on https://github.com/haotian-liu/LLaVA.

import os
import ast
import json
import openai
import argparse
from tqdm import tqdm
from time import sleep
from collections import defaultdict
import multiprocessing
from llava.eval.azure_openai_evaluation import testOpenaiChatCompletions

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--input_json", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--num_process", type=int, default=5, help="Number of splits.")
    args = parser.parse_args()
    return args


def annotate(dct):
    id, question, answer, pred = dct['id'], dct['q'], dct['a'], dct['pred']
    """
    Evaluates question and answer pairs using GPT-4o-mini
    Returns a score for correctness.
    """
    
    system = "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. " + \
    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:" + \
    "------" + \
    "##INSTRUCTIONS: " + \
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n" + \
    "- Consider synonyms or paraphrases as valid matches.\n" + \
    "- Evaluate the correctness of the prediction compared to the answer."
    
    text = "Please evaluate the following video-based question-answer pair:\n\n" + \
    f"Question: {question}\n" + \
    f"Correct Answer: {answer}\n" + \
    f"Predicted Answer: {pred}\n\n" + \
    "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. " + \
    "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING." + \
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. " + \
    "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
    while True:
        try:
            response_message = testOpenaiChatCompletions(system, text)
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = response_dict
            assert 'pred' in result_qa_pair, f"Error: {id} don't has key=pred"
            assert 'score' in result_qa_pair, f"Error: {id} don't has key=score"
            dct['gpt4_result'] = result_qa_pair
            return dct
        except Exception as e:
            print(e)
            sleep(1)
    


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    pred_path = args.input_json
    pred_contents = [json.loads(line) for line in open(pred_path)]

    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        info = json.loads(json.loads(sample['meta'][0])['meta'])
        info['pred'] = sample['pred']
        sample = info
        video_id = sample['id']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['id'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)

    # Generating list of id's and corresponding files
    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]
    # Preparing dictionary of question-answer sets
    prediction_list = []
    for sample in new_pred_contents:
        id = sample['id']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        qa_set = {"id": id, "q": question, "a": answer, "pred": pred, "a_type": sample['answer_type'] if 'answer_type' in sample else None}
        prediction_list.append(qa_set)

    # Combine all the processed files into one
    combined_contents = {}

    num_process = args.num_process
    pool = multiprocessing.Pool(num_process)
    result = []

    with tqdm(total=len(prediction_list)) as pbar:
        for i, res in tqdm(enumerate(pool.imap_unordered(annotate, prediction_list))):
            pbar.update()
            result.append(res)
    pbar.close()
    pool.close()
    pool.join()
    
    
    for res in result:
        if res != None:
            combined_contents[res['id']] = res

    json_path = args.output_json
    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    class ScoreMeter:
        def __init__(self):
            self.score_sum = 0
            self.count = 0
            self.yes_count = 0
            self.no_count = 0
            self.score_dict = {'yes': defaultdict(int), 'no': defaultdict(int)}

        def add_score(self, score, pred):
            self.score_sum += score
            self.count += 1
            pred_lower = pred.lower()
            if 'yes' in pred_lower:
                self.yes_count += 1
                self.score_dict['yes'][score] += 1
            elif 'no' in pred_lower:
                self.no_count += 1
                self.score_dict['no'][score] += 1

        def get_average_score(self):
            res = (self.score_sum / self.count) if self.count else 0
            return f"{res:.6f}"

        def get_accuracy(self, response_type):
            if response_type == 'yes':
                res =  (self.yes_count / self.count) if self.count else 0
            elif response_type == 'no':
                res = (self.no_count / self.count) if self.count else 0
            else:
                res = 0
            return f"{res:.6f}"

    meter_dic = {'total': ScoreMeter()}
    for key, value in combined_contents.items():
        # Computing score
        result = value['gpt4_result']
        score_match = result['score']
        score = int(score_match)
        pred = result['pred']

        meter_dic["total"].add_score(score, pred)
        if 'answer_type' in value and value['answer_type'] is not None:
            typ = str(value['answer_type'][0])
            if typ not in meter_dic:
                meter_dic[typ] = ScoreMeter()
            meter_dic[typ].add_score(score, pred)

    csv_dic = {'acc': meter_dic["total"].get_accuracy('yes'), 'score': meter_dic["total"].get_average_score()}

    output = ""
    output += "Yes count: " + str(meter_dic["total"].yes_count) + "\n"
    output += "No count: " + str(meter_dic["total"].no_count) + "\n"
    output += "Accuracy: " + str(meter_dic["total"].get_accuracy('yes')) + "\n"
    output += "Average score: " + str(meter_dic["total"].get_average_score()) + "\n"
    output += "\n"
    output += "Total Score Yes/No distribution:\n"
    for key, value in meter_dic["total"].score_dict.items():
        output += f"{key}:\n"
        for k in range(0, 6):
            v = value[k]
            output += f"{k}: {v}\n"
    output += "\n"
    output += "Answer Type Score distribution:\n"
    output += 'Type, Accuracy, Avg_score\n'
    key_list = sorted([k for k in meter_dic.keys()])
    for key in key_list:
        output += f"{key}, {meter_dic[key].get_accuracy('yes')}, {meter_dic[key].get_average_score()}\n"
        csv_dic[key] = meter_dic[key].get_accuracy('yes')

    output += "\n"
    for k in csv_dic.keys():
        output += f"{k}, "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"

    for k in csv_dic.keys():
        output += str(csv_dic[k]) + ", "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"

    print(output)
    args.output_csv = args.output_json.replace(".jsonl", ".csv").replace(".json", ".csv")
    with open(args.output_csv, 'w') as f:
        f.write(output)

if __name__ == "__main__":
    main()