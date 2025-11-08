from time_r1.utils.io import load_jsonl
import re


def parse_order(text):
    # return a list of integers indicating the correct step
    pred_order=[]
    try:
        for part in str2int(text):
            int_n=None
            try:
                int_n=int(re.search(r'\d+', part).group())
            except:
                pass

            if int_n is not None:
                pred_order.append(int_n)
    except Exception as e:
        # print('[Error]: ', text)
        # raise ValueError(e)
        pass
    
    return pred_order


def str2int(text):
    parts = text.split(',')
    parts = [part.strip() for part in parts]
    return parts


def sequence_answer_reward(data):
    """
    Sequence answer reward
    """
    rewards = []
    for item in data:
        messages = item['prediction']
        sol = item['target']
        reward = 0.0
        last_content = messages[-1]['content'][-1]['text']
        pattern_answer = r'<answer>(.*?)</answer>'
        # 使用 search 方法查找首个匹配项
        match_answer = re.search(pattern_answer, last_content, re.DOTALL)
        if match_answer:
            answer = match_answer.group(1)
            pred_order = parse_order(answer)
            print(f"pred_order: {pred_order}, sol['answer']: {sol['answer']}")
            if pred_order == sol['answer']:
                reward = 1.0
        else:
            print(f"Format error: {last_content}")
        rewards.append(reward)
    return rewards


def main(input_path):
    data = load_jsonl(input_path)
    rewards = sequence_answer_reward(data)
    print(f"Accuracy: {sum(rewards) / len(rewards)}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)