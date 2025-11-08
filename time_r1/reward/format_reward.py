import re
from time_r1.utils.reward_utils import extract_completion


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>', re.DOTALL)
    rewards = []
    for completion in completions:
        completion = extract_completion(completion)
        match = re.fullmatch(pattern, completion.strip())
        rewards.append(1.0 if match else 0.0)
        print(f"------------- match: {match}, reward: {rewards[-1]} -------------\n")
    return rewards
