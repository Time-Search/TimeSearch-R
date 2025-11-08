import os
import json
import base64
from PIL import Image
from time_r1.utils.video_tools import video_tool_call
from transformers import AutoProcessor
import re
from copy import deepcopy
from io import BytesIO
import json_repair
import torch
from contextlib import contextmanager
from codetiming import Timer
from typing import List, Any, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from time_r1.utils.qwen_vl_utils import process_vision_info, replace_vision_info_with_placeholder
from time_r1.environment.base import Environment
from vllm import SamplingParams


INVALID_TOOL_CALL_PROMPT = (
    '\nThe previous response is invalid. '
    'If I want to use tools, I should output a json object with function name and arguments within <tool_call></tool_call> tags: \n'
    'If I want to give the final answer, I should put the answer between <answer> and </answer>.\n'
    'Let me try again.\n'
)


def cleanup_llm_response(response_str: str) -> str:
    if '<think>' in response_str:
        response_str = '<think>' + response_str.split('<think>')[-1]
    if '</tool_call>' in response_str:
        return response_str.split('</tool_call>')[0] + '</tool_call>'
    elif '</answer>' in response_str:
        return response_str.split('</answer>')[0] + '</answer>'
    else:
        return response_str


def tool_response_placeholder():
    """
    Placeholder for tool call response.
    This is used to avoid FSDP / vllm hang when some examples miss multimodal.
    """
    func = {
        "name": "seek_video_frames",
        "arguments": {
            "query": "any",
            "start_time": 0,
            "end_time": 3600,
            "num_frames": 1,
        }
    }
    return f"<tool_call>\n{json.dumps(func)}\n</tool_call>\n"


def parse_actions_and_contents(predictions: List[Any]) -> Tuple[List[int], List[bool]]:
    """
    Process (text-based) predictions from llm into actions and validity flags.
    
    Args:
        predictions: List of raw predictions
        
    Returns:
        Tuple of (actions list, validity flags list)
    """
    actions = []
    contents = []
    for prediction in predictions:
        pattern = r'<(tool_call|answer)>(.*?)</\1>'
        match = re.search(pattern, prediction, re.DOTALL)
        if match:
            action = match.group(1)
            content = match.group(2).strip()  # Return only the content inside the tags
            if action == "tool_call":
                try:
                    # JSON to dict
                    func = json_repair.loads(content)
                    # arguments 可能是嵌套的 JSON 字符串
                    if isinstance(func.get("arguments"), str):
                        func["arguments"] = json_repair.loads(func["arguments"])
                    action = "tool_call"
                    content = {"type": "function", "function": func}
                except Exception as e:
                    # Parse failed
                    action = None
                    content = ""
        else:
            content = ''
            action = None
        
        actions.append(action)
        contents.append(content)
    return actions, contents


def batch_tool_call(valid_tool_calls: List[Dict], valid_mm_info: List[Dict]) -> List[Dict]:
    """
    不需要并行
    """
    outputs = []
    for tool_call, mm_info in zip(valid_tool_calls, valid_mm_info):
        outputs.append(video_tool_call(tool_call, mm_info))
    return outputs


def invalid_tool_call_message():
    return {
        "role": "tool",
        "name": "parse_error",
        "content": [{
            "type": "text", 
            "text": INVALID_TOOL_CALL_PROMPT
        }]
    }


def execute_predictions(predictions: List[str], multimodal_cache: List[Dict], active_mask=None) -> List[str]:
    """
    Execute predictions across multiple environments.
    Args:
        predictions: List of action predictions
    Returns:
        List of observation strings
    """
    cur_actions, contents = parse_actions_and_contents(predictions)

    valid_tool_calls = [content for action, content in zip(cur_actions, contents) if action == 'tool_call']
    valid_mm_info = [multimodal_cache[i] for i, action in enumerate(cur_actions) if action == 'tool_call']
    search_results = batch_tool_call(valid_tool_calls, valid_mm_info)

    next_obs, dones, valid_action, is_search = [], [], [], []

    for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
        if not active:
            next_obs.append(None)
            dones.append(1)
            valid_action.append(0)
            is_search.append(0)
        else:
            if action == 'answer':
                next_obs.append(None)
                dones.append(1)
                valid_action.append(1)
                is_search.append(0)
            elif action == 'tool_call':
                res = search_results.pop(0)
                next_obs.append(res if res else invalid_tool_call_message())
                dones.append(0)
                valid_action.append(1 if res else 0)
                is_search.append(1 if res else 0)
            else:
                next_obs.append(None)
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)

    assert len(search_results) == 0
    return next_obs, dones, valid_action, is_search


@contextmanager
def _timer(name: str, profiling_metrics: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    profiling_metrics[name] = timer.last


class VideoInteraction(Environment):
    def __init__(self, processor=None, model=None,
                 max_turns=2, max_new_tokens_per_turn=256, use_vllm=False, avoid_mm_missing=False):
        """
        model: 可以是vllm / hf model
        avoid_mm_missing: !!! Training only.
        """
        super().__init__()
        self.processor = processor
        self.model = model
        self.max_turns = max_turns
        self.max_new_tokens_per_turn = max_new_tokens_per_turn
        self.use_vllm = use_vllm
        self.avoid_mm_missing = avoid_mm_missing

    def generate(self, messages_batch: List[List[Dict]], multimodal_cache: List[Dict], profiling_metrics={}, **kwargs) -> Tuple[Dict, Dict]:
        """
        Run main LLM generation loop. 
        The only effective and interactive key is the messages.
        multimodal_cache: List[Dict], e.g. [{
            "video": List[torch.Tensor],
            "fps": List[int],
            "embedding": List[torch.Tensor],
        }, ...]
        """
        batch_size = len(messages_batch)

        active_mask: torch.Tensor = torch.ones(batch_size, dtype=torch.bool)
        turns_stats: torch.Tensor = torch.ones(batch_size, dtype=torch.int)
        valid_action_stats: torch.Tensor = torch.zeros(batch_size, dtype=torch.int)
        valid_search_stats: torch.Tensor = torch.zeros(batch_size, dtype=torch.int)
        active_num_list: List[int] = [active_mask.sum().item()]
        rolling_messages: List[List[Dict]] = deepcopy(messages_batch)
        for step in range(1, self.max_turns + 1):
            if not active_mask.sum():
                break
            active_idxs = torch.where(active_mask)[0]
            # print(f"active_idxs: {active_idxs}")
            messages_active = [rolling_messages[i] for i in active_idxs]
            with _timer(f"profiling/env/generate_step{step}", profiling_metrics):
                responses_str = self.single_turn_generate(messages_active, **kwargs)
            # NOTE: clean up
            responses_str = [cleanup_llm_response(s) for s in responses_str]
            responses_str, responses_msg = self._example_level_pad(responses_str, active_mask)
            with _timer(f"profiling/env/tooluse_step{step}", profiling_metrics):
                # Execute in environment and process observations
                next_obs, dones, valid_action, is_search = execute_predictions(
                    responses_str, multimodal_cache, active_mask
                )
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            # TODO: Avoid too long generation, Update active mask by attention_mask_active
            # curr_active_mask[active_idxs] = curr_active_mask[active_idxs] & attention_mask_active
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            # Update states
            rolling_messages = self._update_rolling_messages(
                rolling_messages,
                responses_msg,
                next_obs,
            )
        # After final MLLM rollout, if any example miss multimodal, append useless multimodal info to avoid FSDP hang
        if self.avoid_mm_missing and torch.any(valid_search_stats == 0):
            without_search_mask = valid_search_stats == 0
            without_search_idxs = torch.where(without_search_mask)[0]
            responses_str = [
                tool_response_placeholder() for _ in without_search_idxs
            ]
            responses_str, responses_msg = self._example_level_pad(responses_str, without_search_mask)
            with _timer(f"profiling/env/tooluse_placeholder", profiling_metrics):
                # Execute in environment and process observations
                next_obs, dones, valid_action, is_search = execute_predictions(
                    responses_str, multimodal_cache, without_search_mask
                )
            without_search_mask = without_search_mask * (~torch.tensor(is_search, dtype=torch.bool))
            if torch.any(without_search_mask == 1):
                idx = torch.where(without_search_mask == 1)[0][0]
                print(
                    f"The examples {torch.where(without_search_mask == 1)} still without multimodal." + \
                    f"The corresponding response is {responses_str[idx]=}"
                )
            # NOTE: responses_msg 需要重置为None，避免用于Policy update
            responses_msg = [None] * len(responses_msg)
            rolling_messages = self._update_rolling_messages(
                rolling_messages,
                responses_msg,
                next_obs,
            )
        # 更新统计信息
        profiling_metrics["env/num_turns"] = turns_stats.float().mean().item()
        profiling_metrics["env/num_valid_actions"] = valid_action_stats.float().mean().item()
        profiling_metrics["env/num_valid_calls"] = valid_search_stats.float().mean().item()
        return rolling_messages

    def _update_rolling_messages(self, messages_batch, responses_msg: List[dict], next_obs: List[dict]) -> Dict:
        """Update rolling state with new responses and observations."""
        for i, messages in enumerate(messages_batch):
            if responses_msg[i]:
                messages.append(responses_msg[i])
            if next_obs[i]:
                messages.append(next_obs[i])
        return messages_batch

    def _example_level_pad(self, responses_str: List[str], 
                           active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        Pad responses for non-active examples with empty messages.
        """
        assert active_mask.sum() == len(responses_str)
        batch_size = len(active_mask)
        padded_responses_str = [''] * batch_size
        padded_responses_msg = [None] * batch_size
        
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                padded_responses_msg[i] = {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": responses_str[s],
                    }]
                }
                s += 1
        return padded_responses_str, padded_responses_msg
    
    def single_turn_generate(self, messages_batch, **kwargs):
        if self.use_vllm:
            return self._single_turn_generate_vllm(messages_batch, **kwargs)
        else:
            return self._single_turn_generate_hf(messages_batch, **kwargs)

    @torch.no_grad()
    def _single_turn_generate_hf(self, messages_batch):
        """
        Generate one step of the generation for multiple cases.
        Args:
            messages_batch: List of message lists, each for one case
        Returns:
            List of generated texts
        """
        # 将messages_batch中的每个messages转换为json字符串
        text = self.processor.apply_chat_template(messages_batch, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages_batch, return_video_kwargs=True)

        inputs = self.processor(
            text=text, 
            images=image_inputs,
            videos=video_inputs, 
            fps=video_kwargs["fps"],
            padding=True, 
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = inputs.to(self.model.device)
        completion_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens_per_turn,
        )
        completion_ids = [completion_ids[i][len(inputs.input_ids[i]):] for i in range(len(completion_ids))]
        output_texts = self.processor.batch_decode(completion_ids, skip_special_tokens=True)
        return output_texts

    def _single_turn_generate_vllm(self, messages_batch, sampling_params=None):
        """
        Generate one step of the generation for multiple cases.
        Args:
            messages_batch: List of message lists, each for one case
        Returns:
            List of generated texts
        """
        # 将messages_batch中的每个messages转换为json字符串
        prompt_text_batch = self.processor.apply_chat_template(messages_batch, tokenize=False, add_generation_prompt=True)
        llm_inputs = []
        for idx, prompt in enumerate(prompt_text_batch):
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages_batch[idx], return_video_kwargs=True)
            mm_data = {}
            mm_kw = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
                # mm_kw.update(video_kwargs)
                for key, value in video_kwargs.items():
                    mm_kw[key] = value[0]
            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": mm_kw,
            })
        if sampling_params is None:
            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=1.0,
                max_tokens=self.max_new_tokens_per_turn,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                seed=42,
                bad_words=["matchCondition", "addCriterion", "_Parms", "actionDate", "fkk", "↤\n↤", " addCriterion"]
            )
            print(f"vLLM sampling_params is None, will use default sampling_params {sampling_params}")
        # NOTE: GRPO max_completion_length is not used
        sampling_params.max_tokens = self.max_new_tokens_per_turn
        all_outputs = self.model.generate(llm_inputs, sampling_params=sampling_params, use_tqdm=False)
        completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
        output_texts = self.processor.batch_decode(completion_ids, skip_special_tokens=True)
        return output_texts

    def update_model_and_processor(self, model=None, processor=None):
        if model is not None:
            self.model = model
        if processor is not None:
            self.processor = processor
