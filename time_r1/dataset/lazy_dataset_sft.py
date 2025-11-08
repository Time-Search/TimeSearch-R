import json
import re
from time_r1.dataset.lazy_dataset import LazyVLDataset
from time_r1.utils.video_tools import video_tool_call
from time_r1.utils.qwen_vl_utils import replace_vision_info_with_placeholder, process_vision_info


class LazyVLDatasetSFT(LazyVLDataset):
    def qwen_vl_preprocess_fn(self, item):
        original_messages = item['messages']
        processed = super().qwen_vl_preprocess_fn(item)
        messages = processed['messages']
        multimodal_cache = processed['multimodal_cache']
        multi_turn_messages = self.construct_multi_turn_messages(original_messages, multimodal_cache)
        messages = messages + multi_turn_messages[1:]   # remove the first message from user
        return {
            "messages": messages,
            "multimodal_cache": multimodal_cache,
        }
    
    def construct_multi_turn_messages(self, messages, multimodal_cache):
        """
        补全所有的tool消息
        """
        updated_messages = []
        for i, message in enumerate(messages):
            updated_messages.append(message)
            if message['role'] == 'assistant':
                last_content = message['content'][-1]['text']
                # "<tool_call> {\"name\": \"seek_video_frames\", \"arguments\": {\"query\": null, \"start_time\": 0.0, \"end_time\": 59.0}}\n</tool_call>"
                pattern = r'<tool_call>(.*?)</tool_call>'
                match = re.search(pattern, last_content, re.DOTALL)
                if match:
                    func = json.loads(match.group(1))
                    content = {"type": "function", "function": func}
                    resp = video_tool_call(content, multimodal_cache)
                    if resp is not None:
                        updated_messages.append(resp)
        return updated_messages


if __name__ == "__main__":
    data_path = "workdir/datasets/qvhighlights/converted/train_messages_balance_mini.json"
    data_root = "workdir/datasets/"
    from tqdm import tqdm
    from time_r1.prompt import get_prompt_fn
    dataset = LazyVLDatasetSFT(data_path, data_root, "v3", tool_name_list=["seek_video_frames"])
    item = dataset[233]
    # print(item)
    # print(replace_vision_info_with_placeholder(item["messages"]))
    
    # for i in range(len(dataset)):
        # print(i)
        # item = dataset[i]
    # for k in item:
    #     if k == "messages":
    #         print(replace_vision_info_with_placeholder(item["messages"]))
    #     else:
    #         print(f"{k}: {item[k]}")
