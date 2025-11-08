import json
import re
from time_r1.dataset.lazy_dataset import LazyVLDataset
from time_r1.utils.video_tools import video_tool_call
from time_r1.utils.qwen_vl_utils import replace_vision_info_with_placeholder, process_vision_info


class LazyVLDatasetBaseline(LazyVLDataset):
    def qwen_vl_preprocess_fn(self, item):
        processed = super().qwen_vl_preprocess_fn(item)
        messages = processed['messages']
        multimodal_cache = processed['multimodal_cache']
        question = item["question"]
        messages = self.construct_baseline_messages(messages, multimodal_cache, question=question)
        return {
            "messages": messages,
            "multimodal_cache": multimodal_cache,
        }
    
    def construct_baseline_messages(self, messages, multimodal_cache, question):
        """
        append key frames
        original
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt}
                ],
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "video", 
                        "video": preview_video_frames,
                        "fps": preview_video_sample_fps,
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                ]
            },
        ]
        """
        content = messages[-1]['content']
        video_ele = content[0]
        text_ele = content[1]
        req = {"type": "function", "function": {"name": "seek_video_frames", "arguments": {"query": question}}}
        resp = video_tool_call(req, multimodal_cache)
        assert resp is not None
        selected_frames_prompt = {"type": "text", "text": "Here are some key frames.\n"}
        selected_frames = resp['content'][1]
        messages[-1]['content'] = [
            video_ele,
            selected_frames_prompt,
            selected_frames,
            text_ele,
        ]
        return messages

if __name__ == "__main__":
    data_path = "workdir/datasets/videomme/bes_videomme_withouttitle_test.json"
    data_root = "dataset/evaluation/llava_next/videomme/data/"
    from tqdm import tqdm
    from time_r1.prompt import get_prompt_fn
    dataset = LazyVLDatasetBaseline(data_path, data_root, "plain", tool_name_list=[])
    item = dataset[0]
    print(item)
    print(replace_vision_info_with_placeholder(item["messages"]))
    # for k in item:
    #     if k == "messages":
    #         print(replace_vision_info_with_placeholder(item["messages"]))
    #     else:
    #         print(f"{k}: {item[k]}")
