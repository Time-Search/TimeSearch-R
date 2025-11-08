import json_repair as json
import re
from time_r1.dataset.lazy_dataset import LazyVLDataset
from time_r1.utils.video_tools import video_tool_call


class LazyVLDatasetPostFrameSelection(LazyVLDataset):
    def qwen_vl_preprocess_fn(self, item):
        prediction_messages = item['prediction']
        processed = super().qwen_vl_preprocess_fn(item)
        messages = processed['messages']
        multimodal_cache = processed['multimodal_cache']
        selected_frames_content = self.select_frames(prediction_messages, multimodal_cache)
        selected_frames_prompt = {"type": "text", "text": "Here are some key frames.\n"}
        video_ele = messages[-1]['content'][0]
        text_ele = messages[-1]['content'][1]
        # 将selected_frames_content插入到messages[-1]['content']中
        messages[-1]['content'] = [video_ele, selected_frames_prompt, *selected_frames_content, text_ele]
        return {
            "messages": messages,
            "multimodal_cache": multimodal_cache,
        }
    
    def select_frames(self, messages, multimodal_cache):
        """
        根据multimodal_cache中的视频，选择关键帧
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
        selected_frames_content = []
        for i, message in enumerate(messages):
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
                        for c in resp['content']:
                            if isinstance(c, dict) and c.get("type") in ["text"] and c.get("text").startswith("Here are selected frames."):
                                # 跳过tool response 中的text prompt
                                continue
                            selected_frames_content.append(c)
        return selected_frames_content


if __name__ == "__main__":
    data_path = "workdir/experiments2/v7-coldstart-mlvu_cgbench_vcrbench_neptune-10k-preview-replay/2025-07-07-11/videomme_test_768f_15k_256.jsonl"
    data_root = "workdir/datasets/"
    from tqdm import tqdm
    from time_r1.prompt import get_prompt_fn
    dataset = LazyVLDatasetPostFrameSelection(data_path, data_root, "baseline", tool_name_list=[])
    item = dataset[233]
