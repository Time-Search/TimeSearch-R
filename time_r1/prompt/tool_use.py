from time_r1.utils.video_tools import get_video_tool_by_name
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt


def get_tool_use_prompt(tool_name_list):
    # Build messages
    tool_list = [get_video_tool_by_name(tool_name) for tool_name in tool_name_list]
    function_list = [tool.function for tool in tool_list]
    assert len(function_list) > 0, "function_list must be provided"
    system_message = NousFnCallPrompt().preprocess_fncall_messages(
        messages=[],
        functions=function_list,
        lang=None,
    )
    system_message = system_message[0].model_dump()
    return system_message['content'][0]['text']
