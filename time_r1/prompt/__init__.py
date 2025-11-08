from .time_zero import make_prompt as time_zero_prompt
from .vtg_v2 import make_prompt as vtg_v2_prompt
from .video_r1 import make_prompt as video_r1_prompt
from .video_chat_r1 import make_prompt as video_chat_r1_prompt
from .nextgqa_mmgen import make_prompt as nextgqa_mmgen_prompt
from .v1 import make_prompt as v1_prompt
from .v2 import make_prompt as v2_prompt
from .v3 import make_prompt as v3_prompt
from .v3_1 import make_prompt as v3_1_prompt
from .v4 import make_prompt as v4_prompt
from .v5 import make_prompt as v5_prompt
from .v6 import make_prompt as v6_prompt
from .plain import make_prompt as plain_prompt
from .baseline import make_prompt as baseline_prompt
from .tool_response import get_tool_response_prompt

PROMPT_TEMPLATES = {
    "time_zero": time_zero_prompt,
    "vtg_v2": vtg_v2_prompt,
    "video_r1": video_r1_prompt,
    "video_chat_r1": video_chat_r1_prompt,
    "nextgqa_mmgen": nextgqa_mmgen_prompt,
    "v1": v1_prompt,
    "v2": v2_prompt,
    "v3": v3_prompt,
    "v3_1": v3_1_prompt,
    "v4": v4_prompt,
    "v5": v5_prompt,
    "v6": v6_prompt,
    "baseline": baseline_prompt,
    "plain": plain_prompt,
    "tool_response": get_tool_response_prompt,
}

def get_prompt_fn(prompt_name):
    return PROMPT_TEMPLATES[prompt_name]
