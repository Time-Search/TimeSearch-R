from .v1 import reward_functions as v1, reward_weights as v1_weights
from .v2 import reward_functions as v2, reward_weights as v2_weights
from .v3 import reward_functions as v3, reward_weights as v3_weights
from .v3_1 import reward_functions as v3_1, reward_weights as v3_1_weights
from .v4 import reward_functions as v4, reward_weights as v4_weights
from .v5 import reward_functions as v5, reward_weights as v5_weights
from .v5_1 import reward_functions as v5_1, reward_weights as v5_1_weights
from .v6 import reward_functions as v6, reward_weights as v6_weights
from .v7 import reward_functions as v7, reward_weights as v7_weights

REWARD_FUNCTIONS_REGISTRY = {
    "v1": (v1, v1_weights),
    "v2": (v2, v2_weights),
    "v3": (v3, v3_weights),
    "v3_1": (v3_1, v3_1_weights),
    "v4": (v4, v4_weights),
    "v5": (v5, v5_weights),
    "v5_1": (v5_1, v5_1_weights),
    "v6": (v6, v6_weights),
    "v7": (v7, v7_weights),
}


def get_reward_functions(version: str):
    if version not in REWARD_FUNCTIONS_REGISTRY:
        raise ValueError(f"Invalid reward version: {version}")
    return REWARD_FUNCTIONS_REGISTRY[version]
