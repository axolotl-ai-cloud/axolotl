"""B200 (sm_100)-tuned LoRA kernels.

The factored LoRA MLP kernel here is certified on B200 (compute capability
(10, 0)) only — see `support.py` for the hard arch gate and
`docs/lora_optims.qmd` for the certified envelope and limitations.
"""

from .support import (
    CUBLAS_WORKSPACE_CONFIG_RECOMMENDED,
    is_sm100,
    lora_mlp_b200_config_eligible,
    lora_mlp_b200_module_supported,
    maybe_set_cublas_workspace_config,
)

__all__ = [
    "CUBLAS_WORKSPACE_CONFIG_RECOMMENDED",
    "is_sm100",
    "lora_mlp_b200_config_eligible",
    "lora_mlp_b200_module_supported",
    "maybe_set_cublas_workspace_config",
]
