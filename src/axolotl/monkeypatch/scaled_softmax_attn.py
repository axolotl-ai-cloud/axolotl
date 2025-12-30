"""
Scaled Softmax (SSMax) attention patch.
SSMax: softmax(scores * log(n))
Ref: https://arxiv.org/abs/2501.19399
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_original_flash_fn = None
_original_eager_fns = {}

SUPPORTED_MODEL_MODULES = [
    "transformers.models.mistral.modeling_mistral",
    "transformers.models.llama.modeling_llama",
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.phi3.modeling_phi3",
    "transformers.models.gemma2.modeling_gemma2",
]


def patch_scaled_softmax_attention(scaling_factor: float = 1.0, model_type: str = None):
    global _original_flash_fn
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def ssmax_scale(seq_len):
        return scaling_factor * math.log(max(seq_len, 2))

    # Patch flash_attention_2
    if "flash_attention_2" in ALL_ATTENTION_FUNCTIONS:
        _original_flash_fn = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        def flash_with_ssmax(
            module, query, key, value, attention_mask, scaling=None, **kw
        ):
            modified_scaling = (scaling or 1.0) * ssmax_scale(query.size(2))
            return _original_flash_fn(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=modified_scaling,
                **kw,
            )

        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_with_ssmax
        LOG.info(f"Patched flash_attention_2 with SSMax (factor={scaling_factor})")

    # Patch eager attention for specific models
    modules_to_patch = SUPPORTED_MODEL_MODULES
    if model_type:
        modules_to_patch = [m for m in SUPPORTED_MODEL_MODULES if model_type in m]

    for module_path in modules_to_patch:
        try:
            import importlib

            mod = importlib.import_module(module_path)
            if not hasattr(mod, "eager_attention_forward"):
                continue

            _original_eager_fns[module_path] = mod.eager_attention_forward
            original_fn = mod.eager_attention_forward

            def make_eager_ssmax(orig_fn):
                def eager_with_ssmax(
                    module: nn.Module,
                    query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor,
                    attention_mask: Optional[torch.Tensor],
                    scaling: float,
                    dropout: float = 0.0,
                    **kwargs,
                ):
                    n_rep = module.num_key_value_groups
                    key = key.repeat_interleave(n_rep, dim=1)
                    value = value.repeat_interleave(n_rep, dim=1)

                    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
                    if attention_mask is not None:
                        attn_weights = (
                            attn_weights + attention_mask[:, :, :, : key.shape[-2]]
                        )

                    attn_weights = attn_weights * ssmax_scale(attn_weights.size(-1))
                    attn_weights = F.softmax(
                        attn_weights, dim=-1, dtype=torch.float32
                    ).to(query.dtype)
                    attn_weights = F.dropout(
                        attn_weights, p=dropout, training=module.training
                    )

                    return torch.matmul(attn_weights, value).transpose(
                        1, 2
                    ).contiguous(), attn_weights

                return eager_with_ssmax

            mod.eager_attention_forward = make_eager_ssmax(original_fn)
            LOG.info(f"Patched {module_path}.eager_attention_forward with SSMax")
        except ImportError:
            pass


def unpatch_scaled_softmax_attention():
    global _original_flash_fn, _original_eager_fns
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if _original_flash_fn:
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _original_flash_fn
        _original_flash_fn = None

    for module_path, orig_fn in _original_eager_fns.items():
        try:
            import importlib

            mod = importlib.import_module(module_path)
            mod.eager_attention_forward = orig_fn
        except ImportError:
            pass
    _original_eager_fns.clear()
