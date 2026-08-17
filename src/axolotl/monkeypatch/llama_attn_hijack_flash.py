"""Flash attention monkey patch for llama model"""

import importlib.util

import transformers
from transformers.models.llama.modeling_llama import LlamaMLP

from axolotl.monkeypatch.utils import set_module_name
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def is_xformers_available() -> bool:
    return importlib.util.find_spec("xformers") is not None


def is_xformers_swiglu_available() -> bool:
    if not is_xformers_available():
        return False

    from xformers.ops.common import get_xformers_operator

    try:
        get_xformers_operator("swiglu_packedw")()
        return True
    except RuntimeError as exc:
        if "No such operator xformers::swiglu_packedw " in str(exc):
            return False
        return True


def replace_llama_mlp_with_swiglu(model):
    if is_xformers_swiglu_available():
        from axolotl.monkeypatch.xformers_ import FusedMLP
    else:
        raise RuntimeError("xformers SwiGLU not available for this environment")

    for name, module in model.named_modules():
        if isinstance(module, LlamaMLP):
            mlp = FusedMLP(
                module.config, module.gate_proj, module.up_proj, module.down_proj
            )
            set_module_name(model, name, mlp)


def patch_fa_llama_cross_entropy():
    LOG.info(
        "patching transformers.loss.loss_utils.fixed_cross_entropy with flash_attn.ops.triton.cross_entropy"
    )
    from flash_attn.ops.triton.cross_entropy import (
        cross_entropy_loss as flash_attn_cross_entropy_loss,
    )

    def fa2_fixed_cross_entropy(
        source,
        target,
        num_items_in_batch: int = None,
        ignore_index: int = -100,
        **kwargs,
    ):
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss, _ = flash_attn_cross_entropy_loss(
            source, target, ignore_index=ignore_index
        )
        if reduction == "sum":
            loss = loss.sum() / num_items_in_batch
        else:
            loss = loss.sum() / (target != ignore_index).sum()
        return loss

    transformers.loss.loss_utils.fixed_cross_entropy = fa2_fixed_cross_entropy
