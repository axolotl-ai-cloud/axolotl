"""Repairs for transformers' kernelization under ``use_kernels=True``.

``set_use_kernels(True)`` calls ``kernelize()``, which swaps each module's
``_hidden_kernels`` entries for hub kernels, but two upstream defects crash it:
~30 architectures (gemma4, qwen3.5, glm4, olmo, ...) stash a bare function it
refuses to register, and gpt-oss's rotary ``Func`` keeps a deprecated
``position_ids`` parameter that fails the kernels library's signature check
against ``kernels-community/rotary``.

Wraps ``PreTrainedModel.set_use_kernels`` to repair the stashes right before the
swap: bare functions are dropped (the model's forward still calls them
directly) and ``position_ids`` is removed from rotary signature *metadata*
(call behavior unchanged). Both repairs no-op once fixed upstream.
"""

import inspect

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_ORIG_SET_USE_KERNELS = None


def _fix_hidden_kernels(module):
    import torch.nn as nn

    for name, fn in list(module.__dict__.get("_hidden_kernels", {}).items()):
        if not isinstance(fn, nn.Module):
            del module._hidden_kernels[name]
        elif getattr(fn, "kernel_layer_name", None) == "rotary_pos_emb":
            forward = type(fn).forward
            signature = inspect.signature(forward)
            if "position_ids" in signature.parameters:
                forward.__signature__ = signature.replace(
                    parameters=[
                        p
                        for p in signature.parameters.values()
                        if p.name != "position_ids"
                    ]
                )


def patch_kernelize_fixes() -> bool:
    global _ORIG_SET_USE_KERNELS
    if _ORIG_SET_USE_KERNELS is not None:
        return True

    from transformers.modeling_utils import PreTrainedModel

    orig = getattr(PreTrainedModel, "set_use_kernels", None)
    if orig is None:
        LOG.warning(
            "kernelize_fixes: PreTrainedModel.set_use_kernels not found, skipping"
        )
        return False

    def set_use_kernels(self, use_kernels, *args, **kwargs):
        if use_kernels:
            self.apply(_fix_hidden_kernels)
        return orig(self, use_kernels, *args, **kwargs)

    PreTrainedModel.set_use_kernels = set_use_kernels
    _ORIG_SET_USE_KERNELS = orig
    LOG.info("kernelize_fixes: patched PreTrainedModel.set_use_kernels")
    return True


def unpatch_kernelize_fixes() -> None:
    global _ORIG_SET_USE_KERNELS
    if _ORIG_SET_USE_KERNELS is None:
        return
    from transformers.modeling_utils import PreTrainedModel

    PreTrainedModel.set_use_kernels = _ORIG_SET_USE_KERNELS
    _ORIG_SET_USE_KERNELS = None
