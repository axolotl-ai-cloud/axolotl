"""Keep MoE router (gate) linears out of torchao fp8 conversion.

accelerate converts every ``nn.Linear`` except the model's first/last to a torchao
``Float8Linear``. When an MoE model exposes its router as an ``nn.Linear`` (e.g. Mixtral
``block_sparse_moe.gate``, Qwen3-MoE ``mlp.gate`` on transformers < 5.x), that casts the
router logits to fp8; the tensorwise-quantized logits feed ``softmax``/``topk``, so routing
decisions flip between steps and full-finetune training diverges to NaN grads. torchtitan
and torchao's own fp8 recipes keep the router in high precision for exactly this reason.

This wraps ``accelerate.accelerator.convert_model_to_fp8_ao`` so router linears are always
skipped, preserving the existing first/last-linear exclusion. Idempotent; a no-op if the
accelerate fp8 helpers are unavailable.
"""

from __future__ import annotations

from functools import partial

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)
_PATCHED = False

# Leaf attribute names used for MoE routers across HF architectures. ``gate_proj`` /
# ``gate_up_proj`` (expert projections we DO want in fp8) don't match — only the exact leaf.
_ROUTER_LEAF_NAMES = {"gate", "router"}


def _is_router(fqn: str) -> bool:
    return fqn.rsplit(".", 1)[-1] in _ROUTER_LEAF_NAMES


def patch_fp8_exclude_moe_router():
    global _PATCHED
    if _PATCHED:
        return
    try:
        import accelerate.accelerator as acc
        from accelerate.utils.ao import (
            filter_linear_layers,
            find_first_last_linear_layers,
        )
    except ImportError:
        return

    orig = acc.convert_model_to_fp8_ao

    def patched(model, config=None, module_filter_func=None):
        if module_filter_func is None:
            first, last = find_first_last_linear_layers(model)
            inner = partial(filter_linear_layers, layers_to_filter=[first, last])
        else:
            inner = module_filter_func

        def _router_aware_filter(module, fqn, _inner=inner):
            if _is_router(fqn):
                return False
            return _inner(module, fqn)

        return orig(model, config=config, module_filter_func=_router_aware_filter)

    acc.convert_model_to_fp8_ao = patched
    _PATCHED = True
    LOG.info("Patched accelerate fp8 conversion to keep MoE router (gate) in bf16")
