"""In-place NVFP4 W4A16 quantization of gemma4 non-expert linears (Marlin path).

Mirrors :mod:`gemma4_nf4_nonexpert` but swaps each frozen non-expert ``nn.Linear`` for a
:class:`MarlinW4A16Linear` — 4-bit NVFP4 weights run through the same validated Marlin kernel as
the grouped MoE experts (bf16 act/out, no FP4 hardware needed; sm80+). Vs the bnb NF4 path this
keeps experts and non-experts on ONE tensor-core 4-bit code path instead of bnb's dequant kernel.

Targets every ``nn.Linear`` that is NOT inside an expert block; norms, embeddings, routers,
vision/audio towers, and lm_head are skipped by name. No LoRA is attached (experts-only).
"""

from __future__ import annotations

import torch.nn as nn

from axolotl.utils.logging import get_logger

from .gemma4_fp8_nonexpert import _should_skip_by_name
from .gemma4_nf4_nonexpert import _expert_paths

LOG = get_logger(__name__)


def quantize_gemma4_nonexpert_nvfp4(model: nn.Module) -> int:
    """Swap non-expert ``nn.Linear`` modules for frozen NVFP4 ``MarlinW4A16Linear``, in place.
    Returns the count swapped. Experts (NVFP4Tensor) are untouched. Idempotent."""
    from .marlin_w4a16 import marlin_w4a16_available
    from .marlin_w4a16.nonexpert_linear import MarlinW4A16Linear

    if not marlin_w4a16_available():
        raise RuntimeError(
            "nonexpert_quantization=nvfp4 needs the Marlin W4A16 kernel (sm80+ GPU + CUDA "
            "toolkit). Use nonexpert_quantization=nf4 (bitsandbytes) on unsupported setups."
        )

    expert_paths = _expert_paths(model)

    def _under_expert(path: str) -> bool:
        return any(path == ep or path.startswith(ep + ".") for ep in expert_paths)

    targets: list[tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, MarlinW4A16Linear):
            continue
        if not isinstance(mod, nn.Linear):
            continue
        if _under_expert(name) or _should_skip_by_name(name):
            continue
        if not isinstance(mod.weight, nn.Parameter) or mod.weight.ndim != 2:
            continue
        targets.append((name, mod))

    for name, mod in targets:
        new = MarlinW4A16Linear(
            mod.weight.data, mod.bias.data if mod.bias is not None else None
        )
        parent_path, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, new)

    if targets:
        LOG.info(
            "Gemma4 NVFP4 non-expert: swapped %d non-expert linears to Marlin W4A16 "
            "(4-bit weight, bf16 compute)",
            len(targets),
        )
    return len(targets)
