"""In-place NF4 (bitsandbytes) quantization of gemma4 non-expert linears.

This mirrors :mod:`gemma4_fp8_nonexpert` but uses bitsandbytes 4-bit NF4 — the
SAME non-expert compute path unsloth's QLoRA uses — so an experts-only LoRA run
can be compared apples-to-apples against unsloth: identical frozen-NF4 non-expert
matmuls, the only difference being the routed-expert path (our NVFP4
grouped/marlin + scatter LoRA vs unsloth's NF4 experts).

Targets every ``nn.Linear`` that is NOT inside a ``Gemma4TextExperts`` block
(experts stay NVFP4Tensor, untouched).  Norms, embeddings, router projections,
vision/audio towers, and lm_head are skipped by name.  Each target becomes a
frozen ``bnb.nn.Linear4bit`` (nf4, double-quant, bf16 compute) — no LoRA is
attached there (experts-only), exactly like unsloth's non-expert state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from axolotl.utils.logging import get_logger

from .gemma4_fp8_nonexpert import _should_skip_by_name

LOG = get_logger(__name__)


def _expert_paths(model: nn.Module) -> set[str]:
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

        cls = Gemma4TextExperts
    except ImportError:
        cls = None
    paths: set[str] = set()
    if cls is not None:
        for path, mod in model.named_modules():
            if isinstance(mod, cls):
                paths.add(path)
    return paths


def quantize_gemma4_nonexpert_nf4(
    model: nn.Module, compute_dtype: torch.dtype = torch.bfloat16
) -> int:
    """Swap all non-expert ``nn.Linear`` modules for frozen bnb NF4 ``Linear4bit``
    (double-quant, bf16 compute), in place.  Returns the count swapped.  Experts
    (NVFP4Tensor) are untouched.  Idempotent: existing ``Linear4bit`` are skipped.
    """
    import bitsandbytes as bnb

    expert_paths = _expert_paths(model)

    def _under_expert(path: str) -> bool:
        return any(path == ep or path.startswith(ep + ".") for ep in expert_paths)

    device = next(
        (p.device for p in model.parameters() if p.is_cuda),
        torch.device("cuda", torch.cuda.current_device()),
    )

    targets: list[tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, bnb.nn.Linear4bit):
            continue
        if not isinstance(mod, nn.Linear):
            continue
        if _under_expert(name) or _should_skip_by_name(name):
            continue
        if not isinstance(mod.weight, nn.Parameter) or mod.weight.ndim != 2:
            continue
        targets.append((name, mod))

    by_type: dict[str, int] = {}
    for name, mod in targets:
        has_bias = mod.bias is not None
        new = bnb.nn.Linear4bit(
            mod.in_features,
            mod.out_features,
            bias=has_bias,
            compute_dtype=compute_dtype,
            quant_type="nf4",
        )
        new.weight = bnb.nn.Params4bit(
            mod.weight.data.to(torch.bfloat16),
            requires_grad=False,
            quant_type="nf4",
            compress_statistics=True,
        )
        if has_bias:
            new.bias = nn.Parameter(
                mod.bias.data.to(compute_dtype), requires_grad=False
            )
        new = new.to(device)  # triggers NF4 quantization of Params4bit

        parent_path, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, new)
        by_type[type(mod).__name__] = by_type.get(type(mod).__name__, 0) + 1

    if targets:
        LOG.info(
            "Gemma4 NF4 frankenstein: swapped %d non-expert linears to bnb NF4 "
            "(double-quant, bf16 compute): %s",
            len(targets),
            by_type,
        )
    return len(targets)
