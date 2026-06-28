"""Clean NVFP4-MoE loading for DeepSeek-V4-Flash-NVFP4 via a FineGrainedFP8 subclass.

The checkpoint declares ``quant_method: fp8`` (→ ``FineGrainedFP8HfQuantizer``) with
``quant_algo: MIXED_PRECISION`` — the non-expert weights are blockwise FP8 ``[128,128]``,
but the routed experts are NVFP4 (``moe_quant_algo: NVFP4``, ``group_size: 16``,
``fmt: e4m3``). ``FP8Experts`` has an FP4 path (``expert_dtype == "fp4"``) but allocates a
blockwise-FP8 ``*_scale_inv`` buffer at the *wrong* granularity (``sf_gran_k=32`` + the
model-global ``scale_fmt: ue8m0``); it has no notion of NVFP4's group-16 ``*_scale``
(E4M3) + per-tensor ``*_scale_2``.

The deepseek_v4 weight converter fuses ``experts.*.w1/w3.weight`` → ``gate_up_proj`` with
*unanchored* source patterns (``mlp.experts.*.w1.weight``), so by regex search they also
match the sibling ``.weight_scale`` / ``.weight_scale_2`` keys. The group-16 ``weight_scale``
(2-D) fuses fine, but ``weight_scale_2`` is a 0-D scalar — ``Concatenate`` on stacked
scalars raises (a spurious conversion error) — and in any case there's no ``*_scale`` /
``*_scale_2`` param to receive the fused scales, so they come in UNEXPECTED and the
``*_scale_inv`` placeholders end up MISSING (random init → invalid).

This subclass loads everything in place, with no checkpoint re-read of weights/scales:
  * ``update_weight_conversions`` — anchors the experts' ``*.weight`` converters with ``$``
    (so the scalar-scale concat never runs) and adds twin ``*.weight_scale`` converters so
    the 2-D group-16 scales fuse into ``gate_up_proj_scale`` / ``down_proj_scale``;
  * ``_process_model_before_weight_loading`` — swaps the wrong ``*_scale_inv`` placeholders
    for correctly-shaped NVFP4 ``*_scale`` params, and marks the 0-D ``*_scale_2`` /
    dynamic ``*.input_scale`` keys ignorable;
  * ``_process_model_after_weight_loading`` — folds the in-place qdata + ``*_scale`` plus
    the per-expert scalar ``*_scale_2`` (the only thing re-read — 4 bytes/expert) into a
    torchao ``NVFP4Tensor`` for the scattermoe fused-LoRA path.

Net: no UNEXPECTED/MISSING warning, no random init, correct NVFP4 experts, no multi-GB
re-read. Installed by swapping it into ``AUTO_QUANTIZER_MAPPING["fp8"]`` before
``from_pretrained``.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_GROUP_SIZE = 16

# How to store the blockwise-FP8 non-expert linears after load (set from the
# `dsv4_fp8_nonexpert_mode` config via `configure_nonexpert_mode`):
#   "float8tensor" (default): wrap each plain FP8Linear weight as a torchao Float8Tensor
#       (1-byte qdata + block scale). Forward/backward + PEFT LoRA run via subclass autograd;
#       the fused LoRA kernels dequant it through axolotl.kernels.quantize.dequantize.
#   "bf16": dequantize to bf16 in place (safe path; 2 bytes/param). Also sidesteps transformers'
#       @triton_op FP8 matmul (no autograd formula; slow discovery).
# Grouped linears (o_a_proj) always go to bf16: their view+bmm forward isn't subclass-safe.
_FP8_NONEXPERT_MODE = "float8tensor"


def configure_nonexpert_mode(mode: str | None) -> None:
    """Set the FP8 non-expert storage mode from cfg before the model loads (the quantizer
    reads the module global in ``_process_model_after_weight_loading``)."""
    global _FP8_NONEXPERT_MODE
    _FP8_NONEXPERT_MODE = (mode or "float8tensor").lower()


# The only per-expert keys left unmatched once weight + weight_scale fuse in place: the 0-D
# per-tensor `weight_scale_2` (folded after load) and the unused dynamic `input_scale`.
_IGNORE_UNEXPECTED = (
    r"experts\..*\.weight_scale_2$",
    r"experts\..*\.input_scale$",
)


def _nvfp4_cls():
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        return NVFP4Tensor
    except ImportError:
        return None


def _float8_cls():
    try:
        from torchao.quantization import Float8Tensor

        return Float8Tensor
    except ImportError:
        return None


def _drop_param(mod: nn.Module, name: str) -> None:
    if name in mod._parameters:
        del mod._parameters[name]
    else:
        setattr(mod, name, None)


def _is_nvfp4_experts(module: nn.Module) -> bool:
    """An ``FP8Experts`` carrying FP4-packed (int8/uint8, K-halved) expert weights."""
    try:
        from transformers.integrations.finegrained_fp8 import FP8Experts
    except ImportError:
        return False
    if not isinstance(module, FP8Experts):
        return False
    w = getattr(module, "gate_up_proj", getattr(module, "up_proj", None))
    return (
        isinstance(w, torch.Tensor)
        and w.dtype in (torch.int8, torch.uint8)
        and w.ndim == 3
    )


def _is_expert_weight_converter(conv) -> bool:
    pats = getattr(conv, "_original_source_patterns", None) or []
    return any("experts" in p and p.endswith(".weight") for p in pats)


def _resolve_repo_file(repo: str, filename: str) -> str:
    """Resolve a checkpoint file path from a local snapshot dir or the HF hub.

    A local snapshot dir (offline/air-gapped axolotl usage) would fail ``hf_hub_download``
    with an ``HFValidationError``, so read straight from disk in that case.
    """
    if os.path.isdir(repo):
        return os.path.join(repo, filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename)


def _scale_2_path(repo: str):
    """Return ``(weight_map, opener)`` for re-reading per-expert scalar ``weight_scale_2``."""
    import json

    from safetensors import safe_open

    with open(_resolve_repo_file(repo, "model.safetensors.index.json")) as _f:
        wmap = json.load(_f)["weight_map"]
    cache: dict[str, object] = {}

    def opener(shard):
        f = cache.get(shard)
        if f is None:
            f = cache[shard] = safe_open(
                _resolve_repo_file(repo, shard), framework="pt"
            )
        return f

    return wmap, opener


def _dequantize_fp8_linears(model: nn.Module, quantizer) -> int:
    """Dequantize blockwise-FP8 linear weights to bf16 IN PLACE (preserving each module's
    class + custom forward, e.g. the grouped ``o_a_proj``). A module is targeted by the
    presence of an fp8/int8 ``weight`` + a ``weight_scale_inv`` companion — not by class —
    so ``FP8Linear`` and any quantized ``nn.Linear`` subclass are both handled. After this,
    ``FP8Linear.forward`` auto-uses plain ``F.linear`` (weight.element_size() > 1), and the
    FP8 matmul custom op (no autograd formula; slow triton kernel discovery) is gone."""
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    deq = Fp8Dequantize(quantizer)
    n = 0
    by_type: dict[str, int] = {}
    for name, mod in model.named_modules():
        wsi = getattr(mod, "weight_scale_inv", None)
        w = getattr(mod, "weight", None)
        if (
            wsi is None
            or not isinstance(w, torch.Tensor)
            or w.dtype not in (torch.float8_e4m3fn, torch.int8)
        ):
            continue
        w_bf16 = deq._dequantize_one(w.data, wsi.data, torch.bfloat16)
        if tuple(w_bf16.shape) != tuple(w.shape):
            # FP4-packed (int8, half-K) unpacks to 2x width (expected); otherwise log.
            LOG.warning(
                "dequant %s: weight %s -> %s (dtype %s)",
                name,
                tuple(w.shape),
                tuple(w_bf16.shape),
                w.dtype,
            )
        mod.weight = nn.Parameter(w_bf16, requires_grad=False)
        if "weight_scale_inv" in mod._parameters:
            del mod._parameters["weight_scale_inv"]
        else:
            mod.weight_scale_inv = None
        by_type[type(mod).__name__] = by_type.get(type(mod).__name__, 0) + 1
        n += 1
    if n:
        LOG.info("Dequantized %d FP8 linear weights to bf16 in place: %s", n, by_type)
    return n


def _wrap_fp8_linears_as_float8tensor(model: nn.Module, quantizer) -> int:
    """Wrap blockwise-FP8 *plain* linear weights as torchao ``Float8Tensor`` (1-byte qdata +
    block scale) so the frozen base keeps the FP8 memory footprint while forward/backward and
    PEFT LoRA run through the tensor subclass. The checkpoint's exact qdata + ``weight_scale_inv``
    are reused (no re-quant). Grouped linears (``o_a_proj``), whose ``view+bmm`` forward isn't
    subclass-safe, are dequantized to bf16 in place instead."""
    Float8Tensor = _float8_cls()
    if Float8Tensor is None:
        LOG.warning(
            "torchao Float8Tensor unavailable; dequantizing FP8 non-experts to bf16"
        )
        return _dequantize_fp8_linears(model, quantizer)
    from transformers.integrations.finegrained_fp8 import (
        Fp8Dequantize,
        FP8GroupedLinear,
    )

    deq = Fp8Dequantize(quantizer)
    wrapped = bf16ed = 0
    for _name, mod in model.named_modules():
        wsi = getattr(mod, "weight_scale_inv", None)
        w = getattr(mod, "weight", None)
        if (
            wsi is None
            or not isinstance(w, torch.Tensor)
            or w.dtype != torch.float8_e4m3fn
        ):
            continue
        if isinstance(mod, FP8GroupedLinear) or (getattr(mod, "n_groups", 1) or 1) > 1:
            mod.weight = nn.Parameter(
                deq._dequantize_one(w.data, wsi.data, torch.bfloat16),
                requires_grad=False,
            )
            _drop_param(mod, "weight_scale_inv")
            bf16ed += 1
            continue
        block = list(mod.block_size) if getattr(mod, "block_size", None) else [128, 128]
        scale = wsi.data
        if scale.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            scale = scale.to(
                torch.float32
            )  # ue8m0 scales have no float ops; torchao wants float
        f8 = Float8Tensor(w.data, scale, block_size=block, dtype=torch.bfloat16)
        mod.weight = nn.Parameter(f8, requires_grad=False)
        _drop_param(mod, "weight_scale_inv")
        wrapped += 1
    if wrapped or bf16ed:
        LOG.info(
            "FP8 non-experts: wrapped %d plain linears as Float8Tensor (1-byte), %d grouped -> bf16",
            wrapped,
            bf16ed,
        )
    return wrapped + bf16ed


def _enable_torchao_lora_dispatch(quantizer) -> None:
    """Let PEFT attach LoRA to the Float8Tensor non-expert bases.

    PEFT's ``dispatch_torchao`` fires because the base weights are torchao ``Float8Tensor``
    and routes to ``TorchaoLoraLinear``, which needs
    ``model.hf_quantizer.quantization_config.get_apply_tensor_subclass``. PEFT sources it via
    ``operator.attrgetter`` and silently skips on ``AttributeError`` (we ship a FineGrained-FP8
    config, not a ``TorchAoConfig``), but the dispatcher then errors on the missing kwarg.
    It's used ONLY by merge/unmerge (re-quantize the merged weight) — never in training (frozen
    base, LoRA kept separate) — so provide a callable that unblocks training and fails loudly
    on merge (blockwise-FP8 merge isn't supported yet)."""
    cfg = getattr(quantizer, "quantization_config", None)
    if cfg is None:
        return
    cls = type(cfg)
    if hasattr(cls, "get_apply_tensor_subclass"):
        return

    def _no_merge():
        raise NotImplementedError(
            "merge-lora on a blockwise-FP8 (Float8Tensor) base is not supported; train/serve "
            "with the adapter kept separate, or set `dsv4_fp8_nonexpert_mode: bf16` to merge."
        )

    # Attach to the CLASS as a staticmethod (not the instance): PEFT's attrgetter still resolves
    # it via attribute lookup, but it stays out of the instance __dict__ so `config.to_json`
    # (model.config save) doesn't try to serialize a function object.
    try:
        cls.get_apply_tensor_subclass = staticmethod(_no_merge)
    except Exception:  # pragma: no cover - some configs are frozen/slotted
        LOG.warning(
            "Could not attach get_apply_tensor_subclass; non-expert LoRA may fail to inject"
        )


def make_nvfp4_fp8_quantizer():
    """Build the ``FineGrainedFP8HfQuantizer`` subclass (deferred import)."""
    from transformers.core_model_loading import WeightConverter
    from transformers.quantizers.quantizer_finegrained_fp8 import (
        FineGrainedFP8HfQuantizer,
    )

    class NVFP4MoEFP8HfQuantizer(FineGrainedFP8HfQuantizer):
        """FineGrained-FP8 quantizer that also loads NVFP4 routed experts correctly."""

        def update_weight_conversions(self, weight_conversions):
            rebuilt = []
            for conv in weight_conversions:
                if not (
                    isinstance(conv, WeightConverter)
                    and _is_expert_weight_converter(conv)
                ):
                    rebuilt.append(conv)
                    continue
                ops = [type(op)(op.dim) for op in conv.operations]
                # weight: anchor `.weight$` so the converter fuses ONLY the packed weight
                # and stops swallowing `.weight_scale*` (the 0-D `_scale_2` concat raises).
                w_conv = WeightConverter(
                    source_patterns=[
                        p + "$" if p.endswith(".weight") else p
                        for p in conv._original_source_patterns
                    ],
                    target_patterns=list(conv._original_target_patterns),
                    operations=ops,
                )
                # weight_scale: twin converter so the 2-D group-16 scales fuse the same way
                # into `*_scale`, loading in place (no re-read).
                s_conv = WeightConverter(
                    source_patterns=[
                        p[: -len(".weight")] + ".weight_scale$"
                        if p.endswith(".weight")
                        else p
                        for p in conv._original_source_patterns
                    ],
                    target_patterns=[
                        t + "_scale" for t in conv._original_target_patterns
                    ],
                    operations=[type(op)(op.dim) for op in conv.operations],
                )
                for new in (w_conv, s_conv):
                    new.scope_prefix = conv.scope_prefix
                    new.base_model_prefix = conv.base_model_prefix
                rebuilt.extend((w_conv, s_conv))
            return super().update_weight_conversions(rebuilt)

        def _process_model_before_weight_loading(self, model, **kwargs):
            super()._process_model_before_weight_loading(model, **kwargs)
            n = 0
            for mod in model.modules():
                if not _is_nvfp4_experts(mod):
                    continue
                E, H, I = mod.num_experts, mod.hidden_dim, mod.intermediate_dim
                dev = (mod.gate_up_proj if mod.has_gate else mod.up_proj).device
                # drop the wrong-granularity blockwise-FP8 placeholders; register NVFP4
                # group-16 E4M3 `*_scale` params so the fused scales land in place.
                pfx, out_dim = (
                    ("gate_up_proj", 2 * I) if mod.has_gate else ("up_proj", I)
                )
                for stale in (pfx + "_scale_inv", "down_proj_scale_inv"):
                    mod._parameters.pop(stale, None)
                mod.register_parameter(
                    pfx + "_scale",
                    nn.Parameter(
                        torch.empty(
                            E,
                            out_dim,
                            H // _GROUP_SIZE,
                            dtype=torch.float8_e4m3fn,
                            device=dev,
                        ),
                        requires_grad=False,
                    ),
                )
                mod.register_parameter(
                    "down_proj_scale",
                    nn.Parameter(
                        torch.empty(
                            E,
                            H,
                            I // _GROUP_SIZE,
                            dtype=torch.float8_e4m3fn,
                            device=dev,
                        ),
                        requires_grad=False,
                    ),
                )
                n += 1
            if n:
                ignore = set(model._keys_to_ignore_on_load_unexpected or ())
                ignore.update(_IGNORE_UNEXPECTED)
                model._keys_to_ignore_on_load_unexpected = ignore
                LOG.info(
                    "Prepared %d NVFP4 experts modules for clean in-place loading", n
                )

        def _process_model_after_weight_loading(self, model, **kwargs):
            super()._process_model_after_weight_loading(model, **kwargs)
            # The swap into AUTO_QUANTIZER_MAPPING['fp8'] is process-global, so this runs for EVERY
            # fp8 checkpoint loaded after install. Gate all NVFP4-specific work on actually finding
            # NVFP4 experts; a plain fp8 model behaves exactly like the original quantizer (super()
            # above) and is NOT re-wrapped as Float8Tensor.
            mods = [
                (name, m) for name, m in model.named_modules() if _is_nvfp4_experts(m)
            ]
            if not mods:
                return
            if _FP8_NONEXPERT_MODE == "bf16":
                n = _dequantize_fp8_linears(model, self)
                if n:
                    LOG.info("Dequantized %d non-expert FP8Linear modules to bf16", n)
            else:
                n = _wrap_fp8_linears_as_float8tensor(model, self)
                if n:
                    _enable_torchao_lora_dispatch(self)
            NVFP4Tensor = _nvfp4_cls()
            if NVFP4Tensor is None:
                LOG.warning("torchao NVFP4Tensor unavailable; skipping expert wrap")
                return
            repo = getattr(model, "name_or_path", None) or getattr(
                model.config, "_name_or_path", None
            )
            wmap, opener = _scale_2_path(repo)
            import re

            for name, mod in mods:
                layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
                pfx = "gate_up_proj" if mod.has_gate else "up_proj"
                projs = {
                    "gate_up_proj": ("w1",),
                    "up_proj": ("w1",),
                    "down_proj": ("w2",),
                }
                for proj in (pfx, "down_proj"):
                    qdata = getattr(mod, proj).data.view(torch.uint8)
                    scale = getattr(mod, proj + "_scale").data
                    # w1/w3 share weight_scale_2; one scalar per expert → [E,1,1] broadcast.
                    src = projs[proj][0]
                    s2 = (
                        torch.stack(
                            [
                                opener(
                                    wmap[
                                        f"layers.{layer}.ffn.experts.{e}.{src}.weight_scale_2"
                                    ]
                                ).get_tensor(
                                    f"layers.{layer}.ffn.experts.{e}.{src}.weight_scale_2"
                                )
                                for e in range(mod.num_experts)
                            ]
                        )
                        .to(device=qdata.device, dtype=torch.float32)
                        .view(-1, 1, 1)
                    )
                    setattr(
                        mod,
                        proj,
                        nn.Parameter(
                            NVFP4Tensor(
                                qdata,
                                scale,
                                _GROUP_SIZE,
                                torch.bfloat16,
                                per_tensor_scale=s2,
                            ),
                            requires_grad=False,
                        ),
                    )
                    mod._parameters.pop(proj + "_scale", None)
            LOG.info(
                "Wrapped %d NVFP4 experts modules as NVFP4Tensor (in-place scales)",
                len(mods),
            )

    return NVFP4MoEFP8HfQuantizer


_ORIGINAL_FP8_QUANTIZER = None


def install_nvfp4_fp8_quantizer() -> None:
    """Swap the NVFP4-aware quantizer into ``AUTO_QUANTIZER_MAPPING["fp8"]`` so any
    ``quant_method: fp8`` checkpoint with NVFP4 experts loads correctly. Idempotent."""
    global _ORIGINAL_FP8_QUANTIZER
    from transformers.quantizers import auto as _auto

    cls = make_nvfp4_fp8_quantizer()
    if getattr(_auto.AUTO_QUANTIZER_MAPPING.get("fp8"), "__name__", "") == cls.__name__:
        return
    _ORIGINAL_FP8_QUANTIZER = _auto.AUTO_QUANTIZER_MAPPING.get("fp8")
    _auto.AUTO_QUANTIZER_MAPPING["fp8"] = cls
    LOG.info("Installed NVFP4-aware FP8 quantizer into AUTO_QUANTIZER_MAPPING['fp8']")


def uninstall_nvfp4_fp8_quantizer() -> None:
    """Restore the original ``AUTO_QUANTIZER_MAPPING["fp8"]`` entry the swap replaced (so a long-lived
    process can return to stock fp8 loading). Idempotent; a no-op if install was never called."""
    global _ORIGINAL_FP8_QUANTIZER
    if _ORIGINAL_FP8_QUANTIZER is None:
        return
    from transformers.quantizers import auto as _auto

    _auto.AUTO_QUANTIZER_MAPPING["fp8"] = _ORIGINAL_FP8_QUANTIZER
    _ORIGINAL_FP8_QUANTIZER = None
    LOG.info("Restored original FP8 quantizer in AUTO_QUANTIZER_MAPPING['fp8']")
