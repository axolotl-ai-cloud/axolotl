"""Native WeightConverter so Gemma-4 NVFP4 MoE experts load as NVFP4Tensor.

nvidia/Gemma-4-26B-A4B-NVFP4 ships per-expert weights under
  ``model.language_model.layers.N.experts.E.{gate_proj,up_proj,down_proj}.{weight,weight_scale,weight_scale_2}``
but ``quant_method: modelopt`` is not a recognized transformers quantizer, so
the model loads as a BF16 skeleton with the per-expert NVFP4 tensors landing
as UNEXPECTED and the fused ``gate_up_proj``/``down_proj`` remaining random BF16.

This module registers a ``WeightConverter`` for the ``gemma4_text`` model type
that fuses the per-expert raw uint8 qdata + E4M3 block scales + per-tensor
scalar into a single ``NVFP4Tensor`` (packed 4-bit) and assigns it in-place to
the ``Gemma4TextExperts`` module — exactly like ``Mxfp4Deserialize`` does for
MXFP4.  ``is_nvfp4_param(param)`` returns ``True`` on the result, activating
the scattermoe fused NVFP4 path.

The fusion itself (stack experts, cat gate/up on the N axis, reconcile the per-tensor scales)
is the shared ``nvfp4_moe_loading.fuse_nvfp4_experts`` core, so the WeightConverter (modelopt
skeleton load) and the post-load scale-attach path use one implementation instead of duplicating
the NVFP4-expert math.

Registration is done via ``transformers.conversion_mapping.register_checkpoint_conversion_mapping``
— no site-packages edits.  The registration helper is gated: call it only when
the model is gemma4 + NVFP4 modelopt.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_FP8_DTYPES = tuple(
    getattr(torch, n)
    for n in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if hasattr(torch, n)
)


def _nvfp4_cls():
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        return NVFP4Tensor
    except ImportError:
        return None


# Set True only by patch_conversion_loader_rank0_only() (installed solely under FSDP
# cpu_ram_efficient_loading). Gates the converter meta-load so non-rank-0 stays on meta ONLY when the
# FSDP broadcast will later fill it — otherwise (e.g. DDP, or the example without the flag) every rank
# loads real weights as normal.
_RANK0_ONLY_ACTIVE = False


def _nonrank0_meta_load() -> bool:
    """True when cpu_ram_efficient broadcast loading wants this rank to stay on meta.

    The safetensors are mmap'd, so a converter that calls ``torch.stack`` on the checkpoint
    tensors copies real data into RAM on EVERY rank — violating the rank0-only contract and
    blowing CPU RAM up by the world size. On non-local-rank-0 we instead emit correctly-shaped
    META params; FSDP's ``fsdp2_load_full_state_dict`` then broadcasts rank 0's real weights.
    (With the ``_materialize_copy`` patch active the converter already receives meta inputs; this
    stays as a defensive second gate.)"""
    return _RANK0_ONLY_ACTIVE and _is_nonrank0_process()


def _to_meta(t):
    import torch as _torch

    return _torch.empty(t.shape, dtype=t.dtype, device="meta")


def _is_nonrank0_process() -> bool:
    """True on every process except local-rank-0 of the node.

    torchrun always exports ``LOCAL_RANK``; we key off it directly rather than transformers'
    ``is_fsdp_enabled()`` because that ALSO requires ``ACCELERATE_USE_FSDP`` /
    ``FSDP_CPU_RAM_EFFICIENT_LOADING`` in the env, which ``axolotl train`` does NOT set at
    ``from_pretrained`` time — so transformers' own rank0-gate is dormant during the load. ``-1``
    (unset) means a single non-distributed process → load normally."""
    import os

    return int(os.environ.get("LOCAL_RANK", "-1")) > 0


def patch_conversion_loader_rank0_only() -> None:
    """Make transformers' conversion-based loader load rank0-only (for FSDP cpu_ram_efficient).

    transformers' new ``core_model_loading`` loader (engaged whenever a ``conversion_mapping`` is
    registered — i.e. our NVFP4 converters) materializes every tensor on every rank: it only shards
    per-rank when a ``tp_plan`` is set, and otherwise has NO rank gating. Under FSDP that loads the
    full model on all ``world_size`` ranks → an N× CPU-RAM blowup that OOMs large models. (And even
    the legacy gate it would mirror is dormant here — see :func:`_is_nonrank0_process`.)

    On non-local-rank-0 return a META tensor from ``_materialize_copy`` so only rank 0 holds real
    weights; ``fsdp2_load_full_state_dict`` then broadcasts them. The mmap'd slice read still happens
    (cheap, shared OS page cache) but the copy is dropped to meta, so non-rank-0 never accumulates.
    Install ONLY when cpu_ram_efficient broadcast loading is active, else it would starve DDP ranks
    that each need their own real weights. Idempotent."""
    global _RANK0_ONLY_ACTIVE
    _RANK0_ONLY_ACTIVE = True

    import os

    import torch as _torch
    import transformers.core_model_loading as cml

    # transformers caps the conversion-loader ThreadPoolExecutor at min(4, cpu_count). Measured NOT
    # to help GLM-5.2 load time (the bottleneck is the SERIAL main-thread converter/match loop over
    # ~240k source tensors, not the materialize I/O — disk is fast NVMe). Left as an explicit opt-in
    # (AXOLOTL_LOAD_WORKERS=<n>) in case it helps on a slow-disk/network box; no default change.
    try:
        _w = int(os.environ.get("AXOLOTL_LOAD_WORKERS", "0") or 0)
        if _w > getattr(cml, "GLOBAL_WORKERS", 4):
            cml.GLOBAL_WORKERS = _w
            LOG.info("Raised transformers conversion-loader workers to %d", _w)
    except Exception:  # pylint: disable=broad-except
        pass

    if getattr(cml._materialize_copy, "_axolotl_rank0_patched", False):
        return

    _orig = cml._materialize_copy

    # safetensors dtype-string -> torch dtype, for building meta tensors without reading the slice.
    _ST_DTYPE = {
        "F64": _torch.float64,
        "F32": _torch.float32,
        "F16": _torch.float16,
        "BF16": _torch.bfloat16,
        "F8_E4M3": _torch.float8_e4m3fn,
        "F8_E5M2": _torch.float8_e5m2,
        "I64": _torch.int64,
        "I32": _torch.int32,
        "I16": _torch.int16,
        "I8": _torch.int8,
        "U8": _torch.uint8,
        "BOOL": _torch.bool,
    }

    def _meta_from_slice(tensor, dtype):
        """Build a meta tensor matching what ``_orig`` WOULD return, without reading from disk."""
        if not (hasattr(tensor, "get_shape") and hasattr(tensor, "get_dtype")):
            return None
        out_dtype = dtype if dtype is not None else _ST_DTYPE.get(tensor.get_dtype())
        if out_dtype is None:
            return None
        return _torch.empty(tensor.get_shape(), dtype=out_dtype, device="meta")

    _stats = {"meta": 0, "real": 0, "fallback": 0, "logged": False}

    def _dbg(msg):
        import sys

        print(
            f"[MATCOPY rank={os.environ.get('LOCAL_RANK', '?')}] {msg}",
            file=sys.stderr,
            flush=True,
        )

    def _patched_materialize_copy(tensor, device=None, dtype=None):
        # dtype-aware: load the RAW NVFP4 components (uint8 qdata / fp8 e4m3|e5m2 scales) at their
        # native dtype instead of the bf16 skeleton dtype. modelopt is an unrecognized quantizer so
        # transformers' own native-dtype branch (hf_quantizer.pre_quantized) never fires and it casts
        # everything to bf16 — doubling the bytes of the largest (uint8) expert tensors and adding a
        # cast the converter's _recast_weight/_recast_scale would just undo. Applied before the
        # rank0/meta split so both produce the same native dtype. (F32/F16/BF16 sources unaffected.)
        if dtype is not None and hasattr(tensor, "get_dtype"):
            _st = tensor.get_dtype()
            if _st == "U8" or _st.startswith("F8"):
                dtype = None
        nonrank0 = _is_nonrank0_process()
        if not _stats["logged"]:
            _dbg(f"first call: nonrank0={nonrank0} slice_type={type(tensor).__name__}")
            _stats["logged"] = True
        # Non-rank-0: never touch the data — produce a same-shape/dtype META tensor so the slice is
        # never read into RAM (reading-then-discarding leaves glibc holding the freed pages).
        if nonrank0:
            meta = _meta_from_slice(tensor, dtype)
            if meta is not None:
                _stats["meta"] += 1
                if _stats["meta"] % 2000 == 0:
                    _dbg(f"meta={_stats['meta']} fallback={_stats['fallback']}")
                return meta
            # Fallback (unknown slice type): read then drop to meta — correctness over memory.
            _stats["fallback"] += 1
            if _stats["fallback"] % 500 == 0:
                _dbg(
                    f"FALLBACK count={_stats['fallback']} type={type(tensor).__name__}"
                )
            out = _orig(tensor, device=device, dtype=dtype)
            return _torch.empty(out.shape, dtype=out.dtype, device="meta")
        _stats["real"] += 1
        return _orig(tensor, device=device, dtype=dtype)

    _patched_materialize_copy._axolotl_rank0_patched = True  # type: ignore[attr-defined]
    cml._materialize_copy = _patched_materialize_copy
    LOG.info(
        "Patched transformers core_model_loading._materialize_copy for rank0-only loading "
        "(LOCAL_RANK=%s)",
        os.environ.get("LOCAL_RANK", "-1"),
    )


class Nvfp4ExpertsDeserialize:
    """ConversionOps that fuses per-expert NVFP4 tensors into a single NVFP4Tensor.

    For gate_up_proj, ``input_dict`` contains four keys (the source_patterns):
      - ``"experts.*.gate_proj.weight"``  → list of E uint8 tensors  [I, H/2]
      - ``"experts.*.up_proj.weight"``    → list of E uint8 tensors  [I, H/2]
      - ``"experts.*.gate_proj.weight_scale"``  → list of E e4m3 tensors [I, H/16]
      - ``"experts.*.up_proj.weight_scale"``    → list of E e4m3 tensors [I, H/16]
      - ``"experts.*.gate_proj.weight_scale_2"`` → list of E float32 scalars (use first)

    For down_proj, ``input_dict`` contains:
      - ``"experts.*.down_proj.weight"``         → list of E uint8 tensors  [H, I/2]
      - ``"experts.*.down_proj.weight_scale"``   → list of E e4m3 tensors [H, I/16]
      - ``"experts.*.down_proj.weight_scale_2"`` → list of E float32 scalars

    The op attaches the fused NVFP4Tensor to the module in-place and returns ``{}``
    so the loader does not try to materialize the original meta-parameter names.
    """

    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str] | None = None,
        target_patterns: list[str] | None = None,
        full_layer_name: str | None = None,
        model: nn.Module | None = None,
        missing_keys: set | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        from transformers.quantizers.quantizers_utils import get_module_from_name

        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
            fuse_nvfp4_experts,
        )

        if full_layer_name is not None and "gate_up_proj" in full_layer_name:
            proj = "gate_up_proj"
        elif full_layer_name is not None and full_layer_name.endswith("up_proj"):
            # non-gated (nemotron_h) layout: single up projection, no gate to fuse
            proj = "up_proj"
        else:
            proj = "down_proj"

        # cpu_ram_efficient_loading: only local-rank-0 materializes; others stay on meta and get
        # filled by the FSDP broadcast. Drop the mmap'd checkpoint data to meta BEFORE fusing.
        if _nonrank0_meta_load():
            input_dict = {
                k: (
                    [_to_meta(t) for t in v]
                    if isinstance(v, (list, tuple))
                    else _to_meta(v)
                )
                for k, v in input_dict.items()
            }

        def _find(pat_suffix: str) -> list[torch.Tensor]:
            """Find the tensor list for a source pattern that ends with pat_suffix."""
            for key, tensors in input_dict.items():
                if key.endswith(pat_suffix):
                    return tensors
            raise KeyError(
                f"Nvfp4ExpertsDeserialize: could not find '{pat_suffix}' in "
                f"input_dict keys: {list(input_dict.keys())}"
            )

        # spawn_materialize casts all checkpoint tensors to the skeleton dtype (bf16) before
        # the converter sees them. uint8 qdata (0-255) and float8_e4m3fn scales both roundtrip
        # exactly through bf16, so recast back to the raw dtypes NVFP4Tensor needs.
        def _recast_weight(t: torch.Tensor) -> torch.Tensor:
            if t.dtype != torch.uint8:
                return t.to(torch.int32).to(torch.uint8)
            return t

        def _recast_scale(t: torch.Tensor) -> torch.Tensor:
            if t.dtype != torch.float8_e4m3fn:
                return t.to(torch.float8_e4m3fn)
            return t

        def _proj_parts(proj_name: str) -> dict:
            return {
                "qd": [_recast_weight(t) for t in _find(f"{proj_name}.weight")],
                "sc": [_recast_scale(t) for t in _find(f"{proj_name}.weight_scale")],
                "pts": list(_find(f"{proj_name}.weight_scale_2")),
            }

        # gate/up fuse on the N axis (each ships its own per-tensor scale, reconciled in the core);
        # up (non-gated) and down are single projections. Fusion + scale reconciliation live in
        # fuse_nvfp4_experts.
        if proj == "gate_up_proj":
            projs = [_proj_parts("gate_proj"), _proj_parts("up_proj")]
        else:
            projs = [_proj_parts(proj)]
        nvfp4 = fuse_nvfp4_experts(projs)

        module, _ = get_module_from_name(model, full_layer_name)

        setattr(module, proj, nn.Parameter(nvfp4, requires_grad=False))

        if missing_keys is not None:
            missing_keys.discard(full_layer_name)

        module._is_hf_initialized = True

        LOG.debug(
            "Nvfp4ExpertsDeserialize: set %s as NVFP4Tensor [%s]",
            full_layer_name,
            list(nvfp4.shape),
        )
        return {}

    # No meaningful reverse op (packed NVFP4 → checkpoint would need to unfuse).
    @property
    def reverse_op(self):
        from transformers.core_model_loading import _IdentityOp

        return _IdentityOp()


class Nvfp4LinearDequantize:
    """ConversionOps that dequantizes one non-routed ``nn.Linear`` weight to bf16 in place.

    ``input_dict`` holds a single module's source tensors (each a 1-element list under a
    non-wildcard source pattern); ``full_layer_name`` is the target ``....weight``.

    modelopt MIXED_PRECISION exports quantize per LAYER, not per module name, while converters
    are registered per suffix, so the branch comes from the keys THIS layer ships:
    ``weight_scale_2`` -> NVFP4 (uint8 qdata ``[out, in/2]`` + e4m3 group-16 scale + per-tensor
    scalar), ``weight_scale`` alone -> static FP8 (``weight · scale``), neither -> passthrough.
    """

    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str] | None = None,
        target_patterns: list[str] | None = None,
        full_layer_name: str | None = None,
        model: nn.Module | None = None,
        missing_keys: set | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        from transformers.quantizers.quantizers_utils import get_module_from_name

        def _get(pat_suffix: str):
            for key, val in input_dict.items():
                if key.endswith(pat_suffix):
                    return val[0] if isinstance(val, (list, tuple)) else val
            return None

        def _one(pat_suffix: str) -> torch.Tensor:
            val = _get(pat_suffix)
            if val is None:
                raise KeyError(
                    f"Nvfp4LinearDequantize: could not find '{pat_suffix}' in "
                    f"input_dict keys: {list(input_dict.keys())}"
                )
            return val

        w = _one(".weight")
        # key presence, not dtype: dtype is unavailable on the meta path.
        is_nvfp4 = _get(".weight_scale_2") is not None
        is_fp8 = not is_nvfp4 and _get(".weight_scale") is not None

        module, param_name = get_module_from_name(model, full_layer_name)
        if is_nvfp4 and _nvfp4_cls() is None:
            # checked before the meta branch so every rank fails, not just rank 0
            raise RuntimeError(
                "torchao.prototype.mx_formats.nvfp4_tensor.NVFP4Tensor not found; "
                "install torchao with NVFP4 support"
            )

        pts = None
        # cpu_ram_efficient_loading: non-local-rank-0 stays on meta (FSDP broadcasts rank 0's
        # weights). qdata is packed [out, in/2] uint8 -> dequantized weight is [out, in] bf16.
        if _nonrank0_meta_load():
            shape = (w.shape[0], w.shape[1] * 2) if is_nvfp4 else tuple(w.shape)
            weight = torch.empty(shape, dtype=torch.bfloat16, device="meta")
        elif is_nvfp4:
            NVFP4Tensor = _nvfp4_cls()
            # spawn_materialize casts checkpoint tensors to the skeleton dtype (bf16); recast the
            # raw uint8 qdata / e4m3 scale back (both roundtrip exactly through bf16).
            qdata = w if w.dtype == torch.uint8 else w.to(torch.int32).to(torch.uint8)
            sc = _one(".weight_scale")
            scale = (
                sc if sc.dtype == torch.float8_e4m3fn else sc.to(torch.float8_e4m3fn)
            )
            # rank-3 fp32 pts: keeps NVFP4Tensor's fp32 block-scale promotion, satisfies torchao>=0.18 rank-in-(0,3) assert
            pts = _one(".weight_scale_2").to(torch.float32).view(1, 1, 1)

            weight = NVFP4Tensor(
                qdata, scale, 16, torch.bfloat16, per_tensor_scale=pts
            ).dequantize(torch.bfloat16)
        elif is_fp8:
            scale = _one(".weight_scale").to(torch.float32).reshape(-1)
            # per-tensor (scalar) or per-row scales both broadcast over the last dim
            scale = scale if scale.numel() == 1 else scale.view(-1, 1)
            weight = (w.to(torch.float32) * scale).to(torch.bfloat16)
        else:
            # The loader casts sources to the skeleton dtype before we see them, so a packed
            # weight is only detectable by shape; a scaleless fp8 one only when that cast is
            # bypassed (rank0-only loading). Both would otherwise load as garbage.
            expected = getattr(module, param_name, None)
            if (
                expected is not None
                and not isinstance(expected, property)
                and tuple(w.shape) != tuple(expected.shape)
            ) or w.dtype in _FP8_DTYPES:
                raise KeyError(
                    f"{full_layer_name}: quantized weight ({w.dtype}, shape {tuple(w.shape)}) "
                    f"with no scale in the checkpoint (keys: {list(input_dict.keys())}); "
                    "it would load unscaled"
                )
            weight = w

        setattr(module, param_name, nn.Parameter(weight, requires_grad=False))
        if pts is not None and not weight.is_meta:
            # merge-aware LoRA needs the original grid's per-tensor scale after dequant
            module._nvfp4_pts = pts.detach().reshape(()).clone()
        if missing_keys is not None:
            missing_keys.discard(full_layer_name)
        module._is_hf_initialized = True
        return {}

    @property
    def reverse_op(self):
        from transformers.core_model_loading import _IdentityOp

        return _IdentityOp()


def _nvfp4_linear_dequant_converter(target_weight: str):
    """A WeightConverter that dequantizes one non-routed linear (``<target_weight>``) to bf16.

    ``target_weight`` is the linear's ``.weight`` suffix (e.g. ``"shared_experts.gate_proj.weight"``).
    Claims the NVFP4 triple, the static-FP8 pair and ``input_scale`` (dropped: training activations
    stay bf16) so none lands UNEXPECTED; longest-suffix-first so ``.weight`` can't steal a scale.
    """
    from transformers.core_model_loading import WeightConverter

    base = target_weight[: -len(".weight")]
    return WeightConverter(
        source_patterns=[
            f"{base}.weight_scale_2",
            f"{base}.weight_scale",
            f"{base}.input_scale",
            f"{base}.weight",
        ],
        target_patterns=target_weight,
        operations=[Nvfp4LinearDequantize()],
    )


def nonrouted_dequant_converters(nonrouted_suffixes: list[str]) -> list:
    """Dequant converters for the non-routed linears a checkpoint quantizes (NVFP4 or static FP8).

    ``nonrouted_suffixes`` are layer-relative module paths detected from the safetensors index
    (e.g. ``"mixer.fc1_latent_proj"``), NOT hardcoded. Exactly one converter per suffix: two
    would share source-pattern strings and transformers resolves those to a single op."""
    return [
        _nvfp4_linear_dequant_converter(f"{suf}.weight") for suf in nonrouted_suffixes
    ]


def nvfp4_experts_weight_converters(routed_projs: list[str] | None = None) -> list:
    """The two WeightConverter instances for NVFP4 routed experts.

    Gated (default): per-expert ``gate/up`` fuse into ``experts.gate_up_proj``. When
    ``routed_projs`` has no gate, the up converter targets ``experts.up_proj`` instead."""
    from transformers.core_model_loading import WeightConverter

    op = Nvfp4ExpertsDeserialize()
    nongated = routed_projs is not None and "gate_proj" not in set(routed_projs)

    # Source patterns MUST be ordered longest-suffix-first. transformers compiles them into a
    # single ``(?P<g0>...)|(?P<g1>...)`` alternation and resolves a key with ``re.search`` +
    # first-non-None group (core_model_loading.py). The patterns are NOT end-anchored when the
    # converter is many-to-one (the ^...$ anchoring only runs for equal-length source/target
    # lists), so ``...weight`` would substring-match inside ``...weight_scale``/``...weight_scale_2``
    # and steal those keys unless the more specific suffixes appear first.
    if nongated:
        up_converter = WeightConverter(
            source_patterns=[
                "experts.*.up_proj.weight_scale_2",
                "experts.*.up_proj.weight_scale",
                "experts.*.up_proj.weight",
            ],
            target_patterns="experts.up_proj",
            operations=[op],
        )
    else:
        up_converter = WeightConverter(
            source_patterns=[
                # gate and up each ship their own weight_scale_2 scalar; claim BOTH so neither lands
                # as an UNEXPECTED key (the op reconciles them — equal in practice, folded if not).
                "experts.*.gate_proj.weight_scale_2",
                "experts.*.up_proj.weight_scale_2",
                "experts.*.gate_proj.weight_scale",
                "experts.*.up_proj.weight_scale",
                "experts.*.gate_proj.weight",
                "experts.*.up_proj.weight",
            ],
            target_patterns="experts.gate_up_proj",
            operations=[op],
        )

    down_converter = WeightConverter(
        source_patterns=[
            "experts.*.down_proj.weight_scale_2",
            "experts.*.down_proj.weight_scale",
            "experts.*.down_proj.weight",
        ],
        target_patterns="experts.down_proj",
        operations=[op],
    )

    return [up_converter, down_converter]


def register_nvfp4_expert_converters(
    model_type: str,
    include_routed: bool = True,
    extra: list | None = None,
    routed_projs: list[str] | None = None,
) -> None:
    """Seed the transformers conversion_mapping cache with NVFP4 converters for ``model_type``.

    The routed-expert converters (``Nvfp4ExpertsDeserialize`` + the ``experts.*.{proj}.weight*``
    source patterns) fuse per-expert ``gate/up/down`` (or non-gated ``up/down`` when
    ``routed_projs`` shows no gate) into the model's 3D expert params; they are registered when
    ``include_routed`` (gate this on the checkpoint actually exporting per-expert NVFP4).
    ``extra`` carries any per-checkpoint non-routed dequant converters (built from the detected
    index layout). Non-expert entries of the model's EXISTING mapping (e.g. nemotron_h's
    ``backbone.`` → ``model.`` renames) are preserved — registration replaces the whole
    per-model list, so dropping them would break the load. The model's built-in bf16 expert-merge
    converters are replaced (they'd fight ours for the same source keys). Safe to call repeatedly
    (our converters are tagged and filtered out of the preserved set on re-entry).
    """
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    from axolotl.utils.weight_conversions import register_weight_conversions

    converters = (
        nvfp4_experts_weight_converters(routed_projs) if include_routed else []
    ) + list(extra or [])
    if not converters:
        return

    def _targets_experts(conv) -> bool:
        tp = getattr(conv, "target_patterns", None) or []
        if isinstance(tp, str):
            tp = [tp]
        return any("experts." in t for t in tp)

    def _is_ours(conv) -> bool:
        ours = (Nvfp4ExpertsDeserialize, Nvfp4LinearDequantize)
        return any(
            isinstance(op, ours) for op in (getattr(conv, "operations", None) or [])
        )

    # The direct-expert-load fast path (include_routed=False) needs the stock bf16 expert-fusion
    # converters gone, so keep only non-expert entries and replace rather than merge.
    existing = get_checkpoint_conversion_mapping(model_type) or []
    keep = [c for c in existing if not _targets_experts(c) and not _is_ours(c)]
    converters = keep + converters
    register_weight_conversions(model_type, converters, replace_existing=True)

    LOG.info(
        "Registered %s NVFP4 WeightConverters (%d) in transformers conversion_mapping",
        model_type,
        len(converters),
    )


def register_gemma4_nvfp4_converters() -> None:
    """Register NVFP4 expert converters for ``gemma4_text`` (see register_nvfp4_expert_converters)."""
    register_nvfp4_expert_converters("gemma4_text")


def register_nvfp4_converters_for_layout(model_type: str, layout: dict) -> None:
    """Register NVFP4 converters built from a detected checkpoint ``layout`` (see
    :func:`...nvfp4_moe_loading.inspect_nvfp4_layout`): routed experts fused into packed
    NVFP4Tensor when present, plus every non-routed quantized linear dequantized to bf16.

    The two suffix lists overlap (a suffix can be NVFP4 in one layer, FP8 in the rest), so they
    are unioned into ONE converter per suffix; the op picks its branch per layer.
    """
    nonrouted = sorted(
        set(layout.get("nonrouted_suffixes", [])) | set(layout.get("fp8_suffixes", []))
    )
    register_nvfp4_expert_converters(
        model_type,
        include_routed=layout.get("routed_present", False),
        routed_projs=layout.get("routed_projs"),
        extra=nonrouted_dequant_converters(nonrouted),
    )
