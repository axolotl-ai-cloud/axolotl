"""Fix NVFP4 MoE-expert loading for DeepSeek-V4-Flash-NVFP4 (and similar checkpoints).

transformers' finegrained_fp8 quantizer treats the model as FP8 blockwise and has NO
NVFP4 path, so for a MIXED_PRECISION checkpoint (FP8 elsewhere, ``moe_quant_algo: NVFP4``
for the routed experts) it fuses the expert *weights* into ``gate_up_proj``/``down_proj``
(uint8 qdata) correctly but drops the NVFP4 *scales* — the per-expert ``weight_scale``
(E4M3 group-16 block scale) + ``weight_scale_2`` (per-tensor) come in UNEXPECTED and the
model's ``*_scale_inv`` placeholders are MISSING → randomly initialized → invalid model.

This re-reads the per-expert ``weight_scale``/``weight_scale_2`` from the checkpoint, fuses
them the same way as the weights (gate_up = cat(w1, w3) on the 2*intermediate axis; down =
w2), and wraps the already-loaded fused qdata as a torchao ``NVFP4Tensor`` so the
scattermoe NVFP4 path dequantizes correctly. Self-contained until an upstream transformers
NVFP4-MoE path lands.
"""

from __future__ import annotations

import os
import re

import torch
import torch.nn as nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _nvfp4_cls():
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        return NVFP4Tensor
    except ImportError:
        return None


def _resolve_repo_file(repo_id: str, filename: str) -> str:
    """Resolve a checkpoint file path from a local snapshot dir or the HF hub.

    A local snapshot dir (offline/air-gapped axolotl usage) would fail ``hf_hub_download``
    with an ``HFValidationError``, so read straight from disk in that case.
    """
    if os.path.isdir(repo_id):
        return os.path.join(repo_id, filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename)


def _load_index(repo_id: str):
    import json

    with open(_resolve_repo_file(repo_id, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _shard_open(repo_id: str, shard: str):
    from safetensors import safe_open

    return safe_open(_resolve_repo_file(repo_id, shard), framework="pt")


def _safetensors_metadata(repo_id: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Return ``{tensor_name: (dtype_str, shape)}`` for every checkpoint tensor, read from the
    safetensors *headers* only (no weight download). Works for a hub repo or a local snapshot dir.
    """
    if os.path.isdir(repo_id):
        from safetensors import safe_open

        out: dict[str, tuple[str, tuple[int, ...]]] = {}
        index = os.path.join(repo_id, "model.safetensors.index.json")
        if os.path.exists(index):
            shards = sorted(set(_load_index(repo_id).values()))
        else:  # single-file checkpoint
            shards = ["model.safetensors"]
        for shard in shards:
            with safe_open(os.path.join(repo_id, shard), framework="pt") as f:
                for name in f.keys():
                    sl = f.get_slice(name)
                    out[name] = (sl.get_dtype(), tuple(sl.get_shape()))
        return out

    from huggingface_hub import HfApi

    meta = HfApi().get_safetensors_metadata(repo_id)
    out = {}
    for fmeta in meta.files_metadata.values():
        for name, tinfo in fmeta.tensors.items():
            out[name] = (str(tinfo.dtype), tuple(tinfo.shape))
    return out


# Known per-module leaf names across NVFP4 export conventions. The packed 4-bit weight is the
# robust NVFP4 signal (uint8 qdata) — distinguishing NVFP4 from FP8 (which also carries a
# ``weight_scale`` but stores an fp8 weight, not uint8). Names differ by exporter:
#   modelopt          : weight        / weight_scale / weight_scale_2
#   compressed-tensors: weight_packed / weight_scale / weight_global_scale
_QDATA_LEAVES = ("weight", "weight_packed")
_GROUP_SCALE_LEAVES = ("weight_scale",)
_PER_TENSOR_LEAVES = ("weight_scale_2", "weight_global_scale")
_ALL_LEAVES = (
    "weight_scale_2",
    "weight_global_scale",
    "weight_scale",
    "weight_packed",
    "weight",
    "input_scale",
    "input_global_scale",
)


def inspect_nvfp4_layout(repo_id: str) -> dict:
    """Detect a checkpoint's NVFP4 layout from its safetensors headers — no layout assumptions.

    A module is treated as **NVFP4** only when it has a *packed* 4-bit weight (uint8 qdata, under
    ``weight`` or ``weight_packed``) plus a group ``weight_scale`` — this distinguishes NVFP4 from
    FP8 modules (which also carry a ``weight_scale`` but store an fp8, not uint8, weight) so a mixed
    NVFP4+FP8 checkpoint is classified correctly. The NVFP4 modules are split into:
      * ``routed_projs``  — proj names under ``...experts.<int>.<proj>`` (fused into 3D expert params),
      * ``nonrouted_suffixes`` — layer-relative paths of every other NVFP4 linear THIS checkpoint
        quantizes (shared experts, dense MLPs, ...),
    plus the detected key ``naming`` (``modelopt`` vs ``compressed-tensors`` vs ``mixed``) so the
    caller can tell whether its converters (which assume modelopt ``weight``/``weight_scale_2``)
    apply. Everything is derived from the headers; differing exports are described, not assumed.
    """
    import re

    meta = _safetensors_metadata(repo_id)
    bases: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
    for name, dt_shape in meta.items():
        for leaf in _ALL_LEAVES:
            if name.endswith("." + leaf):
                bases.setdefault(name[: -len(leaf) - 1], {})[leaf] = dt_shape
                break

    def _qdata(parts):
        for leaf in _QDATA_LEAVES:
            if leaf in parts and parts[leaf][0] == "U8":
                return leaf
        return None

    routed_re = re.compile(r"\.experts\.\d+\.([A-Za-z_][A-Za-z0-9_]*)$")
    layer_re = re.compile(r"^.*?layers\.\d+\.")
    routed_projs: list[str] = []
    routed_sample: dict[str, tuple | None] = {}
    nonrouted: dict[str, dict] = {}
    qdata_names: set[str] = set()
    per_tensor_names: set[str] = set()
    for base, parts in bases.items():
        qd = _qdata(parts)
        is_nvfp4 = qd is not None and any(g in parts for g in _GROUP_SCALE_LEAVES)
        if not is_nvfp4:  # bf16 (excluded) or fp8 module — not NVFP4
            continue
        qdata_names.add(qd)
        for leaf in _PER_TENSOR_LEAVES:
            if leaf in parts:
                per_tensor_names.add(leaf)
        m = routed_re.search(base)
        if m:
            proj = m.group(1)
            if proj not in routed_projs:
                routed_projs.append(proj)
                routed_sample[proj] = parts.get(qd)
        else:
            nonrouted.setdefault(layer_re.sub("", base), parts)

    modelopt = qdata_names <= {"weight"} and per_tensor_names <= {"weight_scale_2"}
    ct = qdata_names <= {"weight_packed"} and per_tensor_names <= {
        "weight_global_scale"
    }
    naming = (
        "modelopt" if modelopt else ("compressed-tensors" if ct else "mixed/unknown")
    )

    return {
        "routed_present": bool(routed_projs),
        "routed_projs": sorted(routed_projs),
        "routed_sample_shapes": routed_sample,
        "nonrouted_suffixes": sorted(nonrouted),
        "nonrouted_sample_shapes": nonrouted,
        "qdata_names": sorted(qdata_names),
        "per_tensor_names": sorted(per_tensor_names),
        "naming": naming,
    }


def fuse_nvfp4_experts(projs: list[dict], *, block_size: int = 16, dtype=None):
    """Shared NVFP4-expert fusion core (one implementation for every load path).

    ``projs`` is the ordered list of projections to fuse on the N (row) axis — one entry for a
    single proj (``down``), two for a fused ``gate_up`` — each a dict of per-expert lists::

        {"qd": [E uint8 [N, K/2]], "sc": [E e4m3 [N, K/16]], "pts": [E f32 scalar]}

    Returns a torchao ``NVFP4Tensor`` ``[E, ΣN, K/2]``. The fused tensor carries ONE per-expert
    per-tensor scale (``pts``), so when the projections export different ``pts`` the ratio is
    folded into the later proj's group scale (dequant = qdata · group_scale · per_tensor_scale, so
    this is exact up to e4m3 rounding) rather than silently dropping it. Used by both the
    ``WeightConverter`` (modelopt skeleton load) and the post-load scale-attach path so the fusion
    + scale reconciliation live in exactly one place.
    """
    import torch as _torch

    NVFP4Tensor = _nvfp4_cls()
    if NVFP4Tensor is None:
        raise RuntimeError("torchao NVFP4Tensor not available")
    dtype = dtype or _torch.bfloat16

    # cpu_ram_efficient_loading runs the converter once on META placeholders (shape inference) and
    # again on the real tensors on rank 0. On meta the scale-reconciliation value ops (`allclose`,
    # `.item()`) can't run, so skip them — only the OUTPUT SHAPE matters on the meta pass, and the
    # stack/cat below produce correctly-shaped meta tensors regardless.
    is_meta = bool(projs[0]["qd"]) and projs[0]["qd"][0].is_meta

    pts0 = _torch.stack([t.to(_torch.float32) for t in projs[0]["pts"]]).view(-1, 1, 1)
    qdatas, scales = [], []
    for i, p in enumerate(projs):
        qd = _torch.stack(list(p["qd"]), dim=0)  # [E, N, K/2]
        sc = _torch.stack(list(p["sc"]), dim=0)  # [E, N, K/16]
        if i > 0 and not is_meta:
            pts_i = _torch.stack([t.to(_torch.float32) for t in p["pts"]]).view(
                -1, 1, 1
            )
            if not _torch.allclose(pts_i, pts0):
                sc = (sc.to(_torch.float32) * (pts_i / pts0)).to(_torch.float8_e4m3fn)
                LOG.warning(
                    "fuse_nvfp4_experts: proj #%d per-tensor scale differs from the first; "
                    "folded the ratio into its group scale (max %.4g)",
                    i,
                    (pts_i / pts0).max().item(),
                )
        qdatas.append(qd)
        scales.append(sc)
    qdata = qdatas[0] if len(qdatas) == 1 else _torch.cat(qdatas, dim=1)
    scale = scales[0] if len(scales) == 1 else _torch.cat(scales, dim=1)
    return NVFP4Tensor(qdata, scale, block_size, dtype, per_tensor_scale=pts0)


def patch_nvfp4_tensor_meta_ops() -> None:
    """Register ``zeros_like`` / ``empty_like`` / ``new_zeros`` on torchao's NVFP4Tensor.

    FSDP2's ``cpu_ram_efficient_loading`` keeps non-rank-0 params on ``meta`` and materializes the
    receive buffers with ``zeros_like`` / ``empty_like`` before scattering rank 0's shards. torchao
    doesn't implement those for NVFP4Tensor (only matmul / view / slice / copy), so a 4-bit expert
    param hits 'unimplemented operator'. Each just applies the op to the packed data + scales and
    rebuilds the tensor, preserving block_size / dtype / per-tensor scale. Idempotent."""
    import torch as _torch

    NVFP4Tensor = _nvfp4_cls()
    if NVFP4Tensor is None or getattr(NVFP4Tensor, "_axolotl_meta_ops", False):
        return
    from torchao.utils import return_and_correct_aliasing

    aten = _torch.ops.aten

    @NVFP4Tensor.implements([aten.zeros_like.default, aten.empty_like.default])
    def _nvfp4_like(func, types, args, kwargs):
        # FSDP2 passes device / pin_memory / memory_format; forward them to the packed data + scale
        # tensors. dtype/layout would break the int-packed payload, so drop them.
        passthru = {
            k: v
            for k, v in kwargs.items()
            if k in ("device", "pin_memory", "memory_format")
        }
        out = args[0]._apply_fn_to_data(lambda x: func(x, **passthru))
        return return_and_correct_aliasing(func, args, kwargs, out)

    @NVFP4Tensor.implements([aten.new_zeros.default])
    def _nvfp4_new_zeros(func, types, args, kwargs):
        out = args[0]._apply_fn_to_data(lambda x: _torch.zeros_like(x))
        return return_and_correct_aliasing(func, args, kwargs, out)

    NVFP4Tensor._axolotl_meta_ops = True


# Per-architecture checkpoint naming for the unfused per-expert NVFP4 tensors.
# base_fmt formats with (layer, e, proj); gate_up/down are the proj names fused
# (gate_up = cat on the N/row axis in this order; down is single).
_NVFP4_MOE_SCHEMES = {
    # DeepSeek-V4-Flash-NVFP4: w1=gate, w3=up, w2=down under ``layers.N.ffn.experts.M``
    "dsv4": {
        "base_fmt": "layers.{layer}.ffn.experts.{e}.{proj}",
        "gate_up": ("w1", "w3"),
        "down": ("w2",),
    },
    # Gemma-4-A4B-NVFP4: separate gate/up/down under ``model.language_model.layers.N.experts.M``
    "gemma4": {
        "base_fmt": "model.language_model.layers.{layer}.experts.{e}.{proj}",
        "gate_up": ("gate_proj", "up_proj"),
        "down": ("down_proj",),
    },
}


def _detect_scheme(wmap):
    """Pick the checkpoint naming scheme whose layer-0 expert-0 weight key is present."""
    for name, sch in _NVFP4_MOE_SCHEMES.items():
        probe = sch["base_fmt"].format(layer=0, e=0, proj=sch["gate_up"][0]) + ".weight"
        if probe in wmap:
            return name, sch
    return None, None


def _build_expert_nvfp4(repo_id, wmap, base_fmt, layer, projs, n_experts, device):
    """Rebuild a fused NVFP4Tensor for one expert projection group straight from the raw
    checkpoint (no dependence on transformers' own fusion), reading every proj's per-expert qdata,
    E4M3 block scale and per-tensor scale, then delegating the stack/concat/scale-reconcile to the
    shared :func:`fuse_nvfp4_experts` core. Returns an ``NVFP4Tensor`` ``[E, ΣN, K/2]`` on ``device``."""
    proj_parts = [{"qd": [], "sc": [], "pts": []} for _ in projs]
    opened: dict[str, object] = {}
    for e in range(n_experts):
        for pi, proj in enumerate(projs):
            base = base_fmt.format(layer=layer, e=e, proj=proj)
            shard = wmap[f"{base}.weight"]
            f = opened.get(shard) or opened.setdefault(
                shard, _shard_open(repo_id, shard)
            )
            proj_parts[pi]["qd"].append(f.get_tensor(f"{base}.weight").to(device))
            proj_parts[pi]["sc"].append(f.get_tensor(f"{base}.weight_scale").to(device))
            proj_parts[pi]["pts"].append(
                f.get_tensor(f"{base}.weight_scale_2").to(torch.float32).to(device)
            )
    return fuse_nvfp4_experts(proj_parts)


def attach_nvfp4_expert_scales(model: nn.Module, repo_id: str) -> int:
    """Wrap each MoE experts module's fused gate_up_proj/down_proj as an NVFP4Tensor rebuilt from
    the checkpoint's per-expert E4M3 scales. Handles both checkpoint layouts (DeepSeek-V4 ``ffn.
    experts``/w1-w3-w2 with uint8 fused params from the FP8 quantizer, and Gemma-4 ``language_model.
    layers.N.experts.M``/gate-up-down where transformers leaves the fused param a random bf16
    placeholder). Returns the number of expert modules fixed."""
    NVFP4Tensor = _nvfp4_cls()
    if NVFP4Tensor is None:
        return 0
    wmap = _load_index(repo_id)
    scheme_name, scheme = _detect_scheme(wmap)
    if scheme is None:
        LOG.warning(
            "attach_nvfp4_expert_scales: no known NVFP4 MoE naming scheme in %s",
            repo_id,
        )
        return 0
    base_fmt, projs_gu, projs_dn = scheme["base_fmt"], scheme["gate_up"], scheme["down"]
    fixed = 0
    for name, mod in model.named_modules():
        # Match the fused experts param by shape (3D stacked experts), not dtype/class: DSV4's
        # FP8 quantizer leaves it uint8, gemma4 leaves it a random bf16 placeholder.
        gup = getattr(mod, "gate_up_proj", None)
        dn = getattr(mod, "down_proj", None)
        if not (
            isinstance(gup, torch.Tensor)
            and isinstance(dn, torch.Tensor)
            and gup.ndim == 3
            and dn.ndim == 3
        ):
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if not m:
            continue
        layer = int(m.group(1))
        # Skip layers whose experts aren't in the checkpoint under this scheme (e.g. dense layers).
        if base_fmt.format(layer=layer, e=0, proj=projs_gu[0]) + ".weight" not in wmap:
            continue
        E = gup.shape[0]
        mod.gate_up_proj = nn.Parameter(
            _build_expert_nvfp4(
                repo_id, wmap, base_fmt, layer, projs_gu, E, gup.device
            ),
            requires_grad=False,
        )
        mod.down_proj = nn.Parameter(
            _build_expert_nvfp4(repo_id, wmap, base_fmt, layer, projs_dn, E, dn.device),
            requires_grad=False,
        )
        # Drop the stale FP8 scale params the quantizer created for these experts
        # (randomly-initialized `*_scale_inv` from the MISSING-key path, plus any FP8
        # `*_scale`/`input_scale`). They're unused once the experts are NVFP4Tensor routed
        # through scattermoe, and would otherwise be sharded by FSDP / waste memory.
        for stale in (
            "gate_up_proj_scale_inv",
            "down_proj_scale_inv",
            "gate_up_proj_scale",
            "down_proj_scale",
            "gate_up_proj_scale_2",
            "down_proj_scale_2",
            "gate_up_proj_input_scale",
            "down_proj_input_scale",
        ):
            for store in (mod._parameters, mod._buffers):
                if stale in store:
                    del store[stale]
        del gup, dn
        fixed += 1
    if fixed:
        LOG.info(
            "Attached NVFP4 expert scales (rebuilt %d %s experts as NVFP4Tensor)",
            fixed,
            scheme_name,
        )
    return fixed


def direct_load_nvfp4_experts(model, repo_id: str, routed_projs: list[str]) -> int:
    """FAST routed-expert load that BYPASSES transformers' conversion loader.

    Transformers' loader spends ~7 min on GLM-5.2 iterating/matching ~240k per-expert source tensors
    through its Python machinery; a DIRECT read+fuse of the same experts is ~25s (profiled). This
    reads each MoE layer's routed-expert ``weight``(uint8)/``weight_scale``(fp8)/``weight_scale_2``
    components straight from the safetensors (native dtype) and fuses them into the 3D NVFP4 expert
    params, mirroring :class:`Nvfp4ExpertsDeserialize` but without the per-tensor loader overhead.

    Only the LOCAL-RANK-0 process materializes (others stay meta for the FSDP broadcast). Requires the
    routed converters to be SKIPPED at registration (so transformers leaves the fused params unfilled).
    Returns the number of fused params filled. Safe no-op on non-rank0 / no routed experts.
    """
    import os
    import re

    import torch

    if str(os.environ.get("LOCAL_RANK", "0")) not in ("0", ""):
        return 0
    if not routed_projs:
        return 0
    NVFP4Tensor = _nvfp4_cls()
    if NVFP4Tensor is None:
        return 0
    wmap = _load_index(repo_id)
    # per-layer expert count, from the index (no reads): count ...experts.<e>.<proj0>.weight keys.
    proj0 = routed_projs[0]
    layer_E: dict[int, int] = {}
    pat = re.compile(
        r"\.layers\.(\d+)\.mlp\.experts\.(\d+)\." + re.escape(proj0) + r"\.weight$"
    )
    for key in wmap:
        m = pat.search(key)
        if m:
            L, e = int(m.group(1)), int(m.group(2))
            layer_E[L] = max(layer_E.get(L, 0), e + 1)

    handles: dict[str, object] = {}

    def gt(key):
        sh = wmap[key]
        if sh not in handles:
            handles[sh] = _shard_open(repo_id, sh)
        return handles[sh].get_tensor(key)  # native dtype (uint8/fp8/fp32)

    fused_map = (
        ("gate_up_proj", ["gate_proj", "up_proj"]),
        ("down_proj", ["down_proj"]),
    )
    n = 0
    for mod_name, mod in model.named_modules():
        if not (hasattr(mod, "gate_up_proj") and hasattr(mod, "down_proj")):
            continue
        m = re.search(r"\.layers\.(\d+)\.", "." + mod_name)
        if m is None:
            continue
        L = int(m.group(1))
        if L not in layer_E:
            continue
        E = layer_E[L]
        base = f"model.layers.{L}.mlp.experts"
        for fused, parts in fused_map:
            sel = [p for p in parts if p in routed_projs]
            if not sel:
                continue
            projs = [
                {
                    "qd": [gt(f"{base}.{e}.{p}.weight") for e in range(E)],
                    "sc": [gt(f"{base}.{e}.{p}.weight_scale") for e in range(E)],
                    "pts": [gt(f"{base}.{e}.{p}.weight_scale_2") for e in range(E)],
                }
                for p in sel
            ]
            nvfp4 = fuse_nvfp4_experts(projs)
            setattr(mod, fused, torch.nn.Parameter(nvfp4, requires_grad=False))
            n += 1
    return n


if __name__ == "__main__":  # local self-consistency test on real layer-0 data
    REPO = "nvidia/DeepSeek-V4-Flash-NVFP4"
    NVFP4Tensor = _nvfp4_cls()
    wmap = _load_index(REPO)
    _, _scheme = _detect_scheme(wmap)
    _base_fmt = _scheme["base_fmt"]
    dev = "cuda"
    # fused gate_up qdata+scale from w1+w3 (expert 0 only via n_experts=1 slice below)
    gqd, gscale, gpts = _build_expert_nvfp4(
        REPO, wmap, _base_fmt, 0, ("w1", "w3"), 4, dev
    )
    print(
        "fused gate_up qdata",
        gqd.shape,
        "scale",
        gscale.shape,
        "per_tensor",
        gpts.item(),
    )
    # self-consistency: NVFP4 of fused must dequant to cat(dequant(w1), dequant(w3))
    f = _shard_open(REPO, wmap["layers.0.ffn.experts.0.w1.weight"])
    qd1 = f.get_tensor("layers.0.ffn.experts.0.w1.weight").to(dev)
    qd3 = f.get_tensor("layers.0.ffn.experts.0.w3.weight").to(dev)
    s1 = f.get_tensor("layers.0.ffn.experts.0.w1.weight_scale").to(dev)
    s3 = f.get_tensor("layers.0.ffn.experts.0.w3.weight_scale").to(dev)
    p = f.get_tensor("layers.0.ffn.experts.0.w1.weight_scale_2").to(dev).float()
    d1 = NVFP4Tensor(qd1, s1, 16, torch.bfloat16, per_tensor_scale=p).dequantize(
        torch.bfloat16
    )
    d3 = NVFP4Tensor(qd3, s3, 16, torch.bfloat16, per_tensor_scale=p).dequantize(
        torch.bfloat16
    )
    fused = NVFP4Tensor(
        gqd[0], gscale[0], 16, torch.bfloat16, per_tensor_scale=gpts
    ).dequantize(torch.bfloat16)
    ref = torch.cat([d1, d3], dim=0)
    print(
        "fusion self-consistent:",
        torch.equal(fused, ref),
        "max_err",
        (fused - ref).abs().max().item(),
    )
