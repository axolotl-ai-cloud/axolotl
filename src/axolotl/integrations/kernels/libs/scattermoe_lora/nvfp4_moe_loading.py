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
    """Rebuild fused qdata + E4M3 block scale + per-tensor scale for all experts of one
    fused projection straight from the raw checkpoint (no dependence on transformers' own
    fusion). Fused on the N (row) axis: qdata [E, sumN, K/2] uint8, scale [E, sumN, K/16]
    E4M3, per_tensor scalar."""
    qd_proj, sc_proj, pts_list = [[] for _ in projs], [[] for _ in projs], []
    opened: dict[str, object] = {}
    for e in range(n_experts):
        for pi, proj in enumerate(projs):
            base = base_fmt.format(layer=layer, e=e, proj=proj)
            shard = wmap[f"{base}.weight"]
            f = opened.get(shard) or opened.setdefault(
                shard, _shard_open(repo_id, shard)
            )
            qd_proj[pi].append(f.get_tensor(f"{base}.weight"))
            sc_proj[pi].append(f.get_tensor(f"{base}.weight_scale"))
            if pi == 0:  # gate/up share weight_scale_2; one per expert
                pts_list.append(
                    f.get_tensor(f"{base}.weight_scale_2").to(torch.float32)
                )
    qdata = torch.cat([torch.stack(q, 0) for q in qd_proj], dim=1).to(device)
    scale = torch.cat([torch.stack(s, 0) for s in sc_proj], dim=1).to(device)
    # Per-expert weight_scale_2 stacked to [E,1,1], not expert-0's scalar broadcast to all experts.
    pts = torch.stack(pts_list).view(-1, 1, 1).to(device)
    return qdata, scale, pts


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
        gqd, gscale, gpts = _build_expert_nvfp4(
            repo_id, wmap, base_fmt, layer, projs_gu, E, gup.device
        )
        dqd, dscale, dpts = _build_expert_nvfp4(
            repo_id, wmap, base_fmt, layer, projs_dn, E, dn.device
        )
        mod.gate_up_proj = nn.Parameter(
            NVFP4Tensor(gqd, gscale, 16, torch.bfloat16, per_tensor_scale=gpts),
            requires_grad=False,
        )
        mod.down_proj = nn.Parameter(
            NVFP4Tensor(dqd, dscale, 16, torch.bfloat16, per_tensor_scale=dpts),
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
