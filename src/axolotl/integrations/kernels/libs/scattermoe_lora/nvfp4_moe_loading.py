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


def _load_index(repo_id: str):
    from huggingface_hub import hf_hub_download

    import json

    path = hf_hub_download(repo_id, "model.safetensors.index.json")
    return json.load(open(path))["weight_map"]


def _shard_open(repo_id: str, shard: str):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    return safe_open(hf_hub_download(repo_id, shard), framework="pt")


def _build_expert_nvfp4(repo_id, wmap, layer, projs, n_experts, device):
    """Rebuild fused qdata + E4M3 block scale + per-tensor scale for all experts of one
    fused projection straight from the raw checkpoint (no dependence on transformers' own
    fusion). `projs` is ("w1","w3") for gate_up or ("w2",) for down; fused on the N (row)
    axis: qdata [E, sumN, K/2] uint8, scale [E, sumN, K/16] E4M3, per_tensor scalar."""
    qd_proj, sc_proj, pts = [[] for _ in projs], [[] for _ in projs], None
    opened: dict[str, object] = {}
    for e in range(n_experts):
        for pi, proj in enumerate(projs):
            base = f"layers.{layer}.ffn.experts.{e}.{proj}"
            shard = wmap[f"{base}.weight"]
            f = opened.get(shard) or opened.setdefault(shard, _shard_open(repo_id, shard))
            qd_proj[pi].append(f.get_tensor(f"{base}.weight"))
            sc_proj[pi].append(f.get_tensor(f"{base}.weight_scale"))
            if pts is None:  # w1/w3 share weight_scale_2; capture once
                pts = f.get_tensor(f"{base}.weight_scale_2").to(torch.float32)
    qdata = torch.cat([torch.stack(q, 0) for q in qd_proj], dim=1).to(device)
    scale = torch.cat([torch.stack(s, 0) for s in sc_proj], dim=1).to(device)
    return qdata, scale, pts.to(device)


def attach_nvfp4_expert_scales(model: nn.Module, repo_id: str) -> int:
    """Wrap each DeepseekV4Experts gate_up_proj/down_proj (uint8 qdata loaded by
    transformers) as an NVFP4Tensor with the checkpoint's fused E4M3 scales. Returns the
    number of expert modules fixed."""
    NVFP4Tensor = _nvfp4_cls()
    if NVFP4Tensor is None:
        return 0
    wmap = _load_index(repo_id)
    fixed = 0
    for name, mod in model.named_modules():
        # The FP8 quantizer replaces DeepseekV4Experts with FP8Experts; match by the
        # packed uint8 fused expert params rather than the class name.
        gup = getattr(mod, "gate_up_proj", None)
        dn = getattr(mod, "down_proj", None)
        if not (
            isinstance(gup, torch.Tensor)
            and isinstance(dn, torch.Tensor)
            and gup.dtype == torch.uint8
            and dn.dtype == torch.uint8
            and gup.ndim == 3
        ):
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if not m:
            continue
        layer = int(m.group(1))
        E = gup.shape[0]
        gqd, gscale, gpts = _build_expert_nvfp4(repo_id, wmap, layer, ("w1", "w3"), E, gup.device)
        dqd, dscale, dpts = _build_expert_nvfp4(repo_id, wmap, layer, ("w2",), E, dn.device)
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
            "gate_up_proj_scale_inv", "down_proj_scale_inv",
            "gate_up_proj_scale", "down_proj_scale",
            "gate_up_proj_scale_2", "down_proj_scale_2",
            "gate_up_proj_input_scale", "down_proj_input_scale",
        ):
            for store in (mod._parameters, mod._buffers):
                if stale in store:
                    del store[stale]
        del gup, dn
        fixed += 1
    if fixed:
        LOG.info("Attached NVFP4 expert scales (rebuilt %d DeepseekV4Experts as NVFP4Tensor)", fixed)
    return fixed


if __name__ == "__main__":  # local self-consistency test on real layer-0 data
    REPO = "nvidia/DeepSeek-V4-Flash-NVFP4"
    NVFP4Tensor = _nvfp4_cls()
    wmap = _load_index(REPO)
    dev = "cuda"
    # fused gate_up qdata+scale from w1+w3 (expert 0 only via n_experts=1 slice below)
    gqd, gscale, gpts = _build_expert_nvfp4(REPO, wmap, 0, ("w1", "w3"), 4, dev)
    print("fused gate_up qdata", gqd.shape, "scale", gscale.shape, "per_tensor", gpts.item())
    # self-consistency: NVFP4 of fused must dequant to cat(dequant(w1), dequant(w3))
    f = _shard_open(REPO, wmap["layers.0.ffn.experts.0.w1.weight"])
    qd1 = f.get_tensor("layers.0.ffn.experts.0.w1.weight").to(dev)
    qd3 = f.get_tensor("layers.0.ffn.experts.0.w3.weight").to(dev)
    s1 = f.get_tensor("layers.0.ffn.experts.0.w1.weight_scale").to(dev)
    s3 = f.get_tensor("layers.0.ffn.experts.0.w3.weight_scale").to(dev)
    p = f.get_tensor("layers.0.ffn.experts.0.w1.weight_scale_2").to(dev).float()
    d1 = NVFP4Tensor(qd1, s1, 16, torch.bfloat16, per_tensor_scale=p).dequantize(torch.bfloat16)
    d3 = NVFP4Tensor(qd3, s3, 16, torch.bfloat16, per_tensor_scale=p).dequantize(torch.bfloat16)
    fused = NVFP4Tensor(gqd[0], gscale[0], 16, torch.bfloat16, per_tensor_scale=gpts).dequantize(torch.bfloat16)
    ref = torch.cat([d1, d3], dim=0)
    print("fusion self-consistent:", torch.equal(fused, ref), "max_err", (fused - ref).abs().max().item())
