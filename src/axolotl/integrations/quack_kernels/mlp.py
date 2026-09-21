# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
QuACK fused gated-MLP (SwiGLU / GeGLU) kernel wrapper and model patching.

The kernel fuses the up-projection GEMM with the gated activation via quack's
`mlp_func`. quack keeps a single fused `gate_up` weight `(2*inter, in)`; HF models
keep `gate_proj` / `up_proj` separate, so we concatenate them per forward (block
layout, `concat_layout=True`). At training token counts the extra weight-sized copy
is amortized against the matmul. Autograd flows through the `cat` back to the two
projection weights.
"""

import types

import torch
from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# HF `hidden_act` -> quack gated activation name. Non-gated acts are unsupported
# (this is a gated-MLP kernel) and cause the module to be skipped. quack's `geglu`
# uses the tanh GELU approximation, so only the tanh-approx HF gelus map to it;
# plain erf `"gelu"` is intentionally omitted so those models fall back instead.
QUACK_GATED_ACT = {
    "silu": "swiglu",
    "swish": "swiglu",
    "gelu_new": "geglu",
    "gelu_pytorch_tanh": "geglu",
    "relu": "reglu",
}

# quack's blockscaled/gated CuTe kernels target Hopper and newer.
MIN_SM_MAJOR = 9

_warned_fallback = False


def _cuda_sm_major() -> int | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()[0]


def resolve_gated_activation(config) -> str | None:
    """Map a HF config's activation to a quack gated activation name, or None.

    Prefer the first attr that MAPS: some configs (Gemma) carry a legacy
    ``hidden_act`` alongside the authoritative ``hidden_activation``.
    """
    for attr in ("hidden_act", "hidden_activation"):
        act = getattr(config, attr, None)
        if act and (mapped := QUACK_GATED_ACT.get(act)):
            return mapped
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return resolve_gated_activation(text_config)
    return None


def mlp_is_eligible(mlp: nn.Module) -> bool:
    """A plain dense gated MLP with three bias-free base-weight Linear projections.

    Skips LoRA / quantized projections: using the base ``.weight`` would silently
    drop the adapter delta (that path is `lora_mlp_kernel`'s job), and quantized
    weights are not a plain fp16/bf16 tensor the kernel can consume.
    """
    for name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(mlp, name, None)
        if (
            type(proj) is not nn.Linear
        ):  # exact type: exclude PEFT/bnb Linear subclasses
            return False
        if proj.bias is not None:
            return False

    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    if gate.weight.dtype not in (torch.float16, torch.bfloat16):
        return False

    inter = gate.out_features
    if up.out_features != inter or down.in_features != inter:
        return False
    # quack gated fc1 needs out % 16 == 0 (out == 2*inter), in % 8 == 0, fc2 out % 8 == 0.
    if gate.in_features % 8 or inter % 8 or down.out_features % 8:
        return False
    return True


def quack_gated_mlp_forward(
    mlp: nn.Module, activation: str, x: torch.Tensor
) -> torch.Tensor:
    from quack.mlp import mlp_func

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    if x2d.stride(-1) != 1:
        x2d = x2d.contiguous()
    # Weight is block [gate; up] of shape (2*inter, in); concat_layout=True tells quack
    # the operand is concatenated (it interleaves the two halves internally).
    gate_up = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0)
    # tuned=False: quack's autotuner selects a numerically-wrong config for small token
    # counts (M<=~384) on this gated path; the heuristic config is correct at all M.
    out = mlp_func(
        x2d,
        gate_up,
        mlp.down_proj.weight,
        activation=activation,
        concat_layout=True,
        tuned=False,
    )
    return out.reshape(*orig_shape[:-1], out.shape[-1])


def _make_patched_forward(orig_forward, activation: str):
    def forward(self, x, *args, **kwargs):
        if x.is_cuda and (_cuda_sm_major() or 0) >= MIN_SM_MAJOR:
            try:
                return quack_gated_mlp_forward(self, activation, x)
            except Exception as exc:  # pragma: no cover - GPU-only path
                global _warned_fallback
                if not _warned_fallback:
                    _warned_fallback = True
                    LOG.warning(
                        f"quack fused MLP kernel failed ({exc}); falling back to the "
                        "original MLP forward for the rest of training."
                    )
        return orig_forward(x, *args, **kwargs)

    return forward


def apply_quack_mlp(model: nn.Module, cfg) -> int:
    """Patch each eligible dense gated MLP's forward to the quack fused kernel.

    Matches the decoder layer's `.mlp` / `.feed_forward` module so routed-expert
    containers and MoE routers (handled by the MoE kernels) are not touched.
    """
    activation = resolve_gated_activation(model.config)
    if activation is None:
        LOG.warning(
            "quack_mlp_kernel enabled but the model activation is not a supported "
            "gated activation (silu/gelu/relu family); leaving MLPs unpatched."
        )
        return 0

    patched = 0
    for name, module in model.named_modules():
        if not (name.endswith(".mlp") or name.endswith(".feed_forward")):
            continue
        if getattr(module, "_quack_mlp_patched", False):
            continue
        if not mlp_is_eligible(module):
            continue
        module._quack_mlp_patched = True
        module.forward = types.MethodType(
            _make_patched_forward(module.forward, activation), module
        )
        patched += 1

    if patched:
        LOG.info(
            f"quack fused MLP kernel: patched {patched} MLP module(s) (act={activation})."
        )
    else:
        LOG.warning("quack_mlp_kernel enabled but found no eligible dense MLP modules.")
    return patched
