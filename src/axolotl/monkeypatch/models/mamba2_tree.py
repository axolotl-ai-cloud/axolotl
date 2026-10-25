"""The Mamba2 (SSD) tree component: Nemotron-H, Mamba2, Bamba, Granite-MoE-Hybrid.

mamba-ssm's chunked scan resets its state at ``seq_idx`` boundaries but takes
initial states per batch row only, so the trie cannot be forked: the tree driver
scans every root-to-segment path in one call and keeps each segment's tail. The
projections, the gated norm and the conv (with its path history) still run once
per shared token, and so do the attention and MLP layers around the mixer.

The forward below follows the transformers Mamba2 mixer that these models share
(``in_proj`` -> conv -> SSD scan -> gated norm -> ``out_proj``); the patched
mixer runs it only for tree rows and its own forward for everything else.
"""

from __future__ import annotations

import functools
import importlib

import torch

from axolotl.monkeypatch.models.mamba_utils import assert_mamba2_scan_honours_seq_idx
from axolotl.monkeypatch.models.recurrent_tree import (
    RecurrentComponent,
    TreeFamily,
    mark_tree_forward,
    register_tree_family,
    segment_ids,
    tree_conv,
    tree_plan,
    tree_scan,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

MAMBA2_SSD = RecurrentComponent("mamba2_ssd", initial_state=False)

# model type -> (its modeling module's model type, mixer class)
MAMBA2_MIXERS = {
    "nemotron_h": ("nemotron_h", "NemotronHMamba2Mixer"),
    "mamba2": ("mamba2", "Mamba2Mixer"),
    "bamba": ("bamba", "BambaMixer"),
    "granitemoehybrid": ("granitemoehybrid", "GraniteMoeHybridMambaLayer"),
}

for _model_type, (_, _mixer) in MAMBA2_MIXERS.items():
    register_tree_family(TreeFamily((_model_type,), (_mixer,), MAMBA2_SSD))


def mamba2_tree_forward(mixer, module, hidden_states, plan):
    """The Mamba2 mixer on a tree row (``hidden_states`` [1, T, H], row order)."""
    dtype = hidden_states.dtype
    projected = mixer.in_proj(hidden_states)
    gate, x_b_c, dt = projected.split(
        [mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], dim=-1
    )
    weight = mixer.conv1d.weight.squeeze(1)

    def conv(x, cu_seqlens):
        out = module.causal_conv1d_fn(
            x.transpose(1, 2), weight, mixer.conv1d.bias, activation=mixer.activation
        )
        return out.transpose(1, 2)

    x_b_c = tree_conv(conv, x_b_c, plan, mixer.conv_kernel_size)
    groups_state = mixer.n_groups * mixer.ssm_state_size
    x, b, c = x_b_c.split([mixer.intermediate_size, groups_state, groups_state], dim=-1)
    a = -torch.exp(mixer.A_log.float())

    def scan(inputs, cu_seqlens, initial_state):
        x, dt, b, c = inputs
        tokens = x.shape[1]
        out = module.mamba2_chunk_scan(
            x.reshape(1, tokens, mixer.num_heads, mixer.head_dim),
            dt,
            a,
            b.reshape(1, tokens, mixer.n_groups, mixer.ssm_state_size),
            c.reshape(1, tokens, mixer.n_groups, mixer.ssm_state_size),
            chunk_size=mixer.chunk_size,
            D=mixer.D,
            z=None,
            dt_bias=mixer.dt_bias,
            dt_softplus=True,
            dt_limit=mixer.time_step_limit,
            seq_idx=segment_ids(cu_seqlens, tokens),
        )
        return out.reshape(1, tokens, -1), None

    out = tree_scan(scan, [x, dt, b, c], plan, MAMBA2_SSD)
    out = mixer.norm(out, gate)
    return mixer.out_proj(out.to(dtype))


def patch_mamba2_tree(model_type: str) -> bool:
    """Route ``model_type``'s Mamba2 mixer through the tree driver on tree rows."""
    if model_type not in MAMBA2_MIXERS:
        return False
    module_type, mixer_name = MAMBA2_MIXERS[model_type]
    module = importlib.import_module(
        f"transformers.models.{module_type}.modeling_{module_type}"
    )
    mixer_cls = getattr(module, mixer_name)
    if getattr(mixer_cls.forward, "_axolotl_recurrent_tree", False):
        return True
    original_forward = mixer_cls.forward

    @functools.wraps(original_forward)
    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        plan = tree_plan(kwargs) if cache_params is None else None
        # Nothing shared: the samples restart position_ids, so the packing
        # patches' seq_idx keeps them apart and the fused kernel can run.
        if plan is None or len(plan.levels) == 1:
            return original_forward(
                self,
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )
        # the path scan resets at seq_idx; the torch fallback would ignore it
        assert_mamba2_scan_honours_seq_idx(module)
        return mamba2_tree_forward(self, module, hidden_states, plan)

    mixer_cls.forward = mark_tree_forward(forward)
    LOG.info(f"Routed {mixer_name} through the recurrent tree driver")
    return True
