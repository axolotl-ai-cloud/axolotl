"""The GatedDeltaNet tree component (Qwen3.5, Qwen3.5-MoE, Qwen3-Next).

FLA's chunk_gated_delta_rule takes per-sequence initial states with varlen
boundaries and returns the final states, so the tree driver can fork the
delta-rule state at the trie's branches (``recurrent_tree`` says when it does);
the short conv runs with each segment's path history.
"""

from __future__ import annotations

from axolotl.monkeypatch.models.recurrent_tree import (
    RecurrentComponent,
    TreeFamily,
    register_tree_family,
)

GATED_DELTA_RULE = RecurrentComponent("gated_delta_rule", initial_state=True)

for model_types, mixer in (
    (("qwen3_5", "qwen3_5_text"), "Qwen3_5GatedDeltaNet"),
    (("qwen3_5_moe", "qwen3_5_moe_text"), "Qwen3_5MoeGatedDeltaNet"),
    (("qwen3_next",), "Qwen3NextGatedDeltaNet"),
):
    register_tree_family(TreeFamily(model_types, (mixer,), GATED_DELTA_RULE))


def gated_delta_rule_scan(chunk_gated_delta_rule):
    """A tree scan over FLA's ``chunk_gated_delta_rule`` (inputs q, k, v, g, beta)."""

    def scan(inputs, cu_seqlens, initial_state):
        query, key, value, g, beta = inputs
        return chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

    return scan


def varlen_conv(causal_conv1d, weight, bias, activation):
    """A tree conv over FLA's varlen ``causal_conv1d`` on ``[1, T, D]`` inputs."""

    def conv(x, cu_seqlens):
        out, _ = causal_conv1d(
            x=x, weight=weight, bias=bias, activation=activation, cu_seqlens=cu_seqlens
        )
        return out

    return conv
