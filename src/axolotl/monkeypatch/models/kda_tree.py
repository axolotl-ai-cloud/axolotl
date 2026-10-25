"""The Kimi Delta Attention tree component (Kimi-Linear, Bailing hybrid).

FLA's chunk_kda takes per-sequence initial states with varlen boundaries and
returns the final states, so the tree driver can fork the KDA state at the
trie's branches (``recurrent_tree`` says when it does); the q/k/v short convs
run with each segment's path history.
"""

from __future__ import annotations

from axolotl.monkeypatch.models.recurrent_tree import (
    RecurrentComponent,
    TreeFamily,
    register_tree_family,
)

KDA = RecurrentComponent("kda", initial_state=True)

for model_types, mixer in (
    (("kimi_linear",), "KimiDeltaAttention"),
    (("bailing_hybrid",), "BailingMoeV3KimiDeltaAttention"),
):
    register_tree_family(TreeFamily(model_types, (mixer,), KDA))


def kda_scan(chunk_kda):
    """A tree scan over FLA's ``chunk_kda`` (inputs q, k, v, g, beta)."""

    def scan(inputs, cu_seqlens, initial_state):
        q, k, v, g, beta = inputs
        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

    return scan


def short_convolution(conv):
    """A tree conv over FLA's ``ShortConvolution`` on ``[1, T, D]`` inputs."""

    def run(x, cu_seqlens):
        return conv(x=x, cu_seqlens=cu_seqlens)[0]

    return run
