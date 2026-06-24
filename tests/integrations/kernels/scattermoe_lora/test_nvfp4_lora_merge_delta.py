# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Per-expert LoRA delta reconstruction (step 1 of the NVFP4 expert-LoRA merge).

The CPU test loads ``nvfp4_lora_merge`` by file path so the scattermoe_lora package
``__init__`` (which imports triton) is NOT executed: ``reconstruct_expert_delta`` is pure
torch and must run on a triton-less / no-CUDA host. It checks shape and equality against
a GENUINELY INDEPENDENT reference (an explicit per-expert python loop of B_e @ A_e, slicing
the scattermoe-layout tensors exactly as the kernel does), catching any transpose /
orientation bug without a GPU.

The triton+GPU test is the definitive orientation guard: it runs the real fused
``parallel_linear_lora`` against ``parallel_linear`` (no LoRA) on the same input + base
weight and asserts the difference equals the input pushed through the reconstructed dense
delta. Skips when triton or CUDA is unavailable.
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch


def _load_merge_module():
    """Load nvfp4_lora_merge by file path so the scattermoe_lora package __init__ (which imports
    triton) is NOT executed. The module needs only torch, so it imports triton-free and runs on CPU."""
    import axolotl

    path = os.path.join(
        os.path.dirname(axolotl.__file__),
        "integrations",
        "kernels",
        "libs",
        "scattermoe_lora",
        "nvfp4_lora_merge.py",
    )
    spec = importlib.util.spec_from_file_location(
        "_axolotl_nvfp4_lora_merge_under_test", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


merge_mod = _load_merge_module()


def _make_scattermoe_AB(E, rank, in_features, out_features, seed=0):
    """Random LoRA A/B already in scattermoe layout: A [r*E, in], B [out, r*E]."""
    g = torch.Generator().manual_seed(seed)
    sm_A = torch.randn(rank * E, in_features, generator=g, dtype=torch.float64)
    sm_B = torch.randn(out_features, rank * E, generator=g, dtype=torch.float64)
    return sm_A, sm_B


def _reference_delta_loop(sm_A, sm_B, scaling, E, rank):
    """Independent reference: explicit per-expert loop, slicing A/B exactly as the kernel.

    A_e = sm_A[e*r:(e+1)*r, :]  ([r, in]); B_e = sm_B[:, e*r:(e+1)*r]  ([out, r]).
    delta_e = scaling * (B_e @ A_e)  ([out, in]). Stacked over experts -> [E, out, in].
    """
    out_features = sm_B.shape[0]
    in_features = sm_A.shape[1]
    delta = torch.empty(E, out_features, in_features, dtype=sm_A.dtype)
    for e in range(E):
        a_e = sm_A[e * rank : (e + 1) * rank, :]
        b_e = sm_B[:, e * rank : (e + 1) * rank]
        delta[e] = scaling * (b_e @ a_e)
    return delta


@pytest.mark.parametrize(
    "E,rank,in_features,out_features,scaling",
    [
        (4, 8, 16, 24, 0.5),
        (3, 4, 10, 6, 2.0),
        (1, 16, 32, 8, 1.0),
    ],
    ids=["E4_r8", "E3_r4", "E1_r16"],
)
def test_reconstruct_expert_delta_matches_independent_loop(
    E, rank, in_features, out_features, scaling
):
    """reconstruct_expert_delta returns [E, out, in] and equals an independent per-expert loop."""
    sm_A, sm_B = _make_scattermoe_AB(E, rank, in_features, out_features)

    delta = merge_mod.reconstruct_expert_delta(sm_A, sm_B, scaling, E, rank)
    assert delta.shape == (E, out_features, in_features), delta.shape

    ref = _reference_delta_loop(sm_A, sm_B, scaling, E, rank)
    assert torch.allclose(delta, ref, atol=1e-10), (delta - ref).abs().max().item()


def test_extract_from_state_dict_reproduces_runtime_layout():
    """extract_expert_lora_from_state_dict applies the same B-layout conversion as train-time.

    Builds a fake adapter state dict in PEFT save shape (A [r*E, in], B [out, r*E] rank-major)
    and checks the returned sm_A/sm_B match a direct rank-major -> expert-major B reshape, and that
    the reconstructed delta equals the independent per-expert loop over the converted tensors.
    """
    E, rank, in_features, out_features, scaling = 4, 8, 16, 24, 0.5
    g = torch.Generator().manual_seed(7)
    # PEFT lora_A.weight: [r*E, in]; lora_B.weight: [out, r*E] (rank-major).
    peft_A = torch.randn(rank * E, in_features, generator=g, dtype=torch.float64)
    peft_B = torch.randn(out_features, rank * E, generator=g, dtype=torch.float64)

    sd = {
        "base_model.model.model.layers.5.mlp.experts.gate_up_proj.lora_A.weight": peft_A,
        "base_model.model.model.layers.5.mlp.experts.gate_up_proj.lora_B.weight": peft_B,
    }
    extracted = merge_mod.extract_expert_lora_from_state_dict(sd, E, scaling)
    assert (5, "gate_up_proj") in extracted, extracted.keys()
    sm_A, sm_B, sc = extracted[(5, "gate_up_proj")]
    assert sc == scaling

    # sm_A is unchanged; sm_B is rank-major -> expert-major.
    assert torch.equal(sm_A, peft_A)
    expected_B = (
        peft_B.reshape(out_features, rank, E)
        .permute(0, 2, 1)
        .contiguous()
        .reshape(out_features, E * rank)
    )
    assert torch.equal(sm_B, expected_B)

    delta = merge_mod.reconstruct_expert_delta(sm_A, sm_B, scaling, E, rank)
    ref = _reference_delta_loop(sm_A, sm_B, scaling, E, rank)
    assert torch.allclose(delta, ref, atol=1e-10)


def test_reconstruct_delta_per_expert_against_explicit_BA():
    """Sanity at E=2: delta[e] is exactly scaling * (B_e @ A_e), built without the impl path."""
    E, rank, in_features, out_features, scaling = 2, 3, 5, 4, 1.5
    sm_A, sm_B = _make_scattermoe_AB(E, rank, in_features, out_features, seed=3)
    delta = merge_mod.reconstruct_expert_delta(sm_A, sm_B, scaling, E, rank)
    for e in range(E):
        a_e = sm_A[e * rank : (e + 1) * rank, :]
        b_e = sm_B[:, e * rank : (e + 1) * rank]
        assert torch.allclose(delta[e], scaling * (b_e @ a_e), atol=1e-10)


def test_reconstruct_delta_orientation_via_fused_kernel():
    """Definitive orientation guard: the fused LoRA kernel's delta == x @ reconstructed_delta^T.

    Runs parallel_linear_lora (base + fused LoRA) and parallel_linear (base only) on the same
    input and base weight, then checks (lora_out - base_out) equals applying the reconstructed
    dense per-expert delta to each token. Requires triton + CUDA.
    """
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the fused ScatterMoE LoRA kernel")

    from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
        flatten_sort_count,
        parallel_linear,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_linear_lora import (
        parallel_linear_lora,
    )

    device = "cuda"
    dtype = torch.float32
    E, rank = 4, 8
    in_features, out_features = 64, 96
    top_k = 2
    n_tokens = 128

    torch.manual_seed(0)
    x = torch.randn(n_tokens, in_features, device=device, dtype=dtype)
    # Base expert weight handed to the kernel is [E, in, out] (= stored [E, out, in] transposed).
    W_kernel = torch.randn(E, in_features, out_features, device=device, dtype=dtype)
    sm_A = torch.randn(rank * E, in_features, device=device, dtype=dtype)
    sm_B = torch.randn(out_features, rank * E, device=device, dtype=dtype)
    scaling = 0.5

    sel = torch.randint(0, E, (n_tokens, top_k), device=device)
    sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
        sel, num_experts=E
    )

    base_out = parallel_linear(
        x,
        W_kernel,
        top_k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        grouped_in=False,
        grouped_out=True,
    )
    lora_out = parallel_linear_lora(
        x,
        W_kernel,
        top_k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        lora_A=sm_A,
        lora_B=sm_B,
        scaling=scaling,
        grouped_in=False,
        grouped_out=True,
    )
    kernel_delta_out = lora_out - base_out  # [n_tokens*top_k, out], grouped order

    # Reconstructed dense delta in stored [E, out, in]; apply per token via its expert.
    dense_delta = merge_mod.reconstruct_expert_delta(
        sm_A.cpu(), sm_B.cpu(), scaling, E, rank
    ).to(device)

    # grouped_out: row i corresponds to expert sorted_expert_idxs[i] and token
    # sorted_scattered_idxs[i] // top_k.
    tok = (sorted_scattered_idxs // top_k).long()
    exp = sorted_expert_idxs.long()
    x_rows = x[tok]  # [L, in]
    W_delta_rows = dense_delta[exp]  # [L, out, in]
    ref_delta_out = torch.bmm(W_delta_rows, x_rows.unsqueeze(-1)).squeeze(
        -1
    )  # [L, out]

    assert torch.allclose(kernel_delta_out, ref_delta_out, atol=1e-3, rtol=1e-3), (
        (kernel_delta_out - ref_delta_out).abs().max().item()
    )
