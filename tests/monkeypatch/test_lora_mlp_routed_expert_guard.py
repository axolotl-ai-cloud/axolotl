"""CPU tests for the lora_mlp_kernel routed-expert guard in find_mlp_in_layer.

When a custom MoE expert kernel (ScatterMoE/SonicMoE) owns the routed experts, the generic
lora_mlp_kernel must fuse ONLY the dense shared MLP, never the routed-expert containers (which the
MoE kernel handles). The dense MLP stays patchable in both cases.
"""

from types import SimpleNamespace

import torch.nn as nn

from axolotl.monkeypatch.lora_kernels import find_mlp_in_layer


def _lin():
    return nn.Linear(4, 4)


def _dense_mlp():
    return SimpleNamespace(gate_proj=_lin(), up_proj=_lin(), down_proj=_lin())


def _routed_experts(n):
    return SimpleNamespace(
        gate_projs=[_lin() for _ in range(n)],
        up_projs=[_lin() for _ in range(n)],
        down_projs=[_lin() for _ in range(n)],
    )


def test_dense_shared_mlp_always_found():
    layer = SimpleNamespace(mlp=_dense_mlp())
    for skip in (False, True):
        mlps = list(find_mlp_in_layer(layer, skip_routed_experts=skip))
        assert len(mlps) == 1
        assert mlps[0][3] is layer.mlp  # the dense MLP module itself


def test_routed_experts_skipped_when_moe_kernel_owns_them():
    layer = SimpleNamespace(feedforward=SimpleNamespace(experts=_routed_experts(3)))
    # default (no custom MoE kernel): routed experts are yielded for fusion
    assert len(list(find_mlp_in_layer(layer))) == 3
    # under ScatterMoE/SonicMoE: routed experts must NOT be yielded
    assert list(find_mlp_in_layer(layer, skip_routed_experts=True)) == []


def test_dense_kept_routed_skipped_together():
    layer = SimpleNamespace(
        mlp=_dense_mlp(),
        feedforward=SimpleNamespace(experts=_routed_experts(2)),
    )
    full = list(find_mlp_in_layer(layer, skip_routed_experts=False))
    guarded = list(find_mlp_in_layer(layer, skip_routed_experts=True))
    assert len(full) == 3  # dense + 2 routed
    assert len(guarded) == 1  # only the dense MLP
    assert guarded[0][3] is layer.mlp


def test_dsv4_translation_disables_generic_mlp_kernel():
    # DSV4 keeps its dedicated clamped-SwiGLU kernel: generic lora_mlp_kernel is translated and off.
    from axolotl.integrations.kernels.args import KernelsArgs

    out = KernelsArgs.disable_mlp_kernel(
        {"use_scattermoe": True, "use_dsv4_kernels": True, "lora_mlp_kernel": True}
    )
    assert out["lora_mlp_kernel"] is False
    assert out["dsv4_shared_mlp_lora_kernel"] is True
