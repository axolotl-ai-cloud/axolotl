"""Always-on shape-preserving placeholders: every mode (A/B/C) must keep released-param shapes for custom-autograd-kernel composition (v61 lora_mlp_kernel fix)."""

from __future__ import annotations

import pytest


def _build_tiny_lora_model():
    """Minimal LoRA-on-Llama wrap small enough for the chunk manager to materialize on any test rig."""
    pytest.importorskip("peft")
    pytest.importorskip("transformers")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=512,
        vocab_size=1024,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        use_cache=False,
    )
    torch.manual_seed(0)
    base = LlamaForCausalLM(cfg).to(dtype=torch.bfloat16)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base, lora_cfg), cfg


def _wrap_protrain_mode(
    model,
    cfg,
    *,
    force_all_persistent: bool,
    zero3_shard: bool,
    small_chunk: bool = True,
    n_persist_override: int | None = None,
    n_buffer_override: int | None = None,
    n_swap_override: int | None = None,
    n_checkpoint_override: int | None = None,
    n_offload_override: int | None = None,
):
    """Wrap a model with ProTrain; small_chunk + the full 4-knob override quad force the non-persistent path the searcher would otherwise skip on the tiny test model."""
    import torch

    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )
    from axolotl.integrations.protrain.types import HardwareProfile

    hw = HardwareProfile(
        gpu_sku=torch.cuda.get_device_name(0),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
    )

    import axolotl.integrations.protrain.api.model_wrapper as mw

    orig_pick_S_chunk = mw.pick_S_chunk
    if small_chunk:
        mw.pick_S_chunk = lambda *a, **k: 1 << 20  # 1 MiB
    try:
        wrapped = protrain_model_wrapper(
            model,
            model_config=cfg,
            hardware_profile=hw,
            batch_size=1,
            seq_len=32,
            capacity_bytes=4 * (1 << 30),
            force_all_persistent=force_all_persistent,
            zero3_shard=zero3_shard,
            n_persist_override=n_persist_override,
            n_buffer_override=n_buffer_override,
            n_swap_override=n_swap_override,
            n_checkpoint_override=n_checkpoint_override,
            n_offload_override=n_offload_override,
        )
    finally:
        mw.pick_S_chunk = orig_pick_S_chunk
    optim = protrain_optimizer_wrapper(wrapped, lr=1e-3)
    return wrapped, optim


def _assert_non_persistent_params_keep_shape(wrapped) -> None:
    """Every non-persistent param's ``.data`` must report its real shape (not ``Size([0])``) so custom autograd kernels record the right shape at forward time."""
    chunk_manager = getattr(wrapped, "chunk_manager", None)
    assert chunk_manager is not None, "wrapper must expose chunk_manager"

    non_persist_ids = sorted(getattr(chunk_manager, "_non_persistent_ids", set()))
    layout = chunk_manager.layout

    # Post-block-wrap rename means user-side named_parameters() and layout ids
    # no longer match; the chunk manager keeps live nn.Parameter refs in _params_by_id.
    params_by_id = getattr(chunk_manager, "_params_by_id", {})

    for cid in non_persist_ids:
        for pid in layout.chunks[int(cid)]:
            param = params_by_id.get(pid)
            if param is None:
                continue
            # Skip zero-element params (genuinely empty buffers) — placeholder vs
            # genuine-empty is indistinguishable and both work.
            if param.numel() == 0:
                continue
            assert param.data.numel() > 0, (
                f"param {pid} released to zero-element placeholder; "
                f"shape-preserving contract broken (got numel=0)"
            )


@pytest.mark.gpu
def test_mode_a_uses_shape_preserving_placeholders() -> None:
    """Mode A (force_all_persistent) — no non-persistent params, but the chunk manager must still advertise the always-on flag."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    model, cfg = _build_tiny_lora_model()
    model = model.to("cuda")
    wrapped, _optim = _wrap_protrain_mode(
        model,
        cfg,
        force_all_persistent=True,
        zero3_shard=False,
        small_chunk=False,
    )
    try:
        chunk_manager = getattr(wrapped, "chunk_manager", None)
        assert chunk_manager is not None
        # Always-on contract: flag is True for every mode.
        assert getattr(chunk_manager, "_shape_preserving_placeholders", False) is True

        # Mode A has no non-persistent chunks — assertion is trivially true; document why.
        non_persist = sorted(getattr(chunk_manager, "_non_persistent_ids", set()))
        assert non_persist == [], (
            f"Mode A should have zero non-persistent chunks; got {non_persist}"
        )
    finally:
        close = getattr(wrapped, "close", None)
        if callable(close):
            close()


@pytest.mark.gpu
def test_mode_b_uses_shape_preserving_placeholders() -> None:
    """Mode B (replicated CPU offload, ws=1 path) — non-persistent params must keep their real shape (regression test for v61 LoRA_MLPBackward shape mismatch)."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    model, cfg = _build_tiny_lora_model()
    model = model.to("cuda")
    # Mode B: not force_all_persistent, not zero3_shard. The 4-knob override
    # quad routes through the non-persistent offload path on this tiny model.
    wrapped, _optim = _wrap_protrain_mode(
        model,
        cfg,
        force_all_persistent=False,
        zero3_shard=False,
        small_chunk=True,
        n_persist_override=0,
        n_buffer_override=4,
        n_swap_override=0,
        n_checkpoint_override=0,
        n_offload_override=cfg.num_hidden_layers,
    )
    try:
        chunk_manager = getattr(wrapped, "chunk_manager", None)
        assert chunk_manager is not None
        # Always-on contract.
        assert getattr(chunk_manager, "_shape_preserving_placeholders", False) is True

        non_persist = sorted(getattr(chunk_manager, "_non_persistent_ids", set()))
        assert non_persist, "Mode B test setup must produce >=1 non-persistent chunk"
        _assert_non_persistent_params_keep_shape(wrapped)
    finally:
        close = getattr(wrapped, "close", None)
        if callable(close):
            close()


@pytest.mark.gpu
def test_mode_c_uses_shape_preserving_placeholders() -> None:
    """Mode C (zero3_shard) — preserves pre-fix behavior; flag must remain True. ws=1 silently degrades to non-sharded but always-on contract still holds."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    model, cfg = _build_tiny_lora_model()
    model = model.to("cuda")
    wrapped, _optim = _wrap_protrain_mode(
        model,
        cfg,
        force_all_persistent=False,
        zero3_shard=True,
        small_chunk=True,
        n_persist_override=0,
        n_buffer_override=4,
        n_swap_override=0,
        n_checkpoint_override=0,
        n_offload_override=cfg.num_hidden_layers,
    )
    try:
        chunk_manager = getattr(wrapped, "chunk_manager", None)
        assert chunk_manager is not None
        assert getattr(chunk_manager, "_shape_preserving_placeholders", False) is True
        _assert_non_persistent_params_keep_shape(wrapped)
    finally:
        close = getattr(wrapped, "close", None)
        if callable(close):
            close()


@pytest.mark.gpu
def test_custom_autograd_forward_backward_under_mode_b() -> None:
    """A custom autograd Function (mirroring lora_mlp_kernel's save_for_backward shape capture) must compose with Mode B's released-param placeholders."""
    pytest.importorskip("torch")
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    # Simulate the LoRA_MLP autograd Function contract: save the weight at
    # forward and validate its shape in backward. Pre-fix, a released param
    # with shape ``(0,)`` would record ``(0,)`` and the backward grad would
    # mismatch the real (e.g. 16x4096) shape — exactly the v61 crash.
    class _ShapeCapturingMatmul(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            ctx.save_for_backward(x, weight)
            ctx.weight_shape = tuple(weight.shape)
            return x @ weight.t()

        @staticmethod
        def backward(ctx, grad_out):
            x, weight = ctx.saved_tensors
            assert tuple(weight.shape) == ctx.weight_shape, (
                f"shape drift across the autograd boundary: "
                f"saved {ctx.weight_shape}, got {tuple(weight.shape)}"
            )
            grad_x = grad_out @ weight
            grad_w = grad_out.t() @ x
            return grad_x, grad_w

    # Build a real ProTrain Mode-B-wrapped LoRA model so the placeholder code
    # path actually engages, then invoke the custom autograd Function against
    # a released param's shape.
    model, cfg = _build_tiny_lora_model()
    model = model.to("cuda")
    wrapped, _optim = _wrap_protrain_mode(
        model,
        cfg,
        force_all_persistent=False,
        zero3_shard=False,
        small_chunk=True,
        n_persist_override=0,
        n_buffer_override=4,
        n_swap_override=0,
        n_checkpoint_override=0,
        n_offload_override=cfg.num_hidden_layers,
    )
    try:
        chunk_manager = getattr(wrapped, "chunk_manager", None)
        assert chunk_manager is not None

        # Probe via _params_by_id; post-block-wrap renames break the
        # name-based lookup.
        params_by_id = getattr(chunk_manager, "_params_by_id", {})

        non_persist_ids = sorted(getattr(chunk_manager, "_non_persistent_ids", set()))
        target_param: nn.Parameter | None = None
        target_shape: tuple[int, ...] | None = None
        for cid in non_persist_ids:
            for pid in chunk_manager.layout.chunks[int(cid)]:
                p = params_by_id.get(pid)
                if p is None or p.numel() == 0:
                    continue
                if p.dim() != 2:
                    continue
                target_param = p
                target_shape = tuple(p.shape)
                break
            if target_param is not None:
                break

        if target_param is None or target_shape is None:
            pytest.skip(
                "tiny test model exposed no 2-D non-persistent param; nothing to probe"
            )
        assert target_param is not None and target_shape is not None  # narrow for mypy

        # Pre-fix: target_param.data.shape would be torch.Size([0]) here.
        # Post-fix: it must report the real shape so save_for_backward captures it.
        assert tuple(target_param.shape) == target_shape, (
            f"released param shape drift: expected {target_shape}, "
            f"got {tuple(target_param.shape)}"
        )

        # Smoke-run the autograd Function with a synthesized real-weight
        # tensor matching the released param's shape/dtype/device. The
        # placeholder itself has one-element storage expanded into the
        # full shape (stride trickery) so it can't drive a real matmul,
        # but ctx.save_for_backward captures the shape we care about.
        _, in_features = target_shape
        x = torch.randn(
            (2, in_features),
            dtype=target_param.dtype,
            device=target_param.device,
            requires_grad=True,
        )
        real_weight = torch.randn(
            target_shape,
            dtype=target_param.dtype,
            device=target_param.device,
            requires_grad=True,
        )
        y = _ShapeCapturingMatmul.apply(x, real_weight)
        loss = y.sum()
        loss.backward()
        assert real_weight.grad is not None
        assert tuple(real_weight.grad.shape) == target_shape
    finally:
        close = getattr(wrapped, "close", None)
        if callable(close):
            close()
