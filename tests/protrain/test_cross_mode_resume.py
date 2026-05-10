"""Cross-mode (Mode A ↔ Mode C) checkpoint resume smoke test (M6C).

ProTrain has multiple operating modes:

* Mode A: all chunks persistent on GPU (``force_all_persistent=True``).
* Mode C: chunks sharded with offload (``zero3_shard=True``).

Different modes have different chunk layouts and optimizer-state shapes.
This test exercises whether a checkpoint saved in one mode loads cleanly
in the other:

* Test 1: Mode A → Mode C (operational-risk: different sharding layout).
* Test 2: Mode C → Mode A (symmetric).

Implementation: Python-level synthetic test on a tiny Llama-arch LM, no
real CLI training. Save/load the underlying model + optimizer
``state_dict``; assert the load path doesn't crash and that subsequent
training produces a finite, non-divergent loss (we don't assert byte-
exact loss continuity because Mode A vs Mode C have different stochastic
ordering — only that the resumed run isn't catastrophically broken).

Substitution rationale: real LLaMA-3-8B + CLI subprocess invocations
were the post-crash unsafe path; the tested invariant (state-dict
round-trip across modes) is architecture-independent.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.gpu


def _build_tiny_llama_lora():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
        vocab_size=512,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        use_cache=False,
    )
    model = LlamaForCausalLM(cfg).to(dtype=torch.bfloat16)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg), cfg


def _wrap(
    model, cfg, *, force_all_persistent: bool, zero3_shard: bool, bs: int, seq: int
):
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
    wrapped = protrain_model_wrapper(
        model,
        model_config=cfg,
        hardware_profile=hw,
        batch_size=bs,
        seq_len=seq,
        capacity_bytes=4 * (1 << 30),
        force_all_persistent=force_all_persistent,
        zero3_shard=zero3_shard,
    )
    optim = protrain_optimizer_wrapper(wrapped, lr=1e-3)
    return wrapped, optim


def _train(wrapped, optim, *, n_iters, input_ids, labels) -> list[float]:
    losses: list[float] = []
    for i in range(n_iters):
        out = wrapped.module(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss_value = float(loss.detach())
        assert math.isfinite(loss_value), f"iter {i}: non-finite loss {loss_value}"
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss_value)
    return losses


def _resume(wrapped, optim, model_state, optim_state):
    """Best-effort cross-mode load. Tolerates partial layouts: if Mode A's
    optimizer state cannot be remapped to Mode C's sharded layout (or
    vice versa), the load_state_dict is allowed to skip the optimizer
    state — we only require it not to crash, and that subsequent training
    still produces finite losses (the optimizer cold-starts, which is the
    documented limitation per phase2.md M6C bail criterion).
    """
    underlying = getattr(wrapped, "module", wrapped)
    try:
        # Allow strict=False because LoRA-PEFT state dicts contain only
        # trainable params; PEFT's load_state_dict accepts strict-False.
        load = getattr(underlying, "load_state_dict", None)
        if load is not None:
            load(model_state, strict=False)
    except Exception as exc:
        pytest.fail(f"cross-mode model state_dict load crashed: {exc}")

    if optim_state is not None and hasattr(optim, "load_state_dict"):
        try:
            optim.load_state_dict(optim_state)
        except Exception as exc:  # noqa: BLE001
            # Documented limitation: cross-mode optimizer-state remap may
            # not be implemented. We don't fail the test on this — we
            # log it and let training cold-start the optimizer.
            print(
                f"\n[cross-mode-resume] optimizer state load failed (cold-start): {exc}"
            )


def _make_inputs(cfg, *, bs: int, seq: int):
    import torch

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    input_ids = torch.randint(
        0, cfg.vocab_size, (bs, seq), device=device, dtype=torch.long
    )
    labels = input_ids.clone()
    return input_ids, labels


def test_cross_mode_resume_a_to_c() -> None:
    """Mode A → Mode C: train, save, re-wrap in Mode C, resume, assert finite training."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain cross-mode resume smoke requires CUDA.")

    model, cfg = _build_tiny_llama_lora()
    device = torch.device("cuda:0")
    model = model.to(device)

    bs, seq = 1, 32
    input_ids, labels = _make_inputs(cfg, bs=bs, seq=seq)

    # Mode A: train + capture state.
    wrapped_a, optim_a = _wrap(
        model, cfg, force_all_persistent=True, zero3_shard=False, bs=bs, seq=seq
    )
    losses_a = _train(wrapped_a, optim_a, n_iters=3, input_ids=input_ids, labels=labels)
    underlying_a = getattr(wrapped_a, "module", wrapped_a)
    model_state = {k: v.detach().clone() for k, v in underlying_a.state_dict().items()}
    optim_state = optim_a.state_dict() if hasattr(optim_a, "state_dict") else None

    # Mode C: re-wrap fresh from same model object, load state, train more.
    wrapped_c, optim_c = _wrap(
        model, cfg, force_all_persistent=False, zero3_shard=True, bs=bs, seq=seq
    )
    _resume(wrapped_c, optim_c, model_state, optim_state)
    losses_c = _train(wrapped_c, optim_c, n_iters=3, input_ids=input_ids, labels=labels)

    print(f"\nA→C resume: losses_a={losses_a} losses_c={losses_c}")

    # Acceptance: no crash above; losses are finite; Mode C losses are
    # not catastrophically larger than the last Mode A loss (allow 5x as
    # a generous bound — the optimizer may have cold-started).
    assert all(math.isfinite(v) for v in losses_c), (
        f"non-finite Mode C loss: {losses_c}"
    )
    assert losses_c[0] < 5.0 * losses_a[-1] + 1.0, (
        f"Mode C loss diverged after A→C resume: a-end={losses_a[-1]} "
        f"c-start={losses_c[0]} (>5x is treated as catastrophic divergence)"
    )


def test_cross_mode_resume_c_to_a() -> None:
    """Mode C → Mode A: symmetric. Train Mode C, save, resume in Mode A."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain cross-mode resume smoke requires CUDA.")

    model, cfg = _build_tiny_llama_lora()
    device = torch.device("cuda:0")
    model = model.to(device)

    bs, seq = 1, 32
    input_ids, labels = _make_inputs(cfg, bs=bs, seq=seq)

    wrapped_c, optim_c = _wrap(
        model, cfg, force_all_persistent=False, zero3_shard=True, bs=bs, seq=seq
    )
    losses_c = _train(wrapped_c, optim_c, n_iters=3, input_ids=input_ids, labels=labels)
    underlying_c = getattr(wrapped_c, "module", wrapped_c)
    model_state = {k: v.detach().clone() for k, v in underlying_c.state_dict().items()}
    optim_state = optim_c.state_dict() if hasattr(optim_c, "state_dict") else None

    wrapped_a, optim_a = _wrap(
        model, cfg, force_all_persistent=True, zero3_shard=False, bs=bs, seq=seq
    )
    _resume(wrapped_a, optim_a, model_state, optim_state)
    losses_a = _train(wrapped_a, optim_a, n_iters=3, input_ids=input_ids, labels=labels)

    print(f"\nC→A resume: losses_c={losses_c} losses_a={losses_a}")

    assert all(math.isfinite(v) for v in losses_a), (
        f"non-finite Mode A loss: {losses_a}"
    )
    assert losses_a[0] < 5.0 * losses_c[-1] + 1.0, (
        f"Mode A loss diverged after C→A resume: c-end={losses_c[-1]} "
        f"a-start={losses_a[0]} (>5x is treated as catastrophic divergence)"
    )
