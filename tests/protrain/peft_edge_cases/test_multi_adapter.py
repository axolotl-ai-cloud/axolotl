"""Multiple-LoRA-adapter + ProTrain composition smoke test (M6A test 2).

PEFT supports loading several named LoRA adapter configs onto a single
base model and switching between them via ``set_adapter``. ProTrain's
chunk manager segments per-chunk regions on a ``(dtype, requires_grad)``
boundary; switching the active adapter changes which sub-Parameters'
``requires_grad`` is True, so the chunk-region split must absorb the
``set_adapter`` transition without state-dict corruption.

Smoke contract:

* Build a tiny Llama-arch LM, attach two named PEFT LoRA adapters
  ("alpha" and "beta") with different ranks.
* Train 3 iters with ``alpha`` active, then 3 iters with ``beta``
  active, against ProTrain in Mode-A.
* Assert: no crash on the ``set_adapter`` switch; per-adapter loss is
  finite and decreases across its 3 iters on a fixed batch.

Substitution rationale: same as ``test_dora.py`` — uses tiny synthetic
Llama (no HF download) to keep the smoke under 30s wall-clock and
avoid any 8B+ memory pressure (which crashed the prior M5 attempt).
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.gpu


def _build_tiny_llama_with_two_adapters():
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

    lora_alpha = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    lora_beta = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_alpha, adapter_name="alpha")
    peft_model.add_adapter("beta", lora_beta)
    return peft_model, cfg


def _wrap_protrain(peft_model, cfg, *, bs: int, seq: int, capacity_bytes: int):
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
        peft_model,
        model_config=cfg,
        hardware_profile=hw,
        batch_size=bs,
        seq_len=seq,
        capacity_bytes=capacity_bytes,
        force_all_persistent=True,
    )
    optim = protrain_optimizer_wrapper(wrapped, lr=1e-3)
    return wrapped, optim


def _train_loop(wrapped, optim, *, n_iters, input_ids, labels) -> list[float]:

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


def test_protrain_multi_lora_adapter_switch() -> None:
    """ProTrain + multi-LoRA adapter switch: alpha 3 iters, beta 3 iters, no crash."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain multi-adapter smoke requires CUDA.")

    peft_model, cfg = _build_tiny_llama_with_two_adapters()
    device = torch.device("cuda:0")
    peft_model = peft_model.to(device)

    # Sanity: both adapters are present.
    adapter_names = set(getattr(peft_model.peft_config, "keys", lambda: [])())
    assert {"alpha", "beta"}.issubset(adapter_names), (
        f"expected both adapters loaded, got {adapter_names}"
    )

    bs, seq = 1, 32
    vocab = int(cfg.vocab_size)
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (bs, seq), device=device, dtype=torch.long)
    labels = input_ids.clone()

    # Wrap once with adapter alpha active. Train 3 iters.
    peft_model.set_adapter("alpha")
    wrapped_a, optim_a = _wrap_protrain(
        peft_model, cfg, bs=bs, seq=seq, capacity_bytes=4 * (1 << 30)
    )
    losses_alpha = _train_loop(
        wrapped_a, optim_a, n_iters=3, input_ids=input_ids, labels=labels
    )
    assert losses_alpha[-1] < losses_alpha[0], (
        f"alpha adapter did not train: {losses_alpha}"
    )

    # Switch to beta. Re-wrap (chunk layout depends on requires_grad which
    # changed) and train another 3 iters. The point of the test is that
    # the set_adapter transition + re-wrap path doesn't crash and beta
    # also makes progress.
    peft_model.set_adapter("beta")
    wrapped_b, optim_b = _wrap_protrain(
        peft_model, cfg, bs=bs, seq=seq, capacity_bytes=4 * (1 << 30)
    )
    losses_beta = _train_loop(
        wrapped_b, optim_b, n_iters=3, input_ids=input_ids, labels=labels
    )
    assert losses_beta[-1] < losses_beta[0], (
        f"beta adapter did not train after switch: {losses_beta}"
    )

    print(
        f"\nProTrain + multi-adapter: losses_alpha={losses_alpha} "
        f"losses_beta={losses_beta}"
    )
