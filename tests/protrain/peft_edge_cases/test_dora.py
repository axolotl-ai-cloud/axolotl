"""DoRA + ProTrain smoke: magnitude vectors must traverse the per-region split alongside LoRA factors."""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.gpu


def _build_tiny_llama_with_dora():
    """Tiny Llama-arch LM with DoRA LoRA; prefers cached SmolLM2-135M, falls back to fresh-init."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
    )

    # Narrow to offline-load failure families so genuine API breakage still surfaces.
    try:
        cfg = AutoConfig.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M", local_files_only=True
        )
        cfg.use_cache = False
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
    except (OSError, ValueError, EnvironmentError):
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
        model = LlamaForCausalLM(cfg).to(dtype=torch.bfloat16)

    # --- DoRA-enabled LoRA config ----------------------------------------
    # Target the standard Llama attention + MLP linears. Use small r/alpha
    # to keep the smoke fast; DoRA's distinguishing feature is the
    # magnitude vector, not its rank.
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_dora=True,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    return peft_model, cfg


def test_protrain_dora_smoke() -> None:
    """ProTrain + DoRA: 5 iters, finite losses, strictly decreasing."""
    pytest.importorskip("torch")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain DoRA smoke requires CUDA.")

    peft_model, cfg = _build_tiny_llama_with_dora()

    device = torch.device("cuda:0")
    peft_model = peft_model.to(device)

    # --- Sanity: DoRA magnitude vectors must exist and be trainable ------
    # If this assertion fails, ``use_dora=True`` silently degraded to
    # plain LoRA and the test wouldn't actually stress the new tensors.
    magnitude_params = [
        (n, p) for n, p in peft_model.named_parameters() if "lora_magnitude_vector" in n
    ]
    assert magnitude_params, (
        "DoRA magnitude vectors not found; LoraConfig(use_dora=True) may "
        "have silently degraded — this test would be testing plain LoRA"
    )
    for n, p in magnitude_params:
        assert p.requires_grad, f"DoRA magnitude vector {n} not trainable"

    # ProTrain wrap: Mode-A (single GPU, all chunks GPU-resident).
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

    bs, seq = 1, 64
    # try/finally ensures hook handles, pinned-host borrows, and CPU adapter threads release on assertion failure.
    wrapped = protrain_model_wrapper(
        peft_model,
        model_config=cfg,
        hardware_profile=hw,
        batch_size=bs,
        seq_len=seq,
        capacity_bytes=20 * (1 << 30),
        force_all_persistent=True,
    )
    try:
        optim = protrain_optimizer_wrapper(wrapped, lr=1e-3)

        vocab = int(getattr(cfg, "vocab_size", 1024))
        torch.manual_seed(0)
        input_ids = torch.randint(0, vocab, (bs, seq), device=device, dtype=torch.long)
        labels = input_ids.clone()

        losses: list[float] = []
        n_iters = 5
        for i in range(n_iters):
            out = wrapped.module(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss_value = float(loss.detach())
            assert math.isfinite(loss_value), (
                f"iter {i}: non-finite loss {loss_value}; losses so far={losses}"
            )
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss_value)

        print(f"\nProTrain + DoRA smoke (tiny Llama): losses={losses}")

        # final < first on a fixed batch confirms DoRA magnitude vectors and LoRA factors actually receive gradient updates.
        assert all(math.isfinite(v) for v in losses), f"non-finite loss in {losses}"
        assert losses[-1] < losses[0], (
            f"DoRA + ProTrain loss did not decrease over {n_iters} iters: "
            f"{losses} — magnitude vectors or LoRA factors may not be "
            f"receiving gradient updates through the chunk-region split"
        )
    finally:
        close = getattr(wrapped, "close", None)
        if callable(close):
            close()
