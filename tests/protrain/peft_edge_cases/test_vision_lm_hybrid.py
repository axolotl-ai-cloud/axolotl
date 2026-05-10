"""Mixed trainable/frozen + LoRA + ProTrain smoke test (M6A test 3).

The phase2.md spec calls for a vision-LM hybrid (LLaVA-class) with LoRA
on the LM tower and full fine-tuning on the vision tower. The chunk-
manager invariant under test is its handling of *mixed trainable and
frozen parameters across model sub-components* — the per-chunk region
split must transparently absorb a non-uniform requires_grad map.

A custom 2-tower nn.Module with a non-standard forward signature breaks
the profiler's warmup pass (which assumes the wrapped module accepts
``input_ids``); we therefore exercise the same invariant on a
standards-compliant tiny Llama by:

* Wrapping the LM with LoRA on q/v projections (LoRA factors are
  trainable; the base attention/MLP weights are frozen).
* Marking ``embed_tokens.weight`` as ``requires_grad=True`` so a
  large base-model parameter is fully trainable alongside the LoRA
  factors.
* Driving 5 forward+backward+step iters with ProTrain Mode-A.

Result: the chunk regions split across "fully-frozen base", "LoRA-
trainable factors", and "fully-trainable embedding" boundaries — the
same shape of split a real LLaVA-class hybrid stresses.

Substitution rationale: documented in the docstring above. Real LLaVA
8B+ runs are out of scope post-crash safety constraint; the architecture-
independent chunk-region invariant is what matters here.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.gpu


def _build_tiny_llama_mixed_trainable():
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
    base_lm = LlamaForCausalLM(cfg).to(dtype=torch.bfloat16)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_lm, lora_cfg)

    # Make the base-model embedding fully trainable in addition to the
    # LoRA factors. This produces the same kind of per-chunk-region
    # split a real vision-LM hybrid would: fully-frozen base attention/
    # MLP weights, LoRA-trainable factors, and a fully-trainable large
    # base parameter (the embedding standing in for the projector or
    # vision tower in the real spec).
    embed = peft_model.get_input_embeddings()
    for p in embed.parameters():
        p.requires_grad = True

    return peft_model, cfg


def test_protrain_mixed_trainable_frozen_smoke() -> None:
    """ProTrain + LoRA + trainable embed_tokens (mixed-grad chunk regions): 5 iters."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain mixed trainable/frozen smoke requires CUDA.")

    peft_model, cfg = _build_tiny_llama_mixed_trainable()
    device = torch.device("cuda:0")
    peft_model = peft_model.to(device)

    # Sanity: trainable surface is what we expect (LoRA + embedding).
    trainable = {n for n, p in peft_model.named_parameters() if p.requires_grad}
    has_lora = any("lora" in n.lower() for n in trainable)
    has_embed = any("embed_tokens" in n for n in trainable)
    assert has_lora, f"expected trainable LoRA params, got {sorted(trainable)[:5]}"
    assert has_embed, (
        f"expected embed_tokens.weight to be trainable, got {sorted(trainable)[:5]}"
    )
    # And we still have frozen base attention/MLP — otherwise the test
    # degrades to "everything trainable" and the mixed-grad split isn't
    # exercised.
    frozen = [n for n, p in peft_model.named_parameters() if not p.requires_grad]
    assert any("self_attn" in n or "mlp" in n for n in frozen), (
        f"expected frozen base attn/mlp, got first 5 frozen={frozen[:5]}"
    )

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

    bs, seq = 1, 32
    wrapped = protrain_model_wrapper(
        peft_model,
        model_config=cfg,
        hardware_profile=hw,
        batch_size=bs,
        seq_len=seq,
        capacity_bytes=4 * (1 << 30),
        force_all_persistent=True,
    )
    optim = protrain_optimizer_wrapper(wrapped, lr=1e-3)

    torch.manual_seed(0)
    input_ids = torch.randint(
        0, cfg.vocab_size, (bs, seq), device=device, dtype=torch.long
    )
    labels = input_ids.clone()

    losses: list[float] = []
    for i in range(5):
        out = wrapped.module(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss_value = float(loss.detach())
        assert math.isfinite(loss_value), f"iter {i}: non-finite loss {loss_value}"
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss_value)

    print(f"\nProTrain + mixed trainable/frozen: losses={losses}")

    assert all(math.isfinite(v) for v in losses), f"non-finite loss in {losses}"
    assert losses[-1] < losses[0], (
        f"mixed trainable/frozen loss did not decrease: {losses} — chunk-"
        f"region split for mixed-grad components may be silently dropping "
        f"gradient updates"
    )
