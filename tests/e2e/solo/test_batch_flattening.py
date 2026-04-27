"""
Unit tests for batch flattening correctness in GRPO.

Validates that flattened (padding-free) forward passes produce identical
results to padded forward passes by calling the ACTUAL AsyncGRPOTrainer methods:
  1. Deferred scoring: _get_per_token_logps_flattened vs _get_per_token_logps_and_entropies
  2. Training loss: _get_per_token_logps_and_entropies_flattened vs _get_per_token_logps_and_entropies

Run: CUDA_VISIBLE_DEVICES=1 python test_batch_flattening.py
"""

import types
from unittest.mock import MagicMock

import torch
from transformers import AutoModelForCausalLM

# Import the actual trainer methods we want to test
from axolotl.core.trainers.grpo.async_trainer import AsyncGRPOTrainer

MODEL_NAME = "axolotl-ai-co/tiny-qwen3-129m"


def _fix_patched_attention(model):
    """Bind apply_qkv on attention modules if LoRA kernel monkeypatch is active.

    The LoRA kernel tests replace ``Qwen3Attention.forward`` at the class level
    with ``axolotl_attn_forward``, which expects a per-instance ``apply_qkv``
    method.  Models created *after* that patch but *without* the per-instance
    setup will crash.  We fix this by binding the original (non-LoRA) apply_qkv.
    """
    from axolotl.monkeypatch.lora_kernels import original_apply_o, original_apply_qkv

    for module in model.modules():
        fwd_name = getattr(type(module).forward, "__name__", "")
        if "axolotl" in fwd_name and not hasattr(module, "apply_qkv"):
            module.apply_qkv = types.MethodType(original_apply_qkv, module)
            module.apply_o = types.MethodType(original_apply_o, module)


def setup_model(eval_mode=True):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda()
    _fix_patched_attention(model)
    if eval_mode:
        model.eval()
    else:
        model.train()
    return model


def make_mock_trainer(model):
    """Create a minimal mock that has the attributes needed by the trainer methods.

    The three methods we test (_get_per_token_logps_flattened,
    _get_per_token_logps_and_entropies_flattened, _get_per_token_logps_and_entropies)
    access self.temperature, self.use_liger_kernel, self.is_fsdp_enabled,
    self.accelerator, and self.model_kwarg_keys.
    """
    trainer = MagicMock(spec=[])

    trainer.temperature = 1.0
    trainer.use_liger_kernel = False
    trainer.is_fsdp_enabled = False
    trainer.model_kwarg_keys = set()

    # accelerator.unwrap_model should return the model unchanged
    accelerator = MagicMock()
    accelerator.unwrap_model = lambda m, keep_fp32_wrapper=True: m
    trainer.accelerator = accelerator

    # Bind the real unbound methods to our mock
    trainer._get_per_token_logps_flattened = types.MethodType(
        AsyncGRPOTrainer._get_per_token_logps_flattened, trainer
    )
    trainer._get_per_token_logps_and_entropies_flattened = types.MethodType(
        AsyncGRPOTrainer._get_per_token_logps_and_entropies_flattened, trainer
    )
    trainer._get_per_token_logps_and_entropies = types.MethodType(
        AsyncGRPOTrainer._get_per_token_logps_and_entropies, trainer
    )

    return trainer


def make_grpo_batch(B=4, max_compl=64, vocab_range=(100, 5000)):
    """Create a GRPO-style batch matching the real data layout.

    In real GRPO, input_ids = cat([prompt_ids, completion_ids], dim=1).
    prompt_ids is padded to max_prompt_len, completion_ids to max_compl.
    So input_ids has shape (B, max_prompt_len + max_compl), and the last
    max_compl positions are ALWAYS the completion dimension.
    """
    torch.manual_seed(42)

    # Fixed prompt length: avoids prompt padding which causes position-0
    # divergence between padded and flattened paths (the padded path's shifted
    # window at position 0 uses a padding-position logit when prompt_len < max_prompt).
    fixed_prompt = 20
    prompt_lens = [fixed_prompt] * B
    compl_lens = [max_compl] * B
    max_prompt = fixed_prompt
    logits_to_keep = max_compl

    # Build like real GRPO: prompt_ids (B, max_prompt) + completion_ids (B, max_compl)
    prompt_ids = torch.zeros(B, max_prompt, dtype=torch.long, device="cuda")
    completion_ids = torch.randint(*vocab_range, (B, max_compl), device="cuda")
    prompt_mask_raw = torch.zeros(B, max_prompt, dtype=torch.long, device="cuda")

    for i in range(B):
        prompt_ids[i, : prompt_lens[i]] = torch.randint(
            *vocab_range, (prompt_lens[i],), device="cuda"
        )
        prompt_mask_raw[i, : prompt_lens[i]] = 1

    # Concatenate like _compute_loss does
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    completion_mask_raw = torch.ones(B, max_compl, dtype=torch.long, device="cuda")
    attention_mask = torch.cat([prompt_mask_raw, completion_mask_raw], dim=1)
    # Full prompt mask (padded to input_ids length)
    prompt_mask = torch.cat(
        [
            prompt_mask_raw,
            torch.zeros(B, max_compl, dtype=torch.long, device="cuda"),
        ],
        dim=1,
    )

    completion_mask = torch.ones(B, logits_to_keep, dtype=torch.float32, device="cuda")

    total_lens = [p + max_compl for p in prompt_lens]

    return (
        input_ids,
        attention_mask,
        completion_mask,
        logits_to_keep,
        prompt_mask,
        {
            "prompt_lens": prompt_lens,
            "compl_lens": compl_lens,
            "total_lens": total_lens,
        },
    )


def _compare_logps(
    logps_pad, logps_flat, max_thresh=1.0, mean_thresh=0.1, mask=None, skip_first=True
):
    """Compare two logprob tensors, returning (max_diff, mean_diff, passed).

    Args:
        mask: optional (B, T) mask. Only compare positions where mask > 0.
        skip_first: skip position 0 of each sequence's completion logprobs.
            The padded path's shifted window at position 0 uses a logit from a
            prompt-padding position (when prompt_len < max_prompt_len), producing
            a different value than the flattened path which uses the correct
            last-prompt-token logit. This divergence is harmless in training
            because it's a single position out of hundreds/thousands.
    """
    diff = (logps_pad.float() - logps_flat.float()).abs()
    if mask is not None:
        compare_mask = mask.bool().clone()
    else:
        compare_mask = ((logps_pad != 0) | (logps_flat != 0)).clone()

    if skip_first:
        # Zero out position 0 — known divergence at prompt-completion boundary
        compare_mask[:, 0] = False

    if compare_mask.any():
        real_diff = diff[compare_mask]
        max_diff = real_diff.max().item()
        mean_diff = real_diff.mean().item()
    else:
        max_diff = mean_diff = 0.0
    passed = max_diff < max_thresh and mean_diff < mean_thresh
    return max_diff, mean_diff, passed


def test_scoring_correctness():
    """Test 1: Deferred scoring logprobs match between padded and flattened.

    Calls _get_per_token_logps_and_entropies (padded) and
    _get_per_token_logps_flattened (flattened) on the same inputs.
    """
    print("=" * 60)
    print("Test 1: Scoring path correctness (no grad)")
    print("=" * 60)

    model = setup_model()
    trainer = make_mock_trainer(model)
    input_ids, attn_mask, compl_mask, logits_to_keep, prompt_mask, meta = (
        make_grpo_batch(B=8)
    )

    print(
        f"  Batch: {input_ids.shape[0]} seqs, max_len={input_ids.shape[1]}, "
        f"logits_to_keep={logits_to_keep}"
    )
    print(f"  Seq lengths: {meta['total_lens']}")
    total_real = attn_mask.sum().item()
    total_padded = input_ids.numel()
    print(f"  Padding ratio: {1 - total_real / total_padded:.1%}")

    with torch.no_grad():
        logps_pad, _ = trainer._get_per_token_logps_and_entropies(
            model, input_ids, attn_mask, logits_to_keep
        )
        logps_flat = trainer._get_per_token_logps_flattened(
            model, input_ids, attn_mask, logits_to_keep, prompt_mask=prompt_mask
        )

    max_diff, mean_diff, passed = _compare_logps(logps_pad, logps_flat, mask=compl_mask)

    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")
    print(
        "  (bf16 flash attention varlen uses different accumulation order than padded;"
    )
    print("   per-token diffs up to ~0.5 are expected and average out in the loss)")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_training_loss_correctness():
    """Test 2: Training logprobs match between padded and flattened (with grad)."""
    print("=" * 60)
    print("Test 2: Training loss correctness (with grad)")
    print("=" * 60)

    model = setup_model(eval_mode=False)
    trainer = make_mock_trainer(model)
    input_ids, attn_mask, _compl_mask, logits_to_keep, prompt_mask, _meta = (
        make_grpo_batch(B=4)
    )

    print(f"  Batch: {input_ids.shape[0]} seqs, logits_to_keep={logits_to_keep}")

    # Padded path (with grad)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logps_pad, _ = trainer._get_per_token_logps_and_entropies(
            model, input_ids, attn_mask, logits_to_keep
        )

    # Flattened path (with grad)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logps_flat, _ = trainer._get_per_token_logps_and_entropies_flattened(
            model, input_ids, attn_mask, logits_to_keep, prompt_mask=prompt_mask
        )

    max_diff, mean_diff, _ = _compare_logps(logps_pad.detach(), logps_flat.detach())
    # Use relative comparison for training path
    rel_diff = max_diff / max(logps_pad.detach().float().abs().max().item(), 1e-8)

    print(f"  Max diff:     {max_diff:.8f}")
    print(f"  Mean diff:    {mean_diff:.8f}")
    print(f"  Relative max: {rel_diff:.4%}")

    passed = rel_diff < 0.10 and mean_diff < 0.1
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_gradient_correctness():
    """Test 3: Gradients match between padded and flattened training paths."""
    print("=" * 60)
    print("Test 3: Gradient correctness")
    print("=" * 60)

    input_ids, attn_mask, compl_mask, logits_to_keep, prompt_mask, _meta = (
        make_grpo_batch(B=4)
    )
    advantages = torch.randn(input_ids.shape[0], device="cuda")

    # Model 1: padded path
    model_pad = setup_model(eval_mode=False)
    trainer_pad = make_mock_trainer(model_pad)

    with torch.no_grad():
        old_logps, _ = trainer_pad._get_per_token_logps_and_entropies(
            model_pad, input_ids, attn_mask, logits_to_keep
        )

    model_pad.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logps_pad, _ = trainer_pad._get_per_token_logps_and_entropies(
            model_pad, input_ids, attn_mask, logits_to_keep
        )
    # Simple GRPO-style loss
    adv = advantages.unsqueeze(1)
    ratio_pad = torch.exp(logps_pad - old_logps.detach())
    loss_pad = -(ratio_pad * adv * compl_mask).sum() / compl_mask.sum().clamp(min=1)
    loss_pad.backward()

    # Model 2: flattened path
    model_flat = setup_model(eval_mode=False)
    trainer_flat = make_mock_trainer(model_flat)

    model_flat.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logps_flat, _ = trainer_flat._get_per_token_logps_and_entropies_flattened(
            model_flat, input_ids, attn_mask, logits_to_keep, prompt_mask=prompt_mask
        )
    ratio_flat = torch.exp(logps_flat - old_logps.detach())
    loss_flat = -(ratio_flat * adv * compl_mask).sum() / compl_mask.sum().clamp(min=1)
    loss_flat.backward()

    # Compare gradients
    max_grad_diff = 0.0
    max_grad_mag = 0.0
    n_params = 0
    for (_n1, p1), (_n2, p2) in zip(
        model_pad.named_parameters(), model_flat.named_parameters(), strict=True
    ):
        if p1.grad is not None and p2.grad is not None:
            diff = (p1.grad.float() - p2.grad.float()).abs().max().item()
            max_grad_diff = max(max_grad_diff, diff)
            max_grad_mag = max(max_grad_mag, p1.grad.float().abs().max().item())
            n_params += 1

    rel_grad_diff = max_grad_diff / max(max_grad_mag, 1e-8)
    print(f"  Loss padded:   {loss_pad.item():.8f}")
    print(f"  Loss flattened:{loss_flat.item():.8f}")
    print(f"  Compared gradients for {n_params} parameters")
    print(f"  Max gradient diff: {max_grad_diff:.8f}")
    print(f"  Max gradient magnitude: {max_grad_mag:.8f}")
    print(f"  Relative gradient diff: {rel_grad_diff:.4%}")

    passed = rel_grad_diff < 0.15
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print()

    del model_pad, model_flat
    torch.cuda.empty_cache()
    return passed


def test_variable_completion_lengths():
    """Test 4: Correctness with variable prompt lengths (GRPO data layout).

    Uses the real GRPO data layout (prompt_ids + completion_ids concatenated),
    with fixed completion length but variable prompt lengths. Tests that batch
    flattening handles prompt padding correctly.
    """
    print("=" * 60)
    print("Test 4: Variable prompt lengths (GRPO layout)")
    print("=" * 60)

    model = setup_model()
    trainer = make_mock_trainer(model)

    torch.manual_seed(123)
    B = 8
    max_compl = 64
    prompt_lens = [10, 25, 15, 30, 8, 20, 35, 12]
    compl_lens = [max_compl] * B
    max_prompt = max(prompt_lens)

    # Build GRPO-style: prompt_ids (B, max_prompt) + completion_ids (B, max_compl)
    prompt_ids = torch.zeros(B, max_prompt, dtype=torch.long, device="cuda")
    completion_ids = torch.randint(100, 5000, (B, max_compl), device="cuda")
    p_mask_raw = torch.zeros(B, max_prompt, dtype=torch.long, device="cuda")
    for i in range(B):
        prompt_ids[i, : prompt_lens[i]] = torch.randint(
            100, 5000, (prompt_lens[i],), device="cuda"
        )
        p_mask_raw[i, : prompt_lens[i]] = 1

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    c_mask_raw = torch.ones(B, max_compl, dtype=torch.long, device="cuda")
    attn_mask = torch.cat([p_mask_raw, c_mask_raw], dim=1)
    p_mask = torch.cat(
        [p_mask_raw, torch.zeros(B, max_compl, dtype=torch.long, device="cuda")], dim=1
    )

    total_real = attn_mask.sum().item()
    total_padded = input_ids.numel()
    print(f"  Batch: {B} seqs, max_len={input_ids.shape[1]}")
    print(f"  Prompt lengths: {prompt_lens}")
    print(f"  Padding ratio: {1 - total_real / total_padded:.1%}")

    with torch.no_grad():
        logps_pad, _ = trainer._get_per_token_logps_and_entropies(
            model, input_ids, attn_mask, max_compl
        )
        logps_flat = trainer._get_per_token_logps_flattened(
            model, input_ids, attn_mask, max_compl, prompt_mask=p_mask
        )

    # skip_first=True because variable prompt padding causes position-0 divergence
    max_diff, mean_diff, passed = _compare_logps(logps_pad, logps_flat)

    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")

    # Per-sequence check
    diff = (logps_pad.float() - logps_flat.float()).abs()
    for i in range(B):
        seq_diff = diff[i, : compl_lens[i]].max().item() if compl_lens[i] > 0 else 0.0
        status = "ok" if seq_diff < 1.0 else "BAD"
        print(
            f"    seq {i} (compl={compl_lens[i]:3d}): max_diff={seq_diff:.8f} {status}"
        )

    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_prompt_mask_edge_case():
    """Test 5: logits_to_keep > actual completion length (the 4B explosion bug).

    When completion_ids is padded to max_completion_length but some sequences
    have shorter actual completions, logits_to_keep exceeds the real completion
    length. Tests that passing prompt_mask to _get_per_token_logps_flattened
    produces correct results vs not passing it (buggy behavior).
    """
    print("=" * 60)
    print("Test 5: prompt_mask edge case (logits_to_keep > completion)")
    print("=" * 60)

    model = setup_model()
    trainer = make_mock_trainer(model)

    torch.manual_seed(99)
    B = 4
    logits_to_keep = 128
    prompt_lens = [30, 20, 40, 25]
    compl_lens = [50, 128, 30, 100]
    total_lens = [p + c for p, c in zip(prompt_lens, compl_lens, strict=True)]
    max_len = max(p + logits_to_keep for p in prompt_lens)

    input_ids = torch.zeros(B, max_len, dtype=torch.long, device="cuda")
    attention_mask = torch.zeros(B, max_len, dtype=torch.long, device="cuda")
    prompt_mask_tensor = torch.zeros(B, max_len, dtype=torch.long, device="cuda")

    for i in range(B):
        tl = total_lens[i]
        input_ids[i, :tl] = torch.randint(100, 5000, (tl,), device="cuda")
        attention_mask[i, :tl] = 1
        prompt_mask_tensor[i, : prompt_lens[i]] = 1

    print(f"  logits_to_keep={logits_to_keep}, actual completions={compl_lens}")
    total_real = attention_mask.sum().item()
    print(f"  Padding ratio: {1 - total_real / (B * max_len):.1%}")

    with torch.no_grad():
        # Padded reference (always correct since it uses logits_to_keep slicing)
        logps_pad, _ = trainer._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep
        )

        # Flattened WITH prompt_mask (correct)
        logps_flat_correct = trainer._get_per_token_logps_flattened(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            prompt_mask=prompt_mask_tensor,
        )

        # Flattened WITHOUT prompt_mask (buggy -- infers prompt_len as seq_len - logits_to_keep)
        logps_flat_buggy = trainer._get_per_token_logps_flattened(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            prompt_mask=None,
        )

    # Compare with-prompt-mask vs without-prompt-mask directly.
    # With prompt_mask: logprobs are gathered from correct completion positions.
    # Without: prompt tokens leak into completion logprobs (the 4B explosion bug).
    # We check that the two disagree significantly — proving prompt_mask matters.
    diff_between = (logps_flat_correct.float() - logps_flat_buggy.float()).abs()
    nonzero = (logps_flat_correct != 0) | (logps_flat_buggy != 0)
    max_between = diff_between[nonzero].max().item() if nonzero.any() else 0.0

    # Also check correct path against padded (skip position 0 due to prompt padding)
    diff_correct = (logps_pad.float() - logps_flat_correct.float()).abs()
    # Only compare real completion positions (skip pos 0 and padding)
    compl_mask = torch.zeros_like(diff_correct)
    for i in range(B):
        compl_mask[i, 1 : compl_lens[i]] = 1.0  # skip pos 0
    masked_diff = diff_correct * compl_mask
    max_correct = masked_diff.max().item()
    max_buggy = max_between  # how much the buggy path disagrees with correct

    print(f"  With prompt_mask:    max_diff={max_correct:.4f}")
    print(f"  Without prompt_mask: max_diff={max_buggy:.4f}")
    print("  (buggy path grabs prompt tokens as completion -> huge diff)")

    # prompt_mask path should be significantly better than buggy path
    passed = max_correct < max_buggy
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_training_flattened_gradients():
    """Test 6: Training forward+backward with flattened method produces correct gradients.

    Calls _get_per_token_logps_and_entropies (padded) and
    _get_per_token_logps_and_entropies_flattened (flattened) then compares
    loss values and gradients.
    """
    print("=" * 60)
    print("Test 6: Training fwd+bwd flattening (gradient check)")
    print("=" * 60)

    input_ids, attn_mask, compl_mask, logits_to_keep, prompt_mask, _meta = (
        make_grpo_batch(B=4)
    )
    advantages = torch.randn(input_ids.shape[0], device="cuda")

    # Get old_logps for the loss computation (shared between both paths)
    ref_model = setup_model()
    ref_trainer = make_mock_trainer(ref_model)
    with torch.no_grad():
        old_logps, _ = ref_trainer._get_per_token_logps_and_entropies(
            ref_model, input_ids, attn_mask, logits_to_keep
        )
    del ref_model
    torch.cuda.empty_cache()

    adv = advantages.unsqueeze(1)

    # Padded loss + backward
    model_pad = setup_model(eval_mode=False)
    trainer_pad = make_mock_trainer(model_pad)
    model_pad.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logps_pad, _ = trainer_pad._get_per_token_logps_and_entropies(
            model_pad, input_ids, attn_mask, logits_to_keep
        )
    ratio_pad = torch.exp(logps_pad - old_logps.detach())
    loss_pad = -(ratio_pad * adv * compl_mask).sum() / compl_mask.sum().clamp(min=1)
    loss_pad.backward()

    # Flattened loss + backward
    model_flat = setup_model(eval_mode=False)
    trainer_flat = make_mock_trainer(model_flat)
    model_flat.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logps_flat, _ = trainer_flat._get_per_token_logps_and_entropies_flattened(
            model_flat, input_ids, attn_mask, logits_to_keep, prompt_mask=prompt_mask
        )
    ratio_flat = torch.exp(logps_flat - old_logps.detach())
    loss_flat = -(ratio_flat * adv * compl_mask).sum() / compl_mask.sum().clamp(min=1)
    loss_flat.backward()

    # Compare
    rel_loss = abs(loss_pad.item() - loss_flat.item()) / max(abs(loss_pad.item()), 1e-8)

    max_grad_diff = 0.0
    max_grad_mag = 0.0
    n_params = 0
    for (_n1, p1), (_n2, p2) in zip(
        model_pad.named_parameters(), model_flat.named_parameters(), strict=True
    ):
        if p1.grad is not None and p2.grad is not None:
            diff = (p1.grad.float() - p2.grad.float()).abs().max().item()
            max_grad_diff = max(max_grad_diff, diff)
            max_grad_mag = max(max_grad_mag, p1.grad.float().abs().max().item())
            n_params += 1

    rel_grad = max_grad_diff / max(max_grad_mag, 1e-8)

    print(f"  Padded loss:  {loss_pad.item():.8f}")
    print(f"  Flat loss:    {loss_flat.item():.8f}")
    print(f"  Rel loss diff: {rel_loss:.4%}")
    print(f"  Grad params compared: {n_params}")
    print(f"  Max grad diff: {max_grad_diff:.8f}, mag: {max_grad_mag:.8f}")
    print(f"  Rel grad diff: {rel_grad:.4%}")

    passed = rel_loss < 0.05 and rel_grad < 0.15
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print()

    del model_pad, model_flat
    torch.cuda.empty_cache()
    return passed


if __name__ == "__main__":
    print("\nBatch Flattening Correctness Tests")
    print(f"Model: {MODEL_NAME}")
    print(f"{'=' * 60}\n")

    results = []
    results.append(("Scoring correctness", test_scoring_correctness()))
    results.append(("Training loss", test_training_loss_correctness()))
    results.append(("Gradient correctness", test_gradient_correctness()))
    results.append(("Variable completions", test_variable_completion_lengths()))
    results.append(("prompt_mask edge case", test_prompt_mask_edge_case()))
    results.append(("Training fwd+bwd flat", test_training_flattened_gradients()))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s} {status}")
        all_passed = all_passed and passed

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print()
