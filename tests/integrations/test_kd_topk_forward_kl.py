"""Tests for the top-k forward-KL KD loss"""

import pytest
import torch

from axolotl.integrations.kd.topk_logprob.forward_kl import ChunkedTopKKDLoss, loss


def reference_loss(
    student_logits,
    target_token_ids,
    target_logprobs,
    target_mask,
    num_items_in_batch=-1,
    kd_temperature=1.0,
):
    """Naive reference using a full log_softmax instead of the
    gather + logsumexp formulation."""
    teacher_seq_len = target_token_ids.shape[1]
    student = (student_logits[:, :teacher_seq_len, :] / kd_temperature).float()
    student_logprobs = torch.log_softmax(student, dim=-1)
    student_logprobs_topk = torch.gather(
        student_logprobs, dim=-1, index=target_token_ids
    )

    valid = target_mask.bool()
    teacher_logprobs = target_logprobs.float()[valid]
    student_logprobs_topk = student_logprobs_topk[valid]

    per_token = teacher_logprobs.exp() * (teacher_logprobs - student_logprobs_topk)
    total = per_token.sum()
    denom = num_items_in_batch if num_items_in_batch > 0 else per_token.size(0)
    return total / denom


def make_inputs(
    batch_size=2,
    student_seq_len=48,
    teacher_seq_len=40,
    top_k=8,
    vocab_size=256,
    dtype=torch.float32,
    seed=0,
):
    torch.manual_seed(seed)
    student_logits = torch.randn(batch_size, student_seq_len, vocab_size, dtype=dtype)
    target_token_ids = torch.randint(
        0, vocab_size, (batch_size, teacher_seq_len, top_k)
    )
    teacher_logits = torch.randn(batch_size, teacher_seq_len, top_k)
    target_logprobs = torch.log_softmax(teacher_logits, dim=-1)
    target_mask = (torch.rand(batch_size, teacher_seq_len, top_k) < 0.8).to(torch.int64)
    target_mask[:, 0, :] = 1  # guarantee at least one valid token
    return student_logits, target_token_ids, target_logprobs, target_mask


@pytest.mark.parametrize("num_items_in_batch", [-1, 7])
def test_loss_matches_reference(num_items_in_batch):
    inputs = make_inputs()

    actual = loss(*inputs, num_items_in_batch=num_items_in_batch)
    expected = reference_loss(*inputs, num_items_in_batch=num_items_in_batch)

    torch.testing.assert_close(actual, expected)


def test_loss_temperature():
    inputs = make_inputs()

    actual = loss(*inputs, kd_temperature=2.0)
    expected = reference_loss(*inputs, kd_temperature=2.0)

    torch.testing.assert_close(actual, expected)
    # temperature must actually change the result
    assert not torch.allclose(actual, loss(*inputs))


def test_masked_entries_do_not_affect_loss():
    student_logits, target_token_ids, target_logprobs, target_mask = make_inputs()
    baseline = loss(student_logits, target_token_ids, target_logprobs, target_mask)

    corrupted_logprobs = target_logprobs.clone()
    corrupted_logprobs[target_mask == 0] = 123.0

    perturbed = loss(student_logits, target_token_ids, corrupted_logprobs, target_mask)
    torch.testing.assert_close(perturbed, baseline)


def test_grad_flow_and_sparsity():
    student_logits, target_token_ids, target_logprobs, target_mask = make_inputs()
    target_mask[:, -3:, :] = 0  # fully mask some teacher positions
    student_logits.requires_grad_(True)

    out = loss(student_logits, target_token_ids, target_logprobs, target_mask)
    out.backward()

    grad = student_logits.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    teacher_seq_len = target_token_ids.shape[1]
    # positions beyond the teacher sequence never enter the loss
    assert (grad[:, teacher_seq_len:, :] == 0).all()
    # fully masked positions get no gradient
    assert (grad[:, teacher_seq_len - 3 : teacher_seq_len, :] == 0).all()
    # unmasked positions do
    assert grad[:, 0, :].abs().sum() > 0


def test_bf16_student_logits():
    inputs = make_inputs()
    expected = loss(*inputs)

    bf16_inputs = (inputs[0].to(torch.bfloat16), *inputs[1:])
    actual = loss(*bf16_inputs)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1e-3)


@pytest.mark.parametrize("num_items_in_batch", [-1, 11])
def test_chunked_matches_unchunked(num_items_in_batch):
    inputs = make_inputs(teacher_seq_len=40, student_seq_len=40)
    unchunked = loss(*inputs, num_items_in_batch=num_items_in_batch)

    chunked_loss = ChunkedTopKKDLoss(num_output_chunks=4)
    chunked = chunked_loss(*inputs, num_items_in_batch=num_items_in_batch)

    torch.testing.assert_close(chunked, unchunked, rtol=1e-5, atol=1e-6)
