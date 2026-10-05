"""
Unit tests for LigerFusedLinearKLTopKLogprobFunction autograd contract.

Regression tests for the has_aux=True contract violation in
LigerFusedLinearKLTopKLogprobFunction: CE loss must contribute to the
backward graph, not be silently dropped as aux.
"""

import torch

from axolotl.integrations.kd.kernels.liger import LigerFusedLinearKLTopKLogprobLoss

BATCH_SIZE = 1
SEQ_LEN = 4
VOCAB_SIZE = 16
HIDDEN_DIM = 8
TOP_K = 4


def make_inputs(seed: int):
    torch.manual_seed(seed)
    student_hidden_states = torch.randn(
        BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, dtype=torch.float32, requires_grad=True
    )
    lm_head_weight = torch.randn(
        VOCAB_SIZE, HIDDEN_DIM, dtype=torch.float32, requires_grad=True
    )
    target_token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN, TOP_K))
    target_logprobs = torch.log_softmax(torch.randn(BATCH_SIZE, SEQ_LEN, TOP_K), dim=-1)
    target_mask = torch.ones(BATCH_SIZE, SEQ_LEN, TOP_K, dtype=torch.bool)
    true_labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    # Set an interior position to ignore_index. After the left-shift inside the
    # kernel (true_labels = pad(...); true_labels[:, 1:]), index [0, 2] in the
    # input becomes index 1 in the flattened CE labels, so this exercises
    # ignore_index handling in F.cross_entropy.
    true_labels[0, 2] = -100
    return {
        "student_hidden_states": student_hidden_states,
        "lm_head_weight": lm_head_weight,
        "target_token_ids": target_token_ids,
        "target_logprobs": target_logprobs,
        "target_mask": target_mask,
        "true_labels": true_labels,
    }


def test_ce_only_gradient_flows_to_student_hidden_states():
    """Regression: with weight_soft_loss=0, weight_hard_loss=1 the CE gradient must reach inputs.

    Pre-fix, has_aux=True dropped CE from the backward graph and student_input.grad was all zero.
    """
    inputs = make_inputs(seed=0)
    loss_fn = LigerFusedLinearKLTopKLogprobLoss(
        weight_hard_loss=1.0,
        weight_soft_loss=0.0,
        temperature=1.0,
        beta=0.0,
        compiled=False,
        chunk_size=2,
        compute_ce_loss=True,
    )
    loss = loss_fn(
        inputs["lm_head_weight"],
        inputs["student_hidden_states"],
        inputs["target_token_ids"],
        inputs["target_logprobs"],
        inputs["target_mask"],
        inputs["true_labels"],
    )
    loss.backward()

    assert inputs["student_hidden_states"].grad is not None
    assert torch.isfinite(inputs["student_hidden_states"].grad).all()
    assert inputs["student_hidden_states"].grad.abs().sum().item() > 0

    assert inputs["lm_head_weight"].grad is not None
    assert torch.isfinite(inputs["lm_head_weight"].grad).all()
    assert inputs["lm_head_weight"].grad.abs().sum().item() > 0


def test_kd_mix_gradient_changes_when_ce_weight_changes():
    """Regression: in KD-mix mode, increasing weight_hard_loss must change the gradient.

    Two runs with identical RNG, differing only in weight_hard_loss (0.0 vs 0.5). Pre-fix
    both runs produced identical grads because CE was silently dropped from backward.
    """
    inputs_a = make_inputs(seed=42)
    loss_fn_a = LigerFusedLinearKLTopKLogprobLoss(
        weight_soft_loss=0.5,
        weight_hard_loss=0.0,
        temperature=1.0,
        beta=0.0,
        compiled=False,
        chunk_size=2,
        compute_ce_loss=True,
    )
    loss_a = loss_fn_a(
        inputs_a["lm_head_weight"],
        inputs_a["student_hidden_states"],
        inputs_a["target_token_ids"],
        inputs_a["target_logprobs"],
        inputs_a["target_mask"],
        inputs_a["true_labels"],
    )
    loss_a.backward()
    grad_a_h = inputs_a["student_hidden_states"].grad.detach().clone()
    grad_a_w = inputs_a["lm_head_weight"].grad.detach().clone()

    inputs_b = make_inputs(seed=42)
    loss_fn_b = LigerFusedLinearKLTopKLogprobLoss(
        weight_soft_loss=0.5,
        weight_hard_loss=0.5,
        temperature=1.0,
        beta=0.0,
        compiled=False,
        chunk_size=2,
        compute_ce_loss=True,
    )
    loss_b = loss_fn_b(
        inputs_b["lm_head_weight"],
        inputs_b["student_hidden_states"],
        inputs_b["target_token_ids"],
        inputs_b["target_logprobs"],
        inputs_b["target_mask"],
        inputs_b["true_labels"],
    )
    loss_b.backward()
    grad_b_h = inputs_b["student_hidden_states"].grad.detach().clone()
    grad_b_w = inputs_b["lm_head_weight"].grad.detach().clone()

    assert grad_a_h is not None
    assert torch.isfinite(grad_a_h).all()
    assert grad_b_h is not None
    assert torch.isfinite(grad_b_h).all()

    assert grad_a_w is not None
    assert torch.isfinite(grad_a_w).all()
    assert grad_b_w is not None
    assert torch.isfinite(grad_b_w).all()

    diff = (grad_b_h - grad_a_h).abs().sum().item()
    assert not torch.allclose(grad_a_h, grad_b_h, atol=1e-6, rtol=1e-5)
    assert diff > 1e-4, f"CE gradient contribution suspiciously small: {diff}"

    diff_w = (grad_b_w - grad_a_w).abs().sum().item()
    assert not torch.allclose(grad_a_w, grad_b_w, atol=1e-6, rtol=1e-5)
    assert diff_w > 1e-4, (
        f"CE weight-gradient contribution suspiciously small: {diff_w}"
    )
