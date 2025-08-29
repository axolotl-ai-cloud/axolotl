"""
Liger Kernels for Chunked Top-K Log-Prob Distillation
"""

import torch
import torch.nn.functional as F
from liger_kernel.chunked_loss.fused_linear_distillation import (
    LigerFusedLinearDistillationBase,
)

from axolotl.integrations.kd.utils import normalize_logprobs


class LigerFusedLinearKLTopKLogprobFunction(LigerFusedLinearDistillationBase):
    """
    Chunked kl-div loss for top-k logprobs
    """

    @staticmethod
    def distillation_loss_fn(
        student_logits_temp_scaled: torch.Tensor,  # [chunk_size, vocab_size], already temp-scaled
        target_token_ids_chunk: torch.Tensor,  # [chunk_size, top_k]
        target_logprobs_chunk: torch.Tensor,  # [chunk_size, top_k], already temp-scaled and normalized logprobs
        target_mask_chunk: torch.Tensor,  # [chunk_size, top_k]
        beta: float = 0.0,
        normalize_topk: bool = True,
    ) -> torch.Tensor:
        """
        Compute Top-K KL divergence loss for a chunk.
        Args:
            student_logits_temp_scaled: Student logits, scaled by temperature. Shape: (N, V).
            target_token_ids_chunk: Top-k teacher token IDs. Shape: (N, K).
            target_logprobs_chunk: Top-k teacher log probabilities (temp-scaled, normalized). Shape: (N, K).
            target_mask_chunk: Mask for valid top-k tokens. Shape: (N, K).
            beta: Controls the type of KL divergence.
                  0.0 for Forward KL (P_teacher || P_student).
                  1.0 for Reverse KL (P_student || P_teacher).
                  0.5 for Symmetric KL (average of Forward and Reverse).
            normalize_topk: Whether to normalize the log probabilities
        Returns:
            Sum of KL divergence losses for the chunk.
        """
        topk = target_token_ids_chunk.shape[-1]
        student_logits_temp_scaled = (  # [chunk_size, vocab_size]
            student_logits_temp_scaled.float()
        )
        target_logprobs_chunk = target_logprobs_chunk.float()

        # Gather student logits for the top-k teacher token IDs
        # target_token_ids_chunk: [chunk_size, top_k]
        # student_logits_topk_temp_scaled: [chunk_size, top_k]
        student_logits_topk_temp_scaled = torch.gather(
            student_logits_temp_scaled, dim=-1, index=target_token_ids_chunk
        )

        # Student log-probabilities for the gathered top-k tokens
        student_lse = torch.logsumexp(
            student_logits_temp_scaled, dim=-1, keepdim=True
        )  # [chunk_size, 1]
        student_logprobs_topk_temp_scaled = (
            student_logits_topk_temp_scaled - student_lse
        )

        # we have the top-k student logprobs, normalize them
        if normalize_topk:
            student_logprobs_topk_temp_scaled = normalize_logprobs(
                student_logprobs_topk_temp_scaled, topk
            )

        valid_mask = target_mask_chunk.to(torch.bool)  # [chunk_size, top_k]

        student_logprobs_topk_valid = student_logprobs_topk_temp_scaled[valid_mask]
        teacher_logprobs_valid = target_logprobs_chunk[valid_mask]

        # Teacher probabilities P(y|x_teacher) from logprobs
        # target_logprobs_valid are already normalized (log(softmax(teacher_logits/T)))
        teacher_probs_valid = teacher_logprobs_valid.exp()
        # Student probabilities P_student from log P_student
        student_probs_topk_valid = student_logprobs_topk_valid.exp()

        # kd_loss_per_token = torch.zeros_like(target_logprobs_valid)

        # KL divergence: sum(P_teacher * (log P_teacher - log P_student))
        # = sum(P_teacher * log P_teacher) - sum(P_teacher * log P_student)
        # The distillation loss is often formulated as -sum(P_teacher * log P_student)
        # or as sum(P_teacher * (log_softmax_teacher - log_softmax_student))
        # Here, target_logprobs_valid are log_softmax_teacher.
        # student_logprobs_topk_valid are log_softmax_student (for the selected K indices).
        if beta == 0.0:  # Contribution from Forward KL
            fwd_kl_per_token = teacher_probs_valid * (
                teacher_logprobs_valid - student_logprobs_topk_valid
            )
            kd_loss = fwd_kl_per_token.sum()
        elif beta == 1.0:  # Contribution from Reverse KL
            rev_kl_per_token = student_probs_topk_valid * (
                student_logprobs_topk_valid - teacher_logprobs_valid
            )
            kd_loss = rev_kl_per_token.sum()
        else:
            # JSD - Jensen-Shannon Divergence / Symmetric
            mean_probs = (
                1 - beta
            ) * student_probs_topk_valid + beta * teacher_probs_valid
            log_mean_probs = mean_probs.log()
            student_kl = F.kl_div(
                log_mean_probs,
                student_logprobs_topk_valid,
                reduction="sum",
                log_target=True,
            )
            teacher_kl = F.kl_div(
                log_mean_probs, teacher_logprobs_valid, reduction="sum", log_target=True
            )
            jsd_loss = beta * teacher_kl + (1 - beta) * student_kl
            kd_loss = jsd_loss

        return kd_loss

    @staticmethod
    def _compute_loss_kl_topk(
        student_input_chunk: torch.Tensor,
        student_weight: torch.Tensor,
        # Args for student_bias, target_token_ids_chunk etc. are passed to the lambda wrapped by grad_and_value
        # or through `partial`. Let's make them explicit here for clarity.
        target_token_ids_chunk: torch.Tensor,
        target_logprobs_chunk: torch.Tensor,
        target_mask_chunk: torch.Tensor,
        target_chunk: torch.Tensor,  # For hard loss (true labels)
        student_bias: torch.Tensor = None,  # This will be one of the grad targets
        # Other params passed via `partial` from `forward`
        distillation_loss_fn=None,
        ignore_index: int = -100,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        compute_ce_loss: bool = True,
        temperature: float = 1.0,
        beta: float = 0.0,
        normalize_topk: bool = True,
    ):
        # Compute student logits for the chunk from hidden states and LM head
        # student_input_chunk: [chunk_size, hidden_dim]
        # student_lm_head_weight: [vocab_size, hidden_dim]
        # student_logits_chunk: [chunk_size, vocab_size]
        student_logits_chunk = F.linear(
            student_input_chunk, student_weight, student_bias
        )

        ce_loss = torch.tensor(
            0.0, device=student_logits_chunk.device, dtype=student_logits_chunk.dtype
        )
        if compute_ce_loss and weight_hard_loss > 0.0:
            ce_loss = F.cross_entropy(
                student_logits_chunk.view(-1, student_logits_chunk.shape[-1]),
                target_chunk.view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )

        soft_loss = torch.tensor(
            0.0, device=student_logits_chunk.device, dtype=student_logits_chunk.dtype
        )
        if weight_soft_loss > 0.0:
            student_logits_chunk_temp_scaled = student_logits_chunk / temperature

            # Assuming student_weight.shape[0] (vocab_size) is adequate for target_token_ids_chunk.max()
            # No explicit padding here; user must ensure vocab alignment or pre-pad student_weight.

            soft_loss = distillation_loss_fn(
                student_logits_chunk_temp_scaled,
                target_token_ids_chunk,
                target_logprobs_chunk,
                target_mask_chunk,
                beta=beta,
                normalize_topk=normalize_topk,
            )

        return soft_loss, ce_loss

    @classmethod
    def forward(
        cls,
        ctx,
        student_input: torch.Tensor,  # [batch_size, seq_len, dim]
        student_lm_head_weight: torch.Tensor,  # [dim, vocab_size]
        target_token_ids: torch.Tensor,  # [batch_size, seq_len, top_k]
        target_logprobs: torch.Tensor,  # [batch_size, seq_len, top_k]
        target_mask: torch.Tensor,  # [batch_size, seq_len, top_k]
        true_labels: torch.Tensor,  # [batch_size, seq_len]
        student_lm_head_bias: torch.Tensor = None,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        beta: float = 0.0,
        compiled: bool = False,
        chunk_size: int = 1024,
        compute_ce_loss: bool = True,
        normalize_topk: bool = True,
    ):
        CHUNK_SIZE = chunk_size
        grad_weight_acc = torch.zeros_like(student_lm_head_weight)
        grad_inputs_list = []
        grad_bias_acc = (
            torch.zeros_like(student_lm_head_bias)
            if student_lm_head_bias is not None
            else None
        )
        kd_loss_acc = torch.zeros(
            (), device=student_input.device, dtype=student_input.dtype
        )
        ce_loss_acc = torch.zeros(
            (), device=student_input.device, dtype=student_input.dtype
        )

        # This function will be what torch.func.grad_and_value differentiates.
        # It takes student_input_chunk, student_weight (full), student_bias (full) as primals.
        # Other necessary data (target_*, etc.) are passed as non-differentiable arguments.
        def loss_fn_for_grad(
            _student_input_chunk,
            _student_lm_head_weight,  # full weight
            _student_lm_head_bias,  # full bias
            # Fixed arguments for a given chunk, not differentiated:
            _target_token_ids_chunk,
            _target_logprobs_chunk,
            _target_mask_chunk,
            _true_labels_chunk,
        ):
            return cls._compute_loss_kl_topk(
                student_input_chunk=_student_input_chunk,
                student_weight=_student_lm_head_weight,
                target_token_ids_chunk=_target_token_ids_chunk,
                target_logprobs_chunk=_target_logprobs_chunk,
                target_mask_chunk=_target_mask_chunk,
                target_chunk=_true_labels_chunk,
                student_bias=_student_lm_head_bias,
                distillation_loss_fn=cls.distillation_loss_fn,
                ignore_index=ignore_index,
                weight_hard_loss=weight_hard_loss,
                weight_soft_loss=weight_soft_loss,
                compute_ce_loss=compute_ce_loss,
                temperature=temperature,
                beta=beta,
                normalize_topk=normalize_topk,
            )

        def accumulate_chunk_grads(
            student_input_chunk_ac,
            target_token_ids_chunk_ac,
            target_logprobs_chunk_ac,
            target_mask_chunk_ac,
            true_labels_chunk_ac,
        ):
            # student_weight and student_bias are closed over from the outer scope (full tensors)
            if student_lm_head_bias is not None:
                (
                    (chunk_grad_input, chunk_grad_weight, chunk_grad_bias),
                    (chunk_kd_loss, chunk_ce_loss),
                ) = torch.func.grad_and_value(
                    loss_fn_for_grad, argnums=(0, 1, 2), has_aux=True
                )(
                    student_input_chunk_ac,
                    student_lm_head_weight,
                    student_lm_head_bias,  # primals
                    target_token_ids_chunk_ac,
                    target_logprobs_chunk_ac,
                    target_mask_chunk_ac,
                    true_labels_chunk_ac,
                )  # non-primals
                grad_bias_acc.add_(chunk_grad_bias)
            else:
                argnums_for_grad = (0, 1)  # Differentiate wrt input_chunk, weight
                (
                    (chunk_grad_input, chunk_grad_weight),  # No grad for bias
                    (chunk_kd_loss, chunk_ce_loss),
                ) = torch.func.grad_and_value(
                    loss_fn_for_grad, argnums=argnums_for_grad, has_aux=True
                )(
                    student_input_chunk_ac,
                    student_lm_head_weight,
                    None,  # Pass None for student_bias primal
                    target_token_ids_chunk_ac,
                    target_logprobs_chunk_ac,
                    target_mask_chunk_ac,
                    true_labels_chunk_ac,
                )

            grad_weight_acc.add_(chunk_grad_weight)
            kd_loss_acc.add_(chunk_kd_loss)
            ce_loss_acc.add_(chunk_ce_loss)

            return chunk_grad_input

        if compiled:
            accumulate_chunk_grads_compiled = torch.compile(
                accumulate_chunk_grads, dynamic=True, backend="inductor"
            )  # dynamic=True often helpful
        else:
            accumulate_chunk_grads_compiled = accumulate_chunk_grads

        # Use the same chunking logic as LigerFusedLinearDistillationBase.forward
        B, N, D = student_input.shape
        K = target_token_ids.shape[-1]

        student_input_flat = student_input.reshape(-1, student_input.shape[-1])
        target_token_ids_flat = target_token_ids.reshape(-1, target_token_ids.shape[-1])
        target_logprobs_flat = target_logprobs.reshape(-1, target_logprobs.shape[-1])
        target_mask_flat = target_mask.reshape(-1, target_mask.shape[-1])
        # pad and shift for cross entropy loss
        true_labels = torch.nn.functional.pad(true_labels, (0, 1), value=ignore_index)
        true_labels_flat = true_labels[:, 1:].contiguous().view(-1)

        num_chunks = max(1, student_input_flat.shape[0] // CHUNK_SIZE)

        _student_input_chunks = torch.chunk(
            student_input_flat, chunks=num_chunks, dim=0
        )
        _target_token_ids_chunks = torch.chunk(
            target_token_ids_flat, chunks=num_chunks, dim=0
        )
        _target_logprobs_chunks = torch.chunk(
            target_logprobs_flat, chunks=num_chunks, dim=0
        )
        _target_mask_chunks = torch.chunk(target_mask_flat, chunks=num_chunks, dim=0)
        _true_labels_chunks = torch.chunk(true_labels_flat, chunks=num_chunks, dim=0)

        for i in range(num_chunks):
            grad_input_chunk = accumulate_chunk_grads_compiled(
                _student_input_chunks[i],
                _target_token_ids_chunks[i],
                _target_logprobs_chunks[i],
                _target_mask_chunks[i],
                _true_labels_chunks[i],
            )
            grad_inputs_list.append(grad_input_chunk)

        grad_inputs_combined = torch.cat(grad_inputs_list, dim=0)
        ctx.save_for_backward(grad_inputs_combined, grad_weight_acc, grad_bias_acc)

        # For matching None returns in backward for non-tensor/non-grad_requiring inputs
        ctx.hyperparams_count = 9  # Corresponds to number of hyperparams after main tensors in fwd signature
        ctx.bias_was_none = student_lm_head_bias is None
        ctx.orig_dims = (B, N, D, K)

        # since this is packed, there is simply a single batch, so batchmean reduction of kl-div is simply the accumulated sum
        # we still need to scale the kd_loss by the temp^2
        kd_loss_acc = kd_loss_acc * (temperature**2)
        final_loss = weight_soft_loss * kd_loss_acc + weight_hard_loss * ce_loss_acc

        return final_loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_input_flat, grad_weight, grad_bias_maybe = (
            ctx.saved_tensors
        )  # grad_input_flat is (B*N, D)

        # Scale gradients by grad_output if it's not 1.0
        if not torch.equal(
            grad_output,
            torch.tensor(1.0, device=grad_output.device, dtype=grad_output.dtype),
        ):
            grad_input_flat = grad_input_flat * grad_output
            grad_weight = grad_weight * grad_output
            if grad_bias_maybe is not None:
                grad_bias_maybe = grad_bias_maybe * grad_output

        # Reshape grad_input_flat to match original student_input shape (B, N, D)
        # ctx.orig_dims stores (B, N, D, K)
        # We need the first three dimensions for student_input's shape.
        # Ensure that orig_dims are not (0,0,0,K) for empty inputs leading to view errors
        if (
            ctx.orig_dims[0] * ctx.orig_dims[1] * ctx.orig_dims[2] == 0
            and grad_input_flat.numel() == 0
        ):
            # If original input was empty, gradient should also be empty with correct shape
            grad_input_reshaped = torch.zeros(
                ctx.orig_dims[0],
                ctx.orig_dims[1],
                ctx.orig_dims[2],
                dtype=grad_input_flat.dtype,
                device=grad_input_flat.device,
            )
        elif grad_input_flat.numel() == 0 and not (
            ctx.orig_dims[0] * ctx.orig_dims[1] * ctx.orig_dims[2] == 0
        ):
            # This case should ideally not happen if forward path is correct (non-empty input -> non-empty flat grad)
            # but as a safeguard:
            grad_input_reshaped = torch.zeros(
                ctx.orig_dims[0],
                ctx.orig_dims[1],
                ctx.orig_dims[2],
                dtype=grad_input_flat.dtype,
                device=grad_input_flat.device,
            )
        else:
            grad_input_reshaped = grad_input_flat.view(
                ctx.orig_dims[0], ctx.orig_dims[1], ctx.orig_dims[2]
            )

        nones_for_hyperparams = [None] * ctx.hyperparams_count
        grad_bias_return = grad_bias_maybe if not ctx.bias_was_none else None

        return (
            grad_input_reshaped,  # Gradient for student_input (reshaped)
            grad_weight,  # Gradient for student_lm_head_weight
            None,  # Gradient for target_token_ids
            None,  # Gradient for target_logprobs
            None,  # Gradient for target_mask
            None,  # Gradient for true_labels
            grad_bias_return,  # Gradient for student_lm_head_bias
            *nones_for_hyperparams,  # Grads for weight_hard_loss, ..., compute_ce_loss
        )


class LigerFusedLinearKLTopKLogprobLoss(torch.nn.Module):
    """
    wrapper for chunked top-k logprob kl-d
    """

    def __init__(
        self,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        temperature: float = 1.0,  # This is the kd_temperature
        beta: float = 1.0,
        ignore_index: int = -100,
        compiled: bool = True,
        chunk_size: int = 1024,
        compute_ce_loss: bool = True,
        normalize_topk: bool = True,
    ):
        super().__init__()
        if not (0.0 <= weight_hard_loss <= 1.0 and 0.0 <= weight_soft_loss <= 1.0):
            raise ValueError("Loss weights must be between 0.0 and 1.0.")
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")

        self.weight_hard_loss = weight_hard_loss
        self.weight_soft_loss = weight_soft_loss
        self.temperature = temperature
        self.beta = beta
        self.ignore_index = ignore_index
        self.compiled = compiled
        self.chunk_size = chunk_size
        self.compute_ce_loss = compute_ce_loss
        self.normalize_topk = normalize_topk

        if not self.compute_ce_loss and self.weight_hard_loss > 0.0:
            print(
                f"Warning: compute_ce_loss is False, but weight_hard_loss ({self.weight_hard_loss}) > 0. Hard loss will effectively be zero."
            )
            # self.weight_hard_loss = 0.0 # Or let user manage this
        if self.weight_soft_loss == 0.0:
            print(
                "Warning: weight_soft_loss is 0.0. Soft (KD) loss will not be computed."
            )

    def forward(
        self,
        lm_head_weight: torch.Tensor,  # Weights of the linear layer in the LM head
        student_hidden_states: torch.Tensor,  # student_hidden_states before the lm_head
        target_token_ids: torch.Tensor,
        target_logprobs: torch.Tensor,
        target_mask: torch.Tensor,
        true_labels: torch.Tensor,
        student_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        return LigerFusedLinearKLTopKLogprobFunction.apply(
            student_hidden_states,
            lm_head_weight,
            target_token_ids,
            target_logprobs,
            target_mask,
            true_labels,
            student_bias,
            self.weight_hard_loss,
            self.weight_soft_loss,
            self.ignore_index,
            self.temperature,
            self.beta,
            self.compiled,
            self.chunk_size,
            self.compute_ce_loss,
            self.normalize_topk,
        )
