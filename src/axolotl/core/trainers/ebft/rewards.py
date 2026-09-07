"""
Feature-matching reward utilities for Energy-Based Fine-Tuning (EBFT).

Ported from: feature-002/ebft_openrlhf/openrlhf/utils/embedding_utils.py
Paper: "Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models"
       (Jelassi et al., 2026) https://arxiv.org/abs/2603.12248
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def extract_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: list[int],
    batch_size: int | None = None,
) -> torch.Tensor:
    """
    Forward pass through model, extracting and concatenating hidden states
    at specified layer indices.

    Args:
        model: The frozen feature network
        input_ids: (B, S) token ids
        attention_mask: (B, S) attention mask
        layer_indices: List of layer indices to extract (e.g., [8, 16, 24] for 32-layer model)
        batch_size: If set, process in chunks to reduce peak memory

    Returns:
        Concatenated hidden states: (B, S, num_layers * H)
    """
    if batch_size is None:
        batch_size = input_ids.shape[0]

    # Use the inner transformer body (skips lm_head) when available.
    # This avoids the expensive hidden_dim × vocab_size matmul whose
    # output (logits) is never used — only hidden_states are needed.
    body = getattr(model, "model", None)
    if body is not None and hasattr(body, "forward"):
        forward_model = body
    else:
        forward_model = model

    all_features = []
    for i in range(0, input_ids.shape[0], batch_size):
        chunk_ids = input_ids[i : i + batch_size]
        chunk_mask = attention_mask[i : i + batch_size]

        outputs = forward_model(
            chunk_ids,
            attention_mask=chunk_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states is a tuple of (num_layers + 1) tensors, each (B, S, H)
        # index 0 is the embedding layer output
        hidden_states = outputs.hidden_states
        chunk_features = []
        for idx in layer_indices:
            chunk_features.append(hidden_states[idx])

        # Concatenate across feature dimension: (B, S, num_layers * H)
        all_features.append(torch.cat(chunk_features, dim=-1))

    return torch.cat(all_features, dim=0)


def apply_embed_method(
    hidden_states: torch.Tensor,
    method: str,
    attention_mask: torch.Tensor | None = None,
    prompt_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pool per-token hidden states into per-sequence embeddings.

    Args:
        hidden_states: (B, S, D) concatenated hidden states
        method: One of "last_token", "mean_pooling", "completion_mean", "concat"
        attention_mask: (B, S) mask for mean pooling
        prompt_lengths: (B,) number of prompt tokens per sample (for completion_mean)

    Returns:
        Sequence embeddings: (B, D) for last_token/mean_pooling/completion_mean,
                             (B, 3*D) for concat
    """
    if method == "last_token":
        if attention_mask is not None:
            # Find last non-padding position per sample
            last_idx = attention_mask.sum(dim=1).long() - 1  # (B,)
            return hidden_states[torch.arange(hidden_states.shape[0]), last_idx]
        return hidden_states[:, -1, :]

    if method == "mean_pooling":
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return hidden_states.mean(dim=1)

    if method == "completion_mean":
        # Mean pool over completion tokens only (exclude prompt)
        if prompt_lengths is None:
            raise ValueError("completion_mean requires prompt_lengths")
        B, S, _ = hidden_states.shape
        positions = torch.arange(S, device=hidden_states.device).unsqueeze(0)  # (1, S)
        comp_mask = positions >= prompt_lengths.unsqueeze(1)  # (B, S)
        if attention_mask is not None:
            comp_mask = comp_mask & attention_mask.bool()
        mask = comp_mask.unsqueeze(-1).float()  # (B, S, 1)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    if method == "concat":
        B, S, D = hidden_states.shape
        if attention_mask is not None:
            valid_lens = attention_mask.sum(dim=1).long()  # (B,)
        else:
            valid_lens = torch.full(
                (B,), S, device=hidden_states.device, dtype=torch.long
            )
        # Compute quartile positions relative to valid length per sample
        # First valid position index for each sample (handles right-padding)
        q1 = (valid_lens // 4).clamp(min=0, max=S - 1)
        q2 = (valid_lens // 2).clamp(min=0, max=S - 1)
        q3 = (3 * valid_lens // 4).clamp(min=0, max=S - 1)
        batch_idx = torch.arange(B, device=hidden_states.device)
        return torch.cat(
            [
                hidden_states[batch_idx, q1],
                hidden_states[batch_idx, q2],
                hidden_states[batch_idx, q3],
            ],
            dim=-1,
        )

    raise ValueError(f"Unknown embed_method: {method}")


@torch.no_grad()
def get_alignment_rewards(
    gen_embedding: torch.Tensor,
    gt_embedding: torch.Tensor,
) -> torch.Tensor:
    """
    Compute alignment reward as cosine similarity between generated
    and ground-truth feature embeddings.

    Args:
        gen_embedding: (B, D) generated sequence embeddings
        gt_embedding: (B, D) ground-truth sequence embeddings
            If num_generations > 1, gt_embedding should be repeated
            to match gen_embedding's batch dim.

    Returns:
        Alignment rewards: (B,) cosine similarities in [-1, 1]
    """
    return F.cosine_similarity(gen_embedding, gt_embedding, dim=-1)


@torch.no_grad()
def get_diversity_rewards(
    gen_embedding: torch.Tensor,
    num_generations: int,
) -> torch.Tensor:
    """
    Compute diversity penalty as mean pairwise dot-product similarity
    between samples from the same prompt (excluding self-similarity).

    Args:
        gen_embedding: (B, D) generated embeddings where B = num_prompts * num_generations
        num_generations: Number of generations per prompt

    Returns:
        Diversity penalties: (B,) mean similarity to other samples from same prompt
    """
    if num_generations <= 1:
        return torch.zeros(gen_embedding.shape[0], device=gen_embedding.device)

    num_prompts = gen_embedding.shape[0] // num_generations

    # Reshape to (num_prompts, num_generations, D)
    reshaped = gen_embedding.view(num_prompts, num_generations, -1)

    # Pairwise dot products within each group: (num_prompts, num_generations, num_generations)
    sims = torch.bmm(reshaped, reshaped.transpose(1, 2))

    # Zero out self-similarity (diagonal)
    eye = torch.eye(num_generations, device=sims.device, dtype=torch.bool)
    sims = sims.masked_fill(eye.unsqueeze(0), 0.0)

    # Mean similarity to other samples: (num_prompts, num_generations)
    diversity = sims.sum(dim=-1) / (num_generations - 1)

    # Flatten back to (B,)
    return diversity.view(-1)


def whiten_embeddings_batched(
    phi: torch.Tensor,
    phi_gt: torch.Tensor,
    whiten_tol: float = 1e-5,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Whiten generated embeddings using SVD, then apply same transform to ground-truth.

    Whitening decorrelates feature dimensions so no single direction dominates
    the feature-matching loss. Uses pseudo-inverse for rank-deficient cases.

    Note: Singular values scale with sqrt(B), so reward magnitudes are
    batch-size dependent. This is acceptable because B = n_samples_per_prompt
    which is fixed during training (typically 2-4).

    Args:
        phi: (B, D) generated embeddings (used to estimate covariance)
        phi_gt: (B, D) ground-truth embeddings
        whiten_tol: Tolerance for singular value cutoff
        normalize: If True, L2-normalize after whitening

    Returns:
        Whitened (phi, phi_gt) tuple, each (B, D)
    """
    phi_f = phi.float()
    phi_gt_f = phi_gt.float()

    # Feature-space SVD: operate on phi_f.T (D, B) so U is (D, D)
    try:
        U, S, _ = torch.linalg.svd(phi_f.T.unsqueeze(0), full_matrices=False)
    except torch._C._LinAlgError:
        # Fallback: add small noise
        noise = 1e-6 * phi_f.abs().mean()
        try:
            U, S, _ = torch.linalg.svd(
                (phi_f.T + noise * torch.randn_like(phi_f.T)).unsqueeze(0),
                full_matrices=False,
            )
        except torch._C._LinAlgError:
            if normalize:
                return (
                    F.normalize(phi, p=2, dim=-1),
                    F.normalize(phi_gt, p=2, dim=-1),
                )
            return phi, phi_gt

    U, S = U.squeeze(0), S.squeeze(0)  # U: (D, min(D,B)), S: (min(D,B),)

    # Safe inverse of singular values
    s_max = S.max()
    inv_s = torch.where(S > whiten_tol * s_max, 1.0 / (S + 1e-12), torch.zeros_like(S))

    # W = U @ diag(inv_s) @ U^T  — feature-space whitening matrix (D, D)
    W = (U * inv_s.unsqueeze(0)) @ U.T  # (D, D)
    phi_w = (phi_f @ W).to(phi.dtype)  # (B, D)
    phi_gt_w = (phi_gt_f @ W).to(phi_gt.dtype)  # (B, D)

    if normalize:
        phi_w = F.normalize(phi_w, p=2, dim=-1)
        phi_gt_w = F.normalize(phi_gt_w, p=2, dim=-1)

    return phi_w, phi_gt_w
