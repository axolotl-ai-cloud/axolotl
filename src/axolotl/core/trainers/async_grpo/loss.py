"""Async GRPO's per-token losses (``loss_type``) and their metrics.

``compute_loss`` is TRL's, with ``_policy_loss`` in place of TRL's clipped objective
and its metrics per loss type.
"""

from __future__ import annotations

import time

import torch
from tree_recurrence.integrations.trl import (
    sample_boundaries,
    tree_kwargs_from_batch,
)

from axolotl.core.trainers.async_grpo.trl_compat import (
    AsyncGRPOTrainer,
    nanmax,
    nanmin,
)


def support_ids(
    inputs: dict[str, torch.Tensor], mask_bool: torch.Tensor
) -> torch.Tensor | None:
    """Each position's top-k/top-p candidate set, padded with -1, which TRL main's
    fused head renormalizes the log-probs over as the sampler did; None without
    truncation, and on TRL releases, which do not record the sets."""
    if "sampling_mask_counts" not in inputs:
        return None
    counts = inputs["sampling_mask_counts"][mask_bool]
    width = int(counts.max()) if counts.numel() else 0
    if not width:
        return None
    candidates = inputs["sampling_mask"][inputs["sampling_mask"] >= 0]
    row = torch.repeat_interleave(
        torch.arange(len(counts), device=counts.device), counts
    )
    col = torch.arange(len(candidates), device=counts.device) - torch.repeat_interleave(
        counts.cumsum(0) - counts, counts
    )
    support = candidates.new_full((len(counts), width), -1)
    support[row, col] = candidates
    return support.unsqueeze(0)


class AsyncGRPOLossMixin(AsyncGRPOTrainer):
    """``compute_loss`` with the loss of ``args.loss_type``."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        mask_bool = inputs["attention_mask"].bool()
        selected = inputs["token_index"] >= 0
        input_ids = inputs["input_ids"][mask_bool].unsqueeze(0)
        position_ids = inputs["position_ids"][mask_bool].unsqueeze(0)
        shift_labels = inputs["shift_labels"][mask_bool].unsqueeze(0)
        token_index = inputs["token_index"][selected]
        old_log_probs = inputs["old_log_probs"][selected]
        advantages = inputs["advantages"][selected]
        segment_id = inputs["segment_id"][selected]
        tree_kwargs = tree_kwargs_from_batch(inputs)
        if not self._share_prefixes:
            tree_kwargs.update(sample_boundaries(tree_kwargs))
        score_centering = self.args.loss_type == "score_centering"
        sampler_kwargs = {}
        support = support_ids(inputs, mask_bool)
        if support is not None:
            sampler_kwargs["support_ids"] = support
        if score_centering:
            sampler_kwargs["score_ids"] = inputs["sampler_ids"][mask_bool].unsqueeze(0)

        forward_start = time.time()
        router_kwargs = {"output_router_logits": True} if self.aux_loss_enabled else {}
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            shift_labels=shift_labels,
            fused_lm_head=True,
            **sampler_kwargs,
            **router_kwargs,
            **tree_kwargs,
        )
        log_probs = outputs.log_probs.flatten()[token_index]
        entropy = outputs.entropy.flatten()[token_index]
        self._last_forward_time_s = time.time() - forward_start
        score_log_probs = sampler_log_probs = None
        if score_centering:
            score_log_probs = outputs.score_log_probs[0][token_index]
            sampler_log_probs = inputs["sampler_log_probs"][mask_bool][token_index]

        per_token_loss, ratio, log_ratio = self._policy_loss(
            log_probs, old_log_probs, advantages, score_log_probs, sampler_log_probs
        )

        loss = per_token_loss.sum()
        global_n_tokens = inputs["global_n_tokens"][0]
        world_size = self.accelerator.num_processes
        tokens_per_rank = (global_n_tokens / world_size).clamp(min=1.0)
        loss = loss / tokens_per_rank.to(torch.float32)
        loss = loss / self.current_gradient_accumulation_steps

        if self.aux_loss_enabled:
            aux_loss = outputs.aux_loss
            loss = (
                loss
                + self.router_aux_loss_coef
                * aux_loss
                / self.current_gradient_accumulation_steps
            )

        with torch.no_grad():
            loss_type = self.args.loss_type
            # Per-token values logged as their mean over trained tokens.
            if loss_type == "cispo":
                token_means = {
                    "sampling/capped_token_frac": ratio
                    > self.args.vllm_importance_sampling_cap
                }
            elif loss_type == "score_centering":
                token_means = {
                    "sampling/masked_mean": (
                        ratio < self.args.vllm_importance_sampling_clip_min
                    )
                    | (ratio > self.args.vllm_importance_sampling_clip_max),
                    "sampling/sampler_mass": sampler_log_probs.exp().sum(-1),
                }
            else:
                is_low_clipped = (ratio < 1 - self.epsilon_low) & (advantages < 0)
                is_high_clipped = (ratio > 1 + self.epsilon_high) & (advantages > 0)
                token_means = {
                    "clip_ratio/low_mean": is_low_clipped,
                    "clip_ratio/high_mean": is_high_clipped,
                    "clip_ratio/region_mean": is_low_clipped | is_high_clipped,
                }
            count = torch.full_like(ratio.sum(), float(log_probs.numel()))
            stats = torch.stack(
                [
                    ratio.sum(),
                    ((ratio - 1) - log_ratio).sum(),
                    entropy.sum(),
                    count,
                    *(value.float().sum() for value in token_means.values()),
                ]
            )
            stats = self.accelerator.reduce(stats, reduction="sum").tolist()
            ratio_sum, kl_sum, entropy_sum, global_count = stats[:4]
            token_sums = stats[4:]
            local_min = ratio.min() if ratio.numel() else ratio.new_tensor(float("inf"))
            ratio_min = self.accelerator.gather(local_min.float().reshape(1)).min()
            metrics = self._metrics["train"]
            # Trainer-side scoring can leave a micro-batch no trained token on any
            # rank. The count is reduced, so every rank skips these together.
            if global_count:
                metrics["ratio"].append(ratio_sum / global_count)
                metrics["kl"].append(kl_sum / global_count)
                metrics["entropy"].append(entropy_sum / global_count)
                for name, total in zip(token_means, token_sums, strict=True):
                    metrics[name].append(total / global_count)
            if torch.isfinite(ratio_min):
                metrics["sampling/ratio_min"].append(ratio_min.item())
            if loss_type == "dapo":
                num_seq = int(segment_id.max()) + 1 if segment_id.numel() else 0

                def seg_sum(vals):
                    return torch.zeros(num_seq, device=vals.device).index_add_(
                        0, segment_id, vals
                    )

                seq_tokens = seg_sum(torch.ones_like(ratio, dtype=torch.float32))
                per_seq_low = seg_sum(is_low_clipped.float()) / seq_tokens
                per_seq_high = seg_sum(is_high_clipped.float()) / seq_tokens
                gathered_low_min = self.accelerator.gather(nanmin(per_seq_low))
                gathered_high_max = self.accelerator.gather(nanmax(per_seq_high))
                if global_count:
                    metrics["clip_ratio/low_min"].append(
                        nanmin(gathered_low_min).item()
                    )
                    metrics["clip_ratio/high_max"].append(
                        nanmax(gathered_high_max).item()
                    )

            if self.aux_loss_enabled:
                gathered_aux = self.accelerator.reduce(
                    aux_loss.detach().to(torch.float32), reduction="sum"
                )
                metrics["aux_loss"].append((gathered_aux / world_size).item())

        n_forward_tokens = float(inputs["global_n_forward_tokens"][0])
        mean_seq_len = float(inputs["mean_seq_len"][0])
        self._step_forward_tokens += n_forward_tokens
        self._step_trained_tokens += float(global_n_tokens)
        self._step_seq_len_weighted += mean_seq_len * n_forward_tokens
        self._step_samples += n_forward_tokens / mean_seq_len
        self._step_forward_s += self._last_forward_time_s
        return loss

    def _policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        score_log_probs: torch.Tensor | None = None,
        sampler_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-token loss, with the policy/sampler ratio and its log (detached).

        ``dapo`` is TRL's objective: PPO's surrogate with the ratio taken against the
        vLLM sampler's log-probabilities, clipped at [1 - epsilon, 1 + epsilon_high].
        That ratio carries staleness and trainer/inference mismatch, so the clip also
        stops the gradient of the tokens whose probability moved furthest since they
        were sampled. ``cispo`` keeps every token's gradient: each rollout is trained
        on once, so the policy ratio is taken against itself (1, never clipped) and the
        policy/sampler ratio is a detached weight, capped at
        ``vllm_importance_sampling_cap`` (``token_truncate``) or zero beyond it
        (``token_mask``). ``score_centering`` is TRL main's: the policy/sampler-weighted
        score, zero outside [``vllm_importance_sampling_clip_min``,
        ``vllm_importance_sampling_clip_max``], minus its expectation over the sampler's
        top tokens (``score_log_probs`` under the policy, ``sampler_log_probs`` under
        the sampler).
        """
        log_ratio = log_probs - old_log_probs
        if self.args.loss_type == "dapo":
            coef_1 = torch.exp(log_ratio)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
            return per_token_loss, coef_1.detach(), log_ratio.detach()
        if self.args.loss_type == "score_centering":
            assert score_log_probs is not None and sampler_log_probs is not None
            low = self.args.vllm_importance_sampling_clip_min
            high = self.args.vllm_importance_sampling_clip_max
            coef_1 = torch.exp(log_ratio)
            is_masked = (coef_1 < low) | (coef_1 > high)
            # arXiv 2609.20807, Appendix A.2, as TRL computes it.
            with torch.no_grad():
                q = sampler_log_probs.exp()
                p = score_log_probs.exp() * (q > 0)
                head = torch.where((p >= low * q) & (p <= high * q), p, 0.0)
                tail_ratio = (1 - p.sum(-1)).clamp(min=1e-6) / (1 - q.sum(-1)).clamp(
                    min=1e-6
                )
                tail = ((tail_ratio >= low) & (tail_ratio <= high)).float()
                residual = head - tail.unsqueeze(-1) * p
                weight = torch.where(is_masked, 0.0, coef_1)
            correction = (residual * score_log_probs).sum(-1)
            per_token_loss = -advantages * (weight * log_probs - correction)
            return per_token_loss, coef_1.detach(), log_ratio.detach()
        log_ratio = log_ratio.detach()
        ratio = torch.exp(log_ratio)
        cap = self.args.vllm_importance_sampling_cap
        if self.args.vllm_importance_sampling_mode == "token_truncate":
            weight = ratio.clamp(max=cap)
        else:
            weight = ratio.masked_fill(ratio > cap, 0.0)
        per_token_loss = (
            -advantages * weight * torch.exp(log_probs - log_probs.detach())
        )
        return per_token_loss, ratio, log_ratio
