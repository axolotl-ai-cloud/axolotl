"""Async GRPO trainer with the SCOPE-RL auxiliary branch."""

import threading

import torch

from axolotl.core.trainers.grpo.trainer import AxolotlAsyncGRPOTrainer
from axolotl.utils.logging import get_logger

from .scope import scope_aux_indices, scope_temperature, scope_weights

LOG = get_logger(__name__)


class ScopeRLAsyncGRPOTrainer(AxolotlAsyncGRPOTrainer):
    """Resamples a fraction of each rollout at an entropy-adjusted temperature and
    trains on its positive completions with an ``alpha``-weighted loss term."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.args.scope_rl:
            raise ValueError(f"{type(self).__name__} requires `scope_rl: true`.")
        if self.args.streaming_partial_batch:
            raise ValueError("scope_rl is not supported with streaming_partial_batch.")
        if not self.use_vllm:
            raise ValueError("scope_rl requires vLLM generation.")
        if not self.args.async_prefetch:
            raise ValueError("scope_rl requires async_prefetch.")
        self._scope_temp_lock = threading.Lock()

    def _scope_generate(self, prompts, rank0_only):
        """Generate the auxiliary branch at an entropy-adjusted temperature.

        The temperature is relative to the policy, so it is applied on top of the
        configured sampling temperature (identical to the paper when that is 1.0).
        Only the generation backend is retuned: ``self.temperature`` also scales
        logits on the training thread, which runs concurrently with this call.
        """
        # Before the first optimizer step there is no entropy yet, so run at T = 1.0.
        entropy = self._last_entropy
        if entropy is None:
            entropy = self.args.scope_target_entropy
        scale = scope_temperature(
            entropy,
            self.args.scope_target_entropy,
            self.args.scope_temperature_min,
            self.args.scope_temperature_max,
        )
        self._metrics["train"]["scope/temperature"].append(scale)

        generation = self.vllm_generation
        with self._scope_temp_lock:
            previous = generation.temperature
            generation.temperature = previous * scale
            try:
                if rank0_only:
                    return self._generate_rank0_only(prompts)
                return self._generate(prompts)
            finally:
                generation.temperature = previous

    def _generate_auxiliary(self, inputs, prompts, images, rank0_only):
        if images is not None:
            LOG.warning_once("scope_rl is skipped for multimodal batches.")
            return [], [], None
        idx = scope_aux_indices(
            len(inputs),
            self.num_generations,
            self.args.scope_alpha,
            self.state.global_step,
        )
        if not idx:
            return [], [], None
        aux_inputs = [inputs[i] for i in idx]
        aux_prompts = [prompts[i] for i in idx]
        return aux_inputs, aux_prompts, self._scope_generate(aux_prompts, rank0_only)

    def _post_advantage_hook(
        self,
        data,
        rewards_per_func,
        advantages,
        inputs,
        num_generations,
        mode,
        s_start=None,
        s_end=None,
        is_last_chunk=True,
        rewards=None,
        process_slice=None,
    ):
        # Runs before the replay buffer so replaced groups keep their advantages.
        if "aux_mask" in data and rewards is not None:
            self._apply_scope(data, advantages, rewards, process_slice, mode)

        super()._post_advantage_hook(
            data,
            rewards_per_func,
            advantages,
            inputs,
            num_generations,
            mode,
            s_start=s_start,
            s_end=s_end,
            is_last_chunk=is_last_chunk,
            rewards=rewards,
            process_slice=process_slice,
        )

    def _apply_scope(self, data, advantages, rewards, process_slice, mode):
        # Auxiliary rows keep positives only, at advantage 1 (Eq. 11).
        aux_mask = data["aux_mask"]
        positive = (rewards >= self.args.scope_positive_threshold).float()
        if aux_mask.size(0) == rewards.size(0):
            aux_mask = aux_mask[process_slice]
            positive = positive[process_slice]
        data["aux_mask"] = aux_mask
        data["scope_weight"] = scope_weights(aux_mask, self.args.scope_alpha)
        data["advantages"] = torch.where(aux_mask.bool(), positive, advantages)
        if "importance_sampling_ratio" in data:
            # Auxiliary rows come from the temperature-scaled policy by design,
            # so the vLLM/policy mismatch correction does not apply to them.
            is_ratio = data["importance_sampling_ratio"]
            data["importance_sampling_ratio"] = torch.where(
                aux_mask.bool().unsqueeze(1), torch.ones_like(is_ratio), is_ratio
            )
        self._metrics[mode]["scope/positive_frac"].append(
            ((positive * aux_mask).sum() / aux_mask.sum().clamp(min=1)).item()
        )

    def _weight_per_token_loss(self, per_token_loss, per_token_kl, inputs):
        per_token_loss, per_token_kl = super()._weight_per_token_loss(
            per_token_loss, per_token_kl, inputs
        )
        if "scope_weight" not in inputs:
            return per_token_loss, per_token_kl
        # Fold the alpha-weighted auxiliary term into the row weights; the KL
        # penalty applies to the main rows only.
        weight = inputs["scope_weight"]
        per_token_loss = per_token_loss * weight.unsqueeze(1)
        if per_token_kl is not None:
            per_token_kl = per_token_kl * (weight * (1 - inputs["aux_mask"])).unsqueeze(
                1
            )
        return per_token_loss, per_token_kl
