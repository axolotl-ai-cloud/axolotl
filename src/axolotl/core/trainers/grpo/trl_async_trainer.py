"""Async GRPO on TRL's experimental ``AsyncGRPOTrainer``.

TRL generates, scores and computes advantages continuously in a CUDA-free child
process that streams requests to a vLLM server, while the trainer consumes scored
samples from a queue (dropping those older than ``max_staleness`` policy versions),
tree-packs them by token budget and syncs weights every ``weight_sync_steps``.

On top of TRL, this trainer:

- takes the model axolotl loaded (quantized, patched, LoRA-wrapped) instead of a
  path, and plugs in axolotl's optimizer, scheduler and distributed mixins;
- drops samples that carry no gradient (``skip_zero_advantage_batches``,
  ``mask_truncated_completions``) before they are packed and forwarded;
- scores rewards in a process pool, several groups at a time
  (``reward_num_workers``);
- honors ``reward_weights``, ``multi_objective_aggregation`` and ``scale_rewards``;
- bounds the policy/sampler ratio of each token
  (``vllm_importance_sampling_cap``).
"""

from __future__ import annotations

import contextlib
import functools
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from accelerate.utils import is_peft_model
from datasets import Dataset
from transformers import PreTrainedModel
from trl.experimental.async_grpo import async_grpo_trainer as trl_async_module
from trl.experimental.async_grpo.async_grpo_config import AsyncGRPOConfig
from trl.experimental.async_grpo.async_grpo_trainer import AsyncGRPOTrainer
from trl.experimental.async_grpo.packing import TREE_ATTENTION
from trl.trainer.utils import add_fused_lm_head, get_config_model_id, nanmax, nanmin

from axolotl.core.trainers.grpo.trl_async_rollout import (
    AGGREGATIONS,
    REWARD_SCALINGS,
    AxolotlAsyncRolloutWorker,
)
from axolotl.core.trainers.mixins import (
    DistributedParallelMixin,
    RngLoaderMixin,
    SchedulerMixin,
)
from axolotl.core.trainers.mixins.optimizer import OptimizerInitMixin, OptimizerMixin
from axolotl.core.training_args_base import AxolotlTrainingMixins
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

IMPORTANCE_SAMPLING_MODES = ("token_truncate", "token_mask")


@dataclass
class AxolotlAsyncGRPOConfig(AxolotlTrainingMixins, AsyncGRPOConfig):
    """TRL's ``AsyncGRPOConfig`` plus axolotl's training fields and extensions."""

    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Weight of each reward function. Defaults to 1.0 for every function."
        },
    )
    multi_objective_aggregation: str = field(
        default="sum_then_normalize",
        metadata={
            "help": "'sum_then_normalize' normalizes the weighted reward sum within each "
            "group; 'normalize_then_sum' (GDPO) normalizes each reward first."
        },
    )
    scale_rewards: str = field(
        default="group",
        metadata={
            "help": "'group' divides advantages by the group's reward std, 'none' only "
            "centers them."
        },
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "Drop completions that hit max_completion_length before packing."
        },
    )
    skip_zero_advantage_batches: bool = field(
        default=True,
        metadata={
            "help": "Drop zero-advantage samples (groups whose rewards are all equal) "
            "before packing; they would cost a forward and backward for no gradient."
        },
    )
    reward_num_workers: int = field(
        default=1,
        metadata={
            "help": "Processes that run synchronous reward functions, scoring as many "
            "groups at once. 1 keeps TRL's thread-based scoring."
        },
    )
    vllm_importance_sampling_correction: bool = field(
        default=True,
        metadata={
            "help": "Bound the per-token ratio between the policy and the vLLM sampler "
            "at vllm_importance_sampling_cap."
        },
    )
    vllm_importance_sampling_mode: str = field(
        default="token_truncate",
        metadata={
            "help": "'token_truncate' caps the unclipped ratio (dual-clip), "
            "'token_mask' drops tokens whose ratio exceeds the cap from the loss."
        },
    )
    vllm_importance_sampling_cap: float = field(
        default=3.0,
        metadata={"help": "Ratio bound used by vllm_importance_sampling_mode."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.multi_objective_aggregation not in AGGREGATIONS:
            raise ValueError(
                f"multi_objective_aggregation must be one of {AGGREGATIONS}, got "
                f"{self.multi_objective_aggregation!r}"
            )
        if self.scale_rewards not in REWARD_SCALINGS:
            raise ValueError(
                f"Async GRPO supports scale_rewards in {REWARD_SCALINGS}, got "
                f"{self.scale_rewards!r}: rewards are normalized per group as each "
                "group finishes, so there is no batch to take a std over."
            )
        if self.vllm_importance_sampling_mode not in IMPORTANCE_SAMPLING_MODES:
            raise ValueError(
                f"Async GRPO supports vllm_importance_sampling_mode in "
                f"{IMPORTANCE_SAMPLING_MODES}, got "
                f"{self.vllm_importance_sampling_mode!r}"
            )
        epsilon_high = self.epsilon if self.epsilon_high is None else self.epsilon_high
        if self.vllm_importance_sampling_cap <= 1.0 + epsilon_high:
            raise ValueError(
                "vllm_importance_sampling_cap must exceed 1 + epsilon_high, or it would "
                "cut into the PPO trust region"
            )
        if self.reward_num_workers < 1:
            raise ValueError("reward_num_workers must be >= 1")


class SampleFilterQueue:
    """A view of the rollout queue that drops samples no gradient can come from.

    A zero-advantage sample adds nothing to the policy loss but costs a full forward
    and backward, and a truncated completion under ``mask_truncated_completions``
    would be masked out of the loss anyway. Dropped samples still count as consumed
    prompt groups, so epoch-bounded training and checkpoint resume stay keyed on the
    prompts that were generated rather than on those that were trained.
    """

    def __init__(
        self,
        inner,
        consumed_groups: set[int],
        metrics: dict[str, list],
        *,
        skip_zero_advantage: bool,
        stop_token_ids: frozenset[int] | None,
    ):
        self._inner = inner
        self._consumed_groups = consumed_groups
        self._metrics = metrics
        self._skip_zero_advantage = skip_zero_advantage
        self._stop_token_ids = stop_token_ids

    def qsize(self) -> int:
        return self._inner.qsize()

    def _drop_reason(self, sample) -> str | None:
        if self._stop_token_ids is not None and (
            len(sample.input_ids) == 0
            or int(sample.input_ids[-1]) not in self._stop_token_ids
        ):
            return "truncated"
        if self._skip_zero_advantage and sample.advantage == 0.0:
            return "zero_adv"
        return None

    def get(self, timeout: float | None = None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            sample = self._inner.get(timeout=remaining)
            reason = self._drop_reason(sample)
            self._metrics["sample/dropped_frac"].append(
                (float(reason is not None), 1.0)
            )
            if reason is None:
                return sample
            self._consumed_groups.add(sample.group_id)
            self._metrics[f"sample/dropped_{reason}_total"].append(1.0)


def _use_loaded_model(model: PreTrainedModel):
    """Stand-in for TRL's ``create_model_from_path`` that returns ``model``."""

    def load(_model_id, *_args, **_kwargs):
        base = model.get_base_model() if is_peft_model(model) else model
        base.set_attn_implementation(TREE_ATTENTION)
        return model

    return load


def _add_fused_lm_head_to_base(model, **kwargs):
    # The fused head reads the backbone and lm_head of the transformers model, so it
    # goes under PEFT's wrapper, as TRL places it before wrapping.
    add_fused_lm_head(
        model.get_base_model() if is_peft_model(model) else model, **kwargs
    )


@contextlib.contextmanager
def _patched_trl_module(**replacements):
    originals = {name: getattr(trl_async_module, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(trl_async_module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(trl_async_module, name, value)


def _is_quantized(model: PreTrainedModel) -> bool:
    base = model.get_base_model() if is_peft_model(model) else model
    return bool(getattr(base, "is_quantized", False)) or any(
        p.dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8)
        for p in base.parameters()
    )


class AxolotlAsyncGRPOTrainer(
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DistributedParallelMixin,
    AsyncGRPOTrainer,
):
    """TRL's ``AsyncGRPOTrainer`` adapted to axolotl and extended (see module doc)."""

    _tag_names = ["trl", "async-grpo", "axolotl"]

    def __init__(
        self,
        model: PreTrainedModel | str,
        reward_funcs=None,
        args: AxolotlAsyncGRPOConfig | None = None,
        train_dataset=None,
        processing_class=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        *,
        eval_dataset=None,
        reward_processing_classes=None,
        rollout_func=None,
        data_collator=None,
        **kwargs,
    ):
        if args is None:
            raise ValueError(
                "AxolotlAsyncGRPOTrainer requires an AxolotlAsyncGRPOConfig"
            )
        if rollout_func is not None:
            raise ValueError(
                "Async GRPO generates in TRL's rollout worker and does not take a "
                "rollout_func."
            )
        if reward_processing_classes:
            raise ValueError("Async GRPO does not support reward models.")
        if data_collator is not None:
            raise ValueError(
                "Async GRPO renders prompts with the tokenizer's chat template and "
                "does not support multimodal datasets."
            )
        if reward_funcs is not None and not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        if not reward_funcs and not (
            kwargs.get("rollout_worker") or kwargs.get("environment_factory")
        ):
            # TRL would only notice once the rollout process starts.
            raise ValueError("Async GRPO needs reward functions (`trl.reward_funcs`).")
        if any(isinstance(func, str) for func in reward_funcs or []):
            raise ValueError(
                "Async GRPO scores rewards in a CUDA-free process, so reward models "
                "are not supported; pass reward functions."
            )
        if eval_dataset is not None:
            LOG.warning("Async GRPO does not evaluate; ignoring the eval dataset.")
        if isinstance(train_dataset, Dataset) and len(train_dataset) > 0:
            prompt = train_dataset[0].get("prompt")
            if not isinstance(prompt, list):
                raise ValueError(
                    "Async GRPO needs conversational prompts (a list of messages in "
                    "the `prompt` column), e.g. from a `chat_template`-style dataset."
                )

        replacements: dict[str, Any] = {
            "AsyncRolloutWorker": functools.partial(
                AxolotlAsyncRolloutWorker,
                reward_weights=args.reward_weights,
                multi_objective_aggregation=args.multi_objective_aggregation,
                scale_rewards=args.scale_rewards,
                reward_num_workers=args.reward_num_workers,
            )
        }
        model_id = model
        if not isinstance(model, str):
            replacements["create_model_from_path"] = _use_loaded_model(model)
            replacements["add_fused_lm_head"] = _add_fused_lm_head_to_base
            model_id = get_config_model_id(model.config)

        with _patched_trl_module(**replacements):
            super().__init__(
                model=model_id,
                reward_funcs=reward_funcs,
                args=args,
                train_dataset=train_dataset,
                processing_class=processing_class,
                callbacks=callbacks,
                optimizers=optimizers,
                peft_config=peft_config,
                **kwargs,
            )

        # Every rank holds the same `_lora_sync` decision, so all of them raise. Merged
        # sync folds the adapter into the base weights, which a quantized base can't take.
        if (
            kwargs.get("weight_transfer") is None
            and not self._lora_sync
            and _is_quantized(self.model)
        ):
            raise ValueError(
                "A quantized base model needs adapter-only weight sync: start the vLLM "
                "server with LoRA enabled (`trl.vllm_lora_sync: true` with "
                "`axolotl vllm-serve`)."
            )

        if self.accelerator.is_main_process and (
            args.skip_zero_advantage_batches or args.mask_truncated_completions
        ):
            stop_token_ids = None
            if args.mask_truncated_completions:
                stop_token_ids = frozenset(
                    [*self.eos_token_ids, self.processing_class.pad_token_id]
                )
            self.rollout_queue = SampleFilterQueue(
                self.rollout_queue,  # type: ignore[has-type]
                self._trained_groups,
                self._metrics["train"],
                skip_zero_advantage=args.skip_zero_advantage_batches,
                stop_token_ids=stop_token_ids,
            )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # TRL's compute_loss with the ratio bound of `_policy_loss`.
        mask_bool = inputs["attention_mask"].bool()
        selected = inputs["token_index"] >= 0
        input_ids = inputs["input_ids"][mask_bool].unsqueeze(0)
        position_ids = inputs["position_ids"][mask_bool].unsqueeze(0)
        shift_labels = inputs["shift_labels"][mask_bool].unsqueeze(0)
        token_index = inputs["token_index"][selected]
        old_log_probs = inputs["old_log_probs"][selected]
        advantages = inputs["advantages"][selected]
        segment_id = inputs["segment_id"][selected]
        tree_kwargs = {
            key: inputs[key][inputs[key] >= 0]
            for key in ("tree_kv_index", "tree_cu_seqlens_q", "tree_cu_seqlens_k")
        }
        tree_kwargs["tree_max_seqlen_q"], tree_kwargs["tree_max_seqlen_k"] = inputs[
            "tree_max_seqlens"
        ][0].tolist()

        forward_start = time.time()
        router_kwargs = {"output_router_logits": True} if self.aux_loss_enabled else {}
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            shift_labels=shift_labels,
            fused_lm_head=True,
            **router_kwargs,
            **tree_kwargs,
        )
        log_probs = outputs.log_probs.flatten()[token_index]
        entropy = outputs.entropy.flatten()[token_index]
        self._last_forward_time_s = time.time() - forward_start

        per_token_loss, coef_1, log_ratio, is_capped = self._policy_loss(
            log_probs, old_log_probs, advantages
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
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped
            count = torch.full_like(coef_1.sum(), float(log_probs.numel()))
            stats = torch.stack(
                [
                    coef_1.sum(),
                    ((coef_1 - 1) - log_ratio).sum(),
                    entropy.sum(),
                    is_low_clipped.float().sum(),
                    is_high_clipped.float().sum(),
                    is_region_clipped.float().sum(),
                    is_capped.float().sum(),
                    count,
                ]
            )
            stats = self.accelerator.reduce(stats, reduction="sum")
            (
                ratio_sum,
                kl_sum,
                entropy_sum,
                low_clip_sum,
                high_clip_sum,
                region_clip_sum,
                capped_sum,
                global_count,
            ) = stats.unbind(0)
            metrics = self._metrics["train"]
            metrics["ratio"].append((ratio_sum / global_count).item())
            metrics["kl"].append((kl_sum / global_count).item())
            metrics["entropy"].append((entropy_sum / global_count).item())
            metrics["clip_ratio/low_mean"].append((low_clip_sum / global_count).item())
            metrics["clip_ratio/high_mean"].append(
                (high_clip_sum / global_count).item()
            )
            metrics["clip_ratio/region_mean"].append(
                (region_clip_sum / global_count).item()
            )
            metrics["sampling/capped_token_frac"].append(
                (capped_sum / global_count).item()
            )

            num_seq = int(segment_id.max()) + 1 if segment_id.numel() else 0

            def seg_sum(vals):
                return torch.zeros(num_seq, device=vals.device).index_add_(
                    0, segment_id, vals
                )

            seq_tokens = seg_sum(torch.ones_like(coef_1, dtype=torch.float32))
            per_seq_low = seg_sum(is_low_clipped.float()) / seq_tokens
            per_seq_high = seg_sum(is_high_clipped.float()) / seq_tokens
            gathered_low_min = self.accelerator.gather(nanmin(per_seq_low))
            gathered_high_max = self.accelerator.gather(nanmax(per_seq_high))
            metrics["clip_ratio/low_min"].append(nanmin(gathered_low_min).item())
            metrics["clip_ratio/high_max"].append(nanmax(gathered_high_max).item())

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-token clipped surrogate against the sampler's log-probabilities.

        The ratio here is the policy over the vLLM sampler, so it carries both
        staleness and trainer/inference mismatch. PPO's clip bounds it only where the
        advantage is positive; for a negative advantage the unclipped term grows with
        the ratio. ``token_truncate`` caps that term at ``vllm_importance_sampling_cap``
        (dual-clip PPO), ``token_mask`` drops tokens beyond the cap from the loss.
        """
        log_ratio = log_probs - old_log_probs
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        unclipped = coef_1
        if self.args.vllm_importance_sampling_correction:
            cap = self.args.vllm_importance_sampling_cap
            is_capped = coef_1.detach() > cap
            if self.args.vllm_importance_sampling_mode == "token_truncate":
                unclipped = torch.clamp(coef_1, max=cap)
        else:
            is_capped = torch.zeros_like(coef_1, dtype=torch.bool)
        per_token_loss = -torch.min(unclipped * advantages, coef_2 * advantages)
        if (
            self.args.vllm_importance_sampling_correction
            and self.args.vllm_importance_sampling_mode == "token_mask"
        ):
            per_token_loss = per_token_loss.masked_fill(is_capped, 0.0)
        return per_token_loss, coef_1, log_ratio, is_capped
