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
- offers ``loss_type: cispo``, which weights each token by its policy/sampler ratio,
  capped at ``vllm_importance_sampling_cap``, instead of clipping that ratio, and,
  experimentally on TRL main, ``loss_type: score_centering``, TRL's score centering;
- syncs merged LoRA weights without merging into the trained model, bitwise equal to
  the ``lora_merge_aware`` forward, and measures how much of the update the merge
  keeps (``vllm_merge_check_interval``);
- runs a custom rollout loop (``rollout_loop``), e.g. one that delegates whole
  rollouts to an agent server, and serves the synced adapter under a fixed name
  for clients that cannot follow TRL's versioned one (``vllm_lora_alias``).
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import time
from collections.abc import Callable, Sized
from dataclasses import dataclass, field
from typing import Any

import requests
import torch
from accelerate.utils import broadcast_object_list, is_peft_model
from datasets import Dataset
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TrainerCallback
from transformers.trainer_utils import SaveStrategy
from tree_recurrence.integrations.trl import (
    ChainFixedCountBatcher,
    ChainTokenBudgetBatcher,
    hidden_layer_types,
    sample_boundaries,
    tree_kwargs_from_batch,
)

from axolotl.core.trainers.grpo.async_progress import (
    STOP_KEY,
    STOP_STALLED,
    RolloutProgress,
    StopAwareDataset,
)
from axolotl.core.trainers.grpo.async_rollout import (
    AGGREGATIONS,
    REWARD_SCALINGS,
    AxolotlAsyncRolloutWorker,
    AxolotlRolloutLoop,
)
from axolotl.core.trainers.grpo.async_scoring import (
    GroupedQueue,
    ScoringCollator,
    StepPlanner,
    TrainerScorer,
    check_step_capacity,
    score_step,
)
from axolotl.core.trainers.grpo.async_traces import (
    TraceCollator,
    TraceRecorder,
    rollout_traces,
)
from axolotl.core.trainers.grpo.merge_check import (
    DEFAULT_MERGE_CHECK_INTERVAL,
    LoraMergeCheckMixin,
)
from axolotl.core.trainers.grpo.trl_compat import (
    SCORE_CENTERING,
    TREE_ATTENTION,
    AsyncGRPOConfig,
    AsyncGRPOTrainer,
    EmptyIterableDataset,
    FixedCountBatcher,
    RolloutQueueDataset,
    TokenBudgetBatcher,
    VLLMClient,
    add_fused_lm_head,
    check_version,
    get_config_model_id,
    install_runtime_patches,
    iter_vllm_named_params,
    nanmax,
    nanmin,
    patched_trainer_module,
    score_centering_unavailable,
)
from axolotl.core.trainers.mixins import (
    ActivationOffloadingMixin,
    CheckpointWriteRetryMixin,
    DistributedParallelMixin,
    RngLoaderMixin,
    SchedulerMixin,
)
from axolotl.core.trainers.mixins.optimizer import OptimizerInitMixin, OptimizerMixin
from axolotl.core.training_args_base import AxolotlTrainingMixins
from axolotl.integrations.base import PluginManager
from axolotl.monkeypatch.models.recurrent_support import recurrent_layer_types
from axolotl.monkeypatch.models.tree_packing import packing_support
from axolotl.utils.logging import get_logger
from axolotl.utils.lora_merge_precision import (
    MergeRetention,
    lora_delta,
    served_dtype_for,
)

LOG = get_logger(__name__)

LOSS_TYPES = ("dapo", "cispo", "score_centering")
IMPORTANCE_SAMPLING_MODES = ("token_truncate", "token_mask")
# Ratio thresholds of mu-GRPO's negative-advantage veto (arXiv 2605.17570), keyed by
# metric suffix. 1e-4 is its default, which trainer/vLLM mismatch alone can reach.
NAV_THRESHOLDS = {"1e-4": 1e-4, "1e-6": 1e-6}


def negative_advantage_veto_counts(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    segment_id: torch.Tensor,
    thresholds: tuple[float, ...] = tuple(NAV_THRESHOLDS.values()),
) -> torch.Tensor:
    """Negative-advantage samples, then how many hold a token below each threshold.

    ``ratio`` and ``advantages`` are per trained token, ``segment_id`` the sample each
    token belongs to. mu-GRPO's veto masks such a sample whole.
    """
    counts = ratio.new_zeros(1 + len(thresholds), dtype=torch.float32)
    if segment_id.numel() == 0:
        return counts
    num_samples = int(segment_id.max()) + 1
    min_ratio = ratio.new_full((num_samples,), float("inf"), dtype=torch.float32)
    min_ratio = min_ratio.scatter_reduce(0, segment_id, ratio.float(), reduce="amin")
    negative = torch.zeros(num_samples, dtype=torch.bool, device=ratio.device)
    negative[segment_id[advantages < 0]] = True
    counts[0] = negative.sum()
    for i, threshold in enumerate(thresholds, start=1):
        counts[i] = (negative & (min_ratio < threshold)).sum()
    return counts


def resolve_rollout_loop(fqn: str) -> type[AxolotlRolloutLoop]:
    module_name, _, attr = fqn.rpartition(".")
    loop_cls = getattr(importlib.import_module(module_name), attr, None)
    if not (isinstance(loop_cls, type) and issubclass(loop_cls, AxolotlRolloutLoop)):
        raise ValueError(f"rollout_loop {fqn!r} is not an AxolotlRolloutLoop subclass")
    return loop_cls


class AliasingVLLMClient(VLLMClient):
    """TRL's vLLM client that also loads every synced adapter under ``alias``.

    vLLM keys its prefix cache on the adapter name, which the alias keeps across
    versions, so a sync drains in-flight requests (pausing with ``mode=wait`` also
    clears the cache) rather than freezing them under the old adapter.
    """

    def __init__(self, *args, alias: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.alias = alias

    def pause(self, timeout: int = 1800) -> None:
        response = requests.post(
            f"{self.server_url}/pause", params={"mode": "wait"}, timeout=timeout
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Pausing vLLM failed: {response.status_code}, {response.text}"
            )

    def load_lora_adapter(
        self, lora_name: str, lora_path: str, timeout: int = 1800
    ) -> None:
        super().load_lora_adapter(lora_name, lora_path, timeout)
        self.unload_lora_adapter(self.alias, timeout)
        super().load_lora_adapter(self.alias, lora_path, timeout)


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
    scoring_token_budget: int | None = field(
        default=None,
        metadata={
            "help": "Max tokens in one forward of a trainer-side scorer (EBFT's "
            "feature network); a row's scored sequences are split into as many "
            "forwards as needed. Defaults to token_budget; <= 0 scores each row in "
            "one forward."
        },
    )
    loss_type: str = field(
        default="dapo",
        metadata={
            "help": "'dapo' is TRL's objective, the PPO surrogate against the vLLM "
            "sampler's log-probabilities clipped at [1 - epsilon, 1 + epsilon_high]. "
            "'cispo' weights each token's policy gradient by the policy/sampler "
            "ratio, bounded at vllm_importance_sampling_cap, and clips nothing. "
            "'score_centering' (TRL main only) is TRL's score centering: the "
            "importance-weighted score minus its expectation over the sampler's "
            "num_sampler_logprobs most likely tokens, dropping tokens whose ratio "
            "falls outside [vllm_importance_sampling_clip_min, "
            "vllm_importance_sampling_clip_max]."
        },
    )
    vllm_importance_sampling_mode: str = field(
        default="token_truncate",
        metadata={
            "help": "With loss_type 'cispo': 'token_truncate' caps the weight at the "
            "cap, 'token_mask' drops tokens whose ratio exceeds the cap from the loss."
        },
    )
    vllm_importance_sampling_cap: float = field(
        default=3.0,
        metadata={"help": "With loss_type 'cispo': the bound on the weight."},
    )
    rollout_loop: str | None = field(
        default=None,
        metadata={
            "help": "Fully qualified name of an AxolotlRolloutLoop subclass to run in "
            "the rollout process instead of TRL's generate-and-score loop."
        },
    )
    rollout_loop_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Extra keyword arguments for rollout_loop."},
    )
    vllm_merge_check_interval: int = field(
        default=DEFAULT_MERGE_CHECK_INTERVAL,
        metadata={
            "help": "With merged weight sync, measure how much of the LoRA update "
            "survives rounding the merged weights at the first sync and then every N "
            "optimizer steps (lora_merge/retained). 0 disables."
        },
    )
    vllm_lora_alias: str | None = field(
        default=None,
        metadata={
            "help": "With adapter-only weight sync, also serve each synced adapter "
            "under this fixed name, for clients that address vLLM by a fixed model "
            "name. Syncs then drain in-flight requests before switching."
        },
    )

    def __post_init__(self):
        if SCORE_CENTERING:
            # TRL main trains with score centering by default; loss_type decides here,
            # so that the rollout worker requests top log-probs only when it is used.
            self.score_centering = self.loss_type == "score_centering"
        elif self.loss_type == "score_centering":
            raise ValueError(score_centering_unavailable())
        super().__post_init__()
        if self.rollout_loop is not None:
            resolve_rollout_loop(self.rollout_loop)
            if self.loss_type == "score_centering":
                raise ValueError(
                    "loss_type 'score_centering' centers over the vLLM sampler's top "
                    "log-probs, which only TRL's own rollout loop requests; "
                    f"rollout_loop {self.rollout_loop!r} does not"
                )
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
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(
                f"Async GRPO supports loss_type in {LOSS_TYPES}, got {self.loss_type!r}"
            )
        if self.vllm_importance_sampling_mode not in IMPORTANCE_SAMPLING_MODES:
            raise ValueError(
                f"Async GRPO supports vllm_importance_sampling_mode in "
                f"{IMPORTANCE_SAMPLING_MODES}, got "
                f"{self.vllm_importance_sampling_mode!r}"
            )
        if self.vllm_importance_sampling_cap <= 1.0:
            raise ValueError(
                "vllm_importance_sampling_cap must exceed 1, or it would down-weight "
                "or drop every token the sampler agrees with"
            )
        if self.reward_num_workers < 1:
            raise ValueError("reward_num_workers must be >= 1")


class SampleFilterQueue:
    """A view of the rollout queue that drops samples which would train nothing.

    - ``stale``: more than ``max_staleness`` policy versions old, TRL's rule, applied
      here before TRL's dataset sees the sample so that the drop is accounted;
    - ``oversize``: longer than ``token_budget``, the rule of TRL's batchers;
    - ``zero_adv``: a zero advantage adds nothing to the policy loss but costs a full
      forward and backward (``skip_zero_advantage_batches``);
    - ``truncated``: masked out of the loss under ``mask_truncated_completions``.

    Each drop counts its prompt group as consumed in ``progress``, so epochs and
    resume stay keyed on the prompts generated, and may end the stream (see
    ``async_progress``).
    """

    def __init__(
        self,
        inner,
        progress: RolloutProgress,
        metrics: dict[str, list],
        *,
        model_version: Callable[[], int] | None = None,
        max_staleness: int | None = None,
        token_budget: Callable[[], int] = lambda: 0,
        skip_zero_advantage: bool = False,
        stop_token_ids: frozenset[int] | None = None,
    ):
        self._inner = inner
        self._progress = progress
        self._metrics = metrics
        self._model_version = model_version
        self._max_staleness = max_staleness
        self._token_budget = token_budget
        self._skip_zero_advantage = skip_zero_advantage
        self._stop_token_ids = stop_token_ids

    def qsize(self) -> int:
        return self._inner.qsize()

    def _drop_reason(self, sample) -> str | None:
        if (
            self._model_version is not None
            and self._max_staleness is not None
            and self._model_version() - sample.model_version > self._max_staleness
        ):
            return "stale"
        budget = self._token_budget()
        if budget > 0 and len(sample.input_ids) > budget:
            LOG.warning_once(
                f"Dropping rollout samples longer than token_budget={budget}; raise "
                "token_budget to keep them (counted in sample/dropped_oversize_total)."
            )
            return "oversize"
        if self._stop_token_ids is not None and (
            len(sample.input_ids) == 0
            or int(sample.input_ids[-1]) not in self._stop_token_ids
        ):
            return "truncated"
        if self._skip_zero_advantage and sample.advantage == 0.0:
            return "zero_adv"
        return None

    def get(self, timeout: float | None = None):
        self._progress.check()
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
            self._metrics[f"sample/dropped_{reason}_total"].append(1.0)
            self._progress.drop(sample.group_id, reason)


def _use_loaded_model(model: PreTrainedModel):
    """Stand-in for TRL's ``create_model_from_path`` that returns ``model``."""

    def load(_model_id, *_args, **_kwargs):
        _set_tree_attention(model.get_base_model() if is_peft_model(model) else model)
        return model

    return load


def _set_tree_attention(base: PreTrainedModel) -> None:
    """Route the attention layers to TRL's tree attention.

    Packed rows rely on it to keep samples (and trie branches) apart, so a model
    that keeps another implementation is refused rather than trained on attention
    across samples; models without attention layers (Mamba2) have none to route.
    """
    # class-name heuristic: transformers' own check scans the modeling source
    has_attention = any(type(m).__name__.endswith("Attention") for m in base.modules())
    try:
        base.set_attn_implementation(TREE_ATTENTION)
    except ValueError:
        if has_attention:
            raise
        return
    implementation = base.config._attn_implementation
    if has_attention and implementation != TREE_ATTENTION:
        raise ValueError(
            f"{type(base).__name__} kept attention implementation {implementation!r} "
            f"instead of {TREE_ATTENTION!r}, so its attention would not keep the packed "
            "samples apart. Its attention layers need to dispatch through "
            "ALL_ATTENTION_FUNCTIONS.get_interface for async GRPO."
        )


def _add_fused_lm_head_to_base(model, **kwargs):
    # The fused head reads the backbone and lm_head of the transformers model, so it
    # goes under PEFT's wrapper, as TRL places it before wrapping.
    add_fused_lm_head(
        model.get_base_model() if is_peft_model(model) else model, **kwargs
    )


class _EmptyStepCallback(TrainerCallback):
    """What an optimizer step whose micro-batches hold no trained token does.

    Trainer-side scoring can exclude every sample of a step (constant rewards, all
    truncated). The step still counts: ``global_step`` and the learning-rate
    schedule advance with the rollouts it consumed. It updates nothing, though:
    without a gradient, Adam's momentum and weight decay would still move the
    weights, so the gradients are dropped and the optimizer skips every parameter,
    leaving its moments and step counts as they were. With the weights as last
    synced, the next weight sync is skipped and the policy version stays.
    """

    def __init__(self, trainer: "AxolotlAsyncGRPOTrainer"):
        self._trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        trainer = self._trainer
        if trainer._step_trains_nothing:
            trainer.model.zero_grad(set_to_none=True)
            trainer._metrics["train"]["batch/empty_steps_total"].append(1.0)
        else:
            trainer._weights_updated = True


def _stop_aware(batcher_cls):
    """A stand-in for one of TRL's batchers that ends its stream on ``RolloutStop``."""

    def build(*args, **kwargs):
        return StopAwareDataset(batcher_cls(*args, **kwargs))

    return build


def _is_quantized(model: PreTrainedModel) -> bool:
    base = model.get_base_model() if is_peft_model(model) else model
    return bool(getattr(base, "is_quantized", False)) or any(
        p.dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8)
        for p in base.parameters()
    )


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor.detach()


def _lora_delta(module, adapter: str) -> torch.Tensor:
    """FP32 delta of ``adapter``. A LoRA linear's comes from its gathered factors, as
    the ``lora_merge_aware`` forward and PEFT's merge compute it; other layers'
    from PEFT."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    if isinstance(module, LoraLinear) and not module.fan_in_fan_out:
        return lora_delta(
            _full(module.lora_A[adapter].weight),
            _full(module.lora_B[adapter].weight),
            module.scaling[adapter],
        )
    return _full(module.get_delta_weight(adapter)).float()


def _measures_merge(module, adapters: list[str]) -> bool:
    """Whether ``measure_lora_merge_retention`` would count this layer's rounding:
    a plain single-adapter LoRA linear that does not already train merge-aware."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    return (
        isinstance(module, LoraLinear)
        and not module.fan_in_fan_out
        and len(adapters) == 1
        and adapters[0] not in module.lora_variant
        and getattr(module, "_axolotl_lora_merge_aware_dtype", None) is None
    )


def _merged_weight(
    module,
    adapters: list[str],
    weight: torch.Tensor,
    retention: MergeRetention | None = None,
) -> torch.Tensor:
    """``weight`` with ``adapters`` folded in and rounded once, leaving ``weight``
    itself untouched; with ``retention``, also how much of the delta the rounding
    keeps."""
    base = _full(weight).float()
    merged = base
    for adapter in adapters:
        if adapter not in module.lora_variant:
            merged = merged + _lora_delta(module, adapter)
            continue
        # Variants (DoRA) rebuild the merge from the module's own sharded weights.
        if isinstance(weight, DTensor) or len(adapters) > 1:
            raise NotImplementedError(
                f"Merged weight sync of a {type(module.lora_variant[adapter]).__name__} "
                "adapter is only supported for a single adapter without FSDP."
            )
        merged = module.lora_variant[adapter].merge_safe(module, adapter, merged)
    if retention is not None and _measures_merge(module, adapters):
        served = served_dtype_for(weight.dtype)
        retention.update(
            merged - base, merged.to(served).float() - base.to(served).float()
        )
        retention.modules += 1
    return merged.to(weight.dtype)


def _merged_bias(
    module,
    adapters: list[str],
    bias: torch.Tensor,
    retention: MergeRetention | None = None,
) -> torch.Tensor:
    merged = _full(bias).float()
    for adapter in adapters:
        if module.lora_bias.get(adapter):
            lora_bias = _full(module.lora_B[adapter].bias).float()
            merged = merged + lora_bias * module.scaling[adapter]
    return merged.to(bias.dtype)


def _out_of_place_merges(model) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    """Map each base parameter a LoRA layer adapts to the function that merges it."""
    from peft.tuners.lora import LoraLayer

    merges: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    for module in model.modules():
        if not isinstance(module, LoraLayer) or module.disable_adapters:
            continue
        present = set(module.lora_A) | set(module.lora_embedding_A)
        adapters = [a for a in module.active_adapters if a in present]
        if not adapters:
            continue
        if module.merged:
            raise RuntimeError(
                f"{type(module).__name__} already has its adapter merged, so its base "
                "weight would be sent with the adapter applied twice."
            )
        base_layer = module.get_base_layer()
        weight = getattr(base_layer, getattr(module, "parameter_name", "weight"), None)
        if not isinstance(weight, torch.Tensor):
            raise NotImplementedError(
                f"Merged weight sync does not support {type(module).__name__} layers."
            )
        merges[id(weight)] = functools.partial(_merged_weight, module, adapters)
        if any(module.lora_bias.get(a) for a in adapters):
            merges[id(base_layer.bias)] = functools.partial(
                _merged_bias, module, adapters
            )
    return merges


def _support_ids(
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


class AxolotlAsyncGRPOTrainer(
    CheckpointWriteRetryMixin,
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DistributedParallelMixin,
    ActivationOffloadingMixin,
    LoraMergeCheckMixin,
    AsyncGRPOTrainer,
):
    """TRL's ``AsyncGRPOTrainer`` adapted to axolotl and extended (see module doc)."""

    _tag_names = ["trl", "async-grpo", "axolotl"]
    rollout_queue: Any

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
        trainer_scorers: list[TrainerScorer] | None = None,
        **kwargs,
    ):
        if args is None:
            raise ValueError(
                "AxolotlAsyncGRPOTrainer requires an AxolotlAsyncGRPOConfig"
            )
        self._trainer_scorers = list(trainer_scorers or [])
        self._step_planner: StepPlanner | None = None
        self._trace_recorder: TraceRecorder | None = None
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
        loop_cls = (
            resolve_rollout_loop(args.rollout_loop)
            if args.rollout_loop
            else AxolotlRolloutLoop
        )
        if not reward_funcs and not (
            kwargs.get("rollout_worker")
            or kwargs.get("environment_factory")
            or loop_cls._provides_rollout_reward
            or self._trainer_scorers
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
        if self._trainer_scorers and args.reward_weights is not None:
            num_columns = (
                len(reward_funcs or [])
                + int(loop_cls._provides_rollout_reward)
                + len(self._trainer_scorers)
            )
            if len(args.reward_weights) != num_columns:
                raise ValueError(
                    f"reward_weights has {len(args.reward_weights)} entries but there "
                    f"are {num_columns} reward sources (reward functions, the rollout "
                    "loop's own reward, then the trainer-side scorers)"
                )

        replacements: dict[str, Any] = {
            "AsyncRolloutWorker": functools.partial(
                AxolotlAsyncRolloutWorker,
                loop_cls=loop_cls,
                reward_weights=args.reward_weights,
                multi_objective_aggregation=args.multi_objective_aggregation,
                scale_rewards=args.scale_rewards,
                reward_num_workers=args.reward_num_workers,
                trainer_scored=bool(self._trainer_scorers),
                **(args.rollout_loop_kwargs or {}),
            )
        }
        if args.vllm_lora_alias:
            replacements["VLLMClient"] = functools.partial(
                AliasingVLLMClient, alias=args.vllm_lora_alias
            )
        model_id = model
        self._share_prefixes = True
        hide_layer_types: contextlib.AbstractContextManager = contextlib.nullcontext()
        loaded: PreTrainedModel | None = None
        trainable: dict[str, bool] = {}
        if not isinstance(model, str):
            # TRL takes a model whose text config is not its config for a VLM and makes
            # its whole text model trainable, expecting to apply PEFT afterwards;
            # axolotl hands it the model already wrapped, trainable as it should be.
            loaded = model
            trainable = {n: p.requires_grad for n, p in model.named_parameters()}
            replacements["create_model_from_path"] = _use_loaded_model(model)
            replacements["add_fused_lm_head"] = _add_fused_lm_head_to_base
            model_id = get_config_model_id(model.config)
            self._share_prefixes = packing_support(model).shares_prefixes
            if recurrent_layer_types(model.config):
                hide_layer_types = hidden_layer_types(model)
        for scorer in self._trainer_scorers:
            scorer.share_prefixes = self._share_prefixes

        check_version()
        install_runtime_patches()
        with patched_trainer_module(**replacements), hide_layer_types:
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
        if loaded is not None:
            for name, param in loaded.named_parameters():
                param.requires_grad_(trainable[name])

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

        # TRL's set of trained groups is what it counts epochs and resumes from, so
        # it holds every consumed group; a pass over the prompts must train one.
        self._progress = RolloutProgress(
            self._trained_groups,
            epochs_done=self._epochs_consumed,
            stall_groups=len(train_dataset)
            if isinstance(train_dataset, Sized)
            else None,
        )
        self._rollouts_stopped = False
        # Both are decided from global token counts, so every rank agrees.
        self._step_trains_nothing = False
        self._weights_updated = True
        self.add_callback(_EmptyStepCallback(self))
        if self.accelerator.is_main_process:
            queue_view = self.rollout_queue  # type: ignore[has-type]
            scored_here = bool(self._trainer_scorers)
            if scored_here:
                # Advantages exist only once the trainer scored a step; `score_step`
                # takes zero-advantage and truncated samples out of the loss instead,
                # and the step planner drops groups too long for a step.
                queue_view = GroupedQueue(queue_view)
            self.rollout_queue = SampleFilterQueue(
                queue_view,
                self._progress,
                self._metrics["train"],
                model_version=lambda: self.model_version,
                max_staleness=args.max_staleness,
                token_budget=(lambda: 0)
                if scored_here
                else (lambda: max(self.args.token_budget or 0, 0)),
                skip_zero_advantage=args.skip_zero_advantage_batches
                and not scored_here,
                stop_token_ids=self._stop_token_ids()
                if args.mask_truncated_completions and not scored_here
                else None,
            )

    def _epochs_consumed(self) -> bool:
        """Whether ``num_train_epochs`` passes over the prompts were consumed."""
        target = self._epoch_stop_groups
        consumed = self._groups_before_resume + len(self._trained_groups)
        return target is not None and consumed >= target

    def _stop_token_ids(self) -> frozenset[int]:
        return frozenset([*self.eos_token_ids, self.processing_class.pad_token_id])

    def get_train_dataloader(self) -> DataLoader:
        self._trace_recorder = (
            TraceRecorder()
            if self.accelerator.is_main_process and PluginManager.get_instance().plugins
            else None
        )
        if not self._trainer_scorers:
            token_batcher, count_batcher = (
                (TokenBudgetBatcher, FixedCountBatcher)
                if self._share_prefixes
                else (ChainTokenBudgetBatcher, ChainFixedCountBatcher)
            )
            replacements: dict[str, Any] = {
                "TokenBudgetBatcher": _stop_aware(token_batcher),
                "FixedCountBatcher": _stop_aware(count_batcher),
                "DataCollatorForRollout": functools.partial(
                    TraceCollator,
                    recorder=self._trace_recorder,
                    progress=self._progress,
                ),
            }
            with patched_trainer_module(**replacements):
                return super().get_train_dataloader()
        # TRL's dataloader with the step planner in place of its batchers.
        num_processes = self.accelerator.num_processes
        steps = self.args.gradient_accumulation_steps
        if self.accelerator.is_main_process:
            dataset = RolloutQueueDataset(
                rollout_queue=self.rollout_queue,
                model_version_fn=lambda: self.model_version,
                check_health_fn=self.rollout_worker.check_health,
                stale_after_s=self.args.heartbeat_stale_after_s,
                metrics=self._metrics["train"],
                max_staleness=self.args.max_staleness,
                report_to=self.args.report_to,
            )
            self._rollout_dataset = dataset
            if self.args.token_budget is None:
                self.vllm_client.wait_for_server_ready()
                self.args.token_budget = self.vllm_client.get_max_model_len()
                LOG.info(
                    "token_budget unset; defaulting to vLLM max_model_len="
                    f"{self.args.token_budget}"
                )
            check_step_capacity(
                group_rows=self.args.num_generations,
                steps=steps,
                num_processes=num_processes,
                token_budget=self.args.token_budget,
                samples_per_row=self.args.per_device_train_batch_size,
            )
            dataset = self._step_planner = StepPlanner(
                dataset,
                num_processes=num_processes,
                steps=steps,
                token_budget=self.args.token_budget,
                samples_per_row=self.args.per_device_train_batch_size,
                scorers=self._trainer_scorers,
                metrics=self._metrics["train"],
                share_prefixes=self._share_prefixes,
                stop_token_ids=self._stop_token_ids()
                if self.args.mask_truncated_completions
                else None,
                progress=self._progress,
            )
            dataset = StopAwareDataset(dataset)
            scoring_budget = self.args.scoring_token_budget
            for scorer in self._trainer_scorers:
                scorer.token_budget = max(
                    self.args.token_budget
                    if scoring_budget is None
                    else scoring_budget,
                    0,
                )
        else:
            dataset = EmptyIterableDataset()
        return self.accelerator.prepare(
            DataLoader(
                dataset,
                batch_size=1,
                collate_fn=ScoringCollator(
                    self.processing_class.pad_token_id,
                    num_processes,
                    groups_trained=self._trained_groups,
                    metrics=self._metrics["train"],
                    token_budget=max(self.args.token_budget or 0, 0),
                    recorder=self._trace_recorder,
                    # `score_step` reports which groups trained.
                    scorers=self._trainer_scorers,
                ),
                num_workers=0,
            )
        )

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        if self._rollouts_stopped:
            return [], None
        batch_samples, num_items_in_batch = super().get_batch_samples(
            epoch_iterator, num_batches, device
        )
        for batch in batch_samples:
            if STOP_KEY in batch:
                self._stop_rollouts(int(batch[STOP_KEY].reshape(-1)[0]))
                return [], None
        scored = None
        if self._trainer_scorers and batch_samples:
            scored = score_step(self, batch_samples)
        self._step_trains_nothing = bool(batch_samples) and not any(
            float(batch["global_n_tokens"][0]) for batch in batch_samples
        )
        if self._trace_recorder is not None and batch_samples:
            self._notify_rollouts_scored(
                self._trace_recorder.pop(batch_samples), scored
            )
        return batch_samples, num_items_in_batch

    def _stop_rollouts(self, code: int) -> None:
        """End training on every rank at the rollout stream's stop micro-batch.

        The micro-batches of an optimizer step it interrupts are not trained: no
        optimizer step would follow them. See ``async_progress``.
        """
        self._rollouts_stopped = True
        if code == STOP_STALLED:
            stop = self._progress.stop if self.accelerator.is_main_process else None
            (error,) = broadcast_object_list([stop.error if stop else None])
            raise RuntimeError(error or "Async GRPO's rollout stream stalled")
        LOG.info(
            "Every prompt group of num_train_epochs is consumed, the last ones "
            "dropped before training; stopping."
        )
        self.control.should_training_stop = True
        if self.args.save_strategy == SaveStrategy.STEPS:
            self.control.should_save = True

    def _notify_rollouts_scored(self, samples, scored) -> None:
        """Hand one optimizer step's rollouts to plugins' ``on_rollouts_scored``."""
        plugins = PluginManager.get_instance()
        cfg = getattr(self, "axolotl_cfg", None) or plugins.cfg
        if cfg is None or not samples:
            return
        prompts, completions, rewards, advantages = rollout_traces(
            self.processing_class, samples, scored
        )
        plugins.on_rollouts_scored(cfg, self, prompts, completions, rewards, advantages)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # TRL's compute_loss with the loss_type-dependent loss of `_policy_loss`.
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
        support_ids = _support_ids(inputs, mask_bool)
        if support_ids is not None:
            sampler_kwargs["support_ids"] = support_ids
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
                    *negative_advantage_veto_counts(ratio, advantages, segment_id),
                ]
            )
            stats = self.accelerator.reduce(stats, reduction="sum").tolist()
            ratio_sum, kl_sum, entropy_sum, global_count = stats[:4]
            token_sums = stats[4 : 4 + len(token_means)]
            negatives, *vetoed = stats[4 + len(token_means) :]
            local_min = ratio.min() if ratio.numel() else ratio.new_tensor(float("inf"))
            ratio_min = self.accelerator.gather(local_min.float().reshape(1)).min()
            metrics = self._metrics["train"]
            # Trainer-side scoring can leave a micro-batch no trained token on any
            # rank. The count is reduced, so every rank skips these together.
            if global_count:
                metrics["ratio"].append(ratio_sum / global_count)
                metrics["kl"].append(kl_sum / global_count)
                metrics["entropy"].append(entropy_sum / global_count)
            if torch.isfinite(ratio_min):
                metrics["sampling/ratio_min"].append(ratio_min.item())
            for name, count_below in zip(NAV_THRESHOLDS, vetoed, strict=True):
                metrics[f"sampling/nav_veto_frac_{name}"].append(
                    (count_below, negatives)
                )
            if global_count:
                for name, total in zip(token_means, token_sums, strict=True):
                    metrics[name].append(total / global_count)
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

    def _sync_weight(self):
        if not self._weights_updated:
            # Only steps that trained nothing since the last sync: same policy.
            self._metrics["train"]["perf/weight_sync_skipped_total"].append(1.0)
            return
        super()._sync_weight()
        self._weights_updated = False

    def _sync_weight_merged(self, t0: float) -> tuple[float, float, float]:
        """TRL's merged sync, without folding the adapter into the trained weights.

        TRL merges the adapter in place for the send and unmerges it afterwards. In
        bf16 the round trip does not restore the base weights, so every sync nudges the
        frozen model; `_streaming_iter` streams ``W + ΔW`` computed out of place instead.
        """
        if self.accelerator.is_main_process and self.weight_transfer:
            self.weight_transfer.pause()
        t_pause = time.time()
        self.accelerator.wait_for_everyone()
        t_barrier = time.time()
        if self.accelerator.is_main_process and self.weight_transfer:
            self.weight_transfer.send_weights(self._streaming_iter())
        else:
            # The other ranks still take part in each parameter's all-gather.
            for _ in self._streaming_iter():
                pass
        t_transfer = time.time()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.weight_transfer:
                self.weight_transfer.resume()
            self.model_version += 1
            if self.rollout_worker:
                self.rollout_worker.update_model_version(self.model_version)
        return t_pause, t_barrier, t_transfer

    def _streaming_iter(self):
        model = self.accelerator.unwrap_model(self.model)
        if not is_peft_model(model):
            yield from super()._streaming_iter()
            return
        merges = _out_of_place_merges(model)
        device = self.accelerator.device
        retention = MergeRetention() if self._lora_merge_check_due() else None
        # The same walk as the weight manifest TRL sends the server at init.
        for name, param in iter_vllm_named_params(model):
            with torch.no_grad():
                merge = merges.get(id(param))
                tensor = merge(param, retention) if merge is not None else _full(param)
            yield name, tensor.to(device)
        if retention is not None:
            self._record_lora_merge_check(retention)

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
