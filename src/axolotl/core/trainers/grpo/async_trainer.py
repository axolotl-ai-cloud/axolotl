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
  capped at ``vllm_importance_sampling_cap``, instead of clipping that ratio;
- syncs merged LoRA weights without merging into the trained model;
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
import torch.nn.functional as F
from accelerate.utils import broadcast_object_list, is_peft_model
from datasets import Dataset
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TrainerCallback
from transformers.trainer_utils import SaveStrategy
from trl.experimental.async_grpo import (
    async_grpo_trainer as trl_async_module,
    packing as trl_packing,
)
from trl.experimental.async_grpo.async_grpo_config import AsyncGRPOConfig
from trl.experimental.async_grpo.async_grpo_trainer import (
    AsyncGRPOTrainer,
    FixedCountBatcher,
    RolloutQueueDataset,
    TokenBudgetBatcher,
    _EmptyIterableDataset,
)
from trl.experimental.async_grpo.packing import TREE_ATTENTION
from trl.experimental.async_grpo.vllm_client import VLLMClient
from trl.trainer.utils import add_fused_lm_head, get_config_model_id, nanmax, nanmin

from axolotl.core.trainers.grpo.async_packing import (
    ChainFixedCountBatcher,
    ChainTokenBudgetBatcher,
    hidden_layer_types,
    sample_boundaries,
    shares_prefixes,
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
from axolotl.core.trainers.mixins import (
    CheckpointWriteRetryMixin,
    DistributedParallelMixin,
    RngLoaderMixin,
    SchedulerMixin,
)
from axolotl.core.trainers.mixins.optimizer import OptimizerInitMixin, OptimizerMixin
from axolotl.core.training_args_base import AxolotlTrainingMixins
from axolotl.integrations.base import PluginManager
from axolotl.monkeypatch.models.recurrent_support import recurrent_layer_types
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

LOSS_TYPES = ("dapo", "cispo")
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
            "ratio, bounded at vllm_importance_sampling_cap, and clips nothing."
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
    vllm_lora_alias: str | None = field(
        default=None,
        metadata={
            "help": "With adapter-only weight sync, also serve each synced adapter "
            "under this fixed name, for clients that address vLLM by a fixed model "
            "name. Syncs then drain in-flight requests before switching."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.rollout_loop is not None:
            resolve_rollout_loop(self.rollout_loop)
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


_trl_tree_flash_attn = trl_packing._flash_attn
_trl_flops_per_token = trl_async_module.compute_flops_per_token


def _flops_per_token(config, seq_len: int) -> int:
    """TRL's per-token FLOPs estimate (for its MFU metrics), dense where it cannot
    read a MoE config.

    TRL treats any config with ``num_experts_per_tok`` as Qwen-MoE-like and reads
    ``decoder_sparse_step``; hybrids such as Nemotron-H have the one without the
    other, which crashed the first step's metrics.
    """
    try:
        return _trl_flops_per_token(config, seq_len)
    except AttributeError as err:
        LOG.warning_once(
            f"MFU uses a dense estimate: TRL cannot read {type(config).__name__}'s "
            f"MoE layout ({err})"
        )
        return _trl_flops_per_token(_DenseView(config), seq_len)


class _DenseView:
    """``config`` read as a dense model (configs are strict dataclasses)."""

    num_experts_per_tok = None

    def __init__(self, config):
        self._config = config

    def __getattr__(self, name):
        return getattr(self._config, name)


@functools.cache
def _tree_flash_attn():
    """TRL's FA3 kernel, or FA2's where FA3 has no build for the GPU (Blackwell)."""
    try:
        return _trl_tree_flash_attn()
    except RuntimeError as err:
        LOG.warning(
            f"Tree attention uses flash-attn2: flash-attn3 failed to load: {err}"
        )
        from kernels import get_kernel

        # Same varlen API, and it aligns the causal mask bottom-right like FA3.
        return _WideValues(get_kernel("kernels-community/flash-attn2", version=3))


class _WideValues:
    """FA2's varlen kernel needs values as wide as the queries, which FA3 (and
    MLA's narrower value heads) do not; pad them and cut the output back."""

    def __init__(self, kernel):
        self.kernel = kernel

    def flash_attn_varlen_func(self, query, key, value, *args, **kwargs):
        width = value.shape[-1]
        if width < query.shape[-1]:
            value = F.pad(value, (0, query.shape[-1] - width))
        out = self.kernel.flash_attn_varlen_func(query, key, value, *args, **kwargs)
        if isinstance(out, tuple):
            return (out[0][..., :width], *out[1:])
        return out[..., :width]


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


def _merged_weight(module, adapters: list[str], weight: torch.Tensor) -> torch.Tensor:
    """``weight`` with ``adapters`` folded in, leaving ``weight`` itself untouched."""
    merged = _full(weight).float()
    for adapter in adapters:
        if adapter not in module.lora_variant:
            merged = merged + _full(module.get_delta_weight(adapter)).float()
            continue
        # Variants (DoRA) rebuild the merge from the module's own sharded weights.
        if isinstance(weight, DTensor) or len(adapters) > 1:
            raise NotImplementedError(
                f"Merged weight sync of a {type(module.lora_variant[adapter]).__name__} "
                "adapter is only supported for a single adapter without FSDP."
            )
        merged = module.lora_variant[adapter].merge_safe(module, adapter, merged)
    return merged.to(weight.dtype)


def _merged_bias(module, adapters: list[str], bias: torch.Tensor) -> torch.Tensor:
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


class AxolotlAsyncGRPOTrainer(
    CheckpointWriteRetryMixin,
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DistributedParallelMixin,
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
            self._share_prefixes = shares_prefixes(model)
            if recurrent_layer_types(model.config):
                hide_layer_types = hidden_layer_types(model)
        for scorer in self._trainer_scorers:
            scorer.share_prefixes = self._share_prefixes

        trl_packing._flash_attn = _tree_flash_attn
        trl_async_module.compute_flops_per_token = _flops_per_token
        with _patched_trl_module(**replacements), hide_layer_types:
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
            with _patched_trl_module(**replacements):
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
            dataset = _EmptyIterableDataset()
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
                    progress=self._progress,
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
        tree_kwargs = {
            key: inputs[key][inputs[key] >= 0]
            for key in ("tree_kv_index", "tree_cu_seqlens_q", "tree_cu_seqlens_k")
        }
        tree_kwargs["tree_max_seqlen_q"], tree_kwargs["tree_max_seqlen_k"] = inputs[
            "tree_max_seqlens"
        ][0].tolist()
        if not self._share_prefixes:
            tree_kwargs.update(sample_boundaries(tree_kwargs))

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

        per_token_loss, ratio, log_ratio = self._policy_loss(
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
            cispo = self.args.loss_type == "cispo"
            if cispo:
                bounded = [ratio > self.args.vllm_importance_sampling_cap]
            else:
                is_low_clipped = (ratio < 1 - self.epsilon_low) & (advantages < 0)
                is_high_clipped = (ratio > 1 + self.epsilon_high) & (advantages > 0)
                bounded = [
                    is_low_clipped,
                    is_high_clipped,
                    is_low_clipped | is_high_clipped,
                ]
            count = torch.full_like(ratio.sum(), float(log_probs.numel()))
            stats = torch.stack(
                [
                    ratio.sum(),
                    ((ratio - 1) - log_ratio).sum(),
                    entropy.sum(),
                    count,
                    *(mask.float().sum() for mask in bounded),
                    *negative_advantage_veto_counts(ratio, advantages, segment_id),
                ]
            )
            stats = self.accelerator.reduce(stats, reduction="sum").tolist()
            ratio_sum, kl_sum, entropy_sum, global_count = stats[:4]
            bounded_sums = stats[4 : 4 + len(bounded)]
            negatives, *vetoed = stats[4 + len(bounded) :]
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
            if cispo:
                if global_count:
                    metrics["sampling/capped_token_frac"].append(
                        bounded_sums[0] / global_count
                    )
            else:
                if global_count:
                    for name, clipped in zip(
                        ("low_mean", "high_mean", "region_mean"),
                        bounded_sums,
                        strict=True,
                    ):
                        metrics[f"clip_ratio/{name}"].append(clipped / global_count)

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
        # The same walk as the weight manifest TRL sends the server at init.
        for name, param in trl_async_module._iter_vllm_named_params(model):
            with torch.no_grad():
                merge = merges.get(id(param))
                tensor = merge(param) if merge is not None else _full(param)
            yield name, tensor.to(device)

    def _policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
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
        (``token_mask``).
        """
        log_ratio = log_probs - old_log_probs
        if self.args.loss_type == "dapo":
            coef_1 = torch.exp(log_ratio)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
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
