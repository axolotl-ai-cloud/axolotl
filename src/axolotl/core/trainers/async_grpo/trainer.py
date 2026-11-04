"""Axolotl's async GRPO trainer on TRL's ``AsyncGRPOTrainer`` (see the package doc)."""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Sized
from typing import Any

import torch
from accelerate.utils import broadcast_object_list, is_peft_model
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TrainerCallback
from transformers.trainer_utils import SaveStrategy
from tree_recurrence.integrations.trl import (
    ChainFixedCountBatcher,
    ChainTokenBudgetBatcher,
    hidden_layer_types,
)

from axolotl.core.trainers.async_grpo.config import AxolotlAsyncGRPOConfig
from axolotl.core.trainers.async_grpo.loss import AsyncGRPOLossMixin
from axolotl.core.trainers.async_grpo.progress import (
    STOP_KEY,
    STOP_STALLED,
    RolloutProgress,
    SampleFilterQueue,
    StopAwareDataset,
)
from axolotl.core.trainers.async_grpo.rollout import (
    AxolotlAsyncRolloutWorker,
    AxolotlRolloutLoop,
    resolve_rollout_loop,
)
from axolotl.core.trainers.async_grpo.scoring import (
    GroupedQueue,
    ScoringCollator,
    StepPlanner,
    TrainerScorer,
    check_step_capacity,
    score_step,
)
from axolotl.core.trainers.async_grpo.traces import (
    TraceCollator,
    TraceRecorder,
    rollout_traces,
)
from axolotl.core.trainers.async_grpo.trl_compat import (
    TREE_ATTENTION,
    AsyncGRPOTrainer,
    EmptyIterableDataset,
    FixedCountBatcher,
    RolloutQueueDataset,
    TokenBudgetBatcher,
    add_fused_lm_head,
    check_version,
    get_config_model_id,
    install_runtime_patches,
    patched_trainer_module,
)
from axolotl.core.trainers.async_grpo.weight_sync import (
    AliasingVLLMClient,
    WeightSyncMixin,
)
from axolotl.core.trainers.mixins import (
    ActivationOffloadingMixin,
    CheckpointWriteRetryMixin,
    DistributedParallelMixin,
    RngLoaderMixin,
    SchedulerMixin,
)
from axolotl.core.trainers.mixins.optimizer import OptimizerInitMixin, OptimizerMixin
from axolotl.integrations.base import PluginManager
from axolotl.monkeypatch.models.recurrent_support import recurrent_layer_types
from axolotl.monkeypatch.models.tree_packing import packing_support
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


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


class AxolotlAsyncGRPOTrainer(
    CheckpointWriteRetryMixin,
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DistributedParallelMixin,
    ActivationOffloadingMixin,
    WeightSyncMixin,
    AsyncGRPOLossMixin,
    AsyncGRPOTrainer,
):
    """TRL's ``AsyncGRPOTrainer`` adapted to axolotl and extended (see the package doc)."""

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
        optimizer step would follow them. See ``async_grpo.progress``.
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
