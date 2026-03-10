"""
Async GRPO training with streaming scoring and IS correction.

Works on stock TRL v0.29.0 and transformers v5.3.0 — no custom branches needed.

Features:
  - Async prefetch: background thread generates completions via vLLM while the main
    thread trains on the previous rollout.
  - Deferred scoring: rewards, advantages, and policy logprobs computed on the main
    thread (thread-safe with GPU forward passes).
  - Streaming group scoring: scores prompt groups incrementally so that reward
    computation overlaps with the next group's logprob computation.
  - Importance sampling (IS) correction: corrects for stale vLLM weights.
  - Off-Policy Sequence Mask (OPSM): drops sequences with high KL + negative advantage.
  - Configurable vLLM weight sync interval.

Classes exported:
  - AsyncGRPOConfig: GRPOConfig extended with async/streaming/IS fields
  - AsyncGRPOTrainer: GRPOTrainer with async prefetch and IS correction
  - ProducerConfig, DataProducer, BaseDataProducer, AsyncDataProducer: data producer protocol
"""

import atexit
import concurrent.futures
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from trl.trainer import GRPOConfig, GRPOTrainer
from trl.trainer.utils import (
    RepeatSampler,
    nanmax,
    nanmin,
    nanstd,
    pad,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)

try:
    from trl.data_utils import (
        apply_chat_template,
        is_conversational,
        prepare_multimodal_messages,
    )
except ImportError:
    from trl.chat_template_utils import apply_chat_template
    from trl.data_utils import is_conversational, prepare_multimodal_messages

try:
    from trl.models.utils import disable_gradient_checkpointing
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def disable_gradient_checkpointing(model, kwargs):
        yield


try:
    from accelerate.utils import gather_object
except ImportError:
    gather_object = None

try:
    from peft import PeftModel
    from trl.trainer.utils import use_adapter
except ImportError:
    PeftModel = None
    use_adapter = nullcontext


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AsyncGRPOConfig(GRPOConfig):
    """GRPOConfig extended with async prefetch, streaming scoring, and IS correction fields.

    Fields already present in stock GRPOConfig (e.g. ``importance_sampling_level``,
    ``multi_objective_aggregation``) are listed here for safety: if the stock version
    does not define them, the defaults below ensure everything works.
    """

    # --- Data producer ---
    use_data_producer: bool = field(
        default=False,
        metadata={
            "help": "Use the GRPODataProducer protocol for online data generation."
        },
    )

    # --- Async data production ---
    async_prefetch: bool = field(
        default=False,
        metadata={
            "help": "Generate rollouts in a background thread while training on the previous rollout."
        },
    )
    prefetch_depth: int = field(
        default=1,
        metadata={"help": "Number of rollouts to prefetch ahead of training."},
    )
    vllm_sync_interval: int = field(
        default=1,
        metadata={
            "help": "Sync model weights to vLLM every N optimizer steps (async mode only)."
        },
    )

    # --- Streaming scoring ---
    streaming_partial_batch: bool = field(
        default=False,
        metadata={
            "help": "Score prompt groups incrementally instead of the full batch at once."
        },
    )
    streaming_min_groups: int = field(
        default=1,
        metadata={"help": "Minimum prompt groups to score per streaming chunk."},
    )

    # --- vLLM importance sampling correction ---
    vllm_importance_sampling_correction: bool = field(
        default=True,
        metadata={
            "help": "Apply IS correction for distribution mismatch between vLLM and training model."
        },
    )
    vllm_importance_sampling_mode: str = field(
        default="token_truncate",
        metadata={
            "help": "IS mode: token_truncate, token_mask, sequence_truncate, or sequence_mask."
        },
    )
    vllm_importance_sampling_cap: float = field(
        default=3.0,
        metadata={"help": "Cap C for IS ratio clipping/masking."},
    )

    # --- Off-policy sequence mask (OPSM) ---
    off_policy_mask_threshold: float | None = field(
        default=None,
        metadata={"help": "KL threshold for OPSM (DeepSeek-V3.2). None = disabled."},
    )

    # --- Bias-corrected KL ---
    use_bias_correction_kl: bool = field(
        default=False,
        metadata={"help": "Apply IS correction to KL divergence term."},
    )


# ---------------------------------------------------------------------------
# Data Producer Protocol (standalone — no transformers branch needed)
# ---------------------------------------------------------------------------

_dp_logger = logging.getLogger(__name__ + ".data_producer")


@dataclass
class ProducerConfig:
    """Configuration for a :class:`DataProducer`.

    Args:
        mini_epochs: Number of training passes over each produced dataset.
        max_rollouts: Maximum number of produce-then-train rounds (None = unlimited).
        steps_per_generation: Optimisation steps per produced dataset before regenerating.
        num_iterations: Number of times to reuse each generation across optimisation steps.
        async_prefetch: Produce the next dataset in a background thread.
        prefetch_depth: How many rollouts to queue ahead when async.
        sync_warmup_rollouts: Initial on-policy rollouts before switching to async.
        eval_during_produce: Switch model to eval() during produce().
        empty_cache_before_produce: torch.cuda.empty_cache() before produce().
        empty_cache_after_produce: torch.cuda.empty_cache() after produce().
    """

    mini_epochs: int = 1
    max_rollouts: int | None = None
    steps_per_generation: int | None = None
    num_iterations: int = 1
    async_prefetch: bool = False
    prefetch_depth: int = 1
    sync_warmup_rollouts: int = 0
    eval_during_produce: bool = True
    empty_cache_before_produce: bool = False
    empty_cache_after_produce: bool = False

    def __post_init__(self):
        if self.mini_epochs < 1:
            raise ValueError(f"mini_epochs must be >= 1, got {self.mini_epochs}")
        if self.max_rollouts is not None and self.max_rollouts < 1:
            raise ValueError(
                f"max_rollouts must be >= 1 or None, got {self.max_rollouts}"
            )
        if self.num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {self.num_iterations}")
        if self.steps_per_generation is not None and self.steps_per_generation < 1:
            raise ValueError(
                f"steps_per_generation must be >= 1 or None, got {self.steps_per_generation}"
            )
        if self.prefetch_depth < 1:
            raise ValueError(f"prefetch_depth must be >= 1, got {self.prefetch_depth}")
        if self.sync_warmup_rollouts < 0:
            raise ValueError(
                f"sync_warmup_rollouts must be >= 0, got {self.sync_warmup_rollouts}"
            )


class DataProducer(ABC):
    """Abstract base class for online data producers.

    Subclass this and implement :meth:`produce` to supply fresh training data
    each rollout round.
    """

    config: ProducerConfig

    @abstractmethod
    def produce(
        self,
        model: Any,
        global_step: int,
        *,
        processing_class: Any = None,
        accelerator: Any = None,
        args: Any = None,
        **kwargs,
    ) -> Dataset:
        """Generate a fresh training dataset."""
        ...


class BaseDataProducer(DataProducer):
    """Convenience base class with a default :class:`ProducerConfig` and lifecycle hooks."""

    def __init__(self, config: ProducerConfig | None = None):
        self.config = config or ProducerConfig()

    def on_rollout_begin(self, global_step: int) -> None:
        """Called before each produce() invocation."""

    def on_rollout_end(self, dataset: Dataset, global_step: int) -> None:
        """Called after each produce() invocation with the produced dataset."""


class AsyncDataProducer:
    """Wraps a synchronous :class:`DataProducer` for background-thread data generation.

    While the Trainer trains on the current rollout, this wrapper produces upcoming
    datasets in a background thread.
    """

    def __init__(
        self, inner: DataProducer, background_produce_kwargs: dict | None = None
    ):
        self._inner = inner
        self._depth = inner.config.prefetch_depth
        self._warmup_remaining = inner.config.sync_warmup_rollouts
        self._background_kwargs = background_produce_kwargs or {}
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="async-producer"
        )
        self._queue: deque[concurrent.futures.Future] = deque()
        self._initialized = False

    @property
    def config(self) -> ProducerConfig:
        return self._inner.config

    def produce(self, model: Any, global_step: int, **kwargs) -> Dataset:
        """Return the next dataset, blocking if the prefetch hasn't finished."""
        # During warmup, produce synchronously (on-policy)
        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            _dp_logger.info(
                f"AsyncDataProducer: sync warmup rollout (remaining={self._warmup_remaining})"
            )
            return self._inner.produce(model, global_step, **kwargs)

        if not self._initialized:
            dataset = self._inner.produce(model, global_step, **kwargs)
            bg_kwargs = {**kwargs, **self._background_kwargs}
            for i in range(1, self._depth + 1):
                self._queue.append(
                    self._executor.submit(
                        self._inner.produce, model, global_step + i, **bg_kwargs
                    )
                )
            self._initialized = True
            return dataset

        dataset = self._queue.popleft().result()
        bg_kwargs = {**kwargs, **self._background_kwargs}
        next_step = global_step + self._depth
        self._queue.append(
            self._executor.submit(self._inner.produce, model, next_step, **bg_kwargs)
        )
        return dataset

    def on_rollout_begin(self, global_step: int) -> None:
        if hasattr(self._inner, "on_rollout_begin"):
            self._inner.on_rollout_begin(global_step)

    def on_rollout_end(self, dataset: Dataset, global_step: int) -> None:
        if hasattr(self._inner, "on_rollout_end"):
            self._inner.on_rollout_end(dataset, global_step)

    def shutdown(self) -> None:
        """Shut down the background thread pool and cancel pending futures."""
        for future in self._queue:
            future.cancel()
        self._queue.clear()
        self._executor.shutdown(wait=False)


class DataProducerCallback:
    """Marker class: if a DataProducer also inherits from this, the Trainer will
    automatically register it as a callback."""

    pass


# ---------------------------------------------------------------------------
# RolloutDataset + GRPODataProducer
# ---------------------------------------------------------------------------


class RolloutDataset(Dataset):
    """A Dataset wrapping the output dict from _generate_and_score_completions.

    Per-sample tensors are sliced by index; shared metadata is passed through.
    """

    _ALWAYS_SHARED = frozenset({"num_items_in_batch", "_pending_policy_logps"})

    def __init__(self, data: dict[str, Any]):
        self._data = data
        self._shared_keys: set[str] = set()
        self._sample_keys: set[str] = set()

        for key, val in data.items():
            if key in self._ALWAYS_SHARED:
                self._shared_keys.add(key)
            elif not isinstance(val, torch.Tensor):
                self._shared_keys.add(key)
            elif val.dim() == 0:
                self._shared_keys.add(key)
            else:
                self._sample_keys.add(key)

        self._num_samples = 0
        for key in self._sample_keys:
            n = data[key].size(0)
            if self._num_samples == 0:
                self._num_samples = n
            elif n != self._num_samples:
                raise ValueError(
                    f"Inconsistent sample count: key '{key}' has {n}, expected {self._num_samples}"
                )
        if self._num_samples == 0:
            raise ValueError("No per-sample tensors found in rollout data")

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {}
        for key in self._sample_keys:
            item[key] = self._data[key][idx]
        for key in self._shared_keys:
            item[key] = self._data[key]
        return item


def make_rollout_collator(shared_keys: set[str]):
    """Return a collator that stacks per-sample tensors and passes shared keys through."""

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in batch[0]:
            if key in shared_keys:
                result[key] = batch[0][key]
            else:
                values = [item[key] for item in batch]
                if isinstance(values[0], torch.Tensor):
                    result[key] = torch.stack(values)
                else:
                    result[key] = values
        return result

    return _collate


class GRPODataProducer(BaseDataProducer):
    """Produces GRPO training rollouts using the trainer's generation pipeline.

    Created before Trainer.__init__ completes; the trainer reference is injected
    later via set_trainer().
    """

    def __init__(
        self,
        config: ProducerConfig,
        prompt_dataset,
        *,
        num_generations: int,
        generation_batch_size: int,
        train_batch_size: int,
        steps_per_generation: int,
        shuffle_dataset: bool,
        seed: int,
    ):
        super().__init__(config)
        self._dataset = prompt_dataset
        self._num_generations = num_generations
        self._generation_batch_size = generation_batch_size
        self._train_batch_size = train_batch_size
        self._steps_per_generation = steps_per_generation
        self._shuffle_dataset = shuffle_dataset
        self._seed = seed
        self._trainer = None
        self._prompt_dl: DataLoader | None = None
        self._prompt_iter = None

    def set_trainer(self, trainer) -> None:
        """Inject the live trainer reference and create the prompt DataLoader."""
        self._trainer = trainer
        self._init_prompt_dataloader()

    def _init_prompt_dataloader(self) -> None:
        from functools import partial

        from transformers.trainer_utils import seed_worker

        trainer = self._trainer
        sampler = RepeatSampler(
            data_source=self._dataset,
            mini_repeat_count=self._num_generations,
            batch_size=self._generation_batch_size // self._num_generations,
            repeat_count=1,
            shuffle=self._shuffle_dataset,
            seed=self._seed,
        )

        # Use identity collator (same as stock GRPOTrainer)
        def _identity(x):
            return x

        dl = DataLoader(
            self._dataset,
            batch_size=self._train_batch_size * self._steps_per_generation,
            sampler=sampler,
            collate_fn=_identity,
            num_workers=trainer.args.dataloader_num_workers,
            pin_memory=trainer.args.dataloader_pin_memory,
            persistent_workers=trainer.args.dataloader_persistent_workers,
            worker_init_fn=partial(
                seed_worker,
                num_workers=trainer.args.dataloader_num_workers,
                rank=trainer.args.process_index,
            ),
        )
        self._prompt_dl = trainer.accelerator.prepare(dl)

        # Don't let accelerator track this dataloader
        acc_dls = trainer.accelerator._dataloaders
        if self._prompt_dl in acc_dls:
            acc_dls.remove(self._prompt_dl)

        self._prompt_iter = iter(self._prompt_dl)

    def produce(
        self,
        model: Any,
        global_step: int,
        *,
        skip_policy_logps: bool = False,
        processing_class: Any = None,
        accelerator: Any = None,
        args: Any = None,
        **kwargs,
    ) -> RolloutDataset:
        """Generate a fresh GRPO training rollout."""
        try:
            inputs = next(self._prompt_iter)
        except StopIteration:
            self._prompt_iter = iter(self._prompt_dl)
            inputs = next(self._prompt_iter)

        if skip_policy_logps:
            # Async path: use _generate_only (generation without scoring) which
            # works on stock TRL (no skip_policy_logps parameter needed).
            output = self._trainer._generate_only(inputs)
        else:
            # Sync path: full generation + scoring
            output = self._trainer._generate_and_score_completions(inputs)

            # Strip non-sequence metadata before shuffling
            metadata = {}
            for key in list(output.keys()):
                val = output[key]
                if not isinstance(val, (torch.Tensor, list)):
                    metadata[key] = output.pop(key)
                elif isinstance(val, torch.Tensor) and val.dim() == 0:
                    metadata[key] = output.pop(key)

            output = shuffle_sequence_dict(output)
            output.update(metadata)

        return RolloutDataset(output)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class AsyncGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with async prefetch, streaming scoring, and IS correction.

    Drop-in replacement: pass ``AsyncGRPOConfig`` as ``args`` and use this trainer
    instead of ``GRPOTrainer``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure custom attributes exist (stock GRPOTrainer.__init__ may not set them).
        for attr, cfg_key, default in [
            (
                "vllm_importance_sampling_correction",
                "vllm_importance_sampling_correction",
                True,
            ),
            (
                "vllm_importance_sampling_mode",
                "vllm_importance_sampling_mode",
                "token_truncate",
            ),
            ("vllm_importance_sampling_cap", "vllm_importance_sampling_cap", 3.0),
            ("off_policy_mask_threshold", "off_policy_mask_threshold", None),
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, getattr(self.args, cfg_key, default))

        # Async state
        self._async_queue: queue.Queue | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._prompt_iter = None
        self._last_synced_step = -1
        self._buffered_inputs: list | None = None  # override stock attr
        self._current_train_step_time = 0.0

        # Data producer (the proper architecture for async generation)
        self.data_producer = None
        if getattr(self.args, "use_data_producer", False):
            self.data_producer = self._create_data_producer(kwargs["args"], kwargs["train_dataset"])

        if self.args.async_prefetch and self.data_producer is None:
            # Legacy path: direct _prepare_inputs override without data producer
            self._setup_async()

    def _create_data_producer(self, args, train_dataset):
        """Create and return the GRPODataProducer (possibly wrapped in AsyncDataProducer)."""
        producer_config = ProducerConfig(
            mini_epochs=args.num_iterations,
            max_rollouts=None,
            eval_during_produce=False,
            empty_cache_before_produce=True,
            empty_cache_after_produce=True,
            async_prefetch=args.async_prefetch,
            prefetch_depth=args.prefetch_depth,
        )
        data_producer = GRPODataProducer(
            config=producer_config,
            prompt_dataset=train_dataset,
            num_generations=self.num_generations,
            generation_batch_size=args.generation_batch_size,
            train_batch_size=args.per_device_train_batch_size,
            steps_per_generation=args.steps_per_generation,
            shuffle_dataset=getattr(self, "shuffle_dataset", True),
            seed=args.seed,
        )
        data_producer.set_trainer(self)

        if args.async_prefetch:
            data_producer = AsyncDataProducer(
                data_producer,
                background_produce_kwargs={"skip_policy_logps": True},
            )
        return data_producer

    # ------------------------------------------------------------------
    # Async setup / teardown
    # ------------------------------------------------------------------

    def _setup_async(self):
        """Create background thread pool, prompt iterator, and pre-fill the async queue."""
        gen_batch_size = getattr(
            self.args,
            "generation_batch_size",
            self._train_batch_size * self.args.gradient_accumulation_steps,
        )
        # RepeatSampler groups prompts with num_generations repetitions each.
        # DataLoader batches the yielded indices into generation-sized batches.
        sampler = RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=gen_batch_size // self.num_generations,
            repeat_count=10_000,  # effectively infinite
            shuffle=True,
            seed=self.args.seed,
        )
        self._prompt_dataloader = DataLoader(
            self.train_dataset,
            batch_size=gen_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=0,
        )
        self._prompt_iter = iter(self._prompt_dataloader)
        self._async_queue = queue.Queue(maxsize=self.args.prefetch_depth)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Pre-submit generations to fill the queue
        for _ in range(self.args.prefetch_depth):
            self._submit_generation()

        atexit.register(self._shutdown_async)

    def _shutdown_async(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _submit_generation(self):
        """Submit the next background generation job."""
        batch = next(self._prompt_iter)
        future = self._executor.submit(self._generate_only, batch)
        self._async_queue.put(future)

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def _maybe_sync_vllm_weights(self):
        """Sync model weights to vLLM if the interval has elapsed."""
        if not (self.use_vllm and self.args.async_prefetch):
            return
        step = self.state.global_step
        interval = self.args.vllm_sync_interval
        if step != self._last_synced_step and step % interval == 0:
            # Wait for in-flight futures to complete (they reference old weights)
            if self._async_queue is not None:
                pending = list(self._async_queue.queue)
                for f in pending:
                    if isinstance(f, concurrent.futures.Future):
                        f.result()
            self.vllm_generation.sync_weights()
            self._last_synced_step = step

    # ------------------------------------------------------------------
    # Background-thread generation (no scoring)
    # ------------------------------------------------------------------

    def _generate_single_turn(self, prompts, **kwargs):
        """Override to prevent weight sync from background thread."""
        is_bg = threading.current_thread() is not threading.main_thread()
        saved_step = None

        if is_bg and self.use_vllm:
            # Trick: match _last_loaded_step so the stock sync check is a no-op
            saved_step = getattr(self, "_last_loaded_step", None)
            self._last_loaded_step = self.state.global_step

        try:
            return super()._generate_single_turn(prompts, **kwargs)
        finally:
            if saved_step is not None:
                self._last_loaded_step = saved_step

    def _generate_only(self, inputs):
        """Generate completions without scoring.  Runs on background thread.

        Mirrors the first half of ``_generate_and_score_completions`` (prompt
        extraction → vLLM generation → tensor padding) and returns a deferred
        output dict for main-thread scoring.

        Args:
            inputs: list of dicts (one per sample), as yielded by the DataLoader
                    with ``identity`` collate_fn.
        """
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]

        # --- Handle images (multimodal) ---
        if "images" in inputs[0]:
            images = [ex.get("images") for ex in inputs]
        elif "image" in inputs[0]:
            images = [
                [ex.get("image")] if ex.get("image") is not None else None
                for ex in inputs
            ]
        else:
            images = None
        if images is not None and all(img == [] for img in images):
            images = None

        if images is not None:
            if not is_conversational(inputs[0]):
                raise ValueError("Multimodal training requires conversational prompts.")
            prompts = [
                prepare_multimodal_messages(p, il)
                for p, il in zip(prompts, images, strict=True)
            ]

        # --- Generate completions ---
        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)

        # --- Pad to tensors ---
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids, padding_value=self.pad_token_id, padding_side="left"
        )
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")

        completion_ids = [
            torch.tensor(ids, device=device) for ids in completion_ids_list
        ]
        completion_mask = [
            torch.ones_like(ids, dtype=torch.long) for ids in completion_ids
        ]
        completion_ids = pad(
            completion_ids, padding_value=self.pad_token_id, padding_side="right"
        )
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        if sampling_per_token_logps_list is not None:
            sampling_logps = [
                torch.tensor(lp, device=device) for lp in sampling_per_token_logps_list
            ]
            sampling_per_token_logps = pad(
                sampling_logps, padding_value=0.0, padding_side="right"
            )
        else:
            sampling_per_token_logps = None

        if tool_mask_list is not None:
            tool_mask = [torch.tensor(m, device=device) for m in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")
        else:
            tool_mask = None

        # --- Mask truncated completions ---
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_trunc = torch.tensor(
                [ids[-1] not in eos_and_pad for ids in completion_ids_list],
                device=device,
            )
            completion_mask = completion_mask * (~is_trunc).unsqueeze(1).int()
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_trunc).unsqueeze(1).int()

        # --- Multimodal forward kwargs ---
        num_images = [len(il) for il in images] if images is not None else None
        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": p},
                    self.processing_class,
                    tools=self.tools,
                    **self.chat_template_kwargs,
                )["prompt"]
                for p in prompts
            ]
            prompt_inputs = self.processing_class(
                images=images, text=prompts_text, padding=True, return_tensors="pt"
            )
            forward_kwargs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in prompt_inputs.items()
                if k not in ("input_ids", "attention_mask")
            }
        else:
            forward_kwargs = {}

        # Extend token_type_ids / mm_token_type_ids for completion tokens
        for ttid_key in ("token_type_ids", "mm_token_type_ids"):
            if ttid_key in forward_kwargs:
                tt = forward_kwargs[ttid_key]
                forward_kwargs[ttid_key] = torch.cat(
                    [tt, tt.new_zeros(completion_ids.shape)], dim=1
                )

        # Merge extra_fields from rollout_func into inputs
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Sync CUDA before crossing thread boundary
        torch.cuda.synchronize()

        # --- Construct deferred output ---
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "num_items_in_batch": num_items_in_batch,
            "advantages": torch.zeros(completion_ids.size(0), device=device),
            # Sentinels for deferred scoring
            "_pending_policy_logps": True,
            "_deferred_inputs": inputs,
            "_deferred_prompts": prompts,
            "_deferred_completions": completions,
            "_deferred_completion_ids_list": completion_ids_list,
        }
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if tool_mask is not None:
            output["tool_mask"] = tool_mask
        if images is not None:
            output["num_images"] = num_images
        for k in (
            "pixel_values",
            "image_grid_thw",
            "pixel_attention_mask",
            "image_sizes",
            "token_type_ids",
            "mm_token_type_ids",
        ):
            if k in forward_kwargs:
                output[k] = forward_kwargs[k]
        return output

    # ------------------------------------------------------------------
    # Hooks (overridden by subclasses like FastAsyncGRPOTrainer)
    # ------------------------------------------------------------------

    def _compute_rewards_for_batch(self, inputs, prompts, completions, completion_ids_list):
        """Compute rewards for a batch. Override for parallel workers, caching, etc."""
        return self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

    def _post_advantage_hook(
        self,
        data: dict,
        rewards_per_func,
        advantages,
        inputs: list,
        num_generations: int,
        mode: str,
        s_start: int | None = None,
        s_end: int | None = None,
        is_last_chunk: bool = True,
    ) -> None:
        """Called after advantages are computed. Override for replay buffer, re-roll, etc."""

    # ------------------------------------------------------------------
    # Main-thread scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_deferred_scores(self, rollout: dict) -> dict:
        """Compute rewards, advantages, policy logprobs, and IS ratio on the main thread.

        Takes the deferred output from ``_generate_only`` and produces a fully
        scored dict ready for ``split_tensor_dict`` → micro-batches.
        """
        device = self.accelerator.device
        batch_size = self.args.per_device_train_batch_size
        num_generations = self.num_generations
        mode = "train"

        # --- Extract deferred data ---
        data = rollout
        inputs = data.pop("_deferred_inputs")
        prompts = data.pop("_deferred_prompts")
        completions = data.pop("_deferred_completions")
        completion_ids_list = data.pop("_deferred_completion_ids_list")
        del data["_pending_policy_logps"]

        prompt_ids = data["prompt_ids"]
        completion_ids = data["completion_ids"]
        prompt_mask = data["prompt_mask"]
        completion_mask = data["completion_mask"]
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Multimodal forward kwargs
        forward_kwargs = {}
        for key in (
            "pixel_values",
            "image_grid_thw",
            "pixel_attention_mask",
            "image_sizes",
            "token_type_ids",
            "mm_token_type_ids",
        ):
            if key in data:
                forward_kwargs[key] = data[key]
        num_images = data.get("num_images")

        # --- Policy logprobs ---
        logprob_batch_size = min(batch_size * 4, len(prompt_ids))
        with disable_gradient_checkpointing(
            self.model, self.args.gradient_checkpointing_kwargs
        ):
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm
                and getattr(self, "vllm_importance_sampling_correction", False)
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    logprob_batch_size,
                    num_images=num_images,
                    **forward_kwargs,
                )
                data["old_per_token_logps"] = old_per_token_logps
            else:
                old_per_token_logps = None

            # Reference model logprobs
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                        num_images=num_images,
                        **forward_kwargs,
                    )
                else:
                    model = self.accelerator.unwrap_model(self.model)
                    adapter_name = (
                        "ref"
                        if hasattr(model, "peft_config") and "ref" in model.peft_config
                        else None
                    )
                    with use_adapter(model, adapter_name=adapter_name):
                        ref_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size,
                            num_images=num_images,
                            **forward_kwargs,
                        )
                data["ref_per_token_logps"] = ref_logps

        # --- IS ratio ---
        if (
            self.use_vllm
            and getattr(self, "vllm_importance_sampling_correction", False)
            and old_per_token_logps is not None
            and "sampling_per_token_logps" in data
        ):
            sampling_logps = data["sampling_per_token_logps"]
            is_mask = (
                completion_mask
                if "tool_mask" not in data
                else completion_mask * data["tool_mask"]
            )
            per_token_logps_diff = (old_per_token_logps - sampling_logps) * is_mask

            is_mode = getattr(self, "vllm_importance_sampling_mode", "token_truncate")
            is_cap = getattr(self, "vllm_importance_sampling_cap", 3.0)
            sequence_level_is = is_mode in ("sequence_mask", "sequence_truncate")
            if sequence_level_is:
                logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
            else:
                logps_diff = per_token_logps_diff

            is_ratio = torch.exp(logps_diff)
            if is_mode in ("sequence_truncate", "token_truncate"):
                is_ratio = torch.clamp(is_ratio, max=is_cap)
            elif is_mode in ("sequence_mask", "token_mask"):
                is_ratio = is_ratio.masked_fill(is_ratio > is_cap, value=0.0)
            data["importance_sampling_ratio"] = is_ratio

        # --- Rewards ---
        rewards_per_func = self._compute_rewards_for_batch(
            inputs, prompts, completions, completion_ids_list
        )

        # --- Advantages ---
        if self.multi_objective_aggregation == "sum_then_normalize":
            rewards = (
                rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
            ).nansum(dim=1)
            mean_grouped = (
                rewards.view(-1, num_generations)
                .mean(dim=1)
                .repeat_interleave(num_generations)
            )
            if self.scale_rewards in ("group", "none"):
                if num_generations > 1:
                    std_rewards = (
                        rewards.view(-1, num_generations)
                        .std(dim=1)
                        .repeat_interleave(num_generations)
                    )
                else:
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                std_rewards = (
                    rewards.std().expand_as(rewards)
                    if rewards.numel() > 1
                    else torch.zeros_like(rewards)
                )
            else:
                raise ValueError(f"Invalid scale_rewards: {self.scale_rewards}")
            advantages = rewards - mean_grouped
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))

        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = (
                nanstd(grouped, dim=1, keepdim=True)
                if num_generations > 1
                else torch.zeros_like(mean_k)
            )
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(
                dim=1
            )
            std_rewards = (
                rewards.std().expand_as(rewards)
                if rewards.numel() > 1
                else torch.zeros_like(rewards)
            )
            advantages = (rewards - rewards.mean()) / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}"
            )

        # Slice for local process
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_advantages = advantages.clone()
        advantages = advantages[process_slice]
        data["advantages"] = advantages

        # --- Post-advantage hook (for replay buffer, re-roll, etc.) ---
        self._post_advantage_hook(
            data, rewards_per_func, advantages, inputs, num_generations, mode,
        )

        # --- Metrics ---
        for i, name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{name}/mean"].append(
                torch.nanmean(rewards_per_func[:, i]).item()
            )
            self._metrics[mode][f"rewards/{name}/std"].append(
                nanstd(rewards_per_func[:, i]).item()
            )
        agg_rewards = rewards_per_func.nansum(dim=1)
        self._metrics[mode]["reward"].append(agg_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(agg_rewards.std().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        # Token counting
        total_prompt = self.accelerator.gather(prompt_mask.sum())
        total_completion = self.accelerator.gather(completion_mask.sum())
        self.state.num_input_tokens_seen += (total_prompt + total_completion).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Completion length metrics
        comp_lengths = completion_mask.sum(dim=1)
        agg_lengths = self.accelerator.gather(comp_lengths)
        self._metrics[mode]["completions/mean_length"].append(
            agg_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_lengths.float().max().item()
        )

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_trunc = torch.tensor(
            [ids[-1].item() not in eos_and_pad for ids in completion_ids], device=device
        )
        agg_trunc = self.accelerator.gather(is_trunc)
        self._metrics[mode]["completions/clipped_ratio"].append(
            agg_trunc.float().mean().item()
        )
        term_lengths = agg_lengths[~agg_trunc]
        if len(term_lengths) == 0:
            term_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_lengths.float().max().item()
        )

        # IS metrics
        if "importance_sampling_ratio" in data and "sampling_per_token_logps" in data:
            old_lp = data["old_per_token_logps"]
            samp_lp = data["sampling_per_token_logps"]
            mask = completion_mask.bool()
            delta = torch.abs(old_lp - samp_lp)
            delta_m = delta[mask]
            md = (
                torch.mean(delta_m)
                if delta_m.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            xd = (
                torch.max(delta_m)
                if delta_m.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(md).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(xd).max().item()
            )
            isr = data["importance_sampling_ratio"]
            is_mode = getattr(self, "vllm_importance_sampling_mode", "token_truncate")
            if is_mode in ("sequence_mask", "sequence_truncate"):
                flat_isr = isr.flatten()
            else:
                flat_isr = isr[mask]
            if flat_isr.numel() > 0:
                self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                    nanmin(self.accelerator.gather(torch.min(flat_isr))).item()
                )
                self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                    self.accelerator.gather(torch.mean(flat_isr)).nanmean().item()
                )
                self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                    nanmax(self.accelerator.gather(torch.max(flat_isr))).item()
                )

        # Log prompt/completion texts
        prompts_text = self.processing_class.batch_decode(
            prompt_ids, skip_special_tokens=True
        )
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if gather_object is not None:
            self._logs["prompt"].extend(gather_object(prompts_text))
            self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_advantages.tolist())

        # Remove deferred keys
        for k in list(data.keys()):
            if k.startswith("_deferred") or k == "_pending_policy_logps":
                data.pop(k, None)

        return data

    @torch.no_grad()
    def _compute_streaming_group_scores(
        self,
        data,
        s_start,
        s_end,
        inputs,
        prompts,
        completions,
        completion_ids_list,
        is_last_chunk,
    ):
        """Score a chunk of prompt groups: rewards, policy logprobs, advantages.

        Called during streaming scoring to incrementally score groups.
        Writes results directly into ``data`` at positions ``s_start:s_end``.
        """
        device = self.accelerator.device
        batch_size = self.args.per_device_train_batch_size
        num_generations = self.num_generations
        mode = "train"
        chunk_size = s_end - s_start

        # --- Policy logprobs for this chunk ---
        chunk_prompt_ids = data["prompt_ids"][s_start:s_end]
        chunk_completion_ids = data["completion_ids"][s_start:s_end]
        chunk_prompt_mask = data["prompt_mask"][s_start:s_end]
        chunk_completion_mask = data["completion_mask"][s_start:s_end]
        prompt_completion_ids = torch.cat(
            [chunk_prompt_ids, chunk_completion_ids], dim=1
        )
        attention_mask = torch.cat([chunk_prompt_mask, chunk_completion_mask], dim=1)
        logits_to_keep = chunk_completion_ids.size(1)

        # Slice multimodal forward kwargs for this chunk
        forward_kwargs = {}
        for key in (
            "pixel_values",
            "image_grid_thw",
            "pixel_attention_mask",
            "image_sizes",
            "token_type_ids",
            "mm_token_type_ids",
        ):
            if key in data:
                val = data[key]
                if (
                    isinstance(val, torch.Tensor)
                    and val.dim() > 0
                    and val.size(0) == len(data["prompt_ids"])
                ):
                    forward_kwargs[key] = val[s_start:s_end]
                else:
                    forward_kwargs[key] = val
        num_images = data.get("num_images")
        if (
            num_images is not None
            and hasattr(num_images, "__getitem__")
            and len(num_images) == len(data["prompt_ids"])
        ):
            num_images = num_images[s_start:s_end]

        logprob_batch_size = min(batch_size * 4, chunk_size)
        with disable_gradient_checkpointing(
            self.model, self.args.gradient_checkpointing_kwargs
        ):
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm
                and getattr(self, "vllm_importance_sampling_correction", False)
            ):
                old_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    logprob_batch_size,
                    num_images=num_images,
                    **forward_kwargs,
                )
                if "old_per_token_logps" not in data:
                    total = len(data["prompt_ids"])
                    data["old_per_token_logps"] = torch.zeros(
                        total, old_logps.size(1), device=device, dtype=old_logps.dtype
                    )
                data["old_per_token_logps"][s_start:s_end] = old_logps

                # Compute IS ratio for this chunk
                if "sampling_per_token_logps" in data:
                    samp_chunk = data["sampling_per_token_logps"][s_start:s_end]
                    is_mask = (
                        chunk_completion_mask
                        if "tool_mask" not in data
                        else (chunk_completion_mask * data["tool_mask"][s_start:s_end])
                    )
                    diff = (old_logps - samp_chunk) * is_mask
                    is_mode = getattr(
                        self, "vllm_importance_sampling_mode", "token_truncate"
                    )
                    is_cap = getattr(self, "vllm_importance_sampling_cap", 3.0)
                    seq_is = is_mode in ("sequence_mask", "sequence_truncate")
                    logps_diff = diff.sum(dim=-1, keepdim=True) if seq_is else diff
                    is_ratio = torch.exp(logps_diff)
                    if is_mode in ("sequence_truncate", "token_truncate"):
                        is_ratio = torch.clamp(is_ratio, max=is_cap)
                    elif is_mode in ("sequence_mask", "token_mask"):
                        is_ratio = is_ratio.masked_fill(is_ratio > is_cap, value=0.0)
                    if "importance_sampling_ratio" not in data:
                        total = len(data["prompt_ids"])
                        shape = (total, 1) if seq_is else (total, is_ratio.size(1))
                        data["importance_sampling_ratio"] = torch.ones(
                            *shape, device=device, dtype=is_ratio.dtype
                        )
                    data["importance_sampling_ratio"][s_start:s_end] = is_ratio

            # Reference logprobs
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                        num_images=num_images,
                        **forward_kwargs,
                    )
                else:
                    model = self.accelerator.unwrap_model(self.model)
                    adapter_name = (
                        "ref"
                        if hasattr(model, "peft_config") and "ref" in model.peft_config
                        else None
                    )
                    with use_adapter(model, adapter_name=adapter_name):
                        ref_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size,
                            num_images=num_images,
                            **forward_kwargs,
                        )
                if "ref_per_token_logps" not in data:
                    total = len(data["prompt_ids"])
                    data["ref_per_token_logps"] = torch.zeros(
                        total, ref_logps.size(1), device=device, dtype=ref_logps.dtype
                    )
                data["ref_per_token_logps"][s_start:s_end] = ref_logps

        # --- Rewards ---
        rewards_per_func = self._compute_rewards_for_batch(
            inputs, prompts, completions, completion_ids_list
        )

        # --- Advantages (group-level normalization) ---
        if self.multi_objective_aggregation == "sum_then_normalize":
            rewards = (
                rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
            ).nansum(dim=1)
            mean_g = (
                rewards.view(-1, num_generations)
                .mean(dim=1)
                .repeat_interleave(num_generations)
            )
            if num_generations > 1:
                std_r = (
                    rewards.view(-1, num_generations)
                    .std(dim=1)
                    .repeat_interleave(num_generations)
                )
            else:
                std_r = torch.zeros_like(rewards)
            advantages = rewards - mean_g
            if self.scale_rewards != "none":
                advantages = advantages / (std_r + 1e-4)
            is_std_zero = torch.isclose(std_r, torch.zeros_like(std_r))

        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = (
                nanstd(grouped, dim=1, keepdim=True)
                if num_generations > 1
                else torch.zeros_like(mean_k)
            )
            reward_k = ((grouped - mean_k) / (std_k + 1e-4)).view(
                -1, len(self.reward_funcs)
            )
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(
                dim=1
            )
            std_r = (
                rewards.view(-1, num_generations)
                .std(dim=1)
                .repeat_interleave(num_generations)
            )
            mean_r = (
                rewards.view(-1, num_generations)
                .mean(dim=1)
                .repeat_interleave(num_generations)
            )
            advantages = (rewards - mean_r) / (std_r + 1e-4)
            is_std_zero = torch.isclose(std_r, torch.zeros_like(std_r))
        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}"
            )

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        if "advantages" not in data or not isinstance(data["advantages"], torch.Tensor):
            data["advantages"] = torch.zeros(len(data["prompt_ids"]), device=device)
        data["advantages"][s_start:s_end] = advantages

        # --- Post-advantage hook (for replay buffer, re-roll, etc.) ---
        self._post_advantage_hook(
            data, rewards_per_func, advantages, inputs, num_generations, mode,
            s_start=s_start, s_end=s_end, is_last_chunk=is_last_chunk,
        )

        # --- Chunk metrics ---
        for i, name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{name}/mean"].append(
                torch.nanmean(rewards_per_func[:, i]).item()
            )
            self._metrics[mode][f"rewards/{name}/std"].append(
                nanstd(rewards_per_func[:, i]).item()
            )
        agg_rewards = rewards_per_func.nansum(dim=1)
        self._metrics[mode]["reward"].append(agg_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(agg_rewards.std().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        # --- Full-batch metrics on last chunk ---
        if is_last_chunk:
            all_prompt_mask = data["prompt_mask"]
            all_completion_mask = data["completion_mask"]
            all_completion_ids = data["completion_ids"]
            total_p = self.accelerator.gather(all_prompt_mask.sum())
            total_c = self.accelerator.gather(all_completion_mask.sum())
            self.state.num_input_tokens_seen += (total_p + total_c).item()
            self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

            comp_lengths = all_completion_mask.sum(dim=1)
            agg_lengths = self.accelerator.gather(comp_lengths)
            self._metrics[mode]["completions/mean_length"].append(
                agg_lengths.float().mean().item()
            )
            self._metrics[mode]["completions/min_length"].append(
                agg_lengths.float().min().item()
            )
            self._metrics[mode]["completions/max_length"].append(
                agg_lengths.float().max().item()
            )

            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_trunc = torch.tensor(
                [ids[-1].item() not in eos_and_pad for ids in all_completion_ids],
                device=device,
            )
            agg_trunc = self.accelerator.gather(is_trunc)
            self._metrics[mode]["completions/clipped_ratio"].append(
                agg_trunc.float().mean().item()
            )
            term = agg_lengths[~agg_trunc]
            if len(term) == 0:
                term = torch.zeros(1, device=device)
            self._metrics[mode]["completions/mean_terminated_length"].append(
                term.float().mean().item()
            )
            self._metrics[mode]["completions/min_terminated_length"].append(
                term.float().min().item()
            )
            self._metrics[mode]["completions/max_terminated_length"].append(
                term.float().max().item()
            )

            # IS metrics
            if (
                self.use_vllm
                and getattr(self, "vllm_importance_sampling_correction", False)
                and "sampling_per_token_logps" in data
                and "old_per_token_logps" in data
            ):
                old_lp = data["old_per_token_logps"]
                samp_lp = data["sampling_per_token_logps"]
                mask = all_completion_mask.bool()
                delta = torch.abs(old_lp - samp_lp)[mask]
                md = (
                    torch.mean(delta)
                    if delta.numel() > 0
                    else torch.tensor(0.0, device=device)
                )
                xd = (
                    torch.max(delta)
                    if delta.numel() > 0
                    else torch.tensor(0.0, device=device)
                )
                self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                    self.accelerator.gather(md).mean().item()
                )
                self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                    self.accelerator.gather(xd).max().item()
                )
                is_mode = getattr(
                    self, "vllm_importance_sampling_mode", "token_truncate"
                )
                isr = data["importance_sampling_ratio"]
                flat = (
                    isr.flatten()
                    if is_mode in ("sequence_mask", "sequence_truncate")
                    else isr[mask]
                )
                if flat.numel() > 0:
                    self._metrics[mode][
                        "sampling/importance_sampling_ratio/min"
                    ].append(nanmin(self.accelerator.gather(torch.min(flat))).item())
                    self._metrics[mode][
                        "sampling/importance_sampling_ratio/mean"
                    ].append(self.accelerator.gather(torch.mean(flat)).nanmean().item())
                    self._metrics[mode][
                        "sampling/importance_sampling_ratio/max"
                    ].append(nanmax(self.accelerator.gather(torch.max(flat))).item())

    def _score_streaming(self, rollout: dict) -> list[dict]:
        """Score a rollout using streaming group scoring.  Returns list of micro-batches."""
        data = rollout
        num_gen = self.num_generations
        n_groups = len(data["prompt_ids"]) // num_gen
        batch_size = self.args.per_device_train_batch_size
        min_groups = max(1, self.args.streaming_min_groups)

        # Extract deferred data
        inputs = data.pop("_deferred_inputs")
        prompts = data.pop("_deferred_prompts")
        completions = data.pop("_deferred_completions")
        completion_ids_list = data.pop("_deferred_completion_ids_list")
        del data["_pending_policy_logps"]

        all_micro_batches = []
        shared_keys = {"num_items_in_batch"}

        for chunk_start_g in range(0, n_groups, min_groups):
            chunk_end_g = min(chunk_start_g + min_groups, n_groups)
            s_start = chunk_start_g * num_gen
            s_end = chunk_end_g * num_gen

            self._compute_streaming_group_scores(
                data=data,
                s_start=s_start,
                s_end=s_end,
                inputs=inputs[s_start:s_end],
                prompts=prompts[s_start:s_end],
                completions=completions[s_start:s_end],
                completion_ids_list=completion_ids_list[s_start:s_end],
                is_last_chunk=(chunk_end_g == n_groups),
            )

            # Yield micro-batches from this scored chunk
            chunk_size = s_end - s_start
            perm = torch.randperm(chunk_size)
            for mb_off in range(0, chunk_size, batch_size):
                mb_idx = perm[mb_off : mb_off + batch_size]
                abs_idx = mb_idx + s_start
                mb = {}
                for key in data:
                    if key.startswith("_"):
                        continue
                    val = data[key]
                    if key in shared_keys:
                        mb[key] = val
                    elif isinstance(val, torch.Tensor) and val.dim() > 0:
                        mb[key] = val[abs_idx]
                    else:
                        mb[key] = val
                all_micro_batches.append(mb)

        # Repeat for num_iterations
        return all_micro_batches * self.num_iterations

    # ------------------------------------------------------------------
    # _prepare_inputs override
    # ------------------------------------------------------------------

    def _prepare_inputs(self, generation_batch):
        """Override to support data producer and async prefetch paths."""
        mode = "train" if self.model.training else "eval"

        # --- Data producer path ---
        if mode == "train" and self.data_producer is not None:
            return self._prepare_inputs_data_producer(generation_batch)

        # --- Legacy async prefetch path (no data producer) ---
        if mode == "train" and self.args.async_prefetch:
            return self._prepare_inputs_legacy_async(generation_batch)

        # --- Stock path ---
        return super()._prepare_inputs(generation_batch)

    def _prepare_inputs_data_producer(self, generation_batch):
        """Data producer path: produce rollout, score deferred logps, split into micro-batches."""
        # Return from buffer if available
        if self._buffered_inputs:
            return self._buffered_inputs.pop(0)

        # Produce a new rollout
        self._maybe_sync_vllm_weights()
        rollout_dataset = self.data_producer.produce(
            self.model,
            self.state.global_step,
            processing_class=self.processing_class,
            accelerator=self.accelerator,
            args=self.args,
        )

        # Convert RolloutDataset back to a dict for scoring/splitting
        rollout = rollout_dataset._data

        # If async (skip_policy_logps=True), score deferred logps on main thread
        if rollout.get("_pending_policy_logps"):
            if self.args.streaming_partial_batch:
                micro_batches = self._score_streaming(rollout)
            else:
                scored = self._compute_deferred_scores(rollout)
                scored = split_pixel_values_by_grid(scored)
                scored = shuffle_sequence_dict(scored)
                batches = split_tensor_dict(scored, self.args.steps_per_generation)
                micro_batches = [unsplit_pixel_values_by_grid(b) for b in batches]
                micro_batches = micro_batches * self.num_iterations
        else:
            # Sync path: data is already fully scored
            rollout = split_pixel_values_by_grid(rollout)
            batches = split_tensor_dict(rollout, self.args.steps_per_generation)
            micro_batches = [unsplit_pixel_values_by_grid(b) for b in batches]
            micro_batches = micro_batches * self.num_iterations

        self._buffered_inputs = micro_batches[1:]
        return micro_batches[0]

    def _prepare_inputs_legacy_async(self, generation_batch):
        """Legacy async path: direct queue-based prefetch without data producer."""
        # Return from buffer if available
        if self._buffered_inputs:
            return self._buffered_inputs.pop(0)

        # Need a new rollout
        self._maybe_sync_vllm_weights()
        future = self._async_queue.get()
        rollout = future.result()
        self._submit_generation()

        if self.args.streaming_partial_batch:
            micro_batches = self._score_streaming(rollout)
        else:
            scored = self._compute_deferred_scores(rollout)
            scored = split_pixel_values_by_grid(scored)
            scored = shuffle_sequence_dict(scored)
            batches = split_tensor_dict(scored, self.args.steps_per_generation)
            micro_batches = [unsplit_pixel_values_by_grid(b) for b in batches]
            micro_batches = micro_batches * self.num_iterations

        self._buffered_inputs = micro_batches[1:]
        return micro_batches[0]

    # ------------------------------------------------------------------
    # Loss override (adds IS ratio + OPSM)
    # ------------------------------------------------------------------

    @staticmethod
    def get_off_policy_mask(
        advantages,
        per_token_logps,
        sampling_per_token_logps,
        mask,
        off_policy_threshold,
    ):
        """OPSM from DeepSeek-V3.2: drop sequences with negative advantage + high KL."""
        kl_div = sampling_per_token_logps - per_token_logps.detach()
        seq_kl = (kl_div * mask).sum(dim=1, keepdim=True) / mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)
        is_pos_adv = advantages >= 0
        is_low_kl = seq_kl <= off_policy_threshold
        return (is_pos_adv | is_low_kl).to(dtype=mask.dtype)

    def _compute_loss(self, model, inputs):
        """Override to add IS ratio correction and off-policy sequence masking."""
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = (
            completion_mask
            if "tool_mask" not in inputs
            else completion_mask * inputs["tool_mask"]
        )

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach()
            if old_per_token_logps is None
            else old_per_token_logps
        )

        # --- OPSM (off-policy sequence mask) ---
        off_policy_mask = None
        if getattr(self, "off_policy_mask_threshold", None) is not None:
            sampling_per_token_logps = inputs.get(
                "sampling_per_token_logps", old_per_token_logps
            )
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        # --- Importance weights ---
        log_ratio = per_token_logps - old_per_token_logps
        is_level = getattr(
            self,
            "importance_sampling_level",
            getattr(self.args, "importance_sampling_level", "token"),
        )
        if is_level == "token":
            log_importance_weights = log_ratio
        elif is_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(
                min=1.0
            )
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {is_level}")

        coef_1 = torch.exp(log_importance_weights)

        # --- KL divergence ---
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            if getattr(self.args, "use_bias_correction_kl", False):
                per_token_kl = per_token_kl * coef_1

        # --- Per-token loss ---
        if self.loss_type == "cispo":
            clamped = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped * advantages * per_token_logps
        elif self.loss_type in ("grpo", "bnpo", "dr_grpo", "dapo", "luspo"):
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1_c = torch.clamp(coef_1, max=self.args.delta)
            else:
                coef_1_c = coef_1
            per_token_loss = -torch.min(coef_1_c * advantages, coef_2 * advantages)
        elif self.loss_type == "sapo":
            temps = torch.where(
                advantages > 0,
                self.args.sapo_temperature_pos,
                self.args.sapo_temperature_neg,
            )
            soft = torch.sigmoid(temps * (coef_1 - 1)) * 4 / temps
            per_token_loss = -soft * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # --- Apply masks ---
        if off_policy_mask is not None:
            per_token_loss = per_token_loss * off_policy_mask
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        # --- IS ratio correction (vLLM distribution mismatch) ---
        if (
            self.use_vllm
            and getattr(self, "vllm_importance_sampling_correction", False)
            and "importance_sampling_ratio" in inputs
        ):
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # --- Aggregate loss ---
        mode = "train" if self.model.training else "eval"
        normalizer = (
            self.current_gradient_accumulation_steps if mode == "train" else 1.0
        )

        if self.loss_type in ("grpo", "sapo"):
            loss = (
                (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            ).mean() / normalizer
        elif self.loss_type == "bnpo":
            loss = (
                (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0) / normalizer
            )
        elif self.loss_type == "dr_grpo":
            loss = (
                (per_token_loss * mask).sum()
                / (per_token_loss.size(0) * self.max_completion_length)
                / normalizer
            )
        elif self.loss_type in ("cispo", "dapo"):
            norm = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / norm
        elif self.loss_type == "luspo":
            loss = (per_token_loss * mask.sum(1, keepdim=True)).mean() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # --- Metrics ---
        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            return (
                x.mean()
                if x.shape[1] == 1
                else (x * mask).sum() / completion_token_count
            )

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )

        if self.loss_type in ("grpo", "bnpo", "dr_grpo", "dapo", "luspo"):
            is_low = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region = is_low | is_high
            low_clip = masked_batch_mean(is_low.float())
            high_clip = masked_batch_mean(is_high.float())
            clip_ratio = masked_batch_mean(is_region.float())
            g_low = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(g_low.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(g_low).item())
            g_high = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(g_high.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(g_high).item())
            g_clip = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(
                g_clip.nanmean().item()
            )
        elif self.loss_type == "cispo":
            is_cispo = (coef_1 > self.epsilon_high) & (advantages > 0)
            cr = masked_batch_mean(is_cispo.float())
            self._metrics[mode]["cispo_clip_ratio"].append(
                self.accelerator.gather(cr).nanmean().item()
            )

        return loss

    # ------------------------------------------------------------------
    # Training step override (timing)
    # ------------------------------------------------------------------

    def training_step(self, model, inputs, num_items_in_batch=None):
        t0 = time.perf_counter()
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        t1 = time.perf_counter()
        self._current_train_step_time += t1 - t0
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
        return output
