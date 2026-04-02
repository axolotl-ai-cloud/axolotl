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
from abc import ABC, abstractmethod
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from trl.extras.profiling import profiling_decorator
from trl.trainer import GRPOConfig, GRPOTrainer
from trl.trainer.utils import (
    RepeatSampler,
    entropy_from_logits,
    nanmax,
    nanmin,
    nanstd,
    pad,
    selective_log_softmax,
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

try:
    from liger_kernel.ops.grpo_loss import (
        fused_selective_log_softmax as _fused_selective_log_softmax,
    )
except ImportError:
    _fused_selective_log_softmax = None


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

    # --- Batch flattening ---
    batch_flattening: bool = field(
        default=False,
        metadata={
            "help": "Use batch flattening for the scoring forward pass. Removes padding tokens "
            "before the forward pass, reducing attention FLOPs proportional to the padding ratio. "
            "Requires flash_attention_2 attention implementation. Incompatible with FSDP and "
            "multimodal models. The per-token logprob results differ by bf16 precision (~0.03 mean) "
            "but produce equivalent loss and gradients."
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

logger = logging.getLogger(__name__)
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

    FSDP compatibility: Background threads must NOT call cross-rank collectives
    (gather_object, broadcast_object_list, FSDP all-gather) because the main thread
    may be doing FSDP forward/backward concurrently, causing deadlocks. When
    ``num_processes > 1``, only rank 0 runs BG generation; results are broadcast
    to other ranks on the main thread when ``produce()`` is next called.
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
        # Lock held by the background thread during vLLM generation.
        # The main thread acquires this lock for weight sync to ensure
        # merge_adapter/unmerge_adapter don't overlap with generation.
        self._generate_lock = threading.Lock()
        # Detected at first produce() call
        self._num_processes: int | None = None
        self._is_main: bool | None = None

    @property
    def config(self) -> ProducerConfig:
        return self._inner.config

    def produce(self, model: Any, global_step: int, **kwargs) -> Dataset:
        """Return the next dataset, blocking if the prefetch hasn't finished."""
        # Detect multi-process on first call
        if self._num_processes is None:
            accelerator = kwargs.get("accelerator")
            if accelerator is not None:
                self._num_processes = accelerator.num_processes
                self._is_main = accelerator.is_main_process
            else:
                self._num_processes = 1
                self._is_main = True

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
            # With FSDP (multi-process), only submit BG tasks on rank 0.
            # Non-rank-0 processes will receive data via broadcast.
            if self._num_processes > 1:
                bg_kwargs["_rank0_only"] = True
            for i in range(1, self._depth + 1):
                self._queue.append(
                    self._executor.submit(
                        self._locked_produce, model, global_step + i, **bg_kwargs
                    )
                )
            self._initialized = True
            return dataset

        # Get the pre-generated dataset from the BG thread
        dataset = self._queue.popleft().result()

        # With FSDP: BG thread only ran on rank 0. Broadcast to all ranks.
        if self._num_processes > 1:
            dataset = self._broadcast_dataset(dataset)

        bg_kwargs = {**kwargs, **self._background_kwargs}
        if self._num_processes > 1:
            bg_kwargs["_rank0_only"] = True
        next_step = global_step + self._depth
        self._queue.append(
            self._executor.submit(self._locked_produce, model, next_step, **bg_kwargs)
        )
        return dataset

    def _broadcast_dataset(self, dataset) -> Dataset:
        """Broadcast a prefetched dataset from rank 0 to all ranks (main thread).

        Rank 0 has a full RolloutDataset from BG generation; other ranks have None.
        After broadcast, tensors are moved to each rank's local device.
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            return dataset

        # Rank 0 sends _data dict; others receive it
        obj_list = [dataset._data if self._is_main else None]
        dist.broadcast_object_list(obj_list, src=0)

        data: dict[str, Any] = obj_list[0]  # type: ignore[assignment]

        # Move tensors to local device (broadcast_object_list deserializes to CPU)
        accelerator = self._inner._trainer.accelerator  # type: ignore[attr-defined]
        device = accelerator.device
        for key, val in data.items():
            if isinstance(val, torch.Tensor) and val.device != device:
                data[key] = val.to(device)

        if not self._is_main:
            from axolotl.core.trainers.grpo.async_trainer import RolloutDataset

            dataset = RolloutDataset(data)
        else:
            # Rank 0 already has the dataset, but update _data with device-moved tensors
            dataset._data = data
        return dataset

    def _locked_produce(self, model: Any, global_step: int, **kwargs) -> Dataset:
        """Run produce while holding the generate lock."""
        with self._generate_lock:
            return self._inner.produce(model, global_step, **kwargs)

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

    _ALWAYS_SHARED = frozenset(
        {"num_items_in_batch", "_pending_policy_logps", "_rank0_only"}
    )

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
        self._trainer: Any = None
        self._prompt_dl: Any = None
        self._prompt_iter: Any = None

    def set_trainer(self, trainer) -> None:
        """Inject the live trainer reference and create the prompt DataLoader."""
        self._trainer = trainer
        # Defer _init_prompt_dataloader if trainer.args is not yet set
        # (happens when set_trainer is called from _create_data_producer during __init__)
        if getattr(trainer, "args", None) is not None:
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
        _rank0_only: bool = False,
        **kwargs,
    ) -> RolloutDataset | None:
        """Generate a fresh GRPO training rollout."""
        # Lazy init: create prompt DataLoader if deferred from set_trainer
        if self._prompt_dl is None and self._trainer is not None:
            self._init_prompt_dataloader()

        is_main = self._trainer.accelerator.is_main_process

        # FSDP rank0-only mode: non-rank-0 returns None (broadcast fills it later)
        if _rank0_only and not is_main:
            return None

        try:
            inputs = next(self._prompt_iter)
        except StopIteration:
            self._prompt_iter = iter(self._prompt_dl)
            inputs = next(self._prompt_iter)

        if skip_policy_logps:
            # Async path: use _generate_only (generation without scoring) which
            # works on stock TRL (no skip_policy_logps parameter needed).
            output = self._trainer._generate_only(inputs, rank0_only=_rank0_only)
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
        # Skip NCCL communicator init when using LoRA sync (filesystem) or HTTP-only
        # merged weight sync. NCCL is only needed for the standard update_named_param
        # path which broadcasts tensors through the communicator.
        training_args = kwargs.get("args") or (args[1] if len(args) > 1 else None)
        _skip_nccl = False
        if training_args is not None:
            if getattr(training_args, "vllm_lora_sync", False):
                _skip_nccl = True  # LoRA sync uses filesystem + HTTP
            elif getattr(training_args, "async_prefetch", False):
                # Skip NCCL at init to avoid DDP param count mismatch in multi-GPU.
                # init_communicator allocates device tensors on rank 0 only, which
                # causes DDP to see different param counts across ranks.
                # The communicator is initialized lazily on first weight sync instead.
                _skip_nccl = True
        if _skip_nccl:
            from trl.generation.vllm_generation import VLLMGeneration

            _orig_init_vllm = VLLMGeneration._init_vllm

            def _init_vllm_no_communicator(self_vllm):
                """Init vLLM client without NCCL communicator (LoRA sync uses filesystem)."""
                if self_vllm.mode == "server" and self_vllm.accelerator.is_main_process:
                    from trl.generation.vllm_client import VLLMClient

                    if self_vllm.server_base_url is not None:
                        base_url = self_vllm.server_base_url
                    else:
                        base_url = (
                            f"http://{self_vllm.server_host}:{self_vllm.server_port}"
                        )
                    self_vllm.vllm_client = VLLMClient(
                        base_url=base_url,
                        group_port=self_vllm.group_port,
                        connection_timeout=self_vllm.server_timeout,
                    )
                    # Deliberately skip init_communicator — no NCCL needed
                elif self_vllm.mode != "server":
                    _orig_init_vllm(self_vllm)

            VLLMGeneration._init_vllm = _init_vllm_no_communicator

        try:
            super().__init__(*args, **kwargs)
        finally:
            # Restore original _init_vllm so other trainers aren't affected
            if _skip_nccl:
                VLLMGeneration._init_vllm = _orig_init_vllm  # type: ignore[possibly-undefined]

        # FP8 models: zero out the pad token embedding so that padding
        # positions have zero hidden states throughout the network.
        # FP8 linear layers produce NaN on non-zero inputs at masked
        # positions (the Triton fp8 matmul kernel can't handle the
        # extreme values that accumulate at unattended positions).
        self._zero_pad_embedding_for_fp8()

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

        # Data producer (the proper architecture for async generation)
        self.data_producer = None
        if getattr(self.args, "use_data_producer", False):
            self.data_producer = self._create_data_producer(
                kwargs["args"], kwargs["train_dataset"]
            )

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
        """Submit the next background generation job.

        With multi-process (DDP/FSDP), only rank 0 generates to avoid
        cross-rank NCCL collectives from background threads.  Non-rank-0
        processes enqueue a sentinel ``None`` that is replaced by a
        broadcast in ``_prepare_inputs_legacy_async``.
        """
        rank0_only = self.accelerator.num_processes > 1
        if rank0_only and not self.accelerator.is_main_process:
            # Non-rank-0: nothing to generate; enqueue a resolved None future
            f: concurrent.futures.Future = concurrent.futures.Future()
            f.set_result(None)
            self._async_queue.put(f)
            return
        batch = next(self._prompt_iter)
        future = self._executor.submit(self._generate_only, batch, rank0_only)
        self._async_queue.put(future)

    # ------------------------------------------------------------------
    # Broadcast rollout (legacy async, multi-process)
    # ------------------------------------------------------------------

    def _broadcast_rollout(self, rollout: dict | None) -> dict:
        """Broadcast a rank0-only rollout dict to all ranks (main thread).

        Rank 0 has the full rollout dict from ``_generate_only``; other ranks
        have ``None``.  After broadcast, tensors are moved to each rank's
        local device.
        """
        import torch.distributed as dist

        obj_list = [rollout if self.accelerator.is_main_process else None]
        dist.broadcast_object_list(obj_list, src=0)
        rollout = obj_list[0]
        assert rollout is not None, "broadcast_object_list failed to deliver rollout"

        # Move tensors to local device (broadcast deserializes to CPU)
        device = self.accelerator.device
        for key, val in rollout.items():
            if isinstance(val, torch.Tensor) and val.device != device:
                rollout[key] = val.to(device)

        return rollout

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def _sync_peft_weights_no_merge(self):
        """Thread-safe weight sync: compute merged LoRA weights without in-place modification.

        Required for FP8 models where merge_adapter() fails (addmm not implemented
        for Float8), and also safe for concurrent use since it never modifies base
        weights in-place.
        """
        accelerator = self.vllm_generation.accelerator
        if not (self.vllm_generation.mode == "server" and accelerator.is_main_process):
            return

        # In multi-GPU async mode, we skip NCCL communicator init to avoid
        # DDP param count mismatch and NCCL device conflicts. Weight sync
        # uses the HTTP-only fallback in batch_update_named_params instead.

        model = self.vllm_generation.model
        vllm_client = self.vllm_generation.vllm_client
        fix_name = self.vllm_generation._fix_param_name_to_vllm

        # Build lookup: module_path -> (A, B, scaling) for all active LoRA layers
        lora_info = {}
        for mod_name, module in model.base_model.model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "active_adapters"):
                continue
            active = module.active_adapters[0]
            if active not in module.lora_A:
                continue
            lora_info[mod_name] = (
                module.lora_A[active].weight.data,
                module.lora_B[active].weight.data,
                module.scaling[active],
            )

        # Build lookup for FP8 scale_inv parameters (needed for dequantization)
        scale_inv_lookup = {}
        for pname, pparam in model.named_parameters():
            if "weight_scale_inv" in pname:
                # Map weight name -> scale_inv tensor
                weight_name = pname.replace(".weight_scale_inv", ".weight")
                scale_inv_lookup[weight_name] = pparam.data

        # Only sync parameters that have LoRA modifications — skip unchanged
        # base weights to avoid OOM on the vLLM GPU from allocating the entire
        # model's worth of NCCL receive buffers.
        params_to_sync = []
        compute_dtype = torch.bfloat16
        for name, param in model.named_parameters():
            vllm_name = name.removeprefix("base_model.model.").replace(
                ".base_layer", ""
            )
            if model.prefix in vllm_name:
                continue
            if "original_module" in vllm_name:
                continue
            if "weight_scale_inv" in vllm_name or "input_scale" in vllm_name:
                continue
            if not vllm_name.endswith(".weight"):
                continue
            # fix_name strips modules_to_save.default. prefix
            raw_mod_path = vllm_name[: -len(".weight")]
            vllm_name = fix_name(vllm_name, extra_prefixes=["modules_to_save.default."])
            mod_path = vllm_name[: -len(".weight")]

            # Sync weights that have LoRA adapters OR are modules_to_save
            is_lora = mod_path in lora_info
            is_modules_to_save = raw_mod_path != mod_path  # fix_name stripped a prefix
            if not is_lora and not is_modules_to_save:
                continue

            data = param.data

            # Dequantize FP8 weights before merging
            if data.dtype == torch.float8_e4m3fn and name in scale_inv_lookup:
                scale_inv = scale_inv_lookup[name]
                fp8_bf16 = data.to(compute_dtype)
                if scale_inv.dim() == 2 and fp8_bf16.dim() == 2:
                    sr, sc = scale_inv.shape
                    br = fp8_bf16.shape[0] // sr
                    bc = fp8_bf16.shape[1] // sc
                    data = (
                        fp8_bf16.reshape(sr, br, sc, bc)
                        * scale_inv[:, None, :, None].to(compute_dtype)
                    ).reshape(fp8_bf16.shape)
                elif scale_inv.dim() <= 1:
                    data = fp8_bf16 * scale_inv.to(compute_dtype)
                else:
                    data = fp8_bf16
            elif data.dtype == torch.float8_e4m3fn:
                data = data.to(compute_dtype)

            if is_lora:
                A, B, s = lora_info[mod_path]
                merged = data.to(compute_dtype) + s * (
                    B.to(compute_dtype) @ A.to(compute_dtype)
                )
                params_to_sync.append((vllm_name, merged))
            else:
                # modules_to_save: send raw weight (no LoRA merge needed)
                params_to_sync.append((vllm_name, data.to(compute_dtype)))

        # Batch sync only LoRA-modified params via HTTP+NCCL
        if params_to_sync:
            sync_mb = sum(t.numel() * t.element_size() for _, t in params_to_sync) / 1e6
            logger.info(
                f"Syncing {len(params_to_sync)} LoRA-modified params ({sync_mb:.0f} MB)"
            )
            vllm_client.batch_update_named_params(params_to_sync)

        # Reset prefix cache after weight update
        vllm_client.reset_prefix_cache()

    def _sync_lora_adapter(self):
        """Sync LoRA adapter to vLLM via filesystem (native LoRA mode).

        Saves the PEFT adapter to a temp directory and POSTs the path to vLLM's
        /set_lora_adapter/ endpoint. vLLM loads the adapter natively using Punica
        kernels, avoiding the need to merge weights and NCCL-broadcast the full model.

        Syncs only the LoRA adapter weights via filesystem instead of the full merged model via NCCL.

        FSDP/DeepSpeed: All ranks must participate in the state_dict gather.
        accelerator.get_state_dict() handles this (FSDP uses FullStateDictConfig
        with rank0_only=True). Only rank 0 gets the full dict, writes files, and
        does the HTTP POST.
        """
        import os
        import tempfile

        accelerator = self.vllm_generation.accelerator
        model = self.vllm_generation.model

        if self.vllm_generation.mode != "server":
            return

        is_main = accelerator.is_main_process

        # Increment adapter version (all ranks, kept in sync)
        if not hasattr(self, "_lora_sync_version"):
            self._lora_sync_version = 0
            if is_main:
                self._lora_sync_dir = tempfile.mkdtemp(prefix="lora_sync_")
            else:
                self._lora_sync_dir = None
            # Broadcast sync dir from rank 0 to all ranks
            if accelerator.num_processes > 1:
                import torch.distributed as dist

                if dist.is_initialized():
                    obj_list = [self._lora_sync_dir]
                    dist.broadcast_object_list(obj_list, src=0)
                    self._lora_sync_dir = obj_list[0]
        self._lora_sync_version += 1

        adapter_path = os.path.join(self._lora_sync_dir, f"v{self._lora_sync_version}")

        # Gather state dict from all ranks (FSDP/DeepSpeed gather, rank0_only)
        # All ranks must participate even though only rank 0 gets the result.
        # Use self.model_wrapped (the DeepSpeed/FSDP engine) for get_state_dict,
        # since it has the necessary hooks (e.g. zero_gather_16bit_weights_on_model_save).
        # self.vllm_generation.model is the unwrapped PEFT model which lacks these.
        wrapped_model = getattr(self, "model_wrapped", model)
        state_dict = accelerator.get_state_dict(wrapped_model)

        if is_main:
            # Unwrap to access PEFT's save_pretrained
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(adapter_path, state_dict=state_dict)

            import requests

            vllm_client = self.vllm_generation.vllm_client
            base_url = vllm_client.base_url
            base_model = getattr(self.args, "model_name_or_path", "axolotl-lora")
            sync_timeout = getattr(self.args, "vllm_server_timeout", 300) or 300

            # Try standard vLLM /v1/load_lora_adapter first, fall back to custom endpoint
            response = requests.post(
                f"{base_url}/v1/load_lora_adapter",
                json={
                    "lora_name": base_model,
                    "lora_path": adapter_path,
                    "load_inplace": True,
                },
                timeout=sync_timeout,
            )
            if response.status_code != 200:
                # Fallback: try custom /set_lora_adapter/ endpoint
                response = requests.post(
                    f"{base_url}/set_lora_adapter/",
                    json={
                        "lora_name": "active_lora",
                        "lora_int_id": self._lora_sync_version,
                        "lora_path": adapter_path,
                    },
                    timeout=30,
                )
                if response.status_code != 200:
                    logger.warning(
                        "Failed to set LoRA adapter: %s %s",
                        response.status_code,
                        response.text,
                    )
                    return

            # Reset prefix cache after adapter update
            try:
                vllm_client.reset_prefix_cache()
            except Exception as exc:
                logger.warning("Failed to reset prefix cache: %s", exc)

            # Clean up old adapter versions (keep only current)
            if self._lora_sync_version > 1:
                old_path = os.path.join(
                    self._lora_sync_dir, f"v{self._lora_sync_version - 1}"
                )
                if os.path.exists(old_path):
                    import shutil

                    shutil.rmtree(old_path, ignore_errors=True)

            logger.info(
                "Synced LoRA adapter v%d to vLLM (%s)",
                self._lora_sync_version,
                adapter_path,
            )

        # Barrier to ensure all ranks complete before resuming forward passes.
        # Without this, rank 1 may start a forward pass (triggering FSDP unshard)
        # while rank 0 is still doing save_pretrained, causing FSDP all-gather deadlock.
        if accelerator.num_processes > 1:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.barrier()

    def _maybe_sync_vllm_weights(self):
        """Sync model weights to vLLM if the interval has elapsed.

        Dispatches to one of three strategies:
        - vllm_lora_sync: saves adapter to filesystem, vLLM loads natively
        - PEFT no-merge: computes merged weights as new tensors, NCCL broadcast
        - Non-PEFT: stock sync_weights via merge_adapter + NCCL
        """
        if not (self.use_vllm and self.args.async_prefetch):
            return
        step = self.state.global_step
        interval = self.args.vllm_sync_interval
        if step != self._last_synced_step and step % interval == 0:
            if step == 0:
                logger.info("Skipping vLLM weight sync at step 0 (no training yet)")
                self._last_synced_step = step
                return
            if getattr(self.args, "vllm_lora_sync", False):
                # Native LoRA sync: save adapter to filesystem, vLLM loads it directly
                self._sync_lora_adapter()
            else:
                from accelerate.utils import is_peft_model

                use_no_merge = is_peft_model(self.vllm_generation.model)

                if use_no_merge:
                    # No-merge sync: computes merged weights as new tensors
                    # (doesn't modify base weights in-place), so it's safe to
                    # run concurrently with BG generation — no lock needed.
                    self._sync_peft_weights_no_merge()
                else:
                    # Non-PEFT: use stock sync (acquires lock to avoid overlap)
                    if self.data_producer is not None and hasattr(
                        self.data_producer, "_generate_lock"
                    ):
                        with self.data_producer._generate_lock:
                            self.vllm_generation.sync_weights()
                    elif self._async_queue is not None:
                        pending = list(self._async_queue.queue)
                        for f in pending:
                            if isinstance(f, concurrent.futures.Future):
                                f.result()
                        self.vllm_generation.sync_weights()
                    else:
                        self.vllm_generation.sync_weights()
            self._last_synced_step = step

    def _zero_pad_embedding_for_fp8(self):
        """Zero out the pad token embedding for FP8 models.

        FP8 linear layers produce NaN when processing positions with
        attention_mask=0 (the hidden states at those positions have
        unconstrained values that overflow FP8 range during
        quantization). By setting the pad token embedding to zeros,
        padding positions start with zero hidden states and stay zero
        through masked attention, preventing NaN from FP8 matmul.
        """
        model = self.accelerator.unwrap_model(self.model)
        # Check if model has FP8 weights
        has_fp8 = any(
            p.dtype == torch.float8_e4m3fn
            for p in model.parameters()
            if not p.requires_grad
        )
        if not has_fp8:
            return

        # Find the embedding layer
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed = model.model.embed_tokens
        elif hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            m = model.base_model.model
            if hasattr(m, "model") and hasattr(m.model, "embed_tokens"):
                embed = m.model.embed_tokens
            else:
                return
        else:
            return

        pad_id = self.processing_class.pad_token_id
        if pad_id is not None and pad_id < embed.weight.shape[0]:
            with torch.no_grad():
                embed.weight.data[pad_id].zero_()
            import logging

            logging.getLogger("async_grpo").info(
                f"Zeroed pad token embedding (id={pad_id}) for FP8 NaN prevention"
            )

    # ------------------------------------------------------------------
    # Background-thread generation (no scoring)
    # ------------------------------------------------------------------

    def _generate_single_turn(self, prompts, *args, **kwargs):
        """Override to prevent weight sync from background thread and to use
        no-merge sync for PEFT models (FP8 models can't merge_adapter)."""
        is_bg = threading.current_thread() is not threading.main_thread()
        saved_step = None

        if is_bg and self.use_vllm:
            # Trick: match _last_loaded_step so the stock sync check is a no-op
            saved_step = getattr(self, "_last_loaded_step", None)
            self._last_loaded_step = self.state.global_step

        # Permanently replace vllm_generation.sync_weights with our custom
        # sync to avoid merge_adapter (fails on FP8 / races with training).
        # For LoRA sync mode, make it a no-op here since _maybe_sync_vllm_weights
        # handles the sync with proper interval tracking.
        if not getattr(self, "_patched_sync_weights", False):
            if self.use_vllm and hasattr(self, "vllm_generation"):
                if getattr(self.args, "vllm_lora_sync", False):
                    # No-op: LoRA sync is driven by _maybe_sync_vllm_weights
                    self.vllm_generation.sync_weights = lambda: None
                    self._patched_sync_weights = True
                else:
                    from accelerate.utils import is_peft_model

                    if is_peft_model(self.vllm_generation.model):

                        def _no_merge_sync():
                            self._sync_peft_weights_no_merge()

                        self.vllm_generation.sync_weights = _no_merge_sync
                        self._patched_sync_weights = True

        try:
            return super()._generate_single_turn(prompts, *args, **kwargs)
        finally:
            if saved_step is not None:
                self._last_loaded_step = saved_step

    def _generate_rank0_only(self, prompts):
        """Generate using vLLM directly on rank 0 without cross-rank collectives.

        Called from BG thread in FSDP mode. Bypasses ``gather_object`` /
        ``broadcast_object_list`` since the main thread may be running FSDP
        collectives concurrently.

        Returns the same tuple as ``_generate``.
        """
        import copy

        prompts = copy.deepcopy(prompts)

        # Duplicate prompts for num_generations (same as TRL's gather+unique pattern)
        num_generations = self.num_generations
        unique_prompts = prompts[::num_generations]

        # Build sampling params
        vg = self.vllm_generation
        sampling_params = {
            "n": num_generations,
            "repetition_penalty": vg.repetition_penalty,
            "temperature": vg.temperature,
            "top_p": vg.top_p,
            "top_k": vg.top_k,
            "min_p": 0.0 if vg.min_p is None else vg.min_p,
            "max_tokens": vg.max_completion_length,
            "logprobs": vg.logprobs,
            "structured_outputs_regex": vg.structured_outputs_regex,
            "generation_kwargs": vg.generation_kwargs,
        }

        # Call vLLM directly (no collectives)
        from trl.data_utils import is_conversational

        if is_conversational({"prompt": unique_prompts[0]}):
            output = vg.vllm_client.chat(
                messages=unique_prompts,
                **sampling_params,
                chat_template_kwargs=self.chat_template_kwargs,
                tools=self.tools,
                chat_template=getattr(self, "chat_template", None),
            )
        else:
            output = vg.vllm_client.generate(prompts=unique_prompts, **sampling_params)

        # vLLM returns 1 prompt_ids per unique prompt, but num_generations completion_ids.
        # Duplicate prompt_ids to match completions (one per generation).
        raw_prompt_ids = output["prompt_ids"]
        prompt_ids = [pid for pid in raw_prompt_ids for _ in range(num_generations)]
        completion_ids = output["completion_ids"]
        logprobs_raw = output["logprobs"]
        extra_fields = {
            k: v
            for k, v in output.items()
            if k
            not in {"prompt_ids", "completion_ids", "logprobs", "logprob_token_ids"}
        }

        # Extract top-1 logprob per token
        logprobs = [[lp[0] for lp in seq] for seq in logprobs_raw]

        # Decode completions
        if is_conversational({"prompt": prompts[0]}):
            contents = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )
            completions = [[{"role": "assistant", "content": c}] for c in contents]
        else:
            completions = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )

        tool_mask = extra_fields.pop("env_mask", None)

        # Compute total completion tokens locally (no gather)
        total_completion_tokens = sum(len(ids) for ids in completion_ids)

        return (
            prompt_ids,
            completion_ids,
            tool_mask,
            completions,
            total_completion_tokens,
            logprobs,
            extra_fields,
        )

    def _generate_only(self, inputs, rank0_only=False):
        """Generate completions without scoring.  Runs on background thread.

        Mirrors the first half of ``_generate_and_score_completions`` (prompt
        extraction → vLLM generation → tensor padding) and returns a deferred
        output dict for main-thread scoring.

        When ``rank0_only=True`` (FSDP mode), bypasses ``gather_object`` /
        ``broadcast_object_list`` collectives and calls vLLM directly on rank 0.
        Results are broadcast to other ranks on the main thread later.

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
        if rank0_only:
            # FSDP mode: call vLLM directly without cross-rank collectives
            (
                prompt_ids_list,
                completion_ids_list,
                tool_mask_list,
                completions,
                num_items_in_batch,
                sampling_per_token_logps_list,
                extra_fields,
            ) = self._generate_rank0_only(prompts)
        else:
            (
                prompt_ids_list,
                completion_ids_list,
                tool_mask_list,
                completions,
                num_items_in_batch,
                sampling_per_token_logps_list,
                extra_fields,
            ) = self._generate(prompts)
            # _generate gathers prompts from all ranks internally. Gather inputs
            # to match the full-batch output size.
            if self.accelerator.num_processes > 1:
                from accelerate.utils import gather_object

                inputs = gather_object(inputs)
                prompts = [x["prompt"] for x in inputs]

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

        # No explicit CUDA sync needed here — both threads share the
        # default stream, so operations are naturally ordered.

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
            "_rank0_only": rank0_only,
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

    def _compute_rewards_for_batch(
        self, inputs, prompts, completions, completion_ids_list
    ):
        """Compute rewards for a batch. Override for parallel workers, caching, etc."""
        return self._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )

    def _launch_reward_workers(self, inputs, prompts, completions, completion_ids_list):
        """Launch reward computation in background. Override for parallel dispatch.

        Default: no-op (rewards computed synchronously in _collect_reward_workers).
        """
        self._pending_reward_args = (inputs, prompts, completions, completion_ids_list)

    def _collect_reward_workers(
        self, inputs, prompts, completions, completion_ids_list
    ):
        """Collect reward results. Override to collect from parallel workers.

        Default: compute rewards synchronously now.
        """
        args = getattr(self, "_pending_reward_args", None)
        if args is not None:
            self._pending_reward_args = None
            return self._compute_rewards_for_batch(*args)
        return self._compute_rewards_for_batch(
            inputs, prompts, completions, completion_ids_list
        )

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

    def _notify_rollouts_scored(
        self,
        prompts: list[str],
        completions: list[str],
        rewards: dict[str, list[float]],
        advantages: list[float],
    ):
        """Dispatch on_rollouts_scored to all registered plugins (rank 0 only)."""
        if not self.accelerator.is_main_process:
            return

        from axolotl.integrations.base import PluginManager

        pm = PluginManager.get_instance()
        if pm and pm.plugins:
            # Try _axolotl_cfg first (set by causal builder), fall back to
            # PluginManager's stored cfg (set during register phase).
            cfg = getattr(self, "_axolotl_cfg", None) or getattr(pm, "_cfg", None)
            if cfg is not None:
                pm.on_rollouts_scored(
                    cfg, self, prompts, completions, rewards, advantages
                )

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
        rank0_only = data.pop("_rank0_only", False)
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

        # --- Launch rewards in parallel with logprobs ---
        self._launch_reward_workers(inputs, prompts, completions, completion_ids_list)

        # --- Policy logprobs ---
        # When batch_flattening is enabled, use the flattened (padding-free) forward
        # pass for the scoring path. This removes padding tokens before the forward
        # pass, reducing attention FLOPs proportional to the padding ratio (20-34%
        # faster in benchmarks). Requires flash_attention_2 and no multimodal inputs.
        can_flatten = (
            getattr(self.args, "batch_flattening", False)
            and not forward_kwargs  # no multimodal inputs
            and not self.is_fsdp_enabled  # FSDP needs wrapped model
        )

        logprob_batch_size = min(batch_size * 4, len(prompt_ids))
        with disable_gradient_checkpointing(
            self.model, self.args.gradient_checkpointing_kwargs
        ):
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm
                and getattr(self, "vllm_importance_sampling_correction", False)
            ):
                if can_flatten:
                    old_per_token_logps = self._get_per_token_logps_flattened(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=logprob_batch_size,
                        prompt_mask=prompt_mask,
                    )
                else:
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
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    adapter_name = (
                        "ref"
                        if hasattr(unwrapped, "peft_config")
                        and "ref" in unwrapped.peft_config
                        else None
                    )
                    with use_adapter(unwrapped, adapter_name=adapter_name):
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
            is_floor = 1.0 / is_cap  # symmetric floor (e.g., cap=3.0 -> floor=0.333)
            if is_mode in ("sequence_truncate", "token_truncate"):
                is_ratio = torch.clamp(is_ratio, min=is_floor, max=is_cap)
            elif is_mode in ("sequence_mask", "token_mask"):
                is_ratio = is_ratio.masked_fill(is_ratio > is_cap, value=0.0)
                is_ratio = is_ratio.clamp(min=is_floor)
            data["importance_sampling_ratio"] = is_ratio

        # --- Collect rewards (launched before logprobs, should be done) ---
        rewards_per_func = self._collect_reward_workers(
            inputs, prompts, completions, completion_ids_list
        )
        # In rank0_only mode, all ranks compute the same rewards on identical data.
        # _calculate_rewards / _collect_reward_workers always `gather()` across ranks,
        # which duplicates the rows (N_local * num_processes).  De-duplicate so that
        # rewards_per_func matches the data dict (which has N_local rows).
        if rank0_only and rewards_per_func.size(0) > len(prompts):
            rewards_per_func = rewards_per_func[: len(prompts)]

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
        # In rank0_only mode, all ranks already have identical data from broadcast,
        # so no slicing needed. Otherwise, each rank takes its portion.
        if rank0_only:
            process_slice = slice(0, len(prompts))
        else:
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
        all_advantages = advantages.clone()
        advantages = advantages[process_slice]
        data["advantages"] = advantages

        # --- Post-advantage hook (for replay buffer, re-roll, etc.) ---
        self._post_advantage_hook(
            data,
            rewards_per_func,
            advantages,
            inputs,
            num_generations,
            mode,
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
        total_prompt = self.accelerator.gather(prompt_mask.sum()).sum()
        total_completion = self.accelerator.gather(completion_mask.sum()).sum()
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

        # Log prompt/completion texts.
        # NB: gather_object merges per-rank local texts into a full-batch list
        # matching rewards_per_func and all_advantages which are already full-batch
        # tensors (gathered/computed earlier in this method). Lengths stay aligned.
        prompts_text = self.processing_class.batch_decode(
            prompt_ids, skip_special_tokens=True
        )
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if gather_object is not None:
            gathered_prompts = gather_object(prompts_text)
            gathered_completions = gather_object(completions_text)
            self._logs["prompt"].extend(gathered_prompts)
            self._logs["completion"].extend(gathered_completions)
        else:
            gathered_prompts = prompts_text
            gathered_completions = completions_text
        rewards_dict = {}
        for i, name in enumerate(self.reward_func_names):
            reward_list = rewards_per_func[:, i].tolist()  # already full-batch
            self._logs["rewards"][name].extend(reward_list)
            rewards_dict[name] = reward_list
        adv_list = all_advantages.tolist()  # already full-batch
        self._logs["advantages"].extend(adv_list)

        # Notify plugins of scored rollouts
        self._notify_rollouts_scored(
            gathered_prompts, gathered_completions, rewards_dict, adv_list
        )

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
        rank0_only=False,
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

        # --- Launch rewards in parallel with logprobs ---
        self._launch_reward_workers(inputs, prompts, completions, completion_ids_list)

        # --- Policy logprobs for this chunk (GPU, overlaps with BG rewards) ---
        can_flatten = (
            getattr(self.args, "batch_flattening", False)
            and not forward_kwargs
            and not self.is_fsdp_enabled
        )
        logprob_batch_size = min(batch_size * 2, chunk_size)
        with disable_gradient_checkpointing(
            self.model, self.args.gradient_checkpointing_kwargs
        ):
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm
                and getattr(self, "vllm_importance_sampling_correction", False)
            ):
                if can_flatten:
                    old_logps = self._get_per_token_logps_flattened(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=logprob_batch_size,
                        prompt_mask=chunk_prompt_mask,
                    )
                else:
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
                    # Symmetric floor clamp (matches non-streaming path at line ~1651)
                    is_floor = 1.0 / is_cap
                    if is_mode in ("sequence_truncate", "token_truncate"):
                        is_ratio = torch.clamp(is_ratio, min=is_floor, max=is_cap)
                    elif is_mode in ("sequence_mask", "token_mask"):
                        is_ratio = is_ratio.masked_fill(is_ratio > is_cap, value=0.0)
                        is_ratio = is_ratio.clamp(min=is_floor)
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
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    adapter_name = (
                        "ref"
                        if hasattr(unwrapped, "peft_config")
                        and "ref" in unwrapped.peft_config
                        else None
                    )
                    with use_adapter(unwrapped, adapter_name=adapter_name):
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

        # --- Collect rewards (should already be done, ran in parallel with logprobs) ---
        rewards_per_func = self._collect_reward_workers(
            inputs, prompts, completions, completion_ids_list
        )
        # De-duplicate gathered rewards when all ranks computed the same data.
        # _calculate_rewards always gather()s, which duplicates rows in rank0_only mode.
        if rewards_per_func.size(0) > chunk_size:
            rewards_per_func = rewards_per_func[:chunk_size]

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

        if rank0_only:
            process_slice = slice(0, len(prompts))
        else:
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
            data,
            rewards_per_func,
            advantages,
            inputs,
            num_generations,
            mode,
            s_start=s_start,
            s_end=s_end,
            is_last_chunk=is_last_chunk,
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
            total_p = self.accelerator.gather(all_prompt_mask.sum()).sum()
            total_c = self.accelerator.gather(all_completion_mask.sum()).sum()
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
        rank0_only = data.pop("_rank0_only", False)
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
                rank0_only=rank0_only,
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
        """Data producer path: produce rollout, score deferred logps, split into micro-batches.

        Architecture (with async_prefetch=True):
          BG thread:  produce(skip_policy_logps=True) → vLLM generation + reward computation
          Main thread: deferred scoring (policy logprobs via GPU forward pass) → training

        Why deferred scoring is necessary for stable training:
          The policy logprobs (old_per_token_logps) must come from the CURRENT
          training model, not the vLLM model (which is N steps behind). Using
          stale vLLM logprobs as old_logps causes the importance sampling ratio
          to start far from 1.0, leading to:
            - Immediate PPO clipping → wasted samples
            - High-variance gradients from IS correction
            - Compounding per-token ratio errors on long sequences
            - In extreme cases, complete training failure (exp-003: accuracy=0)

          Deferred scoring computes old_logps with the latest model weights, so
          the IS ratio starts at exactly 1.0 and drifts gradually — giving
          maximum useful gradient signal before clipping activates.

          Cost: one additional forward pass per scoring round (GPU-bound, cannot
          overlap with training on the same GPU). Use ``batch_flattening: true``
          to reduce this cost by eliminating padding tokens from the forward pass.

        Pipeline:
          [produce(BG)] → [deferred_scores(GPU)] → [train×GA(GPU)] → [weight_sync]
                           ↑ can't overlap with train (same GPU)

        Bottleneck: the produce() wait (generation-limited) dominates when
        generation is slower than training + scoring. Async prefetch hides
        part of this by generating in the BG thread while training runs.
        """
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

        rollout = rollout_dataset._data

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

        # With multi-process, only rank 0 generated. Broadcast to all ranks.
        if self.accelerator.num_processes > 1:
            rollout = self._broadcast_rollout(rollout)

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

        # Release cached CUDA memory from scoring
        # before training allocations begin, reducing peak reserved memory.
        torch.cuda.empty_cache()

        return micro_batches[0]

    def _get_per_token_logps_flattened(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        prompt_mask=None,
    ) -> torch.Tensor:
        """Compute per-token log-probs using batch flattening (padding-free).

        Instead of processing padded batches where attention wastes compute on
        padding tokens, this method:
        1. Chunks the batch into sub-batches of ``batch_size`` sequences
        2. For each chunk, flattens non-padding tokens into [1, chunk_tokens]
        3. Uses FlashAttentionKwargs (cu_seq_lens) for varlen attention
        4. Computes selective_log_softmax on the flat logits
        5. Gathers completion logprobs back to (B, logits_to_keep) padded format

        Args:
            prompt_mask: (B, L) mask where 1 = prompt token, 0 = completion/padding.
                Used to determine the exact prompt length per sequence for correct
                logprob gathering. If None, inferred as seq_len - logits_to_keep.

        Chunking prevents OOM when the total flattened sequence is too long
        (e.g., 32 sequences × 2048 tokens = 65K tokens → 20GB logits tensor).

        Requires flash_attention_2 attention implementation.
        """
        if not self.is_fsdp_enabled:
            model = self.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

        device = input_ids.device
        B, L = input_ids.shape
        if batch_size is None:
            batch_size = max(1, B)

        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        all_logps = torch.zeros(B, logits_to_keep, device=device)

        for chunk_start in range(0, B, batch_size):
            chunk_end = min(chunk_start + batch_size, B)
            chunk_ids = input_ids[chunk_start:chunk_end]
            chunk_mask = attention_mask[chunk_start:chunk_end]
            n = chunk_end - chunk_start

            seq_lens = chunk_mask.sum(dim=1).to(torch.int32)
            total_tokens = seq_lens.sum().item()
            cu_seqlens = torch.zeros(n + 1, dtype=torch.int32, device=device)
            cu_seqlens[1:] = seq_lens.cumsum(0)

            valid = chunk_mask.bool()
            flat_ids = chunk_ids[valid].unsqueeze(0)
            positions = torch.arange(L, device=device).unsqueeze(0).expand(n, L)
            flat_pos = positions[valid].unsqueeze(0)

            with autocast_ctx:
                logits = model(
                    input_ids=flat_ids,
                    position_ids=flat_pos,
                    use_cache=False,
                    cu_seq_lens_q=cu_seqlens,
                    cu_seq_lens_k=cu_seqlens,
                    max_length_q=seq_lens.max().item(),
                    max_length_k=seq_lens.max().item(),
                ).logits
                logits = torch.nan_to_num(logits, nan=0.0)

                # Compute logprobs on the flat shifted tensor
                flat_logits = logits[0, :-1, :] / self.temperature
                flat_targets = flat_ids[0, 1:]
                flat_logps = selective_log_softmax(
                    flat_logits.unsqueeze(0), flat_targets.unsqueeze(0)
                )[0]

                # Mask out cross-sequence boundary positions. In the shifted
                # tensor, position cu_seqlens[i]-1 (for i>0) is where sequence
                # i-1's last token "predicts" sequence i's first token — garbage.
                for boundary in cu_seqlens[1:-1]:
                    idx = boundary.item() - 1
                    if 0 <= idx < flat_logps.size(0):
                        flat_logps[idx] = 0.0

            # Gather completion logprobs per sequence.
            # Use prompt_mask to determine exact prompt length (not logits_to_keep,
            # which is the padded completion dimension and may exceed the actual
            # completion length for shorter sequences).
            for i in range(n):
                slen = seq_lens[i].item()
                abs_i = chunk_start + i  # absolute index in the full batch
                if prompt_mask is not None:
                    plen = int(prompt_mask[abs_i].sum().item())
                else:
                    plen = max(1, slen - logits_to_keep)
                n_compl = slen - plen
                start = cu_seqlens[i].item() + plen - 1
                start = max(0, start)
                actual = min(n_compl, total_tokens - 1 - start)
                if actual > 0:
                    all_logps[chunk_start + i, :actual] = flat_logps[
                        start : start + actual
                    ]

            del logits, flat_logits, flat_logps, flat_ids
            torch.cuda.empty_cache()

        return all_logps

    def _get_per_token_logps_and_entropies_flattened(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        prompt_mask=None,
        compute_entropy=True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Flattened forward pass for training (with gradients).

        Same padding removal as the scoring path, but:
        - Gradients flow through for backward pass
        - Computes entropy alongside logprobs
        - Per-sequence logprob/entropy extraction preserves grad graph
        """
        device = input_ids.device
        B, L = input_ids.shape
        if batch_size is None:
            batch_size = max(1, B)

        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)

        # Pre-allocate output containers (will be filled with grad-carrying slices)
        all_logps_list: list[torch.Tensor] = []
        all_entropy_list: list[torch.Tensor] = []

        for chunk_start in range(0, B, batch_size):
            chunk_end = min(chunk_start + batch_size, B)
            chunk_ids = input_ids[chunk_start:chunk_end]
            chunk_mask = attention_mask[chunk_start:chunk_end]
            n = chunk_end - chunk_start

            seq_lens = chunk_mask.sum(dim=1).to(torch.int32)
            cu_seqlens = torch.zeros(n + 1, dtype=torch.int32, device=device)
            cu_seqlens[1:] = seq_lens.cumsum(0)

            valid = chunk_mask.bool()
            flat_ids = chunk_ids[valid].unsqueeze(0)
            positions = torch.arange(L, device=device).unsqueeze(0).expand(n, L)
            flat_pos = positions[valid].unsqueeze(0)

            with autocast_ctx:
                logits = model(
                    input_ids=flat_ids,
                    position_ids=flat_pos,
                    use_cache=False,
                    cu_seq_lens_q=cu_seqlens,
                    cu_seq_lens_k=cu_seqlens,
                    max_length_q=seq_lens.max().item(),
                    max_length_k=seq_lens.max().item(),
                ).logits
                logits = torch.nan_to_num(logits, nan=0.0)

            # Extract logprobs and entropy per-sequence (avoids cross-sequence targets,
            # preserves gradient graph through selective_log_softmax → logits → model)
            for i in range(n):
                slen = seq_lens[i].item()
                abs_i = chunk_start + i
                if prompt_mask is not None:
                    plen = int(prompt_mask[abs_i].sum().item())
                else:
                    plen = max(1, slen - logits_to_keep)
                n_compl = slen - plen
                s = cu_seqlens[i].item()

                if n_compl <= 0:
                    # No completion tokens — append zeros
                    all_logps_list.append(torch.zeros(logits_to_keep, device=device))
                    if compute_entropy:
                        all_entropy_list.append(
                            torch.zeros(logits_to_keep, device=device)
                        )
                    continue

                with autocast_ctx:
                    # Shifted logits and targets for this sequence only
                    seq_logits = logits[0, s + plen - 1 : s + slen - 1, :]
                    seq_logits = seq_logits / self.temperature
                    seq_targets = flat_ids[0, s + plen : s + slen]

                    # Log probs (differentiable)
                    lps = selective_log_softmax(
                        seq_logits.unsqueeze(0), seq_targets.unsqueeze(0)
                    )[0]  # (n_compl,)

                    # Pad to logits_to_keep
                    if n_compl < logits_to_keep:
                        lps = F.pad(lps, (0, logits_to_keep - n_compl))
                    all_logps_list.append(lps[:logits_to_keep])

                    if compute_entropy:
                        ent = entropy_from_logits(seq_logits)  # (n_compl,)
                        if n_compl < logits_to_keep:
                            ent = F.pad(ent, (0, logits_to_keep - n_compl))
                        all_entropy_list.append(ent[:logits_to_keep])

        # Stack per-sequence results into (B, logits_to_keep) tensors
        all_logps = torch.stack(all_logps_list, dim=0)
        all_entropies = (
            torch.stack(all_entropy_list, dim=0) if compute_entropy else None
        )
        return all_logps, all_entropies

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        mm_token_type_ids=None,
    ) -> tuple[Any, torch.Tensor | None]:
        """Compute log-probs and (optionally) entropies for each token.

        When running under no_grad (scoring path), bypasses accelerate's
        ConvertOutputsToFp32 wrapper to avoid a fp32 copy of the
        logits tensor.
        """
        # Bypass accelerate's ConvertOutputsToFp32 wrapper which converts the
        # entire (B, L, V) logits tensor from bf16 to fp32 — unnecessary and
        # extremely wasteful for large vocabularies.
        # Skip unwrapping for FSDP — parameters are only valid inside FSDP's
        # forward context; unwrapping exposes flattened/sharded tensors.
        if not self.is_fsdp_enabled:
            model = self.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        autocast_ctx = torch.autocast(
            device_type=input_ids.device.type, dtype=torch.bfloat16
        )

        # Use Liger's Triton kernel in scoring path (no grad): fuses
        # temperature + log_softmax + gather into a single kernel pass.
        use_fused = (
            self.use_liger_kernel
            and _fused_selective_log_softmax is not None
            and not torch.is_grad_enabled()
        )

        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []
        with autocast_ctx:
            for start in range(0, input_ids.size(0), batch_size):
                input_ids_batch = input_ids[start : start + batch_size]
                attention_mask_batch = attention_mask[start : start + batch_size]

                # Build model inputs
                model_inputs = {
                    "input_ids": input_ids_batch,
                    "attention_mask": attention_mask_batch,
                }
                if image_grid_thw is not None and pixel_values is not None:
                    rows_per_image = image_grid_thw.prod(dim=-1)
                    rows_per_sample = torch.split(rows_per_image, num_images)
                    rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                    cum_rows = torch.cat(
                        [
                            torch.tensor([0], device=rows_per_sample.device),
                            rows_per_sample.cumsum(0),
                        ]
                    )
                    row_start, row_end = (
                        cum_rows[start].item(),
                        cum_rows[start + batch_size].item(),
                    )
                    model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                    cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                    img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                    model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
                elif pixel_values is not None:
                    model_inputs["pixel_values"] = pixel_values[
                        start : start + batch_size
                    ]
                if pixel_attention_mask is not None:
                    model_inputs["pixel_attention_mask"] = pixel_attention_mask[
                        start : start + batch_size
                    ]
                if image_sizes is not None:
                    model_inputs["image_sizes"] = image_sizes[
                        start : start + batch_size
                    ]
                if token_type_ids is not None:
                    model_inputs["token_type_ids"] = token_type_ids[
                        start : start + batch_size
                    ]
                if mm_token_type_ids is not None:
                    model_inputs["mm_token_type_ids"] = mm_token_type_ids[
                        start : start + batch_size
                    ]

                if "logits_to_keep" in self.model_kwarg_keys:
                    model_inputs["logits_to_keep"] = logits_to_keep + 1

                model_inputs["use_cache"] = False

                logits = model(**model_inputs).logits
                completion_ids = input_ids_batch[:, -logits_to_keep:]
                # FP8 models produce NaN logits at positions where
                # attention_mask=0 (padding). Replace NaN with 0 so
                # log_softmax yields uniform distribution for those positions.
                # The completion_mask ensures these don't affect the loss.
                logits = torch.nan_to_num(logits, nan=0.0)

                if use_fused:
                    logits = logits[:, -(logits_to_keep + 1) :, :]
                    if not logits.is_contiguous():
                        logits = logits.contiguous()
                    logps = _fused_selective_log_softmax(
                        logits, completion_ids, self.temperature
                    )
                    all_logps.append(logps)
                    # Liger fused path doesn't compute entropy — append zeros
                    if compute_entropy:
                        all_entropies.append(torch.zeros_like(logps))
                else:
                    logits = logits[:, :-1, :]
                    logits = logits[:, -logits_to_keep:, :]
                    logits.div_(self.temperature)
                    logps = selective_log_softmax(logits, completion_ids)
                    all_logps.append(logps)

                    if compute_entropy:
                        with torch.no_grad():
                            entropies = entropy_from_logits(logits)
                        all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

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

        # Check for multimodal inputs
        forward_kwargs = {
            k: inputs[k]
            for k in (
                "pixel_values",
                "image_grid_thw",
                "num_images",
                "pixel_attention_mask",
                "image_sizes",
                "token_type_ids",
                "mm_token_type_ids",
            )
            if k in inputs and inputs[k] is not None
        }

        can_flatten = (
            getattr(self.args, "batch_flattening", False)
            and not forward_kwargs
            and not self.is_fsdp_enabled
        )

        if can_flatten:
            per_token_logps, entropies = (
                self._get_per_token_logps_and_entropies_flattened(
                    model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    prompt_mask=prompt_mask,
                    compute_entropy=True,
                )
            )
        else:
            per_token_logps, entropies = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                compute_entropy=True,
                **forward_kwargs,
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
