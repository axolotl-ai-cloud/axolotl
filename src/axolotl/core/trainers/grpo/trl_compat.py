"""Everything axolotl's async GRPO takes from TRL, in one place.

Async GRPO extends ``trl.experimental.async_grpo``, an experimental API with no
stability promise: it subclasses TRL's trainer, rollout loop, batchers and
collator, imports private names, swaps module globals while TRL builds its trainer
and dataloader, and replaces two module functions for good. All of it goes through
this module, so a TRL upgrade starts here and with ``tests/core/test_trl_compat.py``,
which pins the source of every TRL function axolotl overrides or copies and fails,
naming the function, when an upgrade changes one.
"""

from __future__ import annotations

import contextlib
import functools
from importlib.metadata import version

import torch.nn.functional as F
from trl.chat_template_utils import add_response_schema
from trl.experimental.async_grpo import (
    async_grpo_trainer as trainer_module,
    packing as packing_module,
)
from trl.experimental.async_grpo.async_grpo_config import AsyncGRPOConfig
from trl.experimental.async_grpo.async_grpo_trainer import (
    AsyncGRPOTrainer,
    DataCollatorForRollout,
    FixedCountBatcher,
    RolloutQueueDataset,
    TokenBudgetBatcher,
    _EmptyIterableDataset as EmptyIterableDataset,
    _iter_vllm_named_params as iter_vllm_named_params,
)
from trl.experimental.async_grpo.async_rollout_worker import (
    AsyncRolloutWorker,
    RolloutGroup,
    RolloutSample,
    TrainingSequence,
    TurnRecord,
    _AsyncRolloutLoop as AsyncRolloutLoop,
    _chain_to_sequences as chain_to_sequences,
)
from trl.experimental.async_grpo.packing import TREE_ATTENTION, TreeRow
from trl.experimental.async_grpo.vllm_client import VLLMClient
from trl.trainer.utils import (
    add_fused_lm_head,
    get_config_model_id,
    is_async_callable,
    nanmax,
    nanmin,
    pad,
    print_prompt_completions_sample,
)

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# The release the overrides and copies below were written against (pyproject's pin).
TRL_VERSION = "1.16.0"

# TRL code axolotl copies or re-implements rather than subclasses; the drift test
# pins these besides every TRL method an axolotl subclass overrides.
UPSTREAM = {
    "AsyncGRPOTrainer.compute_loss": "copied into AxolotlAsyncGRPOTrainer, with "
    "the loss behind loss_type",
    "AsyncGRPOConfig": "its defaults are mirrored in axolotl's validation "
    "(max_staleness, weight_sync_steps), and its dispatch_batches/split_batches "
    "carry the rollout stream's stop micro-batch to every rank",
    "RolloutQueueDataset.__iter__": "its staleness rule is applied ahead of it by "
    "SampleFilterQueue",
    "TokenBudgetBatcher.__iter__": "its oversize rule is applied ahead of it by "
    "SampleFilterQueue",
    "_EpochStopCallback.on_step_end": "reads the consumed-groups set that "
    "RolloutProgress keeps",
    "_TrainBeginCallback.on_train_begin": "its cold weight sync must run in "
    "AxolotlAsyncGRPOTrainer._sync_weight",
    "_iter_vllm_named_params": "walked by the out-of-place merged weight sync",
    "_flash_attn": "replaced by tree_flash_attn",
    "compute_flops_per_token": "wrapped by flops_per_token",  # nosec B105
    "_chain_to_sequences": "used by NeMo Gym's rollout loop",
}


def check_version() -> None:
    installed = version("trl")
    if installed != TRL_VERSION:
        LOG.warning_once(
            f"Async GRPO extends trl {TRL_VERSION}'s experimental AsyncGRPOTrainer; "
            f"trl {installed} is installed. Re-check "
            "axolotl/core/trainers/grpo/trl_compat.py before trusting a run."
        )


@contextlib.contextmanager
def patched_trainer_module(**replacements):
    """Swap globals of TRL's trainer module while TRL code reads them, e.g. the
    batcher classes its ``get_train_dataloader`` instantiates."""
    originals = {name: getattr(trainer_module, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(trainer_module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(trainer_module, name, value)


_trl_flash_attn = packing_module._flash_attn
_trl_flops_per_token = trainer_module.compute_flops_per_token


def install_runtime_patches() -> None:
    """Replace the two TRL functions read at forward and logging time.

    Unlike the swaps in ``patched_trainer_module``, these stay installed: TRL
    looks them up while training, not while building the trainer. Both fall back
    to TRL's own, so installing them twice, or for TRL's own trainer, is harmless.
    """
    packing_module._flash_attn = tree_flash_attn
    trainer_module.compute_flops_per_token = flops_per_token


def flops_per_token(config, seq_len: int) -> int:
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
def tree_flash_attn():
    """TRL's FA3 kernel, or FA2's where FA3 has no build for the GPU (Blackwell)."""
    try:
        return _trl_flash_attn()
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


__all__ = [
    "TREE_ATTENTION",
    "TRL_VERSION",
    "UPSTREAM",
    "AsyncGRPOConfig",
    "AsyncGRPOTrainer",
    "AsyncRolloutLoop",
    "AsyncRolloutWorker",
    "DataCollatorForRollout",
    "EmptyIterableDataset",
    "FixedCountBatcher",
    "RolloutGroup",
    "RolloutQueueDataset",
    "RolloutSample",
    "TokenBudgetBatcher",
    "TrainingSequence",
    "TreeRow",
    "TurnRecord",
    "VLLMClient",
    "add_fused_lm_head",
    "add_response_schema",
    "chain_to_sequences",
    "check_version",
    "flops_per_token",
    "get_config_model_id",
    "install_runtime_patches",
    "is_async_callable",
    "iter_vllm_named_params",
    "nanmax",
    "nanmin",
    "packing_module",
    "pad",
    "patched_trainer_module",
    "print_prompt_completions_sample",
    "trainer_module",
    "tree_flash_attn",
]
