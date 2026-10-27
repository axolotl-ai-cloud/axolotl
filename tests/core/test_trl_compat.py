"""The TRL code async GRPO overrides or copies, pinned at ``trl_compat.TRL_VERSION``,
and at ``trl_compat.TRL_MAIN_COMMIT`` for TRL main.

A TRL upgrade that changes any of it fails ``test_upstream_sources_are_pinned``,
naming what changed. Re-check the axolotl side (the override, or the copy or rule
``trl_compat.UPSTREAM`` describes), then replace the table the failure names with
the one it prints.
"""

import hashlib
import inspect
import math
import types
from importlib.metadata import version

import numpy as np
import pytest
from tree_recurrence.integrations.trl import (
    ChainFixedCountBatcher,
    ChainRow,
    ChainTokenBudgetBatcher,
)
from trl.experimental.async_grpo import (
    async_grpo_config,
    async_grpo_trainer,
    async_rollout_worker,
    packing,
)

from axolotl.core.trainers.ebft.async_rollout import EBFTRolloutLoop
from axolotl.core.trainers.ebft.async_trainer import AxolotlAsyncEBFTTrainer
from axolotl.core.trainers.grpo import trl_compat
from axolotl.core.trainers.grpo.async_progress import ProgressCollator
from axolotl.core.trainers.grpo.async_rollout import (
    AxolotlAsyncRolloutWorker,
    AxolotlRolloutLoop,
)
from axolotl.core.trainers.grpo.async_scoring import ScoredRow, ScoringCollator
from axolotl.core.trainers.grpo.async_traces import TraceCollator
from axolotl.core.trainers.grpo.async_trainer import (
    AliasingVLLMClient,
    AxolotlAsyncGRPOTrainer,
)
from axolotl.integrations.nemo_gym.rollout import NemoGymRolloutLoop

SUBCLASSES = [
    AxolotlAsyncGRPOTrainer,
    AxolotlAsyncEBFTTrainer,
    AxolotlRolloutLoop,
    EBFTRolloutLoop,
    NemoGymRolloutLoop,
    AxolotlAsyncRolloutWorker,
    AliasingVLLMClient,
    ChainTokenBudgetBatcher,
    ChainFixedCountBatcher,
    ChainRow,
    ScoredRow,
    ProgressCollator,
    TraceCollator,
    ScoringCollator,
]
MODULES = [async_grpo_trainer, async_rollout_worker, packing, async_grpo_config]

# trl_compat.TRL_VERSION
PINNED = {
    "AsyncGRPOConfig": "627858d1dad729f6",
    "AsyncGRPOTrainer.__init__": "8ef1227654a229e7",
    "AsyncGRPOTrainer._streaming_iter": "56159bc6237badc9",
    "AsyncGRPOTrainer._sync_weight": "38e9a8cb3b240a06",
    "AsyncGRPOTrainer._sync_weight_merged": "ab2541763c20cba1",
    "AsyncGRPOTrainer.compute_loss": "5c0c485d130acb07",
    "AsyncGRPOTrainer.get_train_dataloader": "2950938428f1c183",
    "AsyncRolloutWorker.__init__": "4421bac6f1128f93",
    "DataCollatorForRollout": "6538e7c28fadffc2",
    "DataCollatorForRollout.torch_call": "fc1ce59367d57b6f",
    "FixedCountBatcher.__iter__": "121e176022912d3b",
    "RolloutQueueDataset.__iter__": "d92a084c61a3511c",
    "TokenBudgetBatcher.__iter__": "f2321cfe667f9b41",
    "TreeRow.__init__": "cb6cc1d22b4a45c8",
    "TreeRow.add": "59d788cfc6cfe1fd",
    "TreeRow.fits": "e5f4e79d5fdfcab6",
    "TreeRow.pack": "c33de3e4ab8253f0",
    "VLLMClient.__init__": "5e9e16c61d8281cc",
    "VLLMClient.load_lora_adapter": "1651c8d0459f3653",
    "VLLMClient.pause": "b7fca52f2b342f2b",
    "_AsyncRolloutLoop.__init__": "f2f7fc012668b22c",
    "_AsyncRolloutLoop._generate_one": "a412fb0898da0d95",
    "_AsyncRolloutLoop._repeat_iterator": "ed97bff71aee925b",
    "_AsyncRolloutLoop._run_loops": "664f17f542dfcc7d",
    "_AsyncRolloutLoop._score_group": "d032308d2e87d971",
    "_AsyncRolloutLoop._score_loop": "a4356a487c050c7a",
    "_AsyncRolloutLoop.run": "5010a1b9cf1080c7",
    "_EpochStopCallback.on_step_end": "96f8c76d0f00eb4f",
    "_TrainBeginCallback.on_train_begin": "f48592a42d916d12",
    "_chain_to_sequences": "8689016fef7adf3a",
    "_flash_attn": "26d2f930796627de",
    "_iter_vllm_named_params": "7df575a1c1116a8d",
    "compute_flops_per_token": "c24cff3a04b72fdb",
}

# trl_compat.TRL_MAIN_COMMIT
PINNED_MAIN = {
    "AsyncGRPOConfig": "f71d9900d3fe973a",
    "AsyncGRPOTrainer.__init__": "9d79fa63af8c7932",
    "AsyncGRPOTrainer._streaming_iter": "56159bc6237badc9",
    "AsyncGRPOTrainer._sync_weight": "38e9a8cb3b240a06",
    "AsyncGRPOTrainer._sync_weight_merged": "e0a40ea9e210ca13",
    "AsyncGRPOTrainer.compute_loss": "38ddff39cf4d2a24",
    "AsyncGRPOTrainer.get_train_dataloader": "2950938428f1c183",
    "AsyncRolloutWorker.__init__": "4421bac6f1128f93",
    "DataCollatorForRollout": "3a90badab42f2bd1",
    "DataCollatorForRollout.torch_call": "8672ed6a0521baac",
    "FixedCountBatcher.__iter__": "121e176022912d3b",
    "RolloutQueueDataset.__iter__": "daba12e052526226",
    "TokenBudgetBatcher.__iter__": "f2321cfe667f9b41",
    "TreeRow.__init__": "cb6cc1d22b4a45c8",
    "TreeRow.add": "59d788cfc6cfe1fd",
    "TreeRow.fits": "e5f4e79d5fdfcab6",
    "TreeRow.pack": "0d96e77b80a883f9",
    "VLLMClient.__init__": "5e9e16c61d8281cc",
    "VLLMClient.load_lora_adapter": "1651c8d0459f3653",
    "VLLMClient.pause": "813e1c184e025b99",
    "_AsyncRolloutLoop.__init__": "7d5fded881cc9fb8",
    "_AsyncRolloutLoop._generate_one": "625ec24da81a925b",
    "_AsyncRolloutLoop._repeat_iterator": "ed97bff71aee925b",
    "_AsyncRolloutLoop._run_loops": "c449fe01077c1213",
    "_AsyncRolloutLoop._score_group": "15ee65aa76cfc061",
    "_AsyncRolloutLoop._score_loop": "a4356a487c050c7a",
    "_AsyncRolloutLoop.run": "5010a1b9cf1080c7",
    "_EpochStopCallback.on_step_end": "96f8c76d0f00eb4f",
    "_TrainBeginCallback.on_train_begin": "f48592a42d916d12",
    "_chain_to_sequences": "8689016fef7adf3a",
    "_flash_attn": "3476fb91552cfc9a",
    "_iter_vllm_named_params": "7df575a1c1116a8d",
    "compute_flops_per_token": "c24cff3a04b72fdb",
}


def _trl_definition(cls, name):
    for base in cls.__mro__[1:]:
        if base.__module__.startswith("trl.") and name in vars(base):
            member = vars(base)[name]
            try:
                inspect.getsource(_function(member))
            except OSError:
                # Generated by @dataclass: the fields it is made from are the class.
                return base.__name__, base
            return f"{base.__name__}.{name}", member
    return None


def _function(obj):
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    if isinstance(obj, property):
        obj = obj.fget
    return inspect.unwrap(obj) if callable(obj) else obj


# TRL's functions as they were before install_runtime_patches replaced them.
PATCHED = {
    "_flash_attn": trl_compat._trl_flash_attn,
    "compute_flops_per_token": trl_compat._trl_flops_per_token,
}


def _resolve(path):
    if path in PATCHED:
        return PATCHED[path]
    head, *rest = path.split(".")
    for module in MODULES:
        if hasattr(module, head):
            obj = getattr(module, head)
            for part in rest:
                obj = vars(obj)[part]
            return obj
    raise LookupError(path)


def _source_hash(obj):
    source = inspect.getsource(_function(obj))
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def upstream_hashes() -> dict[str, str]:
    upstream = {}
    for cls in SUBCLASSES:
        for name, member in vars(cls).items():
            if not isinstance(
                member, (types.FunctionType, staticmethod, classmethod, property)
            ):
                continue
            found = _trl_definition(cls, name)
            if found is not None:
                upstream[found[0]] = found[1]
    for path in trl_compat.UPSTREAM:
        upstream[path] = _resolve(path)
    return {path: _source_hash(obj) for path, obj in sorted(upstream.items())}


def test_upstream_sources_are_pinned():
    if version("trl") == trl_compat.TRL_MAIN_VERSION:
        name, pinned = "PINNED_MAIN", PINNED_MAIN
        against = f"TRL main at {trl_compat.TRL_MAIN_COMMIT}"
    else:
        name, pinned, against = "PINNED", PINNED, f"trl {trl_compat.TRL_VERSION}"
    current = upstream_hashes()
    changed = sorted(
        path for path in current.keys() & pinned.keys() if current[path] != pinned[path]
    )
    added = sorted(current.keys() - pinned.keys())
    removed = sorted(pinned.keys() - current.keys())
    table = "\n".join(f'    "{path}": "{digest}",' for path, digest in current.items())
    assert not (changed or added or removed), (
        f"TRL code that async GRPO overrides or copies differs from {against}: "
        f"changed {changed}, newly overridden {added}, gone {removed}. Re-check the "
        f"axolotl side, then set {name} to:\n{name} = {{\n{table}\n}}"
    )


def test_trainer_module_swaps_are_undone():
    original = async_grpo_trainer.TokenBudgetBatcher
    with pytest.raises(RuntimeError):
        with trl_compat.patched_trainer_module(TokenBudgetBatcher=object):
            assert async_grpo_trainer.TokenBudgetBatcher is object
            raise RuntimeError
    assert async_grpo_trainer.TokenBudgetBatcher is original


def test_upstream_sources_ignore_installed_patches():
    pinned = upstream_hashes()
    trl_compat.install_runtime_patches()
    assert upstream_hashes() == pinned


def test_runtime_patches_wrap_trls_own_functions():
    trl_compat.install_runtime_patches()
    trl_compat.install_runtime_patches()
    assert packing._flash_attn is trl_compat.tree_flash_attn
    assert async_grpo_trainer.compute_flops_per_token is trl_compat.flops_per_token
    assert trl_compat._trl_flash_attn is not trl_compat.tree_flash_attn
    assert trl_compat._trl_flops_per_token is not trl_compat.flops_per_token


def test_warns_that_trl_main_is_experimental(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        trl_compat, "version", lambda _name: trl_compat.TRL_MAIN_VERSION
    )
    monkeypatch.setattr(trl_compat.LOG, "warning_once", warnings.append)
    trl_compat.check_version()
    assert f"TRL main at {trl_compat.TRL_MAIN_COMMIT}" in warnings[0]


@pytest.mark.skipif(
    not trl_compat.SCORE_CENTERING, reason="TRL releases record no sampler fields"
)
def test_sampler_fields_pad_the_top_tokens():
    sequence = types.SimpleNamespace(
        sampling_mask=[[], [5, 6]], sampler_logprobs=[{}, {7: -0.5, 9: -1.5}]
    )
    fields = trl_compat.sampler_fields(sequence)
    assert fields["sampling_mask"] == [[], [5, 6]]
    assert fields["sampler_ids"].tolist() == [[-1, -1], [7, 9]]
    np.testing.assert_array_equal(
        fields["sampler_log_probs"], [[-math.inf, -math.inf], [-0.5, -1.5]]
    )
    unsampled = trl_compat.unsampled_fields(3)
    assert unsampled["sampler_ids"].shape == (3, 0)
    assert unsampled["sampling_mask"] == [[], [], []]


@pytest.mark.skipif(trl_compat.SCORE_CENTERING, reason="TRL main records them")
def test_trl_releases_take_no_sampler_fields():
    sequence = types.SimpleNamespace(sampling_mask=[[]], sampler_logprobs=[{}])
    assert trl_compat.sampler_fields(sequence) == {}
    assert trl_compat.unsampled_fields(3) == {}
    assert trl_compat.unsampled_sequence_fields(3) == {}


def test_warns_when_another_trl_is_installed(monkeypatch):
    warnings = []
    monkeypatch.setattr(trl_compat, "version", lambda _name: "9.9.9")
    monkeypatch.setattr(trl_compat.LOG, "warning_once", warnings.append)
    trl_compat.check_version()
    assert "trl 9.9.9 is installed" in warnings[0]
