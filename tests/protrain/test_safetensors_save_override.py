"""Force `save_safetensors: False` for ProTrain full-FT + offload paths.

ProTrain's shape-preserving expand-placeholder shares scratch storage across
released chunks. safetensors detects this as "shared tensors" and refuses to
save the full model. LoRA-scope saves are unaffected (only adapter weights
serialized). The save_safetensors auto-override only fires under three
combined conditions: full-FT (all params trainable), offload regime
(non-persistent chunks exist), and the trainer's save_safetensors is True.
"""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from axolotl.integrations.protrain.plugin import (
    _force_pickle_save_for_fullft_offload,
)


def _make_trainer_with_model(all_trainable: bool):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    for p in model.parameters():
        p.requires_grad_(all_trainable)
    args = SimpleNamespace(save_safetensors=True)
    trainer = SimpleNamespace(model=model, args=args)
    return trainer


def _make_wrapped(non_persistent_ids):
    chunk_manager = SimpleNamespace(_non_persistent_ids=non_persistent_ids)
    return SimpleNamespace(chunk_manager=chunk_manager)


def test_full_ft_with_offload_forces_pickle_save():
    """Full-FT + non-persistent chunks → save_safetensors flipped to False."""
    cfg = SimpleNamespace()
    trainer = _make_trainer_with_model(all_trainable=True)
    wrapped = _make_wrapped(non_persistent_ids={1, 2, 3})

    _force_pickle_save_for_fullft_offload(cfg, trainer, wrapped)

    assert trainer.args.save_safetensors is False


def test_no_offload_leaves_safetensors_alone():
    """All-persistent (n_persist == N_chunk) → no override."""
    cfg = SimpleNamespace()
    trainer = _make_trainer_with_model(all_trainable=True)
    wrapped = _make_wrapped(non_persistent_ids=set())

    _force_pickle_save_for_fullft_offload(cfg, trainer, wrapped)

    assert trainer.args.save_safetensors is True


def test_lora_scope_leaves_safetensors_alone():
    """When some params are frozen (LoRA), override does NOT fire."""
    cfg = SimpleNamespace()
    trainer = _make_trainer_with_model(all_trainable=False)
    wrapped = _make_wrapped(non_persistent_ids={1, 2, 3})

    _force_pickle_save_for_fullft_offload(cfg, trainer, wrapped)

    # LoRA-scope save isn't affected by the shared-tensor bug.
    assert trainer.args.save_safetensors is True


def test_existing_false_is_left_alone():
    """User-set save_safetensors=False → no-op (no double-flip, no warning)."""
    cfg = SimpleNamespace()
    trainer = _make_trainer_with_model(all_trainable=True)
    trainer.args.save_safetensors = False
    wrapped = _make_wrapped(non_persistent_ids={1, 2, 3})

    _force_pickle_save_for_fullft_offload(cfg, trainer, wrapped)

    assert trainer.args.save_safetensors is False


def test_missing_chunk_manager_is_safe():
    """Wrapped object without chunk_manager (defensive) is a no-op."""
    cfg = SimpleNamespace()
    trainer = _make_trainer_with_model(all_trainable=True)
    wrapped = SimpleNamespace(chunk_manager=None)

    _force_pickle_save_for_fullft_offload(cfg, trainer, wrapped)

    assert trainer.args.save_safetensors is True


def test_missing_trainer_args_is_safe():
    """Trainer without args attribute (test harness scenarios) is a no-op."""
    cfg = SimpleNamespace()
    trainer = SimpleNamespace(model=nn.Linear(4, 4))
    # Make the model "fully trainable" for the check.
    for p in trainer.model.parameters():
        p.requires_grad_(True)
    wrapped = _make_wrapped(non_persistent_ids={1, 2, 3})

    # Should not raise.
    _force_pickle_save_for_fullft_offload(cfg, trainer, wrapped)
