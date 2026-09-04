# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Pins the FSDP2 FULL_STATE_DICT key naming that axolotl now delegates to accelerate.

Under ``torch_compile: true`` the trainer's model is a dynamo ``OptimizedModule``
(transformers sets ``self.model = self.model_wrapped`` when FSDP is on), so a plain
``state_dict()`` prefixes every key with ``_orig_mod.``. Nothing between there and
``save_pretrained`` strips it, and the resulting checkpoint reloads with every key missing
and a randomly initialized model, silently.

accelerate's FSDP2 branch avoids this by going through ``get_model_state_dict``, whose
``_get_fqns`` drops ``_orig_mod`` and ``_checkpoint_wrapped_module``. axolotl used to
override that branch with a hand-rolled gather that skipped the unwrap, which is where the
bug came from. These tests fail if a future torch/accelerate bump stops normalizing the
names, or if the override comes back.
"""

from __future__ import annotations

import functools

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(4, 4)

    def forward(self, x):
        return self.mlp(x)


def _compiled_checkpointed_model():
    """A tiny stand-in for the trainer's model: activation-checkpointed, then compiled."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    model = nn.Sequential(_Block())
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        auto_wrap_policy=lambda module, recurse, nonwrapped_numel: isinstance(
            module, _Block
        ),
    )
    return torch.compile(model)


def test_compiled_state_dict_carries_the_prefix():
    """The failure mode, so the test below is not vacuous."""
    keys = _compiled_checkpointed_model().state_dict().keys()
    assert keys and all(k.startswith("_orig_mod.") for k in keys)
    # the checkpoint wrapper strips its own infix via a state-dict hook; compile does not
    assert not any("_checkpoint_wrapped_module" in k for k in keys)


def test_full_state_dict_normalizes_names():
    state_dict = get_model_state_dict(
        _compiled_checkpointed_model(), options=StateDictOptions(full_state_dict=True)
    )
    assert set(state_dict) == {"0.mlp.weight", "0.mlp.bias"}


def test_accelerate_fsdp2_patch_leaves_get_state_dict_alone(monkeypatch):
    """axolotl's rank0-only override is gone; accelerate's own gather is already rank0-only."""
    import accelerate

    from axolotl.monkeypatch.accelerate import fsdp2

    assert not hasattr(fsdp2, "get_state_dict")
    # the patch swaps this too; record it so pytest restores it
    monkeypatch.setattr(
        accelerate.accelerator,
        "fsdp2_prepare_model",
        accelerate.accelerator.fsdp2_prepare_model,
    )

    before = accelerate.Accelerator.get_state_dict
    fsdp2.patch_accelerate_fsdp2()
    assert accelerate.Accelerator.get_state_dict is before
