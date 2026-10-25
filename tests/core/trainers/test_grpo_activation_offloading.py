"""GRPO trainers run their step under axolotl's activation offload context."""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers import Trainer
from trl.models.activation_offloading import OffloadActivations

from axolotl.core.trainers.grpo.trainer import (
    AxolotlAsyncGRPOTrainer,
    AxolotlGRPOTrainer,
)
from axolotl.core.trainers.mixins import (
    ActivationOffloadingMixin,
    activation_checkpointing as offload_mixin,
)


@pytest.mark.parametrize("trainer_cls", [AxolotlGRPOTrainer, AxolotlAsyncGRPOTrainer])
def test_grpo_trainers_wrap_the_trl_step(trainer_cls):
    mro = trainer_cls.__mro__
    trl_base = next(c for c in mro if c.__module__.startswith("trl."))
    # the mixin's training_step must run around the TRL trainer's
    assert mro.index(ActivationOffloadingMixin) < mro.index(trl_base)


class _TrlTrainerWithOffload(Trainer):
    """A TRL trainer that builds its own offload context, as DPO does."""

    def __init__(self, args, model):  # pylint: disable=super-init-not-called
        self.args = args
        self.model = model
        self.maybe_activation_offload_context = "trl-offloader"


class _Harness(ActivationOffloadingMixin, _TrlTrainerWithOffload):
    pass


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(4, 4)


@pytest.mark.parametrize("offloading", [True, False])
def test_trl_offload_context_is_replaced(offloading):
    with (
        mock.patch.object(
            torch.accelerator, "current_accelerator", return_value=torch.device("cpu")
        ),
        # CUDA streams are unavailable on a CPU host
        mock.patch.object(
            offload_mixin,
            "get_act_offloading_ctx_manager",
            lambda model: OffloadActivations(use_streams=False),
        ),
    ):
        trainer = _Harness(SimpleNamespace(activation_offloading=offloading), _Model())

    if offloading:
        assert isinstance(trainer.activation_offload_context, OffloadActivations)
        assert isinstance(
            trainer.maybe_activation_offload_context, contextlib.nullcontext
        )
    else:
        assert isinstance(trainer.activation_offload_context, contextlib.nullcontext)
        assert trainer.maybe_activation_offload_context == "trl-offloader"
