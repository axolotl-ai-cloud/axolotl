"""The grad-norm overrides must not walk the parameters on the default path."""

from unittest.mock import MagicMock, patch

import torch
from torch import nn
from transformers import Trainer

from axolotl.core.trainers.mixins import DistributedParallelMixin


class _Trainer(DistributedParallelMixin):
    def __init__(self, ep=False, offload=False):  # pylint: disable=super-init-not-called
        self.args = MagicMock(max_grad_norm=1.0)
        self.accelerator = MagicMock()
        self._ep = ep
        self._offload = offload

    def _expert_parallel_enabled(self):
        return self._ep

    def _cpu_offloaded(self):
        return self._offload

    def _global_mesh(self):
        return None


def _model_that_forbids_parameter_walks():
    model = nn.Linear(2, 2)
    model.parameters = MagicMock(side_effect=AssertionError("parameters() walked"))
    return model


def test_default_path_skips_parameter_walk():
    trainer = _Trainer()
    model = _model_that_forbids_parameter_walks()
    with (
        patch.object(Trainer, "_clip_grad_norm", return_value=torch.tensor(1.0)),
        patch.object(Trainer, "_get_grad_norm", return_value=torch.tensor(2.0)),
    ):
        assert trainer._clip_grad_norm(model).item() == 1.0
        assert trainer._get_grad_norm(model).item() == 2.0


def test_cpu_offload_path_still_walks_parameters():
    trainer = _Trainer(offload=True)
    model = nn.Linear(2, 2)
    with (
        patch.object(Trainer, "_clip_grad_norm", return_value=torch.tensor(1.0)),
        patch.object(Trainer, "_get_grad_norm", return_value=torch.tensor(2.0)),
        patch(
            "axolotl.utils.gradient_clipping.has_cpu_offloaded_dtensor_gradients",
            return_value=False,
        ) as has_grads,
        patch(
            "axolotl.utils.gradient_clipping.has_cpu_offloaded_dtensor_parameters",
            return_value=False,
        ) as has_params,
    ):
        assert trainer._clip_grad_norm(model).item() == 1.0
        assert trainer._get_grad_norm(model).item() == 2.0
    has_grads.assert_called_once()
    has_params.assert_called_once()
