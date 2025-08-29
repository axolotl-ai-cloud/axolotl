"""
monkeypatch for Trainer _get_learning_rate method
"""

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# TODO remove this patch once https://github.com/huggingface/transformers/pull/37881 is included in a release
def _get_learning_rate(self):
    if self.is_deepspeed_enabled:
        # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
        # not run for the first few dozen steps while loss scale is too large, and thus during
        # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
        try:
            last_lr = self.lr_scheduler.get_last_lr()[0]
        except AssertionError as e:
            if "need to call step" in str(e):
                LOG.warning(
                    "tried to get lr value before scheduler/optimizer started stepping, returning lr=0"
                )
                last_lr = 0
            else:
                raise
    else:
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            last_lr = self.optimizer.param_groups[0]["lr"]
        else:
            last_lr = self.lr_scheduler.get_last_lr()[0]

    if torch.is_tensor(last_lr):
        last_lr = last_lr.item()
    return last_lr


def patch_trainer_get_lr():
    from transformers.trainer import Trainer

    Trainer._get_learning_rate = _get_learning_rate
