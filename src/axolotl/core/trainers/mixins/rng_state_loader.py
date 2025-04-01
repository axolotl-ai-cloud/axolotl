"""
Temporary fix/override for bug in resume from checkpoint

See https://github.com/huggingface/transformers/pull/37162

TODO: Remove when upstream added PR to release
"""

import logging
import os
import random

import numpy as np
import torch
from transformers import Trainer, is_torch_npu_available
from transformers.trainer import safe_globals
from transformers.trainer_pt_utils import set_rng_state_for_device
from transformers.training_args import ParallelMode

LOG = logging.getLogger(__name__)


class RngLoaderMixin(Trainer):
    """
    mixin for method override to load RNG states from a checkpoint
    """

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                LOG.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                LOG.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        # Use safe_globals to ensure numpy RNG states can be deserialized safely under PyTorch 2.6+,
        # which requires allowlisted classes when loading with weights_only=True.
        with safe_globals():
            checkpoint_rng_state = torch.load(rng_file)  # nosec B614
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])

        is_distributed = self.args.parallel_mode == ParallelMode.DISTRIBUTED
        if torch.cuda.is_available():
            set_rng_state_for_device(
                "CUDA", torch.cuda, checkpoint_rng_state, is_distributed
            )
        if is_torch_npu_available():
            set_rng_state_for_device(
                "NPU", torch.npu, checkpoint_rng_state, is_distributed
            )
