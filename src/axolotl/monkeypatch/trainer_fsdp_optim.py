"""
fix for FSDP optimizer save in trainer w 4.47.0
"""

import inspect

from transformers import Trainer

from axolotl.monkeypatch.utils import detab_code
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

ORIGINAL_TRAINER_CODE = """
                if delay_optimizer_creation:
                    self.optimizer = self.accelerator.prepare(self.optimizer)
"""

PATCHED_TRAINER_CODE = """
                if delay_optimizer_creation:
                    model = self.accelerator.prepare(self.model)
"""


def get_training_loop_code() -> str:
    training_loop = inspect.getsource(Trainer._inner_training_loop)
    return training_loop


def check_training_loop_is_patchable() -> bool:
    training_loop = get_training_loop_code()
    training_loop, _ = detab_code(training_loop)
    return ORIGINAL_TRAINER_CODE in training_loop


def patch_training_loop_for_fsdp():
    """
    monkeypatch for fixing the training loop for fsdp with optimizer save
    """

    try:
        training_loop = get_training_loop_code()
    except OSError:
        return
    Trainer._original_inner_training_loop = training_loop
    training_loop, _ = detab_code(training_loop)
    if ORIGINAL_TRAINER_CODE not in training_loop:
        return

    training_loop = training_loop.replace(ORIGINAL_TRAINER_CODE, PATCHED_TRAINER_CODE)
    training_loop = training_loop.replace(
        "def _inner_training_loop(",
        "def _fixed_inner_training_loop(",
        1,
    )

    # load imports necessary
    import transformers.trainer

    items_to_import = []
    for item in dir(transformers.trainer):
        if item in training_loop:
            items_to_import.append(item)

    exec(
        "from transformers.trainer import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(training_loop, globals())
    LOG.info("patching _inner_training_loop for fsdp optimizer save")
    Trainer._inner_training_loop = _fixed_inner_training_loop
