"""
fix for FSDP optimizer save in trainer w 4.47.0
"""

import inspect
import logging

from transformers import Trainer

from axolotl.monkeypatch.utils import detab_code

LOG = logging.getLogger("axolotl.monkeypatch.trainer_fsdp_save")

ORIGINAL_TRAINER_CODE = """

    delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled

"""

PATCHED_TRAINER_CODE = """

    delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

"""


def get_training_loop_code() -> str:
    training_loop = inspect.getsource(
        Trainer._inner_training_loop  # pylint: disable=protected-access
    )
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
    Trainer._original_inner_training_loop = (  # pylint: disable=protected-access
        training_loop
    )
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

    exec(  # pylint: disable=exec-used  # nosec B102
        "from transformers.trainer import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(training_loop, globals())  # pylint: disable=exec-used  # nosec B102
    LOG.info("patching _inner_training_loop for fsdp optimizer save")
    Trainer._inner_training_loop = (  # pylint: disable=protected-access
        _fixed_inner_training_loop  # pylint: disable=undefined-variable  # noqa: F821
    )
