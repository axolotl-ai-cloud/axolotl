"""
fix for FSDP2 evals when using torch.compile
"""

import inspect
import logging

from transformers import Trainer

from axolotl.monkeypatch.utils import detab_code

LOG = logging.getLogger(__name__)

ORIGINAL_TRAINER_CODE = """
    model.eval()
"""

PATCHED_TRAINER_CODE = """
    if hasattr(model, "eval") and callable(model.eval):
        self.model.eval()
"""


def get_evaluation_loop_code() -> str:
    training_loop = inspect.getsource(Trainer.evaluation_loop)
    return training_loop


def check_evaluation_loop_is_patchable() -> bool:
    eval_loop = get_evaluation_loop_code()
    eval_loop, _ = detab_code(eval_loop)
    return ORIGINAL_TRAINER_CODE in eval_loop


def patch_evaluation_loop_for_fsdp2():
    """
    monkeypatch for fixing the eval loop for fsdp2 with torch.compile
    """

    try:
        evaluation_loop = get_evaluation_loop_code()
    except OSError:
        return
    Trainer._original_evaluation_loop = (  # pylint: disable=protected-access
        evaluation_loop
    )
    evaluation_loop, _ = detab_code(evaluation_loop)
    if ORIGINAL_TRAINER_CODE not in evaluation_loop:
        return

    evaluation_loop = evaluation_loop.replace(
        ORIGINAL_TRAINER_CODE, PATCHED_TRAINER_CODE
    )
    evaluation_loop = evaluation_loop.replace(
        "def evaluation_loop(",
        "def _fixed_evaluation_loop(",
        1,
    )

    # load imports necessary
    import transformers.trainer

    items_to_import = []
    for item in dir(transformers.trainer):
        if item in evaluation_loop:
            items_to_import.append(item)

    exec(  # pylint: disable=exec-used  # nosec B102
        "from transformers.trainer import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(evaluation_loop, globals())  # pylint: disable=exec-used  # nosec B102
    LOG.info("patching _inner_training_loop for fsdp optimizer save")
    Trainer.evaluation_loop = (  # pylint: disable=protected-access
        _fixed_evaluation_loop  # pylint: disable=undefined-variable  # noqa: F821
    )
