"""
allow adding additional kwargs to Accelerator init
"""

import inspect
import logging

from transformers import Trainer

from axolotl.monkeypatch.utils import detab_code

LOG = logging.getLogger(__name__)

ORIGINAL_TRAINER_CODE = """
    # create accelerator object
    self.accelerator = Accelerator(**args)
"""

PATCHED_TRAINER_CODE = """
    if hasattr(self, "additional_accelerator_args"):
        additional_args = self.additional_accelerator_args(fp8=True, **args)
        if additional_args:
            args.update(additional_args)

    # create accelerator object
    self.accelerator = Accelerator(**args)
"""


def get_create_accelerate_code() -> str:
    training_loop = inspect.getsource(Trainer.create_accelerator_and_postprocess)
    return training_loop


def check_create_accelerate_code_is_patchable() -> bool:
    create_code = get_create_accelerate_code()
    create_code, _ = detab_code(create_code)
    return ORIGINAL_TRAINER_CODE in create_code


def patch_create_accelerate_code_for_fp8():
    """
    monkeypatch create_accelerator_and_postprocess so it checks for additional kwargs
    """

    try:
        create_code = get_create_accelerate_code()
    except OSError:
        return
    Trainer._original_create_accelerator_and_postprocess = (  # pylint: disable=protected-access
        create_code
    )
    create_code, _ = detab_code(create_code)
    if ORIGINAL_TRAINER_CODE not in create_code:
        return

    create_code = create_code.replace(ORIGINAL_TRAINER_CODE, PATCHED_TRAINER_CODE)
    create_code = create_code.replace(
        "def create_accelerator_and_postprocess(",
        "def fixed_create_accelerator_and_postprocess(",
        1,
    )

    # load imports necessary
    import transformers.trainer

    items_to_import = []
    for item in dir(transformers.trainer):
        if item in create_code:
            items_to_import.append(item)

    exec(  # pylint: disable=exec-used  # nosec B102
        "from transformers.trainer import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(create_code, globals())  # pylint: disable=exec-used  # nosec B102
    LOG.info("patching create_accelerator_and_postprocess to allow for overrides")
    Trainer.create_accelerator_and_postprocess = fixed_create_accelerator_and_postprocess  # pylint: disable=protected-access  # pylint: disable=undefined-variable  # noqa: F821
