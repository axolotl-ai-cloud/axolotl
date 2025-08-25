"""
allow adding additional kwargs to Accelerator init
"""

import inspect

from transformers import Trainer

from axolotl.monkeypatch.utils import detab_code
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

ORIGINAL_TRAINER_CODE = """
    # create accelerator object
    self.accelerator = Accelerator(**args)
"""

PATCHED_TRAINER_CODE = """
    if hasattr(self, "additional_accelerator_args"):
        additional_args = self.additional_accelerator_args(fp8=True, enable_fsdp_float8_all_gather={enable_fsdp_float8_all_gather}, **args)
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


def patch_create_accelerate_code_for_fp8(enable_fsdp_float8_all_gather: bool):
    """
    Monkeypatch create_accelerator_and_postprocess so it checks for additional kwargs.
    """

    try:
        create_code = get_create_accelerate_code()
    except OSError:
        return
    Trainer._original_create_accelerator_and_postprocess = create_code
    create_code, _ = detab_code(create_code)
    if ORIGINAL_TRAINER_CODE not in create_code:
        return

    patched_trainer_code = PATCHED_TRAINER_CODE.format(
        enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather
    )
    create_code = create_code.replace(ORIGINAL_TRAINER_CODE, patched_trainer_code)
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

    exec(
        "from transformers.trainer import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(create_code, globals())
    LOG.info("patching create_accelerator_and_postprocess to allow for overrides")
    Trainer.create_accelerator_and_postprocess = (
        fixed_create_accelerator_and_postprocess
    )
