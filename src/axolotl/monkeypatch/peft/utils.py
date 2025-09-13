"""
Patch prepare_model_for_kbit_training to not upcast everything
"""

import inspect

import peft

import axolotl
from axolotl.monkeypatch.utils import detab_code
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

ORIGINAL_PREPARE_CODE = """
        for param in model.parameters():
            if (
                (param.dtype == torch.float16) or (param.dtype == torch.bfloat16)
            ) and param.__class__.__name__ != "Params4bit":
                param.data = param.data.to(torch.float32)
"""

PATCHED_PREPARE_CODE = """
        for name, param in model.named_parameters():
            if (
                (param.dtype == torch.float16) or (param.dtype == torch.bfloat16)
            ) and param.__class__.__name__ != "Params4bit" and all(embed_name not in name for embed_name in ["embed_tokens", "lm_head"]):
                param.data = param.data.to(torch.float32)
"""


def get_peft_prep_code() -> str:
    prepare = inspect.getsource(peft.utils.other.prepare_model_for_kbit_training)
    return prepare


def check_peft_prep_code_is_patchable() -> bool:
    prep_code = get_peft_prep_code()
    prep_code, _ = detab_code(prep_code)
    return ORIGINAL_PREPARE_CODE in prep_code


def patch_peft_prep_code():
    """
    monkeypatch create_accelerator_and_postprocess so it checks for additional kwargs
    """

    try:
        prep_code = get_peft_prep_code()
    except OSError:
        return
    peft.utils.other._original_create_accelerator_and_postprocess = prep_code
    prep_code, _ = detab_code(prep_code)
    if ORIGINAL_PREPARE_CODE not in prep_code:
        return

    prep_code = prep_code.replace(ORIGINAL_PREPARE_CODE, PATCHED_PREPARE_CODE)
    prep_code = prep_code.replace(
        "def prepare_model_for_kbit_training(",
        "def fixed_prepare_model_for_kbit_training(",
        1,
    )

    items_to_import = []
    for item in dir(peft.utils.other):
        if item in prep_code:
            items_to_import.append(item)

    exec(
        "from peft.utils.other import (" + ", ".join(x for x in items_to_import) + ")",
        globals(),
    )
    exec(prep_code, globals())
    LOG.info("patching prepare_model_for_kbit_training to allow for overrides")
    peft.utils.other.prepare_model_for_kbit_training = (
        fixed_prepare_model_for_kbit_training
    )
    axolotl.loaders.model.prepare_model_for_kbit_training = (
        fixed_prepare_model_for_kbit_training
    )
