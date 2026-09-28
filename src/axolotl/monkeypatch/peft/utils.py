"""
Patch prepare_model_for_kbit_training to not upcast everything
"""

import inspect

import peft

import axolotl.loaders.model
from axolotl.loaders.utils import get_linear_embedding_layers
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
            ) and param.__class__.__name__ != "Params4bit" and all(embed_name not in name for embed_name in {embedding_modules}):
                param.data = param.data.to(torch.float32)
"""


def get_peft_prep_code() -> str:
    prepare = peft.utils.other.prepare_model_for_kbit_training
    source = getattr(prepare, "_axolotl_original_source", None)
    return source if source is not None else inspect.getsource(prepare)


def check_peft_prep_code_is_patchable() -> bool:
    prep_code = get_peft_prep_code()
    prep_code, _ = detab_code(prep_code)
    return ORIGINAL_PREPARE_CODE in prep_code


def patch_peft_prep_code(embedding_modules: list[str] | None = None):
    """
    monkeypatch prepare_model_for_kbit_training so it leaves the embeddings alone
    """
    if embedding_modules is None:
        embedding_modules = get_linear_embedding_layers("llama")

    try:
        prep_code = get_peft_prep_code()
    except OSError:
        return
    original_source = prep_code
    prep_code, _ = detab_code(prep_code)
    if ORIGINAL_PREPARE_CODE not in prep_code:
        return

    prep_code = prep_code.replace(
        ORIGINAL_PREPARE_CODE,
        PATCHED_PREPARE_CODE.format(embedding_modules=list(embedding_modules)),
    )
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
    fixed_prepare_model_for_kbit_training._axolotl_original_source = original_source
    LOG.info("patching prepare_model_for_kbit_training to allow for overrides")
    peft.utils.other.prepare_model_for_kbit_training = (
        fixed_prepare_model_for_kbit_training
    )
    axolotl.loaders.model.prepare_model_for_kbit_training = (
        fixed_prepare_model_for_kbit_training
    )
