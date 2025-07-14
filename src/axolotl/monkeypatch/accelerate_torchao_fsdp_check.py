"""
Fix check in accelerate.Accelerator constructor logic.

This can be removed if / when this PR lands in a release:
https://github.com/huggingface/accelerate/pull/3677.
"""

import inspect

from accelerate.accelerator import Accelerator

from axolotl.monkeypatch.utils import detab_code
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

ORIGINAL_ACCELERATE_CODE = """
    if self.fp8_backend == "AO" and self.state.fsdp_plugin.cpu_ram_efficient_loading:
"""

PATCHED_ACCELERATE_CODE = """
    if self.fp8_backend == "AO" and hasattr(self.state, "fsdp_plugin") and self.state.fsdp_plugin.cpu_ram_efficient_loading:
"""


def get_accelerator_constructor_code() -> str:
    constructor = inspect.getsource(Accelerator.__init__)
    return constructor


def check_accelerator_constructor_code_is_patchable() -> bool:
    constructor_code = get_accelerator_constructor_code()
    constructor_code, _ = detab_code(constructor_code)
    return ORIGINAL_ACCELERATE_CODE in constructor_code


def patch_accelerator_constructor_code_for_fp8():
    """
    Monkeypatch for Accelerator constructor so torchao fp8 training works outside of
    FSDP training.
    """
    try:
        constructor_code = get_accelerator_constructor_code()
    except OSError:
        return

    Accelerator._original__init__ = constructor_code  # pylint: disable=protected-access
    constructor_code, _ = detab_code(constructor_code)
    if ORIGINAL_ACCELERATE_CODE not in constructor_code:
        return

    constructor_code = constructor_code.replace(
        ORIGINAL_ACCELERATE_CODE, PATCHED_ACCELERATE_CODE
    )
    constructor_code = constructor_code.replace(
        "def __init__(",
        "def _patched__init__(",
        1,
    )

    # load imports necessary
    import accelerate.accelerator

    items_to_import = []
    for item in dir(accelerate.accelerator):
        if item in constructor_code:
            items_to_import.append(item)

    exec(  # pylint: disable=exec-used  # nosec B102
        "from accelerate.accelerator import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(constructor_code, globals())
    Accelerator.__init__ = _patched__init__  # pylint: disable=protected-access  # pylint: disable=undefined-variable  # noqa: F821
    LOG.info("patched Accelerator.__init__ to fix torchao + FSDP guard")
