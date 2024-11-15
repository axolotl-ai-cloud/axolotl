"""
fix for FSDP gradient accumulation
see https://github.com/huggingface/transformers/pull/34645
"""
import inspect
import logging

from transformers.trainer import Trainer

from axolotl.monkeypatch.unsloth_ import detab_code

LOG = logging.getLogger("axolotl.monkeypatch.trainer_fsdp_grad_accumulation")

ORIGINAL_CONTEXT_CODE = """
                context = (
                    functools.partial(self.accelerator.no_sync, model=model)
                    if i == len(batch_samples) - 1
                    else contextlib.nullcontext
                )
"""

PATCHED_CONTEXT_CODE = """
                context = (
                    functools.partial(self.accelerator.no_sync, model=model)
                    if i != len(batch_samples) - 1
                    else contextlib.nullcontext
                )
"""


def get_training_loop_code() -> str:
    training_loop = inspect.getsource(
        Trainer._inner_training_loop  # pylint: disable=protected-access
    )
    return training_loop


def check_training_loop_is_patchable() -> bool:
    train_loop = get_training_loop_code()
    train_loop, _ = detab_code(train_loop)
    return ORIGINAL_CONTEXT_CODE in train_loop


def patch_training_loop_for_fsdp_grad_accum():
    """
    monkeypatch for fixing the training loop for FSDP gradient accumulation
    """

    train_loop = get_training_loop_code()
    Trainer._original_inner_training_loop = (  # pylint: disable=protected-access
        train_loop
    )
    train_loop, _ = detab_code(train_loop)
    assert (
        ORIGINAL_CONTEXT_CODE in train_loop
    ), "Original _inner_training_loop code not found"

    train_loop = train_loop.replace(ORIGINAL_CONTEXT_CODE, PATCHED_CONTEXT_CODE)
    train_loop = train_loop.replace(
        "def _inner_training_loop(",
        "def _fixed_inner_training_loop(",
        1,
    )

    # load imports necessary
    import transformers.trainer

    items_to_import = []
    for item in dir(transformers.trainer):
        if item in train_loop:
            items_to_import.append(item)

    exec(  # pylint: disable=exec-used  # nosec B102
        "from transformers.trainer import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(train_loop, globals())  # pylint: disable=exec-used  # nosec B102
    LOG.info("patching _inner_training_loop", main_process_only=True)
    Trainer._inner_training_loop = (  # pylint: disable=protected-access
        _fixed_inner_training_loop  # pylint: disable=undefined-variable  # noqa: F821
    )
