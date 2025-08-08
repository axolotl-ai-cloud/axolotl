"""
Module for patching transformers Trainer loss calculation to use nanmean.

This is needed for context parallelism since chunks of the input sequences may be fully
masked and return NaNs in the loss calculation.

Also includes a patch for FSDP2 + torch.compile. We need to bundle this together with
the other evaluation_loop patch because we can't patch the same code twice without
raising an OSError.
"""

import importlib
import inspect

from transformers import Trainer

from axolotl.monkeypatch.utils import detab_code
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

ORIGINAL_EVAL_CODE = {
    "list": 'metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()',
    "array": 'metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()',
}
PATCHED_EVAL_CODE = {
    "list": 'metrics[f"{metric_key_prefix}_loss"] = np.nanmean(np.concatenate(all_losses)).item()',
    "array": 'metrics[f"{metric_key_prefix}_loss"] = np.nanmean(all_losses).item()',
}

ORIGINAL_FSDP2_CODE = """
    model.eval()
"""

PATCHED_FSDP2_CODE = """
    if hasattr(model, "eval") and callable(model.eval):
        self.model.eval()
"""

ORIGINAL_MAYBE_CODE = "tr_loss_scalar = self._nested_gather(tr_loss).mean().item()"
PATCHED_MAYBE_CODE = "tr_loss_scalar = self._nested_gather(tr_loss).nanmean().item()"


def check_evaluation_loop_is_patchable() -> bool:
    evaluation_loop_source = inspect.getsource(Trainer.evaluation_loop)
    return all(value in evaluation_loop_source for value in ORIGINAL_EVAL_CODE.values())


def check_evaluation_loop_is_fsdp2_patchable() -> bool:
    evaluation_loop_source = inspect.getsource(Trainer.evaluation_loop)
    evaluation_loop_source, _ = detab_code(evaluation_loop_source)
    return ORIGINAL_FSDP2_CODE in evaluation_loop_source


# pylint: disable=protected-access
def patch_evaluation_loop(patch_fsdp2: bool):
    """Patch the evaluation_loop method."""
    # Check if already patched
    if hasattr(Trainer, "_original_evaluation_loop"):
        LOG.info("Trainer.evaluation_loop already patched")
        return

    # Check if the patterns exist
    try:
        evaluation_loop_source = inspect.getsource(Trainer.evaluation_loop)
    except OSError:
        return
    Trainer.evaluation = evaluation_loop_source
    evaluation_loop_source, _ = detab_code(evaluation_loop_source)

    # Apply the nanmean patches
    evaluation_loop_source = evaluation_loop_source.replace(
        ORIGINAL_EVAL_CODE["list"], PATCHED_EVAL_CODE["list"]
    )
    evaluation_loop_source = evaluation_loop_source.replace(
        ORIGINAL_EVAL_CODE["array"], PATCHED_EVAL_CODE["array"]
    )

    # Apply FSDP2 eval guard patch if needed
    if patch_fsdp2 and ORIGINAL_FSDP2_CODE in evaluation_loop_source:
        evaluation_loop_source = evaluation_loop_source.replace(
            ORIGINAL_FSDP2_CODE, PATCHED_FSDP2_CODE
        )
        LOG.info("Applied FSDP2 eval guard patch to evaluation_loop")

    # Rename the function to avoid conflicts
    evaluation_loop_source = evaluation_loop_source.replace(
        "def evaluation_loop(",
        "def axolotl_evaluation_loop(",
        1,
    )

    # Get the module for necessary imports
    module_name = Trainer.__module__
    module = importlib.import_module(module_name)

    # Import necessary items from the module
    items_to_import = []
    for item in dir(module):
        if item in evaluation_loop_source:
            items_to_import.append(item)

    # Execute the imports and patched method
    exec(  # pylint: disable=exec-used  # nosec B102
        f"from {module_name} import ({', '.join(items_to_import)})",
        globals(),
    )
    exec(evaluation_loop_source, globals())  # pylint: disable=exec-used  # nosec B102

    LOG.info("Patched Trainer.evaluation_loop with nanmean loss calculation")
    Trainer.evaluation_loop = (
        axolotl_evaluation_loop  # pylint: disable=undefined-variable  # noqa: F821
    )


def check_maybe_log_save_evaluate_is_patchable() -> bool:
    maybe_log_source = inspect.getsource(Trainer._maybe_log_save_evaluate)
    return ORIGINAL_MAYBE_CODE in maybe_log_source


# pylint: disable=protected-access
def patch_maybe_log_save_evaluate():
    """Patch the _maybe_log_save_evaluate method."""
    # Check if already patched
    if hasattr(Trainer, "_original_maybe_log_save_evaluate"):
        LOG.info("Trainer._maybe_log_save_evaluate already patched")
        return

    # Check if the patterns exist
    try:
        maybe_log_source = inspect.getsource(Trainer._maybe_log_save_evaluate)
    except OSError:
        return
    Trainer._original_maybe_log_save_evaluate = maybe_log_source
    maybe_log_source, _ = detab_code(maybe_log_source)

    # Apply the patch
    maybe_log_source = maybe_log_source.replace(ORIGINAL_MAYBE_CODE, PATCHED_MAYBE_CODE)

    # Rename the function to avoid conflicts
    maybe_log_source = maybe_log_source.replace(
        "def _maybe_log_save_evaluate(",
        "def axolotl_maybe_log_save_evaluate(",
        1,
    )

    # Get the module for necessary imports
    module_name = Trainer.__module__
    module = importlib.import_module(module_name)

    # Import necessary items from the module
    items_to_import = []
    for item in dir(module):
        if item in maybe_log_source:
            items_to_import.append(item)

    # Execute the imports and patched method
    exec(  # pylint: disable=exec-used  # nosec B102
        f"from {module_name} import ({', '.join(items_to_import)})",
        globals(),
    )
    exec(maybe_log_source, globals())  # pylint: disable=exec-used  # nosec B102

    LOG.info("Patched Trainer._maybe_log_save_evaluate with nanmean loss calculation")
    Trainer._maybe_log_save_evaluate = axolotl_maybe_log_save_evaluate  # pylint: disable=undefined-variable  # noqa: F821
