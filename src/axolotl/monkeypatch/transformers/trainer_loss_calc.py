"""
Module for patching transformers Trainer loss calculation to use nanmean.

This is needed for context parallelism since chunks of the input sequences may be fully
masked and return NaNs in the loss calculation.
"""

import importlib
import inspect

from transformers import Trainer

from axolotl.monkeypatch.utils import detab_code
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# pylint: disable=protected-access
def patch_evaluation_loop():
    """Patch the evaluation_loop method."""
    # Check if already patched
    if hasattr(Trainer, "_original_evaluation_loop"):
        LOG.info("Trainer.evaluation_loop already patched")
        return

    # Get the original method source
    evaluation_loop_source = inspect.getsource(Trainer.evaluation_loop)
    Trainer._original_evaluation_loop = evaluation_loop_source
    evaluation_loop_source, _ = detab_code(evaluation_loop_source)

    # Define the patterns to replace
    original_list_pattern = 'metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()'
    patched_list_pattern = 'metrics[f"{metric_key_prefix}_loss"] = np.nanmean(np.concatenate(all_losses)).item()'

    original_array_pattern = (
        'metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()'
    )
    patched_array_pattern = (
        'metrics[f"{metric_key_prefix}_loss"] = np.nanmean(all_losses).item()'
    )

    # Check if the patterns exist
    if (
        original_list_pattern not in evaluation_loop_source
        or original_array_pattern not in evaluation_loop_source
    ):
        LOG.warning(
            "Original loss calculation patterns not found in Trainer.evaluation_loop"
        )
        return

    # Apply the patches
    evaluation_loop_source = evaluation_loop_source.replace(
        original_list_pattern, patched_list_pattern
    )
    evaluation_loop_source = evaluation_loop_source.replace(
        original_array_pattern, patched_array_pattern
    )

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


# pylint: disable=protected-access
def patch_maybe_log_save_evaluate():
    """Patch the _maybe_log_save_evaluate method."""
    # Check if already patched
    if hasattr(Trainer, "_original_maybe_log_save_evaluate"):
        LOG.info("Trainer._maybe_log_save_evaluate already patched")
        return

    # Get the original method source
    maybe_log_source = inspect.getsource(Trainer._maybe_log_save_evaluate)
    Trainer._original_maybe_log_save_evaluate = maybe_log_source
    maybe_log_source, _ = detab_code(maybe_log_source)

    # Define the pattern to replace
    original_pattern = "tr_loss_scalar = self._nested_gather(tr_loss).mean().item()"
    patched_pattern = "tr_loss_scalar = self._nested_gather(tr_loss).nanmean().item()"

    # Check if the pattern exists
    if original_pattern not in maybe_log_source:
        LOG.warning(
            "Original tr_loss_scalar pattern not found in Trainer._maybe_log_save_evaluate"
        )
        return

    # Apply the patch
    maybe_log_source = maybe_log_source.replace(original_pattern, patched_pattern)

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
