"""
module to freeze/unfreeze parameters by name
"""
import logging
import re

from axolotl.utils.distributed import is_main_process

LOG = logging.getLogger("axolotl.utils.freeze")


def freeze_parameters_except(model, regex_patterns):
    """
    Freezes all layers of the given model except for the layers that match given regex patterns.
    Periods in the patterns are treated as literal periods, not as wildcard characters.

    Parameters:
    - model (nn.Module): The PyTorch model to be modified.
    - regex_patterns (list of str): List of regex patterns to match layer names to keep unfrozen.

    Returns:
    None; the model is modified in place.
    """
    # Escape periods and compile the regex patterns
    compiled_patterns = [
        re.compile(pattern.replace(".", "\\.")) for pattern in regex_patterns
    ]

    # First, freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layers that match the regex patterns
    for name, param in model.named_parameters():
        if any(pattern.match(name) for pattern in compiled_patterns):
            if is_main_process():
                LOG.debug(f"unfreezing {name}")
            param.requires_grad = True
