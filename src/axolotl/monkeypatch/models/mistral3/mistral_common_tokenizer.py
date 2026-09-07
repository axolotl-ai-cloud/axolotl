"""
Monkeypatch to fix inefficient tensor conversion in MistralCommonBackend.apply_chat_template
"""

import importlib
import inspect

from axolotl.monkeypatch.utils import detab_code
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def apply_mistral_tokenizer_image_patch():
    """Apply patch to MistralCommonBackend.apply_chat_template to fix image tensor conversion."""
    from transformers.tokenization_mistral_common import MistralCommonBackend

    # Get original source
    original_source = inspect.getsource(MistralCommonBackend.apply_chat_template)
    original_source, _ = detab_code(original_source)

    # Define the replacement
    original_tensor_conversion = (
        "                    pixel_values = torch.tensor(images)"
    )

    patched_tensor_conversion = """                    if isinstance(images, list) and len(images) > 0 and isinstance(images[0], np.ndarray):
                        pixel_values = torch.tensor(np.array(images))
                    else:
                        pixel_values = torch.tensor(images)"""

    # Apply the replacement
    if original_tensor_conversion in original_source:
        patched_source = original_source.replace(
            original_tensor_conversion, patched_tensor_conversion
        )
        patched_source = patched_source.replace(
            "def apply_chat_template(",
            "def patched_apply_chat_template(",
            1,
        )

        # Load necessary imports from the module
        module_name = MistralCommonBackend.__module__
        module = importlib.import_module(module_name)

        # Detect what needs to be imported
        items_to_import = []
        for item in dir(module):
            if item in patched_source and not item.startswith("_"):
                items_to_import.append(item)

        # Execute imports in global scope
        if items_to_import:
            exec(  # nosec B102
                f"from {module_name} import ({', '.join(items_to_import)})",
                globals(),
            )

        # Also need standard imports that might be used
        exec("import numpy as np", globals())  # nosec B102
        exec("import torch", globals())  # nosec B102
        exec("from typing import Union, Optional, List, Dict, Any, Callable", globals())  # nosec B102
        exec("from pathlib import Path", globals())  # nosec B102

        # Import other dependencies that might be needed
        try:
            exec("from transformers.utils import is_torch_available", globals())  # nosec B102
            exec(
                "from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, TensorType",
                globals(),
            )  # nosec B102
            exec("from transformers.utils import logging", globals())  # nosec B102
            exec("logger = logging.get_logger(__name__)", globals())  # nosec B102
        except ImportError as e:
            LOG.warning(f"Could not import some dependencies: {e}")

        # Execute the patched source
        exec(patched_source, globals())  # nosec B102

        # Replace the method
        MistralCommonBackend.apply_chat_template = patched_apply_chat_template
        LOG.info("Successfully applied MistralCommonBackend tensor conversion patch")
    else:
        LOG.warning("Could not find target code for MistralCommonBackend patching")
