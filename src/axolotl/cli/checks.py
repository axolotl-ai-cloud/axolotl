"""Various checks for Axolotl CLI."""

import logging
import os
from pathlib import Path

from accelerate.commands.config import config_args
from huggingface_hub import HfApi
from huggingface_hub.utils import LocalTokenNotFoundError

from axolotl.logging_config import configure_logging

configure_logging()
LOG = logging.getLogger(__name__)


def check_accelerate_default_config() -> None:
    """Logs at warning level if no accelerate config file is found."""
    if Path(config_args.default_yaml_config_file).exists():
        LOG.warning(
            f"accelerate config file found at {config_args.default_yaml_config_file}. This can lead to unexpected errors"
        )


def check_user_token() -> bool:
    """Checks for HF user info. Check is skipped if HF_HUB_OFFLINE=1.

    Returns:
        Boolean indicating successful check (i.e., HF_HUB_OFFLINE=1 or HF user info is retrieved).

    Raises:
        LocalTokenNotFoundError: If HF user info can't be retrieved.
    """
    # Skip check if HF_HUB_OFFLINE is set to True
    if os.getenv("HF_HUB_OFFLINE") == "1":
        LOG.info(
            "Skipping HuggingFace token verification because HF_HUB_OFFLINE is set to True. Only local files will be used."
        )
        return True

    # Verify if token is valid
    api = HfApi()
    try:
        user_info = api.whoami()
        return bool(user_info)
    except LocalTokenNotFoundError:
        LOG.warning(
            "Error verifying HuggingFace token. Remember to log in using `huggingface-cli login` and get your access token from https://huggingface.co/settings/tokens if you want to use gated models or datasets."
        )
        return False
