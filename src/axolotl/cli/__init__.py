"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""
import logging
import os
import sys
from pathlib import Path

# add src to the pythonpath so we don't need to pip install this
from accelerate.commands.config import config_args
from art import text2art
from huggingface_hub import HfApi
from huggingface_hub.utils import LocalTokenNotFoundError
from transformers.utils.import_utils import _is_package_available

from axolotl.logging_config import configure_logging
from axolotl.utils.distributed import is_main_process

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

configure_logging()
LOG = logging.getLogger("axolotl.scripts")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

AXOLOTL_LOGO = """
     #@@ #@@      @@# @@#
    @@  @@          @@  @@           =@@#                               @@                 #@    =@@#.
    @@    #@@@@@@@@@    @@           #@#@=                              @@                 #@     .=@@
      #@@@@@@@@@@@@@@@@@            =@# @#     ##=     ##    =####=+    @@      =#####+  =#@@###.   @@
    @@@@@@@@@@/  +@@/  +@@          #@  =@=     #@=   @@   =@#+  +#@#   @@    =@#+  +#@#   #@.      @@
    @@@@@@@@@@  ##@@  ##@@         =@#   @#      =@# @#    @@      @@   @@    @@      #@   #@       @@
     @@@@@@@@@@@@@@@@@@@@          #@=+++#@=      =@@#     @@      @@   @@    @@      #@   #@       @@
                                  =@#=====@@     =@# @#    @@      @@   @@    @@      #@   #@       @@
    @@@@@@@@@@@@@@@@  @@@@        #@      #@=   #@=  +@@   #@#    =@#   @@.   =@#    =@#   #@.      @@
                                 =@#       @#  #@=     #@   =#@@@@#=    +#@@=  +#@@@@#=    .##@@+   @@
    @@@@  @@@@@@@@@@@@@@@@
"""


def print_legacy_axolotl_text_art(suffix=None):
    font = "nancyj"
    ascii_text = "  axolotl"
    if suffix:
        ascii_text += f"  x  {suffix}"
    ascii_art = text2art(ascii_text, font=font)

    if is_main_process():
        print(ascii_art)

    print_dep_versions()


def print_axolotl_text_art(
    **kwargs,  # pylint: disable=unused-argument
):
    if is_main_process():
        print(AXOLOTL_LOGO)


def print_dep_versions():
    packages = ["accelerate", "peft", "transformers", "trl", "torch", "bitsandbytes"]
    max_len = max(len(pkg) for pkg in packages)
    if is_main_process():
        print("*" * 40)
        print("**** Axolotl Dependency Versions *****")
        for pkg in packages:
            pkg_version = _is_package_available(pkg, return_version=True)
            print(f"{pkg: >{max_len}}: {pkg_version[1]: <15}")
        print("*" * 40)


def check_accelerate_default_config():
    if Path(config_args.default_yaml_config_file).exists():
        LOG.warning(
            f"accelerate config file found at {config_args.default_yaml_config_file}. This can lead to unexpected errors"
        )


def check_user_token():
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
