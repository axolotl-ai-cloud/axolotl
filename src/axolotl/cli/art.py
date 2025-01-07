"""Axolotl ASCII logo utils."""

from art import text2art
from transformers.utils.import_utils import _is_package_available

from axolotl.utils.distributed import is_main_process

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


def print_dep_versions():
    """Prints versions of various axolotl dependencies."""
    packages = ["accelerate", "peft", "transformers", "trl", "torch", "bitsandbytes"]
    max_len = max(len(pkg) for pkg in packages)
    if is_main_process():
        print("*" * 40)
        print("**** Axolotl Dependency Versions *****")
        for pkg in packages:
            pkg_version = _is_package_available(pkg, return_version=True)
            print(f"{pkg: >{max_len}}: {pkg_version[1]: <15}")
        print("*" * 40)


def print_legacy_axolotl_text_art(suffix=None):
    """Prints axolotl ASCII art and dependency versions."""
    font = "nancyj"
    ascii_text = "  axolotl"
    if suffix:
        ascii_text += f"  x  {suffix}"
    ascii_art = text2art(ascii_text, font=font)

    if is_main_process():
        print(ascii_art)

    print_dep_versions()


def print_axolotl_text_art():
    """Prints axolotl ASCII art."""
    if is_main_process():
        print(AXOLOTL_LOGO)
