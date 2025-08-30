"""Axolotl ASCII logo utils."""

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

HAS_PRINTED_LOGO = False


def print_axolotl_text_art():
    """Prints axolotl ASCII art."""

    global HAS_PRINTED_LOGO
    if HAS_PRINTED_LOGO:
        return
    if is_main_process():
        HAS_PRINTED_LOGO = True
        print(AXOLOTL_LOGO)
