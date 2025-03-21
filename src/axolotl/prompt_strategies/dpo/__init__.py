"""
module for DPO style dataset transform strategies
"""

from functools import partial

from ..base import load as load_base

load = partial(load_base, module_base="axolotl.prompt_strategies.dpo")
