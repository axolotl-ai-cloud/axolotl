"""
Modeling module for Mamba models
"""

import importlib


def check_mamba_ssm_installed():
    mamba_ssm_spec = importlib.util.find_spec("mamba_ssm")
    if mamba_ssm_spec is None:
        raise ImportError(
            "MambaLMHeadModel requires mamba_ssm. Please install it with `pip install -e .[mamba-ssm]`"
        )


def fix_mamba_attn_for_loss():
    check_mamba_ssm_installed()

    from mamba_ssm.models import mixer_seq_simple

    from .modeling_mamba import MambaLMHeadModel as MambaLMHeadModelFixed

    mixer_seq_simple.MambaLMHeadModel = MambaLMHeadModelFixed
    return mixer_seq_simple.MambaLMHeadModel  # pylint: disable=invalid-name
