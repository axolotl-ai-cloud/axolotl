"""
Modeling module for Mamba models
"""


def fix_mamba_attn_for_loss():
    from mamba_ssm.models import mixer_seq_simple

    from .modeling_mamba import MambaLMHeadModel as MambaLMHeadModelFixed

    mixer_seq_simple.MambaLMHeadModel = MambaLMHeadModelFixed
    return mixer_seq_simple.MambaLMHeadModel  # pylint: disable=invalid-name
