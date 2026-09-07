"""
HF Transformers MambaConfig
"""

from transformers import PretrainedConfig


class MambaConfig(PretrainedConfig):
    """
    modeling configuration for state space model/mamba
    """

    model_type = "mamba"

    def __init__(
        self,
        vocab_size=50280,
        d_model=2560,
        n_layer=64,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
        pad_token_id=50277,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
