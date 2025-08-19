"""Configuration classes for diffusion language models."""

from transformers import LlamaConfig, MistralConfig


class LlamaForDiffusionConfig(LlamaConfig):
    """Configuration class for Llama models with diffusion training."""
    
    model_type = "llama_diffusion"
    
    def __init__(
        self,
        mask_token_id: int = 32000,
        eps: float = 1e-3,
        importance_weighting: bool = False,
        sample_packing: bool = False,
        min_mask_ratio: float = 0.0,
        max_mask_ratio: float = 1.0,
        noise_schedule: str = "linear",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Diffusion-specific parameters
        self.mask_token_id = mask_token_id
        self.eps = eps
        self.importance_weighting = importance_weighting
        self.sample_packing = sample_packing
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.noise_schedule = noise_schedule


class MistralForDiffusionConfig(MistralConfig):
    """Configuration class for Mistral models with diffusion training."""
    
    model_type = "mistral_diffusion"
    
    def __init__(
        self,
        mask_token_id: int = 32000,
        eps: float = 1e-3,
        importance_weighting: bool = False,
        sample_packing: bool = False,
        min_mask_ratio: float = 0.0,
        max_mask_ratio: float = 1.0,
        noise_schedule: str = "linear",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Diffusion-specific parameters
        self.mask_token_id = mask_token_id
        self.eps = eps
        self.importance_weighting = importance_weighting
        self.sample_packing = sample_packing
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.noise_schedule = noise_schedule


# Keep the base class for backward compatibility but mark as deprecated
class DiffusionConfig(LlamaForDiffusionConfig):
    """
    Deprecated: Use LlamaForDiffusionConfig or MistralForDiffusionConfig instead.
    """
    
    model_type = "diffusion"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)