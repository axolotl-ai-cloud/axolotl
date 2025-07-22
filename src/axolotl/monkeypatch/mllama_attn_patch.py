"""
Monkeypatch to add missing is_causal attribute to Mllama attention classes
"""

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_mllama_attention():
    """Add is_causal attribute to Mllama attention classes"""
    try:
        import transformers.models.mllama.modeling_mllama as mllama_modeling

        LOG.debug("Patching Attention in mllama due to missing attributes")

        if hasattr(mllama_modeling, "MllamaVisionAttention"):
            mllama_modeling.MllamaVisionAttention.is_causal = False

        if hasattr(mllama_modeling, "MllamaCrossAttention"):
            mllama_modeling.MllamaCrossAttention.is_causal = False

        if hasattr(mllama_modeling, "MllamaTextAttention"):
            mllama_modeling.MllamaTextAttention.is_causal = True

    except ImportError:
        LOG.debug("Mllama model not available, skipping is_causal patch")
