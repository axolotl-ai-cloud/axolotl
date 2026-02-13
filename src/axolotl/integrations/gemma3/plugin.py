"""Plugin for loading Gemma3 multimodal checkpoints into Gemma3ForCausalLM (text-only).

Uses transformers v5's ``key_mapping`` parameter on ``from_pretrained`` to remap
``model.language_model.*`` keys to ``model.*``, discarding vision tower and projector
weights.  On save, transformers automatically reverses the mapping so saved
checkpoints retain the original ``model.language_model.*`` prefix.
"""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# key_mapping for transformers from_pretrained:
# Remap checkpoint keys matching ^model.language_model -> model
# Vision tower / projector keys won't match any model parameter and are discarded.
GEMMA3_KEY_MAPPING = {"^model.language_model": "model"}


class Gemma3TextFromMultimodalPlugin(BasePlugin):
    """Load a Gemma3 multimodal checkpoint as a text-only Gemma3ForCausalLM.

    Hooks
    -----
    register(cfg)
        Runs before config validation.  Sets the ``_extract_text_config`` flag,
        ensures ``model_type`` is ``Gemma3ForCausalLM``, and injects
        ``key_mapping`` into ``model_kwargs`` so that ``from_pretrained`` remaps
        ``model.language_model.*`` → ``model.*``.

    pre_model_load(cfg)
        Runs after config validation/normalization but before model instantiation.
        Validates that ``model_config_type`` is ``gemma3_text`` and
        ``is_multimodal`` is False (confirming that ``_extract_text_config``
        worked correctly).
    """

    def get_input_args(self) -> str:
        return "axolotl.integrations.gemma3.Gemma3TextFromMultimodalArgs"

    def register(self, cfg: dict):
        """Set up config for multimodal → text-only loading.

        This runs before Pydantic validation, so ``cfg`` is a raw dict.
        """
        if not cfg.get("gemma3_text_from_multimodal", True):
            LOG.info("Gemma3TextFromMultimodalPlugin: disabled via config")
            return

        LOG.info(
            "Gemma3TextFromMultimodalPlugin: configuring multimodal → text-only loading"
        )

        # Flag for load_model_config() to extract the text sub-config
        cfg["extract_text_config"] = True

        # Ensure model_type is set for the text-only model class
        if not cfg.get("model_type"):
            cfg["model_type"] = "Gemma3ForCausalLM"

        # Inject key_mapping into model_kwargs so from_pretrained remaps weights
        model_kwargs = cfg.setdefault("model_kwargs", {})
        model_kwargs["key_mapping"] = GEMMA3_KEY_MAPPING

    def pre_model_load(self, cfg):
        """Validate that config extraction worked before model instantiation."""
        if not getattr(cfg, "gemma3_text_from_multimodal", True):
            return

        if cfg.model_config_type != "gemma3_text":
            LOG.warning(
                "Gemma3TextFromMultimodalPlugin: expected model_config_type='gemma3_text' "
                "but got '%s'. The text config extraction may not have worked.",
                cfg.model_config_type,
            )

        if cfg.is_multimodal:
            LOG.warning(
                "Gemma3TextFromMultimodalPlugin: cfg.is_multimodal is True. "
                "The model will be loaded via the multimodal trainer path, "
                "which may not support sample packing or LoRA kernels."
            )

        LOG.info(
            "Gemma3TextFromMultimodalPlugin: model_config_type=%s, is_multimodal=%s",
            cfg.model_config_type,
            cfg.is_multimodal,
        )
