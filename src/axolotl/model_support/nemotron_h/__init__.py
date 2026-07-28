"""NemotronH model support (hybrid Mamba/attention causal LM)."""

from axolotl.model_support.base import ModelSupport
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _needs_ssm_hybrid_patch(cfg) -> bool:
    return bool(cfg.sample_packing or (cfg.context_parallel_size or 0) > 1)


def _before_model_build(context: ModelHookContext) -> None:
    if not _needs_ssm_hybrid_patch(context.cfg):
        return
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHPreTrainedModel,
    )

    from axolotl.monkeypatch.models.nemotron_h.modeling import (
        patch_nemotron_h_modeling_packing,
    )

    patch_nemotron_h_modeling_packing()
    # supports_gradient_checkpointing is only enabled after
    # patch_nemotron_h_modeling_packing() installs the GC-compatible
    # NemotronHBlock.forward. Without the patch, upstream marks this
    # False because the original block forward is not GC-safe.
    NemotronHPreTrainedModel.supports_gradient_checkpointing = True


def _after_base_model_build(_context: ModelHookContext) -> None:
    # Must run after model build because NemotronHForCausalLM.__init__
    # calls register_nemotron_h_conversion_mapping() with overwrite=True,
    # which would clobber any earlier fix.
    fix_nemotron_h_conversion_mapping()


def fix_nemotron_h_conversion_mapping() -> None:
    """Remove the spurious embedding→embeddings WeightRenaming from the
    nemotron_h checkpoint conversion mapping.

    The nvidia Hub model registers:
        WeightRenaming("embedding.weight", "embeddings.weight")
    to handle a legacy checkpoint variant. Its reverse (applied on save)
    converts ``embeddings`` back to ``embedding``, which silently renames
    ``backbone.embeddings.weight`` → ``backbone.embedding.weight`` when
    merging LoRA adapters back into the base model.
    """
    try:
        from transformers.conversion_mapping import (
            WeightRenaming,
            get_checkpoint_conversion_mapping,
            register_checkpoint_conversion_mapping,
        )
    except ImportError:
        return

    mapping = get_checkpoint_conversion_mapping("nemotron_h")
    if mapping is None:
        return

    filtered = [
        entry
        for entry in mapping
        if not (
            isinstance(entry, WeightRenaming)
            and entry.source_patterns == ["embedding.weight"]
            and entry.target_patterns == ["embeddings.weight"]
        )
    ]
    if len(filtered) != len(mapping):
        register_checkpoint_conversion_mapping("nemotron_h", filtered, overwrite=True)
        LOG.info(
            "Removed embedding→embeddings WeightRenaming from nemotron_h "
            "checkpoint conversion mapping"
        )


@register_model_support
class NemotronHSupport(ModelSupport):
    """Descriptor for NemotronH hybrid SSM/attention models."""

    model_types = ("nemotron_h",)
    profile = ModelProfile(
        family=VANILLA_CAUSAL_LM,
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.BEFORE_MODEL_BUILD: (_before_model_build,),
                ModelHookPhase.AFTER_BASE_MODEL_BUILD: (_after_base_model_build,),
            }
        ),
    )
