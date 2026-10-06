"""Native pinned Dream descriptor."""

from axolotl.model_support.base import ModelSupport, Unsupported
from axolotl.model_support.diffusion import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    EosHandling,
    FirstPositionAlignment,
    GenerationAdapter,
    LogitAlignment,
    MaskTokenPolicy,
    ObjectiveReduction,
    ReductionScope,
    TimeWeighting,
)
from axolotl.model_support.native_adapters import validate_native_diffusion_lora
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelMatchers,
    ModelProfile,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import DIFFUSION_LM

_DREAM_REVISIONS = {
    "Dream-org/Dream-v0-Instruct-7B": "05334cb9faaf763692dcf9d8737c642be2b2a6ae",
    "Dream-org/Dream-v0-Base-7B": "6572adb5535263e4d1a337b56942ba48b6dee2a9",
}


def _matches_cfg(cfg) -> bool:
    source = getattr(cfg, "base_model", None)
    if not isinstance(source, str):
        return False
    expected_revision = _DREAM_REVISIONS.get(source)
    revision = getattr(cfg, "revision_of_model", None)
    return expected_revision is not None and revision in {None, expected_revision}


def _model_class() -> type:
    from .compat import resolve_patched_dream_model_class

    class AutoDreamModel:
        def __new__(cls, config, **kwargs):
            return cls.from_config(config, **kwargs)

        @classmethod
        def from_pretrained(cls, model_source, **kwargs):
            from pathlib import Path

            revision = kwargs.get("revision")
            if revision is None and not Path(model_source).exists():
                revision = _DREAM_REVISIONS.get(str(model_source))
                if revision is not None:
                    kwargs["revision"] = revision
            model_class = resolve_patched_dream_model_class(
                model_source,
                revision=revision,
                local_files_only=bool(kwargs.get("local_files_only", False)),
            )
            model = model_class.from_pretrained(model_source, **kwargs)
            model.reset_rope_parameters()
            return model

        @classmethod
        def from_config(cls, config, **kwargs):
            source = getattr(config, "_name_or_path", None)
            if not source:
                raise ValueError(
                    "Dream from_config requires config._name_or_path to identify audited model code."
                )
            model_class = resolve_patched_dream_model_class(
                source,
                revision=getattr(config, "_commit_hash", None),
                local_files_only=bool(kwargs.get("local_files_only", False)),
            )
            kwargs.pop("trust_remote_code", None)
            return model_class._from_config(config, **kwargs)

    return AutoDreamModel


def _validate(context: ModelHookContext) -> None:
    if not getattr(context.cfg, "trust_remote_code", False):
        raise ValueError(
            "Dream native support requires trust_remote_code: true for its audited implementation."
        )
    validate_native_diffusion_lora(context.cfg, model_name="Dream")


@register_model_support
class DreamSupport(ModelSupport):
    """Descriptor for the audited pinned Dream remote implementation."""

    model_types = ("Dream",)
    profile = ModelProfile(
        family=DIFFUSION_LM,
        diffusion=DiffusionSpec(
            noise=DiffusionNoise.ABSORBING,
            layout=DiffusionLayout.FULL_SEQUENCE,
            logit_alignment=LogitAlignment.SHIFTED,
            first_position_alignment=FirstPositionAlignment.DUPLICATE_FIRST,
            self_conditioning=False,
            max_canvas=None,
            max_context=2048,
            eos_handling=EosHandling.TREAT_EOS_AS_ONE,
            mask_token_policy=MaskTokenPolicy.MODEL,
            default_time_weighting=TimeWeighting.INV_T,
            objective_reduction=ObjectiveReduction.MASKED_TOKEN_MEAN,
            time_floor=0.0,
            reduction_scope=ReductionScope.MICROBATCH,
            generation_adapter=GenerationAdapter.DREAM,
        ),
        capabilities={
            "fused_attn_kernel": Unsupported(
                "Native attention parity is not verified."
            ),
            "lora_kernels": Unsupported(
                "Native diffusion LoRA uses ordinary PEFT layers."
            ),
            "fsdp": Unsupported("Native Dream LoRA has no FSDP validation."),
            "quantized_lora": Unsupported(
                "Native Dream LoRA has no quantized validation."
            ),
        },
        strategies=ModelStrategyOverrides(auto_model_cls=_model_class),
        matchers=ModelMatchers(cfg=_matches_cfg),
        hooks=ModelHooks(by_phase={ModelHookPhase.CONFIGURE_RUN: (_validate,)}),
    )
