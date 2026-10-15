"""Native DiffusionGemma descriptor."""

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
from axolotl.model_support.native_adapters import (
    share_diffusion_gemma_tied_lora,
    validate_native_diffusion_lora,
)
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import DIFFUSION_LM


def _model_class() -> type:
    from .modeling import AxolotlDiffusionGemmaForBlockDiffusion

    return AxolotlDiffusionGemmaForBlockDiffusion


def _validate(context: ModelHookContext) -> None:
    diffusion_cfg = getattr(context.cfg, "diffusion_lm", None)
    self_conditioning = (
        diffusion_cfg.get("self_conditioning")
        if isinstance(diffusion_cfg, dict)
        else getattr(diffusion_cfg, "self_conditioning", None)
    )
    train_module = (
        self_conditioning.get("train_module", False)
        if isinstance(self_conditioning, dict)
        else getattr(self_conditioning, "train_module", False)
    )
    if train_module:
        modules_to_save = list(getattr(context.cfg, "lora_modules_to_save", None) or [])
        module_name = "model.decoder.self_conditioning"
        if module_name not in modules_to_save:
            modules_to_save.append(module_name)
        context.cfg.lora_modules_to_save = modules_to_save
    validate_native_diffusion_lora(context.cfg, model_name="DiffusionGemma")


def _after_adapter_load(context: ModelHookContext) -> None:
    share_diffusion_gemma_tied_lora(
        context.model, validate_saved=bool(getattr(context.cfg, "lora_model_dir", None))
    )


@register_model_support
class DiffusionGemmaSupport(ModelSupport):
    """Descriptor for text-only DiffusionGemma block diffusion."""

    model_types = ("diffusion_gemma",)

    @staticmethod
    def resolve_lora_merge_method(cfg, requested: str) -> str:
        """Use PEFT's loaded merge so tied factors share one merge ledger."""
        del cfg
        if requested == "memory_efficient":
            raise ValueError(
                "DiffusionGemma tied LoRA requires merge_method: legacy; "
                "memory_efficient merging cannot preserve its shared merge ledger."
            )
        return "legacy"

    profile = ModelProfile(
        family=DIFFUSION_LM,
        diffusion=DiffusionSpec(
            noise=DiffusionNoise.UNIFORM,
            layout=DiffusionLayout.ENCODER_CANVAS,
            logit_alignment=LogitAlignment.ALIGNED,
            first_position_alignment=FirstPositionAlignment.REQUIRES_PREDECESSOR,
            self_conditioning=True,
            max_canvas=256,
            max_context=None,
            eos_handling=EosHandling.INDEPENDENT,
            mask_token_policy=MaskTokenPolicy.NONE,
            default_time_weighting=TimeWeighting.NONE,
            objective_reduction=ObjectiveReduction.SUPERVISED_TOKEN_MEAN,
            time_floor=0.001,
            reduction_scope=ReductionScope.GLOBAL_WINDOW,
            generation_adapter=GenerationAdapter.ENCODER_CANVAS,
        ),
        capabilities={
            "fused_attn_kernel": Unsupported(
                "Native attention parity is not verified."
            ),
            "lora_kernels": Unsupported("Native tied LoRA uses ordinary PEFT layers."),
            "fsdp": Unsupported("Native tied LoRA has no FSDP validation."),
            "quantized_lora": Unsupported(
                "Native tied LoRA has no quantized validation."
            ),
        },
        strategies=ModelStrategyOverrides(auto_model_cls=_model_class),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.CONFIGURE_RUN: (_validate,),
                ModelHookPhase.AFTER_ADAPTER_LOAD: (_after_adapter_load,),
            }
        ),
    )
