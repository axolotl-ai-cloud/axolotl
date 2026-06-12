"""DiffusionGemma block-diffusion training plugin for Axolotl."""

from __future__ import annotations

from peft import PeftModel
from transformers import PreTrainedModel

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_MODEL_CLASS = "DiffusionGemmaForBlockDiffusion"


class DiffusionGemmaPlugin(BasePlugin):
    """Plugin enabling block-diffusion fine-tuning of Google's DiffusionGemma.

    DiffusionGemma is an encoder-decoder MoE block-diffusion model. The encoder
    autoregressively consumes the prompt prefix into a KV cache, and the
    bidirectional decoder denoises a fixed-length canvas of target tokens. This
    plugin supplies the matching corruption process, loss, and data collation,
    and forces the non-standard ``DiffusionGemmaForBlockDiffusion`` model class.
    """

    def __init__(self):
        super().__init__()
        self.cfg = None

    def get_input_args(self) -> str:
        return "axolotl.integrations.diffusion_gemma.DiffusionGemmaArgs"

    def register(self, cfg: dict):
        """Force the DiffusionGemma block-diffusion model class.

        ``DiffusionGemmaForBlockDiffusion`` is not an ``AutoModelForCausalLM``
        head, so the loader must be told which transformers class to use. This
        runs at config-load time, before the loader reads ``type_of_model``.
        Loading this plugin is an explicit opt-in to DiffusionGemma training, so
        we set the class unconditionally when unset.
        """
        if not cfg.get("type_of_model"):
            cfg["type_of_model"] = _MODEL_CLASS
            LOG.info(f"DiffusionGemma: set type_of_model={_MODEL_CLASS}")

    def pre_model_load(self, cfg: DictDefault):
        if cfg.model_config_type != "diffusion_gemma":
            LOG.warning(
                "DiffusionGemmaPlugin is loaded but the base model config_type is "
                f"'{cfg.model_config_type}', not 'diffusion_gemma'."
            )
        # DiffusionGemma ties encoder experts to decoder experts; with
        # `quantize_moe_experts` the tied source becomes a parametrized 4-bit param
        # that transformers' tie_weights() can't resolve without help.
        if cfg.quantize_moe_experts:
            from .quant_compat import patch_tie_weights_for_quantized_experts

            patch_tie_weights_for_quantized_experts()

        # The 26B base is large; transformers' caching_allocator_warmup pre-allocates a
        # full bf16 copy of every parameter before loading, which doubles peak host/GPU
        # memory and can OOM a single device. Disable it (mirrors moe_quant's behavior).
        import transformers.modeling_utils as _mu

        if getattr(_mu, "caching_allocator_warmup", None) is not None:
            _mu.caching_allocator_warmup = lambda *args, **kwargs: None

    def pre_lora_load(self, cfg: DictDefault, model: PreTrainedModel):
        """PEFT's ``PeftModelForCausalLM`` (task_type CAUSAL_LM) reads
        ``base_model.prepare_inputs_for_generation``, which DiffusionGemma's custom
        block-diffusion generation mixin does not define. Training never uses it, so a
        stub is enough to let PEFT wrap the model."""
        if not hasattr(model, "prepare_inputs_for_generation"):
            model.prepare_inputs_for_generation = lambda *args, **kwargs: {}
        # HF Trainer's align_special_tokens reads generation_config.bos_token_id, which
        # DiffusionGemma's custom generation config omits (it only defines a subset).
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None and not hasattr(gen_cfg, "bos_token_id"):
            gen_cfg.bos_token_id = getattr(
                model.config.get_text_config(), "bos_token_id", None
            )

    def post_model_build(self, cfg: DictDefault, model: PreTrainedModel):
        """Quantize the fused experts to a frozen torchao FP4 format for ScatterMoE."""
        fmt = cfg.block_diffusion.frozen_fp4_experts
        if not fmt:
            return
        if not cfg.use_scattermoe:
            LOG.warning(
                "block_diffusion.frozen_fp4_experts requires use_scattermoe; skipping."
            )
            return
        from .quant_compat import quantize_experts_to_fp4

        quantize_experts_to_fp4(model, fmt)

    def post_model_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        self.cfg = cfg

    def get_trainer_cls(self, cfg: DictDefault):
        from .trainer import BlockDiffusionTrainer

        return BlockDiffusionTrainer

    def get_collator_cls_and_kwargs(self, cfg: DictDefault, is_eval: bool = False):
        from .collator import CanvasCollator

        canvas_length = cfg.block_diffusion.canvas_length or 256
        return CanvasCollator, {"canvas_length": canvas_length}

    def post_trainer_create(self, cfg: DictDefault, trainer):
        if hasattr(trainer, "axolotl_cfg"):
            trainer.axolotl_cfg = cfg
        trainer.post_set_axolotl_cfg()
