"""Adapter loading functionality, including LoRA / QLoRA and associated utils"""

import os
import types
from typing import Any

import bitsandbytes as bnb
import torch
from bitsandbytes.nn import Params4bit
from peft import (
    AdaptionPromptConfig,
    LoftQConfig,
    LoraConfig,
    PeftConfig,
    PeftMixedModel,
    PeftModel,
    TaskType,
    get_peft_model,
)
from transformers import PreTrainedModel

from axolotl.loaders.adapters.builders.factory import AdapterBuilderFactory
from axolotl.loaders.utils import get_linear_embedding_layers
from axolotl.telemetry.errors import send_errors
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def setup_quantized_meta_for_peft(model: torch.nn.Module):
    """Replaces `quant_state.to` with a dummy function to prevent PEFT from moving `quant_state` to meta device"""

    def temp_to_method(self, *args, **kwargs):
        return self

    for param in model.parameters():
        if isinstance(param, Params4bit) and param.quant_state is not None:
            param.quant_state._orig_to = param.quant_state.to
            param.quant_state.to = types.MethodType(temp_to_method, param.quant_state)


def setup_quantized_peft_meta_for_training(model: torch.nn.Module):
    """Replaces dummy `quant_state.to` method with the original function to allow training to continue"""
    for param in model.parameters():
        if isinstance(param, Params4bit) and hasattr(param.quant_state, "_orig_to"):
            param.quant_state.to = param.quant_state._orig_to
            param.quant_state._orig_to = None


def find_all_linear_names(model):
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    embedding_modules = get_linear_embedding_layers(model.config.model_type)
    output_embedding = embedding_modules[1]
    if output_embedding in lora_module_names:  # needed for 16-bit
        lora_module_names.remove(output_embedding)

    return list(lora_module_names)


def _patch_peft_clippable_linear():
    """Patch PEFT to handle Gemma4ClippableLinear which wraps nn.Linear.

    Gemma4's vision tower uses ClippableLinear (a thin wrapper around nn.Linear
    that clips activations). PEFT doesn't recognise it as a supported layer type,
    so we redirect LoRA injection to the inner ``.linear`` child instead.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ClippableLinear as _cls,
        )
    except ImportError:
        return

    from peft.tuners.lora.model import LoraModel

    if getattr(LoraModel, "_axolotl_clippable_patched", False):
        return
    _orig = LoraModel._create_and_replace

    def _patched(
        self,
        peft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key=None,
        **kw,
    ):
        if isinstance(target, _cls):
            # Redirect to the inner nn.Linear so PEFT can wrap it normally.
            return _orig(
                self,
                peft_config,
                adapter_name,
                target.linear,
                "linear",
                target,
                current_key=current_key,
                **kw,
            )
        return _orig(
            self,
            peft_config,
            adapter_name,
            target,
            target_name,
            parent,
            current_key=current_key,
            **kw,
        )

    LoraModel._create_and_replace = _patched
    LoraModel._axolotl_clippable_patched = True


def _peft_will_auto_convert_target_params(model, lora_config) -> bool:
    """Check whether PEFT will auto-populate target_parameters for this model.

    PEFT 0.19's ``convert_peft_config_for_transformers`` rewrites old MoE
    ``target_modules`` (e.g. ``w1``/``w2``/``w3`` on Mixtral) into
    ``target_parameters`` (``gate_up_proj``/``down_proj``) because
    transformers v5 fused those expert linears into 3D ``nn.Parameter``
    tensors. PEFT wraps the resulting 3D params with ``ParamWrapper``,
    which rejects ``lora_dropout != 0``. This probe runs the conversion on
    a copy of the config so we can detect the situation before
    ``get_peft_model`` blows up.
    """
    if getattr(lora_config, "target_parameters", None):
        return False

    try:
        from peft.utils.transformers_weight_conversion import (
            convert_peft_config_for_transformers,
            get_model_conversion_mapping,
        )
    except ImportError:
        return False

    import copy

    probe_cfg = copy.deepcopy(lora_config)
    try:
        convert_peft_config_for_transformers(
            probe_cfg,
            model=model,
            conversions=get_model_conversion_mapping(model),
        )
    except Exception:  # pylint: disable=broad-except
        return False

    return bool(getattr(probe_cfg, "target_parameters", None))


def _patch_peft_param_wrapper_dropout():
    """Let PEFT's ``ParamWrapper`` silently accept ``lora_dropout != 0``.

    ``ParamWrapper`` wraps 3D expert ``nn.Parameter`` tensors and rejects
    non-zero dropout because dropout can't be factored out of
    ``lora_B(lora_A(dropout(x)))`` when the inner op is an expert-indexed
    matmul. For mixed configs (attention + MoE experts) this is too
    aggressive — the non-expert ``Linear`` LoRA layers *can* apply dropout
    and that's usually what the user intended. We pass a copy of the
    ``LoraConfig`` with ``lora_dropout=0`` only to ``ParamWrapper.__init__``
    so it builds with ``nn.Identity`` for its internal dropout slot while
    every other layer type still receives the real dropout value.
    """
    from peft.tuners.lora.layer import ParamWrapper

    if getattr(ParamWrapper, "_axolotl_dropout_patched", False):
        return

    _orig_init = ParamWrapper.__init__

    def _patched_init(
        self,
        base_layer,
        adapter_name,
        parameter_name,
        config,
        *args,
        **kwargs,
    ):
        if getattr(config, "lora_dropout", 0):
            import copy as _copy

            patched_config = _copy.copy(config)
            patched_config.lora_dropout = 0.0
            return _orig_init(
                self,
                base_layer,
                adapter_name,
                parameter_name,
                patched_config,
                *args,
                **kwargs,
            )
        return _orig_init(
            self,
            base_layer,
            adapter_name,
            parameter_name,
            config,
            *args,
            **kwargs,
        )

    ParamWrapper.__init__ = _patched_init
    ParamWrapper._axolotl_dropout_patched = True


def load_lora(
    model: PreTrainedModel,
    cfg: DictDefault,
    inference: bool = False,
    config_only: bool = False,
) -> tuple[PreTrainedModel | PeftModel | PeftMixedModel | None, PeftConfig | None]:
    _patch_peft_clippable_linear()
    lora_target_modules = cfg.lora_target_modules or []
    lora_target_parameters = cfg.lora_target_parameters or []

    if cfg.lora_target_linear:
        linear_names = find_all_linear_names(model)
        LOG.info(f"found linear modules: {repr(sorted(linear_names))}")
        lora_target_modules_as_list = (
            lora_target_modules
            if isinstance(lora_target_modules, list)
            else [lora_target_modules]
        )
        lora_target_modules = list(set(lora_target_modules_as_list + linear_names))

    lora_config_kwargs = {}
    loftq_bits = cfg.peft and cfg.peft.loftq_config and cfg.peft.loftq_config.loftq_bits
    if loftq_bits:
        lora_config_kwargs["loftq_config"] = LoftQConfig(loftq_bits=loftq_bits)
        lora_config_kwargs["init_lora_weights"] = "loftq"
    if cfg.peft_init_lora_weights:
        lora_config_kwargs["init_lora_weights"] = cfg.peft_init_lora_weights
    if cfg.peft_use_dora:
        lora_config_kwargs["use_dora"] = cfg.peft_use_dora
        LOG.info("Initializing LoRA weights using dora. This might take longer.")
    if cfg.peft_use_rslora:
        lora_config_kwargs["use_rslora"] = cfg.peft_use_rslora
    if cfg.peft_layer_replication:
        lora_config_kwargs["layer_replication"] = cfg.peft_layer_replication
    if cfg.peft_trainable_token_indices:
        lora_config_kwargs["trainable_token_indices"] = cfg.peft_trainable_token_indices
    if cfg.peft_ensure_weight_tying is not None:
        lora_config_kwargs["ensure_weight_tying"] = cfg.peft_ensure_weight_tying

    # Determine the correct PEFT task type
    model_cls = type(model).__name__
    if "SequenceClassification" in model_cls:
        task_type = TaskType.SEQ_CLS
    elif "TokenClassification" in model_cls:
        task_type = TaskType.TOKEN_CLS
    else:
        task_type = TaskType.CAUSAL_LM

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=lora_target_modules,
        target_parameters=lora_target_parameters,
        layers_to_transform=cfg.peft_layers_to_transform,
        layers_pattern=cfg.peft_layers_pattern,
        lora_dropout=cfg.lora_dropout,
        fan_in_fan_out=cfg.lora_fan_in_fan_out,
        modules_to_save=cfg.lora_modules_to_save if cfg.lora_modules_to_save else None,
        exclude_modules=getattr(cfg, "lora_exclude_modules", None) or None,
        bias="none",
        task_type=task_type,
        **lora_config_kwargs,
    )

    if config_only:
        return None, lora_config

    if getattr(
        lora_config, "lora_dropout", 0
    ) and _peft_will_auto_convert_target_params(model, lora_config):
        LOG.warning(
            "lora_dropout=%s requested but PEFT will wrap this model's fused "
            "MoE expert parameters with ParamWrapper, which cannot apply "
            "dropout (the 3D einsum can't factor dropout out of "
            "lora_B(lora_A(dropout(x)))). Dropout will still be applied to "
            "non-expert LoRA layers (e.g. attention), and expert LoRA layers "
            "will use nn.Identity for the dropout slot.",
            lora_config.lora_dropout,
        )
        _patch_peft_param_wrapper_dropout()

    rank = int(os.environ.get("LOCAL_RANK", 0))

    if (
        cfg.fsdp_config
        and cfg.adapter
        and cfg.fsdp_config.cpu_ram_efficient_loading
        and rank != 0
    ):
        setup_quantized_meta_for_peft(model)

    model_kwargs: Any = {}
    if cfg.peft_autocast_adapter_dtype is not None:
        model_kwargs["autocast_adapter_dtype"] = cfg.peft_autocast_adapter_dtype

    if cfg.lora_model_dir:
        LOG.debug("Loading pretrained PEFT - LoRA")
        if cfg.lora_on_cpu:
            model_kwargs["max_memory"] = {"cpu": "256GiB"}
            model_kwargs["device_map"] = {"": "cpu"}
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            is_trainable=(not inference),
            **model_kwargs,
        )
    else:
        model = get_peft_model(model, lora_config, **model_kwargs)

    # FP8 models: LoRA A/B inherit FP8 dtype from base weights, but training
    # requires a compute dtype (bf16/fp16). Cast trainable LoRA params.
    if cfg.torch_dtype:
        _fp8_cast_dtype = cfg.torch_dtype
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        _fp8_cast_dtype = torch.bfloat16
    else:
        _fp8_cast_dtype = torch.float16
    for _name, param in model.named_parameters():
        if param.requires_grad and param.dtype == torch.float8_e4m3fn:
            param.data = param.data.to(_fp8_cast_dtype)

    if rank == 0:
        try:
            model.print_trainable_parameters()
        except AttributeError as exc:
            LOG.warning(
                "Exception caught during model.print_trainable_parameters(): %s", exc
            )
    elif (
        cfg.fsdp_config
        and cfg.adapter
        and cfg.fsdp_config.cpu_ram_efficient_loading
        and rank != 0
    ):
        setup_quantized_peft_meta_for_training(model)

    return model, lora_config


@send_errors
def load_adapter(
    model: PreTrainedModel,
    cfg: DictDefault,
    adapter: str | None,
    inference: bool = False,
    config_only: bool = False,
) -> tuple[PreTrainedModel | PeftModel | PeftMixedModel | None, PeftConfig | None]:
    try:
        if adapter is None:
            return model, None
        builder = AdapterBuilderFactory.create_builder(adapter, cfg)

        config = builder.build_config(model)

        if config_only:
            return None, config

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        model = builder.build_model(model, config, inference=inference)
        return model, config

    except ValueError as e:
        LOG.debug(
            f"Builder pattern failed, falling back to legacy adapter loading: {e}"
        )

        if adapter is None:
            return model, None
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if adapter in ["lora", "qlora"]:
            peft_model, lora_config = load_lora(
                model, cfg, inference=inference, config_only=config_only
            )
            return peft_model, lora_config
        if adapter == "llama-adapter":
            peft_model, lora_config = load_llama_adapter(model, cfg)
            return peft_model, lora_config

        raise NotImplementedError(f"{adapter} PEFT adapter not available") from None


def load_llama_adapter(
    model: PreTrainedModel, cfg: DictDefault
) -> tuple[PeftModel | PeftMixedModel, PeftConfig]:
    peft_config = AdaptionPromptConfig(
        adapter_layers=cfg.peft_adapter.layers,  # layers (L)
        adapter_len=cfg.peft_adapter.len,  # prompt length (K)
        task_type="CAUSAL_LM",
    )

    if cfg.lora_model_dir:
        LOG.debug("Loading pretrained PEFT - llama_adapter")
        peft_model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            torch_dtype=torch.float16,
        )
    else:
        peft_model = get_peft_model(model, peft_config)

    peft_model.print_trainable_parameters()

    return peft_model, peft_config
