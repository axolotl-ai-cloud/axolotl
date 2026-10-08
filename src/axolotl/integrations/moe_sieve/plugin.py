"""Axolotl integration for routing-guided packed-expert LoRA."""

import json
from dataclasses import fields
from pathlib import Path

from peft import PeftModel, get_peft_model
from transformers import set_seed

from axolotl.integrations.base import AdapterCapabilities, BasePlugin

from .args import MoeSieveConfig
from .peft import MoeSieveLoraConfig, register_selected_experts


def validate_runtime(cfg):
    """Keep unsupported weight layouts away from the compact adapter path."""
    unsupported = (
        "load_in_4bit",
        "load_in_8bit",
        "gptq",
        "qat",
        "fp8",
        "deepspeed",
        "relora",
        "lora_on_cpu",
        "torch_compile",
    )
    enabled = [key for key in unsupported if cfg.get(key)]
    for key in (
        "tensor_parallel_size",
        "sequence_parallel_degree",
    ):
        if (cfg.get(key) or 1) > 1:
            enabled.append(key)
    if cfg.get("experts_implementation") not in (
        None,
        "eager",
        "batched_mm",
        "scattermoe",
        "sonicmoe",
        "grouped_mm",
        "expert_parallel",
    ):
        enabled.append("experts_implementation")
    if (cfg.get("fsdp") or cfg.get("fsdp_config")) and cfg.get("fsdp_version") == 1:
        enabled.append("FSDP1")
    if (cfg.get("expert_parallel_size") or 1) > 1:
        if not cfg.get("fsdp_config"):
            enabled.append("expert_parallel_size requires FSDP2")
        if cfg.get("expert_parallel_backend") not in (None, "auto", "torch"):
            enabled.append("expert_parallel_backend (use torch)")
        state = (cfg.get("fsdp_config") or {}).get("state_dict_type")
        if state and state != "FULL_STATE_DICT":
            enabled.append("EP checkpoints require FULL_STATE_DICT")
    if cfg.get("expert_backend"):
        enabled.append("expert_backend")
    if enabled:
        raise ValueError(f"MoE-Sieve does not yet support: {', '.join(enabled)}")


class MoeSievePlugin(BasePlugin):
    """Provide a LoRA-like adapter without changing PEFT global registries."""

    def get_input_args(self):
        return "axolotl.integrations.moe_sieve.MoeSieveArgs"

    def get_adapter_capabilities(self):
        return [AdapterCapabilities(name="moe_sieve", lora_like=True)]

    def pre_model_load(self, cfg):
        if cfg.adapter == "moe_sieve":
            validate_runtime(cfg)

    def load_adapter(self, model, cfg, inference=False, config_only=False):
        if cfg.adapter != "moe_sieve":
            return None
        from axolotl.loaders.adapter import _build_peft_lora_config
        from axolotl.utils.train import determine_last_checkpoint

        validate_runtime(cfg)
        settings = MoeSieveConfig.model_validate(dict(cfg.get("moe_sieve") or {}))
        checkpoint = cfg.lora_model_dir
        if not inference and (
            cfg.resume_from_checkpoint or cfg.auto_resume_from_checkpoints
        ):
            checkpoint = determine_last_checkpoint(cfg) or checkpoint
        if checkpoint:
            config = MoeSieveLoraConfig.from_pretrained(checkpoint)
            if not config.moe_sieve_selection:
                raise ValueError("Checkpoint is missing its MoE-Sieve expert selection")
        else:
            if not settings.selection_file:
                raise ValueError(
                    "MoE-Sieve requires moe_sieve.selection_file; run the calibration command first"
                )
            profile = json.loads(Path(settings.selection_file).read_text())
            if profile.get("version") != 1:
                raise ValueError("Unsupported MoE-Sieve selection file version")
            if profile.get("base_model") != cfg.base_model:
                raise ValueError(
                    "MoE-Sieve selection was profiled on a different base model"
                )
            if profile.get("revision") != cfg.revision_of_model:
                raise ValueError(
                    "MoE-Sieve selection model revision differs from the training config"
                )
            if profile.get("fraction") != settings.fraction:
                raise ValueError(
                    "MoE-Sieve selection fraction differs from the training config"
                )
            base_config = _build_peft_lora_config(model, cfg)
            config = MoeSieveLoraConfig(
                **{
                    item.name: getattr(base_config, item.name)
                    for item in fields(base_config)
                    if item.init
                },
                moe_sieve_selection=profile["selection"],
            )
            if isinstance(config.target_parameters, str):
                raise ValueError(
                    "MoE-Sieve requires lora_target_parameters to be a list"
                )
            config.target_parameters = sorted(
                set(config.target_parameters or [])
                | {
                    f"{name}.{parameter}"
                    for name, spec in config.moe_sieve_selection.items()
                    for parameter in spec["parameter_shapes"]
                }
            )
        if config.lora_dropout:
            raise ValueError("MoE-Sieve packed experts require lora_dropout: 0")
        config.inference_mode = inference
        register_selected_experts(model, config)
        if config_only:
            return None, config
        kwargs = {}
        if cfg.peft_autocast_adapter_dtype is not None:
            kwargs["autocast_adapter_dtype"] = cfg.peft_autocast_adapter_dtype
        if checkpoint:
            from .distributed import local_adapter_checkpoint

            with local_adapter_checkpoint(model, config, checkpoint) as directory:
                model = PeftModel.from_pretrained(
                    model,
                    directory,
                    config=config,
                    is_trainable=not inference,
                    **kwargs,
                )
            cfg.lora_model_dir = checkpoint
        else:
            if cfg.seed is not None:
                set_seed(cfg.seed)
            model = get_peft_model(model, config, **kwargs)
        if cfg.lora_fp32_gradients:
            from axolotl.utils.lora_precision import upcast_lora_parameters

            upcast_lora_parameters(model)
        model.print_trainable_parameters()
        return model, config
