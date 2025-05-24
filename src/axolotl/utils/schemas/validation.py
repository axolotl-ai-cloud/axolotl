"""Module with validation methods for config pydantic model."""

import logging

from pydantic import (
    field_validator,
    model_validator,
)
from transformers.utils.import_utils import is_torch_npu_available

from axolotl.utils.schemas.enums import ChatTemplate, RingAttnFunc, RLType

LOG = logging.getLogger(__name__)

SUPPORTED_METRICS = {"sacrebleu", "comet", "ter", "chrf", "perplexity"}


class DatasetValidationMixin:
    """Validation methods related to dataset configuration."""

    @field_validator("datasets", mode="before")
    @classmethod
    def deprecate_sharegpt_datasets(cls, datasets):
        for _, ds_cfg in enumerate(datasets):
            ds_type = (
                ds_cfg.get("type")
                if isinstance(ds_cfg, dict)
                else getattr(ds_cfg, "type", None)
            )
            if not ds_type:
                continue

            if isinstance(ds_type, dict):
                continue

            if isinstance(ds_type, str) and ds_type.startswith("sharegpt"):
                raise ValueError(
                    "`type: sharegpt.*` is deprecated. Please use `type: chat_template` instead."
                )

        return datasets

    @model_validator(mode="before")
    @classmethod
    def check_dataset_or_pretraining_dataset(cls, data):
        if data.get("datasets") is None and data.get("pretraining_dataset") is None:
            raise ValueError("either datasets or pretraining_dataset is required")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_push_ds_auth(cls, data):
        if (
            data.get("push_dataset_to_hub")
            and data.get("hf_use_auth_token") is not True
        ):
            raise ValueError(
                "Require cfg.hf_use_auth_token to be True for push_dataset_to_hub"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_val_w_test_datasets(cls, data):
        if data.get("test_datasets") and data.get("val_set_size"):
            raise ValueError(
                "non-zero val_set_size should not be used with test_datasets configuration"
            )
        return data


class AttentionValidationMixin:
    """Validation methods related to attention mechanisms."""

    @model_validator(mode="before")
    @classmethod
    def check_attention_fields(cls, data):
        fields = (
            "xformers_attention",
            "sdp_attention",
            "s2_attention",
            "flash_attention",
            "flex_attention",
        )
        non_empty_count = sum(1 for field in fields if data.get(field))

        if non_empty_count > 1:
            raise ValueError(f"Only one of {', '.join(fields)} must be set")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_sample_packing_without_attention(cls, data):
        if (
            data.get("sample_packing")
            and not data.get("flash_attention")
            and not data.get("sdp_attention")
            and not data.get("flex_attention")
            and not data.get("xformers_attention")
        ):
            LOG.warning(
                "sample_packing without flash, sdp, xformers or flex attention does not handle cross sample decontamination."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_sample_packing_with_s2attn(cls, data):
        if data.get("sample_packing") and data.get("s2_attention"):
            raise ValueError(
                "Received `sample_packing=true` and `s2_attention=true`; however, \
                shifted-sparse attention does not currently support sample packing."
            )
        return data


class TrainingValidationMixin:
    """Validation methods related to training configuration."""

    @model_validator(mode="before")
    @classmethod
    def check_batch_size_fields(cls, data):
        fields = ("micro_batch_size", "gradient_accumulation_steps", "batch_size")
        non_empty_count = sum(1 for field in fields if data.get(field))

        if non_empty_count < 2:
            raise ValueError(f"At least two of {', '.join(fields)} must be set")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_gas_bsz(cls, data):
        if data.get("gradient_accumulation_steps") and data.get("batch_size"):
            raise ValueError(
                "please set only one of gradient_accumulation_steps or batch_size"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_warmup(cls, data):
        if data.get("warmup_steps") and data.get("warmup_ratio"):
            raise ValueError("warmup_steps and warmup_ratio are mutually exclusive")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_saves(cls, data):
        if (
            data.get("save_strategy")
            and data.get("save_steps")
            and data.get("save_strategy") != "steps"
        ):
            raise ValueError(
                "save_strategy and save_steps mismatch. Please set save_strategy to 'steps' or remove save_steps."
            )
        if data.get("saves_per_epoch") and data.get("save_steps"):
            raise ValueError(
                "save_steps and saves_per_epoch are mutually exclusive and cannot be used together."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_evals(cls, data):
        if (
            data.get("eval_strategy")
            and data.get("eval_steps")
            and data.get("eval_strategy") != "steps"
        ):
            raise ValueError(
                "eval_strategy and eval_steps mismatch. Please set eval_strategy to 'steps' or remove eval_steps."
            )

        if (
            data.get("val_set_size") == 0
            and (data.get("eval_steps") or data.get("eval_strategy"))
            and not data.get("test_datasets")
            and data.get("eval_strategy") != "no"
        ):
            raise ValueError(
                "eval_steps and eval_strategy are not supported with val_set_size == 0"
            )
        if data.get("evals_per_epoch") and data.get("eval_steps"):
            raise ValueError(
                "eval_steps and evals_per_epoch are mutually exclusive and cannot be used together."
            )
        if (
            data.get("evals_per_epoch")
            and data.get("eval_strategy")
            and data.get("eval_strategy") != "steps"
        ):
            raise ValueError(
                "eval_strategy must be empty or set to `steps` when used with evals_per_epoch."
            )

        if data.get("do_bench_eval") and not (
            data.get("evals_per_epoch") or data.get("eval_steps")
        ):
            raise ValueError(
                "do_bench_eval requires evals_per_epoch or eval_steps to be set."
            )
        return data


class LoRAValidationMixin:
    """Validation methods related to LoRA/QLoRA configuration."""

    @model_validator(mode="before")
    @classmethod
    def check_lr_groups(cls, data):
        if data.get("lr_groups") and data.get("loraplus_lr_ratio"):
            raise ValueError("lr_groups and loraplus_lr_ratio cannot be used together.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_frozen(cls, data):
        if (
            data.get("adapter")
            and data.get("peft_layers_to_transform")
            and data.get("unfrozen_parameters")
        ):
            raise ValueError(
                "`unfrozen_parameters` used with `peft_layers_to_transform` can have unexpected behavior."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_peft_layers_pattern(cls, data):
        if data.get("peft_layers_pattern") and not data.get("peft_layers_to_transform"):
            raise ValueError(
                "peft_layers_pattern requires peft_layers_to_transform to be set"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_qlora_unsloth(cls, data):
        if (
            data.get("unsloth_lora_mlp")
            or data.get("unsloth_lora_qkv")
            or data.get("unsloth_lora_o")
        ):
            if data.get("adapter") == "lora" and data.get("load_in_8bit"):
                raise ValueError(
                    "unsloth_lora_mlp, unsloth_lora_qkv, and unsloth_lora_o are not compatible with 8-bit LoRA"
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_lora_8bit(cls, data):
        if (
            data.get("lora_mlp_kernel")
            or data.get("lora_qkv_kernel")
            or data.get("lora_o_kernel")
        ):
            if data.get("adapter") == "lora" and data.get("load_in_8bit"):
                raise ValueError(
                    "lora_mlp_kernel, lora_mlp_kernel, and lora_mlp_kernel are not compatible with 8-bit LoRA"
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_lora_axolotl_unsloth(cls, data):
        is_lora_kernel = any(
            data.get(k) for k in ["lora_mlp_kernel", "lora_qkv_kernel", "lora_o_kernel"]
        )
        is_unsloth_lora = any(
            data.get(k)
            for k in ["unsloth_lora_mlp", "unsloth_lora_qkv", "unsloth_lora_o"]
        )
        if is_lora_kernel and is_unsloth_lora:
            raise ValueError(
                "both lora_mlp_kernel and unsloth_lora_mlp cannot be true (similarly for lora_qkv_kernel, lora_o_kernel)"
            )
        return data


class RLValidationMixin:
    """Validation methods related to RL training configuration."""

    @model_validator(mode="before")
    @classmethod
    def check_sample_packing_w_rl(cls, data):
        if data.get("sample_packing") and data.get("rl"):
            raise ValueError("`sample_packing: true` does not work with RLHF training")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_kto_config(cls, data):
        if data.get("rl") == "kto":
            if data.get("sample_packing") or data.get("eval_sample_packing"):
                raise ValueError("sample_packing is not supported with kto")

            if data.get("remove_unused_columns") is not False:
                raise ValueError("Set `remove_unused_columns: False` when using kto")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_grpo_liger_sequence_parallel(cls, data):
        if (
            data.get("rl") == "grpo"
            and data.get("trl", {})
            and data.get("trl").get("use_liger_loss")
            and data.get("sequence_parallel_degree", 1) > 1
        ):
            raise ValueError("GRPO + SP + Liger not currently supported")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_rl_config_gradient_checkpointing(cls, data):
        # TODO: SalmanMohammadi
        # Distributed RL with QLoRA + gradient checkpointing
        # and use_reentrant = True is broken upstream in TRL
        # pylint: disable=too-many-boolean-expressions
        if (
            data.get("rl")
            and data.get("gradient_checkpointing")
            and data.get("gradient_checkpointing_kwargs")
            and data.get("gradient_checkpointing_kwargs").get("use_reentrant")
            and data.get("load_in_4bit")
            and data.get("adapter") == "qlora"
            and data.get("capabilities")
            and data.get("capabilities").get("n_gpu", 1) > 1
        ):
            raise ValueError(
                "The `use_reentrant: True` implementation of gradient checkpointing "
                "is not supported for distributed RL training with QLoRA. Please set "
                "`use_reentrant: False` in `gradient_checkpointing_kwargs`."
            )
        return data


class OptimizationValidationMixin:
    """Validation methods related to optimization and performance."""

    @model_validator(mode="before")
    @classmethod
    def check_batch_flattening_fa(cls, data):
        if data.get("batch_flattening"):
            batch_flattening_auto = data.get("batch_flattening") == "auto"
            if not data.get("flash_attention") and not batch_flattening_auto:
                raise ValueError("batch_flattening requires flash attention")
            if data.get("sample_packing") and not batch_flattening_auto:
                raise ValueError("batch_flattening not compatible with sample_packing")
            if data.get("micro_batch_size") == 1 and not batch_flattening_auto:
                LOG.warning("batch_flattening has no effect with micro_batch_size == 1")

            if (
                batch_flattening_auto
                and data.get("flash_attention")
                and not data.get("sample_packing")
                and data.get("micro_batch_size") > 1
            ):
                data["batch_flattening"] = True
            elif batch_flattening_auto:
                data["batch_flattening"] = False

        return data

    @model_validator(mode="before")
    @classmethod
    def check_torch_compile_deepspeed(cls, data):
        if data.get("deepspeed") and data.get("torch_compile"):
            raise ValueError(
                "torch_compile should be set within your deepspeed config file"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_xentropy_patch_conflicts(cls, data):
        if data.get("flash_attn_cross_entropy") and data.get(
            "unsloth_cross_entropy_loss"
        ):
            raise ValueError(
                "flash_attn_cross_entropy and unsloth_cross_entropy_loss cannot be both enabled"
            )
        return data


class SystemValidationMixin:
    """Validation methods related to system and hardware configuration."""

    @model_validator(mode="before")
    @classmethod
    def check_mem_mismatch(cls, data):
        if (
            data.get("max_memory") is not None
            and data.get("gpu_memory_limit") is not None
        ):
            raise ValueError(
                "max_memory and gpu_memory_limit are mutually exclusive and cannot be used together."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_fsdp_deepspeed(cls, data):
        if data.get("deepspeed") and data.get("fsdp"):
            raise ValueError("deepspeed and fsdp cannot be used together.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_npu_config(cls, data):
        if is_torch_npu_available():
            # check attention config
            attn_list = ["flash_attention", "sdp_attention", "s2_attention"]
            for attn in attn_list:
                if data.get(attn):
                    raise NotImplementedError(
                        f"{attn} is currently not supported in Ascend npu, please disable this configuration."
                    )

            # check quant config
            if data.get("optimizer") is not None and "bit" in data.get("optimizer"):
                optimizer = data.get("optimizer")
                raise NotImplementedError(
                    f"{optimizer} is currently not supported in Ascend npu, choose another one please."
                )

            quant_list = ["load_in_8bit", "load_in_4bit"]
            for quant in quant_list:
                if data.get(quant):
                    raise NotImplementedError(
                        f"Quantification is currently not supported in Ascend npu, please disable {quant}."
                    )

            # check dtype config
            if data.get("tf32"):
                raise NotImplementedError(
                    "tf32 dtype is currently not supported in Ascend npu, please disable this configuration"
                )

        return data


class ChatTemplateValidationMixin:
    """Validation methods related to chat template configuration."""

    @model_validator(mode="before")
    @classmethod
    def check_chat_template_config(cls, data):
        # if chat_template is set to jinja, chat_template_jinja is required
        if data.get("chat_template") == ChatTemplate.jinja and not data.get(
            "chat_template_jinja"
        ):
            raise ValueError(
                "chat_template_jinja is required when chat_template is set to jinja"
            )

        # If chat_template_jinja is set, set chat_template to jinja
        if data.get("chat_template_jinja") and not data.get("chat_template"):
            data["chat_template"] = ChatTemplate.jinja

        return data


class PretrainingValidationMixin:
    """Validation methods related to pretraining configuration."""

    @model_validator(mode="before")
    @classmethod
    def check_pretraining_w_max_steps(cls, data):
        if data.get("pretraining_dataset") and not data.get("max_steps"):
            raise ValueError(
                "max_steps must be set when using iterable pretraining_dataset, Trainer can't infer length and schedule optimizer/learning rate without it!"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_pretraining_w_group_by_length(cls, data):
        if data.get("pretraining_dataset") and data.get("group_by_length"):
            LOG.warning(
                "You probably want to disable group_by_length as it will force a streamed dataset to download completely."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_pretraining_split_batches_accelerate(cls, data):
        # alternatively set ACCELERATE_SPLIT_BATCHES=False
        if data.get("pretraining_dataset"):
            accelerator_config = data.get("accelerator_config", {})
            if not accelerator_config:
                data["accelerator_config"] = {
                    "split_batches": False,
                    "dispatch_batches": False,
                }
            else:
                if accelerator_config.get("split_batches") is None:
                    data["accelerator_config"]["split_batches"] = False
                if accelerator_config.get("dispatch_batches") is None:
                    data["accelerator_config"]["dispatch_batches"] = False
        return data


class ModelCompatibilityValidationMixin:
    """Validation methods for specific model compatibility."""

    @model_validator(mode="after")
    def check_falcon_fsdp(self):
        if (self.base_model and "falcon" in self.base_model.lower()) and self.fsdp:
            raise ValueError("FSDP is not supported for falcon models")
        return self

    @model_validator(mode="after")
    def check_mpt_checkpointing(self):
        if (
            self.base_model and "mpt" in self.base_model.lower()
        ) and self.gradient_checkpointing:
            raise ValueError("gradient_checkpointing is not supported for MPT models")
        return self

    @model_validator(mode="before")
    @classmethod
    def check_gptq_w_revision(cls, data):
        if data.get("gptq") and data.get("revision_of_model"):
            raise ValueError(
                "revision_of_model is not supported for GPTQ models. "
                + "Please download the model from HuggingFace Hub manually for correct branch, "
                + "point to its path, and remove revision_of_model from the config."
            )
        return data


class ComplexValidationMixin:
    """Complex validation methods that involve multiple systems."""

    @field_validator("neftune_noise_alpha")
    @classmethod
    def validate_neftune_noise_alpha(cls, neftune_noise_alpha):
        if neftune_noise_alpha is not None and neftune_noise_alpha <= 0.0:
            raise ValueError("neftune_noise_alpha must be > 0.0")
        return neftune_noise_alpha

    @model_validator(mode="after")
    def check_rl_beta(self):
        if self.dpo_beta and not self.rl_beta:
            self.rl_beta = self.dpo_beta
            del self.dpo_beta
        return self

    @model_validator(mode="after")
    def check_simpo_warmup(self):
        if self.rl is RLType.SIMPO and self.warmup_ratio:
            raise ValueError(
                "warmup_ratio is not supported with the simpo trainer. Please use `warmup_steps` instead"
            )
        return self

    @model_validator(mode="after")
    def check_relora(self):
        if self.relora_steps:
            if self.adapter not in ("lora", "qlora"):
                raise ValueError("cfg.adapter must be lora or qlora to use ReLoRA")

            if self.fsdp:
                raise ValueError("fsdp not supported with ReLoRA")

            if self.deepspeed:
                raise ValueError("deepspeed not supported with ReLoRA")

            if self.lr_scheduler == "one_cycle":
                raise ValueError(
                    "ReLoRA is not compatible with the one_cycle scheduler"
                )

            if self.flash_attn_fuse_qkv or self.flash_attn_fuse_mlp:
                raise ValueError("Fused modules are not supported with ReLoRA")
        return self

    @model_validator(mode="after")
    def check_early_stopping(self):
        if self.early_stopping_patience:
            if not self.save_steps or not self.eval_steps:
                raise ValueError(
                    "`early_stopping_patience` requires save_steps and eval_steps to be set. eval_steps should evenly divide save_steps."
                )
            if self.save_steps % self.eval_steps != 0:
                raise ValueError(
                    "`early_stopping_patience` requires that eval_steps should evenly divide save_steps."
                )
        return self

    @model_validator(mode="after")
    def check_sequence_parallel_degree(self):
        if not self.sequence_parallel_degree:
            self.sequence_parallel_degree = 1
        elif self.sequence_parallel_degree > 1:
            if not self.flash_attention:
                raise ValueError(
                    "flash_attention: true must be set with sequence_parallel_degree > 1"
                )

            if self.sample_packing and self.micro_batch_size > 1:
                raise ValueError(
                    "micro_batch_size must be set to 1 when sample_packing is enabled "
                    "due to a `ring-flash-attn` requirement"
                )

            try:
                import ring_flash_attn  # noqa: F401 # pylint:disable=unused-import
            except ImportError as exception:
                raise ImportError(
                    "sequence_parallel_degree > 1 but ring_flash_attn is not installed. "
                    "Please install it with `pip install axolotl[ring-flash-attn] "
                    "or `pip install ring-flash-attn>=0.1.4`."
                ) from exception

            LOG.warning(
                "Sequence parallelism (SP) is enabled with "
                f"sequence_parallel_degree={self.sequence_parallel_degree}. "
                "Please note that logged losses may differ slightly to the non-SP "
                "losses due to transformers Trainer implementation details. "
                "Please see https://github.com/axolotl-ai-cloud/axolotl/pull/2495#issuecomment-2784022042 "
                "for more details."
            )

        return self

    @model_validator(mode="after")
    def validate_ring_attn_func(self):
        if getattr(self, "sequence_parallel_degree", 1) == 1:
            return self

        if self.ring_attn_func is not None:
            self.ring_attn_func = RingAttnFunc(self.ring_attn_func)
        else:
            # Default ring attention function selection
            sample_packing = getattr(self, "sample_packing", False)
            self.ring_attn_func = (
                RingAttnFunc.VARLEN_LLAMA3
                if sample_packing
                else RingAttnFunc.BATCH_RING
            )

        return self


# pylint: disable=too-many-ancestors
class ValidationMixin(
    DatasetValidationMixin,
    AttentionValidationMixin,
    TrainingValidationMixin,
    LoRAValidationMixin,
    RLValidationMixin,
    OptimizationValidationMixin,
    SystemValidationMixin,
    ChatTemplateValidationMixin,
    PretrainingValidationMixin,
    ModelCompatibilityValidationMixin,
    ComplexValidationMixin,
):
    """Full validation mixin for Axolotl configuration."""
