"""Module with validation methods for config pydantic model."""

# pylint: disable=too-many-lines

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

    @field_validator("seed", mode="after")
    @classmethod
    def set_default_seed(cls, seed):
        if seed is None:
            LOG.info("`seed` not set in config; setting to 42")
            seed = 42
        return seed

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

    @model_validator(mode="before")
    @classmethod
    def check_test_datasets_bench(cls, data):
        if (
            data.get("do_bench_eval")
            and not data.get("test_datasets")
            and not data.get("val_set_size")
        ):
            LOG.warning(
                "`do_bench_eval` needs a test dataset to run evals, adding an empty test_dataset."
            )
            data["test_datasets"] = [{"path": "axolotl-ai-co/empty-test-ds"}]
        return data

    @model_validator(mode="before")
    @classmethod
    def check_eval_packing(cls, data):
        # TODO also should check test_datasets and val_set_size as we can skip
        # if there are no eval datasets/splits
        if (
            data.get("sample_packing")
            and data.get("eval_table_size")
            and data.get("eval_sample_packing") is not False
        ):
            raise ValueError(
                "eval_table_size and eval_sample_packing are not supported together with sample_packing. Please set 'eval_sample_packing' to false."
            )
        if (
            data.get("sample_packing")
            and data.get("eval_sample_packing") is None
            and not data.get("eval_table_size")
        ):
            LOG.info(
                "explicitly setting `eval_sample_packing` to match `sample_packing`"
            )
            data["eval_sample_packing"] = True

        if (
            data.get("sample_packing")
            and data.get("eval_sample_packing") is False
            and data.get("remove_unused_columns") is None
        ):
            LOG.info(
                "setting `remove_unused_columns: false` for when sample_packing and eval_sample_packing don't match"
            )
            data["remove_unused_columns"] = False

        return data

    @model_validator(mode="before")
    @classmethod
    def check_mm_prepare(cls, data):
        if data.get("skip_prepare_dataset"):
            if data.get("remove_unused_columns") is None:
                LOG.info(
                    "setting `remove_unused_columns: false` for skip_prepare_dataset"
                )
                data["remove_unused_columns"] = False

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
    def hint_sample_packing_padding(cls, data):
        if data.get("sample_packing"):
            pad_to_sequence_len = data.get("pad_to_sequence_len")
            if pad_to_sequence_len is False:
                LOG.warning(
                    "`pad_to_sequence_len: true` is recommended when using sample_packing"
                )
            elif pad_to_sequence_len is None:
                LOG.info(
                    "Setting `pad_to_sequence_len: true` to prevent memory leaks when sample_packing"
                )
                data["pad_to_sequence_len"] = True
        return data

    @model_validator(mode="before")
    @classmethod
    def hint_reward_model_pad(cls, data):
        if data.get("reward_model") and not data.get("pad_to_sequence_len"):
            LOG.warning(
                "`pad_to_sequence_len: true` is recommended when using reward_model"
            )
            if data.get("pad_to_sequence_len") is None:
                data["pad_to_sequence_len"] = True
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
    def hint_eval_train_mbsz(cls, data):
        if (
            data.get("eval_batch_size")
            and data.get("micro_batch_size")
            and data.get("eval_batch_size") != data.get("micro_batch_size")
        ):
            LOG.warning(
                "eval_batch_size != micro_batch_size. This can lead to VRAM instability."
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
    def check_push_save(cls, data):
        if data.get("hub_model_id") and (
            data.get("save_strategy") not in ["steps", "epoch", None]
        ):
            LOG.warning(
                "hub_model_id is set without any models being saved. To save a model, set save_strategy."
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

    @model_validator(mode="before")
    @classmethod
    def check_neftune(cls, data):
        if data.get("noisy_embedding_alpha") and not data.get("neftune_noise_alpha"):
            data["neftune_noise_alpha"] = data["noisy_embedding_alpha"]
            del data["noisy_embedding_alpha"]
        elif data.get("noisy_embedding_alpha") and data.get("neftune_noise_alpha"):
            raise ValueError(
                "noisy_embedding_alpha is deprecated, use neftune_noise_alpha; both are set, please remove the deprecated noisy_embedding_alpha setting"
            )
        return data

    @model_validator(mode="after")
    def check_fft_possible_bad_config(self):
        if (
            # pylint: disable=too-many-boolean-expressions
            not (self.bf16 or self.bfloat16)
            and (self.fp16 or self.float16)
            and not self.adapter
            and not self.flash_attention
            and self.sample_packing
        ):
            LOG.warning(
                "Full fine tune w/o FA2 w/ sample packing and fp16/float16 is likely to raise errors. Try LoRA."
            )
            # ValueError: Attempting to unscale FP16 gradients.
            # OR
            # RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half
        return self

    @model_validator(mode="before")
    @classmethod
    def check_use_reentrant_mismatch(cls, data):
        if (
            data.get("unfrozen_parameters")
            and data.get("gradient_checkpointing_kwargs")
            and data.get("gradient_checkpointing_kwargs", {}).get("use_reentrant")
            is True
        ):
            # https://github.com/huggingface/transformers/issues/21381
            raise ValueError(
                "`use_reentrant` must be false when used with partially frozen model."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_eval_strategy(cls, data):
        if (
            data.get("evaluation_strategy") is not None
            and data.get("eval_strategy") is None
        ):
            LOG.info(
                "explicitly setting `eval_strategy` from the `evaluation_strategy`"
            )
            data["eval_strategy"] = data.get("evaluation_strategy")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_causal_lm_evals(cls, data):
        if data.get("do_causal_lm_eval") and data.get("eval_sample_packing"):
            raise ValueError(
                "do_causal_lm_eval is enabled, eval_sample_packing must be set to False"
            )

        if data.get("eval_causal_lm_metrics"):
            if not isinstance(data.get("eval_causal_lm_metrics"), list):
                raise ValueError("eval_causal_lm_metrics must be a list")
            # only ["sacrebleu", "comet", "ter", "chrf"] supported
            if set(data.get("eval_causal_lm_metrics")) - SUPPORTED_METRICS:
                raise ValueError(
                    f"eval_causal_lm_metrics must be one of {SUPPORTED_METRICS}"
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_tokenizer_use_mistral_common(cls, data):
        if data.get("tokenizer_use_mistral_common") is None:
            if any(
                "magistral" in name.lower()
                for name in [
                    data.get("base_model", ""),
                    data.get("base_model_config", ""),
                    data.get("tokenizer_config", ""),
                ]
            ):
                LOG.warning(
                    "tokenizer_use_mistral_common auto inferred to True for Magistral models. Please set it to True explicitly if you want to use mistral-common tokenizer."
                )
                data["tokenizer_use_mistral_common"] = True

        return data

    @field_validator("tokenizer_use_mistral_common", mode="after")
    @classmethod
    def check_mistral_common_import(cls, tokenizer_use_mistral_common):
        if tokenizer_use_mistral_common:
            try:
                import mistral_common  # noqa: F401 # pylint:disable=unused-import
            except ImportError as exception:
                raise ImportError(
                    "mistral-common is required for mistral models. Please install it with `pip install axolotl` or `pip install -e .`."
                ) from exception

        return tokenizer_use_mistral_common

    @model_validator(mode="before")
    @classmethod
    def check_mistral_common_incompatible_options(cls, data):
        if not data.get("tokenizer_use_mistral_common"):
            return data

        # NOTE: mistral-common tokenizer is not compatible with editing tokenizer at the moment

        if data.get("added_tokens_overrides"):
            raise ValueError(
                "added_tokens_overrides is not supported with mistral-common tokenizer"
            )

        if data.get("special_tokens"):
            raise ValueError(
                "special_tokens override is not supported with mistral-common tokenizer"
            )

        if data.get("tokens"):
            raise ValueError(
                "tokens override is not supported with mistral-common tokenizer"
            )

        if data.get("chat_template"):
            raise ValueError(
                "Setting chat_template is not supported with mistral-common tokenizer"
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def pretrain_with_tps(cls, data):
        if data.get("pretraining_dataset") and data.get(
            "include_tokens_per_second", False
        ):
            # combining these would raise `TypeError: cannot pickle 'dict_keys' object`
            # due to trying to count the number of tokens total in the dataset
            raise ValueError(
                "pretraining_dataset and include_tokens_per_second cannot be used together."
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_tiled_mlp_deepspeed(cls, data):
        if data.get("tiled_mlp", False) and not data.get("deepspeed"):
            raise ValueError("tiled_mlp requires deepspeed ZeRO to be enabled")
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
                    "lora_mlp_kernel, lora_qkv_kernel, and lora_o_kernel are not compatible with 8-bit LoRA"
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

    @model_validator(mode="after")
    def check_fused_lora(self):
        if self.adapter in ["lora", "qlora"] and (
            self.flash_attn_fuse_qkv or self.flash_attn_fuse_mlp
        ):
            raise ValueError("Fused modules are not supported with LoRA/QLoRA")
        return self

    @model_validator(mode="after")
    def hint_lora_8bit(self):
        loftq = (
            self.peft and self.peft.loftq_config and self.peft.loftq_config.loftq_bits
        )
        if not self.load_in_8bit and self.adapter == "lora" and not loftq:
            LOG.warning("We recommend setting `load_in_8bit: true` for LORA finetuning")
        return self

    @model_validator(mode="before")
    @classmethod
    def warn_qlora_zero3_w_use_reentrant(cls, data):
        if (
            data.get("adapter") == "qlora"
            and data.get("gradient_checkpointing_kwargs", {})
            and data.get("gradient_checkpointing_kwargs", {}).get("use_reentrant")
            is False
            and data.get("deepspeed", "") is not None
            and "zero3" in data.get("deepspeed", "")
        ):
            # may result in:
            # torch.utils.checkpoint.CheckpointError: torch.utils.checkpoint:
            # Recomputed values for the following tensors have different metadata
            # than during the forward pass.
            LOG.warning(
                "qlora + zero3 with use_reentrant: false may result in a CheckpointError about recomputed values"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_lora_kernel_8bit(cls, data):
        if (
            data.get("lora_mlp_kernel")
            or data.get("lora_qkv_kernel")
            or data.get("lora_o_kernel")
        ):
            if data.get("adapter") == "lora" and data.get("load_in_8bit"):
                raise ValueError(
                    "lora_mlp_kernel, lora_qkv_kernel, and lora_o_kernel are not compatible with 8-bit LoRA"
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_lora_kernel_rl(cls, data):
        if (
            data.get("lora_mlp_kernel")
            or data.get("lora_qkv_kernel")
            or data.get("lora_o_kernel")
        ) and data.get("rl"):
            raise ValueError(
                "lora_mlp_kernel, lora_qkv_kernel, and lora_o_kernel are not compatible with RL at the moment."
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

    @model_validator(mode="after")
    def check_adamw_optimizer_params(self):
        if any([self.adam_beta1, self.adam_beta2, self.adam_epsilon]) and (
            not self.optimizer or "adamw" not in str(self.optimizer).lower()
        ):
            LOG.warning("adamw hyperparameters found, but no adamw optimizer set")
        return self

    @model_validator(mode="before")
    @classmethod
    def check_muon_deepspeed_fsdp(cls, data):
        if data.get("optimizer") == "muon" and (
            data.get("deepspeed") or data.get("fsdp") or data.get("fsdp_config")
        ):
            raise ValueError(
                "Muon optimizer is currently incompatible with DeepSpeed and FSDP"
            )
        return data

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

    @model_validator(mode="before")
    @classmethod
    def check_fsdp_offload_w_8bit_optimizer(cls, data):
        if (
            data.get("fsdp")
            and "8bit" in data.get("optimizer", "")
            and data.get("fsdp_config")
            and data["fsdp_config"].get("fsdp_offload_params")
            and str(data["fsdp_config"].get("fsdp_version")) != "2"
        ):
            raise ValueError(
                f"FSDP Offload not compatible with {data.get('optimizer')}"
            )
        if (
            data.get("fsdp")
            and "8bit" in data.get("optimizer", "")
            and data.get("fsdp_config")
            and str(data["fsdp_config"].get("fsdp_version")) == "2"
        ):
            if data.get("optimizer", "") in ["adamw_8bit", "adamw_bnb_8bit"]:
                # CUDA ops errors with bnb 8bit optimizer + FSDP2
                raise ValueError(
                    f"FSDP2 not compatible with {data.get('optimizer')}, use `adamw_torch_8bit` instead"
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_fsdp_sharded_state_dict_w_safetensors(cls, data):
        if (
            data.get("fsdp")
            and data.get("save_safetensors")
            and data.get("fsdp_config")
            and data["fsdp_config"].get("fsdp_state_dict_type") == "SHARDED_STATE_DICT"
        ):
            raise ValueError(
                "FSDP SHARDED_STATE_DICT not compatible with save_safetensors"
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

    @model_validator(mode="after")
    def check_offload_grad_checkpointing(self):
        if self.gradient_checkpointing and self.gradient_checkpointing == "unsloth":
            LOG.warning(
                "`unsloth` is deprecated for gradient_checkpointing, use `offload`"
            )
            self.gradient_checkpointing = "offload"
        return self

    @model_validator(mode="after")
    def check_better_transformers(self):
        if self.flash_optimum is True:
            if self.adapter:
                LOG.warning(
                    "BetterTransformers probably doesn't work with PEFT adapters"
                )
            if self.fp16 or self.bf16:
                raise ValueError("AMP is not supported with BetterTransformer")
            if self.float16 is not True and self.bfloat16 is not True:
                LOG.warning(
                    "You should probably set bfloat16 or float16 to true to "
                    "load the model in float16 for BetterTransformers"
                )
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
