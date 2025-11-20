"""Module with validation methods for config pydantic model."""

import json
import sys
import tempfile
from pathlib import Path

from pydantic import (
    field_validator,
    model_validator,
)
from transformers.utils.import_utils import is_torch_npu_available

from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.enums import ChatTemplate, RingAttnFunc, RLType

LOG = get_logger(__name__)

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
    def check_pretraining_streaming_deprecation(cls, data):
        # TODO(djsaunde): remove this check + implement change for 0.13.0 release
        if data.get("pretraining_dataset") and not data.get("streaming"):
            LOG.warning(
                "Setting `pretraining_dataset` without explicitly setting `streaming: "
                "true` is deprecated. In a future release, streaming will not be "
                "automatically enabled when using pretraining_dataset. Please "
                "explicitly set `streaming: true` in your configuration to maintain "
                "current behavior."
            )
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
                "explicitly setting `eval_sample_packing` to match `sample_packing`",
                main_process_only=True,
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

    @model_validator(mode="before")
    @classmethod
    def check_multipack_buffer_size(cls, data):
        if data.get("pretrain_multipack_buffer_size") and not data.get(
            "streaming_multipack_buffer_size"
        ):
            LOG.warning(
                "`pretrain_multipack_buffer_size` is deprecated in v0.13.0, will be "
                "removed in v0.14.0. Use `streaming_multipack_buffer_size` instead."
            )
            data["streaming_multipack_buffer_size"] = data[
                "pretrain_multipack_buffer_size"
            ]
            del data["pretrain_multipack_buffer_size"]
        elif data.get("pretrain_multipack_buffer_size") and data.get(
            "streaming_multipack_buffer_size"
        ):
            raise ValueError(
                "pretrain_multipack_buffer_size is deprecated, use "
                "streaming_multipack_buffer_size; both are set, please remove the "
                "deprecated pretrain_multipack_buffer_size setting"
            )
        return data

    @model_validator(mode="after")
    def check_fft_possible_bad_config(self):
        if (
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
    def check_fp8_config(cls, data):
        if data.get("fp8") and not data.get("torch_compile"):
            LOG.warning(
                "torch_compile is strongly recommended for FP8 training in order to "
                "see speed improvements. Please consider setting `torch_compile: "
                "true` in your config."
            )
        fsdp_config = data.get("fsdp_config") or {}
        if data.get("fp8") and (
            fsdp_config.get("activation_checkpointing", False) is True
            or fsdp_config.get("fsdp_activation_checkpointing", False) is True
        ):
            LOG.warning(
                "FP8 + FSDP2 + activation checkpointing may be slower than BF16 "
                "training. Please considering setting `activation_checkpointing: false` "
                "in your FSDP config."
            )
        if (
            data.get("fp8_enable_fsdp_float8_all_gather")
            and not data.get("fsdp_version", None) == 2
        ):
            raise ValueError(
                "fp8_enable_fsdp_float8_all_gather requires FSDP2 (fsdp_version: 2) "
                "to be used."
            )

        return data

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
            import importlib.util

            if importlib.util.find_spec("mistral_common") is None:
                raise ImportError(
                    "mistral-common is required for mistral models. Please install it with `pip install axolotl` or `pip install -e .`."
                )

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
        if self.adapter in ["lora", "qlora"] and self.flash_attn_fuse_mlp:
            raise ValueError("Fused modules are not supported with LoRA/QLoRA")
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
    def check_lora_kernels_8bit(cls, data):
        if (
            data.get("lora_mlp_kernel")
            or data.get("lora_qkv_kernel")
            or data.get("lora_o_kernel")
        ):
            if data.get("adapter") == "lora" and data.get("load_in_8bit"):
                raise ValueError(
                    "lora_mlp_kernel, lora_qkv_kernel, and lora_o_kernel are not "
                    "compatible with 8-bit LoRA a the moment."
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_lora_kernels_dora(cls, data):
        if (
            data.get("lora_mlp_kernel")
            or data.get("lora_qkv_kernel")
            or data.get("lora_o_kernel")
        ) and data.get("peft_use_dora"):
            raise ValueError(
                "lora_mlp_kernel, lora_qkv_kernel, and lora_o_kernel are not "
                "compatible with DoRA at the moment."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_lora_kernels_rl(cls, data):
        if (
            data.get("lora_mlp_kernel")
            or data.get("lora_qkv_kernel")
            or data.get("lora_o_kernel")
        ) and data.get("rl"):
            raise ValueError(
                "lora_mlp_kernel, lora_qkv_kernel, and lora_o_kernel are not "
                "compatible with RL at the moment."
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
            and data.get("context_parallel_size", 1) > 1
        ):
            raise ValueError("GRPO + SP + Liger not currently supported")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_rl_config_gradient_checkpointing(cls, data):
        # TODO: SalmanMohammadi
        # Distributed RL with QLoRA + gradient checkpointing
        # and use_reentrant = True is broken upstream in TRL

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
    def check_fsdp_version(cls, data):
        fsdp_config = data.get("fsdp_config", {})
        if fsdp_config and str(data.get("fsdp_version")) != "2":
            LOG.info(
                "FSDP1 will be deprecated in an upcoming release of Axolotl."
                "We recommend that you use FSDP version 2 for better performance and compatibility. "
                "Please see this link for more details: https://docs.axolotl.ai/docs/multi-gpu.html#sec-fsdp "
                "For more details on migrating your config. "
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_fsdp2_cpu_offload_pin_memory(cls, data):
        if not (fsdp_config := data.get("fsdp_config")):
            return data

        if fsdp_config.get("cpu_offload_pin_memory") is False:
            if str(data.get("fsdp_version")) != "2":
                raise ValueError(
                    "FSDP1 does not support disabling cpu_offload_pin_memory, please set `fsdp_version` to 2"
                )
            if not fsdp_config.get("offload_params"):
                raise ValueError(
                    "disabling cpu_offload_pin_memory requires enabling offload_params"
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_fsdp2_base_model_quant_rl(cls, data):
        if data.get("fsdp_version") == 2 and data.get("rl") in [
            RLType.DPO,
            RLType.KTO,
            RLType.ORPO,
            RLType.IPO,
        ]:
            if data.get("load_in_8bit") or data.get("load_in_4bit"):
                raise ValueError(
                    f"FSDP2 does not support load_in_8bit or load_in_4bit with {data.get('rl')}. Please use DeepSpeed or set `fsdp_version` to 1."
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_fsdp_version_in_fsdp_config(cls, data):
        fsdp_config = data.get("fsdp_config") or {}
        if fsdp_config and fsdp_config.get("fsdp_version"):
            LOG.warning(
                "Configuring `fsdp_version` in `fsdp_config` is deprecated. "
                "Please configure `fsdp_version` as a top-level field."
            )
            data["fsdp_version"] = fsdp_config.pop("fsdp_version")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_fsdp_config_kwargs_prefix(cls, data):
        if fsdp_config := data.get("fsdp_config"):
            should_fix = False
            for key, _ in fsdp_config.items():
                if key.startswith("fsdp_"):
                    should_fix = True
                    LOG.warning_once(
                        "Configuring FSDP fields with the `fsdp_` prefix is deprecated. "
                        "Please omit the `fsdp_` prefix from the any fields in `fsdp_config`."
                    )
            if should_fix:
                update_fsdp_config = {}
                for key, value in fsdp_config.items():
                    if key.startswith("fsdp_") and key != "fsdp_version":
                        update_fsdp_config[key.replace("fsdp_", "")] = value
                    else:
                        update_fsdp_config[key] = value
                data["fsdp_config"] = update_fsdp_config
        return data

    @model_validator(mode="after")
    def check_fsdp_offload_w_8bit_optimizer(self):
        if (
            hasattr(self, "fsdp_config")
            and self.fsdp_config
            and self.optimizer
            and "8bit" in self.optimizer.value
            and self.fsdp_config.offload_params
            and str(self.fsdp_version) != "2"
        ):
            raise ValueError(
                f"FSDP Offload not compatible with {str(self.optimizer.value)}"
            )
        return self

    @model_validator(mode="after")
    def check_fsdp2_w_8bit_optimizer(self):
        if (
            hasattr(self, "fsdp_config")
            and self.fsdp_config
            and self.optimizer
            and "8bit" in self.optimizer.value
            and str(self.fsdp_version) == "2"
        ):
            if self.optimizer in ["adamw_8bit", "adamw_bnb_8bit"]:
                # CUDA ops errors with bnb 8bit optimizer + FSDP2
                raise ValueError(
                    f"FSDP2 not compatible with {self.optimizer.value}, use `adamw_torch_8bit` instead"
                )

        return self

    @model_validator(mode="after")
    def lr_groups_ao_optimizer(self):
        if (
            self.loraplus_lr_ratio is not None
            or self.embedding_lr_scale is not None
            or self.embedding_lr is not None
            or self.lr_groups is not None
        ) and self.optimizer.value in ["adamw_torch_8bit", "adamw_torch_4bit"]:
            # TODO(wing): remove this once ao>0.12.0
            # requires https://github.com/pytorch/ao/pull/2606 in an ao release
            raise ValueError(
                "lr groups (`loraplus_lr_ratio`, `embedding_lr_scale`, `embedding_lr`, `lr_groups`) are not "
                "supported with ao low-bit optimizers until ao>0.12.0. "
                "Please refer to https://github.com/pytorch/ao/pull/2606."
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def check_tensor_parallel_size_update_ds_json(cls, data):
        tensor_parallel_size = data.get("tensor_parallel_size")
        if tensor_parallel_size is not None and tensor_parallel_size > 1:
            if data.get("deepspeed"):
                with open(data.get("deepspeed"), "r", encoding="utf-8") as ds_fin:
                    ds_config = json.load(ds_fin)
                    should_save = False
                    if "tensor_parallel" not in ds_config:
                        ds_config["tensor_parallel"] = {
                            "autotp_size": tensor_parallel_size
                        }
                        should_save = True
                    if (
                        "gather_16bit_weights_on_model_save"
                        not in ds_config["zero_optimization"]
                    ):
                        ds_config["zero_optimization"][
                            "gather_16bit_weights_on_model_save"
                        ] = True
                        should_save = True
                    if should_save:
                        temp_dir = tempfile.mkdtemp()
                        with open(
                            Path(temp_dir) / "autotp_ds.json", "w", encoding="utf-8"
                        ) as ds_fout:
                            json.dump(ds_config, ds_fout, indent=4)
                        data["deepspeed"] = str(Path(temp_dir) / "autotp_ds.json")

        return data

    @model_validator(mode="before")
    @classmethod
    def check_deepcompile(cls, data):
        deepcompile = data.get("deepcompile")
        if deepcompile:
            if not data.get("deepspeed"):
                raise ValueError("DeepCompile is only supported with DeepSpeed")
            with open(data.get("deepspeed"), "r", encoding="utf-8") as ds_fin:
                ds_config = json.load(ds_fin)
                if "compile" not in ds_config:
                    ds_config["compile"] = {"deepcompile": True}
                    temp_dir = tempfile.mkdtemp()
                    with open(
                        Path(temp_dir) / "deepcompile_ds.json", "w", encoding="utf-8"
                    ) as ds_fout:
                        json.dump(ds_config, ds_fout, indent=4)
                    data["deepspeed"] = str(Path(temp_dir) / "deepcompile_ds.json")

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
    def check_model_quantization_config_vs_bnb(cls, data):
        if data.get("model_quantization_config"):
            if data.get("load_in_8bit") or data.get("load_in_4bit"):
                raise ValueError(
                    "model_quantization_config and load_in_8bit or load_in_4bit cannot be used together."
                )
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

    @model_validator(mode="before")
    @classmethod
    def check_pretraining_w_val_set_size(cls, data):
        if data.get("pretraining_dataset") and data.get("val_set_size"):
            raise ValueError(
                "val_set_size is not supported with pretraining_dataset. "
                "Use test_datasets to specify evaluation datasets for pretraining."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_streaming_w_val_set_size(cls, data):
        if data.get("streaming") and data.get("val_set_size"):
            raise ValueError(
                "val_set_size is not supported with streaming datasets. "
                "Use test_datasets to specify evaluation datasets when streaming is enabled."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_streaming_w_max_steps(cls, data):
        if data.get("streaming") and not data.get("max_steps"):
            raise ValueError(
                "max_steps must be set when using streaming datasets. "
                "Trainer cannot infer dataset length for iterable datasets."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_streaming_w_multiple_datasets(cls, data):
        if (
            data.get("streaming")
            and data.get("sample_packing")
            and data.get("datasets")
            and len(data.get("datasets")) > 1
        ):
            raise NotImplementedError(
                "Sample packing with multiple streaming datasets is not yet supported"
            )
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
    def check_gradient_checkpointing_w_offload(self):
        if self.gradient_checkpointing == "offload":
            LOG.warning(
                "`offload` is deprecated for gradient_checkpointing, use `activation_offloading: true` or `activation_offloading: legacy`"
            )
            self.gradient_checkpointing = True
            LOG.warning(
                "`offload` now uses a new stream implementation; to use the previous implementation, use `activation_offloading: legacy`"
            )
            self.activation_offloading = True
        if self.gradient_checkpointing == "offload_disk":
            LOG.warning(
                "`offload_disk` is deprecated for gradient_checkpointing, use `activation_offloading: disk`"
            )
            self.gradient_checkpointing = True
            self.activation_offloading = "disk"
        return self

    @model_validator(mode="after")
    def check_activation_offloading_wo_gc(self):
        if self.activation_offloading and not self.gradient_checkpointing:
            raise ValueError("activation_offloading requires gradient_checkpointing")
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

    @model_validator(mode="before")
    @classmethod
    def check_gpt_oss_fsdp_loading(cls, data):
        if data.get("model_quantization_config", "") == "Mxfp4Config":
            fsdp_config = data.get("fsdp_config") or {}
            if fsdp_config.get("cpu_ram_efficient_loading", False) is True:
                raise ValueError(
                    "FSDP cpu_ram_efficient_loading is not supported for Mxfp4Config model quantization."
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
        if self.relora:
            if not self.jagged_restart_steps:
                raise ValueError("jagged_restart_steps must be set to use ReLoRA")
            if self.adapter not in ("lora", "qlora"):
                raise ValueError("cfg.adapter must be lora or qlora to use ReLoRA")

            if self.fsdp or self.fsdp_config:
                raise ValueError("fsdp not supported with ReLoRA")

            if self.deepspeed:
                raise ValueError("deepspeed not supported with ReLoRA")

            if self.lr_scheduler == "one_cycle":
                raise ValueError(
                    "ReLoRA is not compatible with the one_cycle scheduler"
                )

            if self.flash_attn_fuse_mlp:
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
    def check_tensor_parallel_size(self):
        if not self.tensor_parallel_size:
            self.tensor_parallel_size = 1
        return self

    @model_validator(mode="after")
    def check_expert_parallel_size(self):
        if not getattr(self, "expert_parallel_size", None):
            self.expert_parallel_size = 1
        elif self.expert_parallel_size < 1:
            raise ValueError("expert_parallel_size must be >= 1")
        return self

    @model_validator(mode="after")
    def check_context_parallel_size(self):
        if self.sequence_parallel_degree and not self.context_parallel_size:
            LOG.warning(
                "`sequence_parallel_degree` is deprecated, use `context_parallel_size`"
            )
            self.context_parallel_size = self.sequence_parallel_degree
        if not self.context_parallel_size:
            self.context_parallel_size = 1
        elif self.context_parallel_size > 1:
            if not self.flash_attention:
                raise ValueError(
                    "flash_attention: true must be set with context_parallel_size > 1"
                )

            if self.sample_packing and self.micro_batch_size > 1:
                raise ValueError(
                    "micro_batch_size must be set to 1 when sample_packing is enabled "
                    "due to a `ring-flash-attn` requirement"
                )

            try:
                import transformers.modeling_flash_attention_utils
                from transformers.utils import is_flash_attn_greater_or_equal

                transformers.modeling_flash_attention_utils._flash_supports_window = (
                    True
                )
                sys.modules[
                    "transformers.modeling_flash_attention_utils"
                ]._flash_supports_window = True
                sys.modules[
                    "transformers.modeling_flash_attention_utils"
                ]._flash_supports_window_size = True
                sys.modules[
                    "transformers.modeling_flash_attention_utils"
                ].is_flash_attn_greater_or_equal = is_flash_attn_greater_or_equal
                import ring_flash_attn  # noqa: F401  # Required after monkey-patching
            except ImportError as exception:
                raise ImportError(
                    "context_parallel_size > 1 but ring_flash_attn is not installed. "
                    "Please install it with `pip install axolotl[ring-flash-attn] "
                    "or `pip install ring-flash-attn>=0.1.4`."
                ) from exception

            LOG.warning(
                "Sequence parallelism (SP) is enabled with "
                f"context_parallel_size={self.context_parallel_size}. "
                "Please note that logged losses may differ slightly to the non-SP "
                "losses due to transformers Trainer implementation details. "
                "Please see https://github.com/axolotl-ai-cloud/axolotl/pull/2495#issuecomment-2784022042 "
                "for more details."
            )

        return self

    @model_validator(mode="after")
    def validate_ring_attn_func(self):
        if getattr(self, "context_parallel_size", 1) == 1:
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

    def hint_gradient_checkpointing_dpo_lora_ddp(self):
        if (
            (self.gradient_checkpointing is True or self.gradient_checkpointing is None)
            and self.capabilities
            and self.capabilities.get("n_gpu", 1) > 1
            and self.adapter in ("lora", "qlora")
            and self.rl == RLType.DPO
            and not self.fsdp
            and not self.deepspeed
        ):
            LOG.warning(
                "gradient_checkpointing with DPO + DDP + LoRA is not recommended."
            )
        return self


class DistributedValidationMixin:
    """validation for distributed training."""

    @model_validator(mode="after")
    def check_tensor_parallel_optimizer(self):
        if self.tensor_parallel_size > 1:
            if self.optimizer in ["paged_adamw_8bit", "adamw_8bit", "adamw_bnb_8bit"]:
                raise ValueError(
                    "tensor_parallel_size is not supported with paged_adamw_8bit, adamw_8bit, and adamw_bnb_8bit optimizers"
                )

        return self


class GRPOVllmValidationMixin:
    """Validation mixin for vllm when using GRPO."""

    @model_validator(mode="after")
    def check_vllm_mode_set(self):
        if self.trl and self.trl.use_vllm and not self.trl.vllm_mode:
            LOG.warning(
                "vllm_mode must be set to either `server` or `colocate` when using vllm, using default value `server`"
            )
            self.trl.vllm_mode = "server"
        return self


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
    GRPOVllmValidationMixin,
):
    """Full validation mixin for Axolotl configuration."""
