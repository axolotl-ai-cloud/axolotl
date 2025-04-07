"""Module with Pydantic models for configuration."""

# pylint: disable=too-many-lines

import logging
import os
from typing import Annotated, Any, Literal

from annotated_types import MinLen
from packaging import version
from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)
from transformers.utils.import_utils import is_torch_npu_available

from axolotl.utils.schemas.datasets import (
    DatasetConfig,
    DPODataset,
    KTODataset,
    PretrainingDataset,
    SFTDataset,
    StepwiseSupervisedDataset,
)
from axolotl.utils.schemas.deprecated import DeprecatedParameters, RemappedParameters
from axolotl.utils.schemas.enums import ChatTemplate, RLType
from axolotl.utils.schemas.integrations import (
    CometConfig,
    GradioConfig,
    LISAConfig,
    MLFlowConfig,
    RayConfig,
    WandbConfig,
)
from axolotl.utils.schemas.internal import EnvCapabilities, GPUCapabilities
from axolotl.utils.schemas.model import (
    ModelInputConfig,
    ModelOutputConfig,
    SpecialTokensConfig,
)
from axolotl.utils.schemas.multimodal import MultiModalConfig
from axolotl.utils.schemas.peft import LoraConfig, ReLoRAConfig
from axolotl.utils.schemas.training import HyperparametersConfig
from axolotl.utils.schemas.trl import TRLConfig
from axolotl.utils.schemas.vllm import VllmConfig

LOG = logging.getLogger(__name__)

SUPPORTED_METRICS = {"sacrebleu", "comet", "ter", "chrf", "perplexity"}


# pylint: disable=too-many-public-methods,too-many-ancestors
class AxolotlInputConfig(
    ModelInputConfig,
    ModelOutputConfig,
    LoraConfig,
    ReLoRAConfig,
    HyperparametersConfig,
    WandbConfig,
    MLFlowConfig,
    CometConfig,
    LISAConfig,
    GradioConfig,
    RayConfig,
    MultiModalConfig,
    RemappedParameters,
    DeprecatedParameters,
    BaseModel,
):
    """Wrapper of all config options"""

    model_config = {"populate_by_name": True}

    strict: bool | None = Field(default=False)
    resume_from_checkpoint: str | None = None
    auto_resume_from_checkpoints: bool | None = None
    resize_token_embeddings_to_32x: bool | None = None
    mean_resizing_embeddings: bool | None = False
    # optionally shrink the embeddings when the tokenizer vocab size is smaller
    shrink_embeddings: bool | None = None

    rl: RLType | None = None
    trl: TRLConfig | None = Field(
        default_factory=lambda: TRLConfig(),  # pylint: disable=unnecessary-lambda
    )
    vllm: VllmConfig | None = Field(
        default_factory=lambda: VllmConfig(),  # pylint: disable=unnecessary-lambda
    )
    reward_model: bool | None = None
    process_reward_model: bool | None = None
    num_labels: int | None = None
    # Whether to use weighting in DPO trainer.
    # If `None`, default is `False` in the trainer.
    dpo_use_weighting: bool | None = None
    dpo_use_logits_to_keep: bool | None = None

    datasets: (
        Annotated[
            list[SFTDataset | DPODataset | KTODataset | StepwiseSupervisedDataset],
            MinLen(1),
        ]
        | None
    ) = None

    test_datasets: (
        Annotated[
            list[SFTDataset | DPODataset | KTODataset | StepwiseSupervisedDataset],
            MinLen(1),
        ]
        | None
    ) = None
    shuffle_merged_datasets: bool | None = True
    dataset_prepared_path: str | None = None
    dataset_shard_num: int | None = None
    dataset_shard_idx: int | None = None
    skip_prepare_dataset: bool | None = False

    pretraining_dataset: (
        Annotated[list[PretrainingDataset | SFTDataset], MinLen(1)] | None
    ) = Field(
        default=None,
        json_schema_extra={"description": "streaming dataset to use for pretraining"},
    )
    dataset_processes: int | None = Field(default=min(32, os.cpu_count()))  # type: ignore[type-var]
    dataset_exact_deduplication: bool | None = None
    dataset_keep_in_memory: bool | None = None
    dataloader_pin_memory: bool | None = None
    dataloader_num_workers: int | None = None
    dataloader_prefetch_factor: int | None = None
    dataloader_drop_last: bool | None = None

    accelerator_config: dict[str, Any] | None = None

    remove_unused_columns: bool | None = None

    push_dataset_to_hub: str | None = None
    hf_use_auth_token: bool | None = None

    device: Any | None = None
    device_map: Any | None = None
    world_size: int | None = None
    local_rank: int | None = None
    ddp: bool | None = None

    seed: int | None = None
    ddp_timeout: int | None = None
    ddp_bucket_cap_mb: int | None = None
    ddp_broadcast_buffers: bool | None = None
    ddp_find_unused_parameters: bool | None = None

    eval_table_size: int | None = None
    eval_max_new_tokens: int | None = None
    do_causal_lm_eval: bool | None = None
    eval_causal_lm_metrics: list[str] | None = None
    do_bench_eval: bool | None = None
    bench_dataset: str | None = None
    bench_split: str | None = None
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None

    loss_watchdog_threshold: float | None = None
    loss_watchdog_patience: int | None = None

    gc_steps: int | None = None

    bf16: Literal["auto"] | bool | None = "auto"
    fp16: bool | None = None
    fp8: bool | None = None
    bfloat16: bool | None = None  # for non-AMP cases
    float16: bool | None = None  # for non-AMP cases
    tf32: bool | None = None
    float32: bool | None = None

    # torch_dtype: torch.dtype | None

    gradient_checkpointing: Literal["unsloth", "offload"] | bool | None = Field(
        default=False
    )
    gradient_checkpointing_kwargs: dict[str, Any] | None = None

    unfrozen_parameters: list[str] | None = None

    sequence_len: int = Field(default=512)
    min_sample_len: int | None = None
    max_prompt_len: int = Field(
        default=512,
        json_schema_extra={"description": "maximum prompt length for RL training"},
    )
    sample_packing: bool | None = None
    sample_packing_group_size: int | None = 100_000
    sample_packing_bin_size: int | None = 200
    sample_packing_sequentially: bool | None = None
    eval_sample_packing: bool | None = None
    pad_to_sequence_len: bool | None = None
    curriculum_sampling: bool | None = None
    multipack_real_batches: bool | None = None
    pretraining_sample_concatenation: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "whether to soft pack/concatenate samples during pretraining",
        },
    )

    batch_flattening: Literal["auto"] | bool | None = None

    # for PoSE context length extension
    use_pose: bool | None = None
    pose_split_on_token_ids: list[int] | None = None
    pose_max_context_len: int | None = None
    pose_num_chunks: int | None = None

    pretrain_multipack_buffer_size: int | None = 10_000
    pretrain_multipack_attn: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "whether to prevent cross attention for packed sequences during pretraining",
        },
    )

    xformers_attention: bool | None = None
    sdp_attention: bool | None = None
    s2_attention: bool | None = None
    flex_attention: bool | None = None
    flash_attention: bool | None = None
    flash_attn_cross_entropy: bool | None = None
    flash_attn_rms_norm: bool | None = None
    flash_attn_fuse_qkv: bool | None = None
    flash_attn_fuse_mlp: bool | None = None
    flash_optimum: bool | None = None

    eager_attention: bool | None = None

    unsloth_cross_entropy_loss: bool | None = None
    unsloth_lora_mlp: bool | None = None
    unsloth_lora_qkv: bool | None = None
    unsloth_lora_o: bool | None = None
    unsloth_rms_norm: bool | None = None
    unsloth_rope: bool | None = None

    lora_mlp_kernel: bool | None = None
    lora_qkv_kernel: bool | None = None
    lora_o_kernel: bool | None = None

    llama4_linearized_experts: bool | None = None

    deepspeed: str | dict[str, Any] | None = None
    fsdp: list[str] | None = None
    fsdp_config: dict[str, Any] | None = None
    fsdp_final_state_dict_type: (
        Literal["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"] | None
    ) = None

    val_set_size: float | None = Field(default=0.0)

    sequence_parallel_degree: int | None = None
    heads_k_stride: int | None = None

    special_tokens: SpecialTokensConfig | None = None
    tokens: list[str] | None = None
    added_tokens_overrides: dict[int, str] | None = None

    torch_compile: Literal["auto"] | bool | None = None
    torch_compile_backend: str | None = None
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] | None = (
        None
    )

    max_steps: int | None = None
    warmup_steps: int | None = None
    warmup_ratio: float | None = None
    eval_steps: int | float | None = None
    evals_per_epoch: int | None = None
    eval_strategy: str | None = None
    save_steps: int | float | None = None
    saves_per_epoch: int | None = None
    save_strategy: str | None = None
    save_total_limit: int | None = None
    logging_steps: int | None = None
    early_stopping_patience: int | None = None
    load_best_model_at_end: bool | None = False
    save_only_model: bool | None = False
    use_tensorboard: bool | None = None
    profiler_steps: int | None = None
    include_tokens_per_second: bool | None = None

    neftune_noise_alpha: float | None = None

    orpo_alpha: float | None = None
    rpo_alpha: float | None = None
    simpo_gamma: float | None = None
    cpo_alpha: float | None = None

    kto_desirable_weight: float | None = None
    kto_undesirable_weight: float | None = None
    rl_beta: float | None = None

    max_memory: dict[int | Literal["cpu", "disk"], int | str] | None = None
    gpu_memory_limit: int | str | None = None
    low_cpu_mem_usage: bool | None = None

    chat_template: (
        ChatTemplate
        | Annotated[str, StringConstraints(pattern="^tokenizer_default_fallback_")]
    ) | None = None
    chat_template_jinja: str | None = None
    default_system_message: str | None = None

    fix_untrained_tokens: int | list[int] | None = None

    # INTERNALS - document for now, generally not set externally
    is_preprocess: bool | None = None
    preprocess_iterable: bool | None = None

    total_num_tokens: int | None = None
    total_supervised_tokens: int | None = None
    sample_packing_eff_est: float | None = None
    axolotl_config_path: str | None = None

    is_falcon_derived_model: bool | None = Field(default=None)
    is_llama_derived_model: bool | None = Field(default=None)
    is_mistral_derived_model: bool | None = Field(default=None)
    is_qwen_derived_model: bool | None = Field(default=None)

    plugins: list[str] | None = Field(default=None)

    @field_validator("datasets", mode="before")
    @classmethod
    def deprecate_sharegpt_datasets(cls, datasets):
        for _, ds_cfg in enumerate(datasets):
            # Handle both dict and pydantic model cases
            ds_type = (
                ds_cfg.get("type")
                if isinstance(ds_cfg, dict)
                else getattr(ds_cfg, "type", None)
            )
            if not ds_type:
                continue

            # skip if it's a dict (for custom user instruction prompt)
            if isinstance(ds_type, dict):
                continue

            if isinstance(ds_type, str) and ds_type.startswith("sharegpt"):
                raise ValueError(
                    "`type: sharegpt.*` is deprecated. Please use `type: chat_template` instead."
                )

        return datasets

    @field_serializer("datasets")
    def datasets_serializer(
        self, ds_configs: list[DatasetConfig] | None
    ) -> list[dict[str, Any]] | None:
        if ds_configs:
            return [ds_config.model_dump(exclude_none=True) for ds_config in ds_configs]
        return None

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
    def check_batch_size_fields(cls, data):
        fields = ("micro_batch_size", "gradient_accumulation_steps", "batch_size")
        non_empty_count = sum(1 for field in fields if data.get(field))

        if non_empty_count < 2:
            raise ValueError(f"At least two of {', '.join(fields)} must be set")
        return data

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
    def check_sample_packing_w_xformers(cls, data):
        if data.get("sample_packing") and data.get("xformers_attention"):
            raise ValueError(
                "sample_packing not compatible with xformers_attention. Use flash_attention"
            )

        return data

    @model_validator(mode="before")
    @classmethod
    # pylint: disable=duplicate-code
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

    @model_validator(mode="before")
    @classmethod
    def check_sample_packing_wo_flash(cls, data):
        if (
            data.get("sample_packing")
            and not data.get("flash_attention")
            and not data.get("sdp_attention")
            and not data.get("flex_attention")
        ):
            LOG.warning(
                "sample_packing without flash, sdp or flex attention does not handle cross sample decontamination."
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
    def check_sample_packing_w_rl(cls, data):
        if data.get("sample_packing") and data.get("rl"):
            raise ValueError("`sample_packing: true` does not work with RLHF training")
        return data

    @model_validator(mode="before")
    @classmethod
    def hint_sample_packing_padding(cls, data):
        if data.get("sample_packing") and not data.get("pad_to_sequence_len"):
            LOG.warning(
                "`pad_to_sequence_len: true` is recommended when using sample_packing"
            )
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
    def check_push_ds_auth(cls, data):
        if (
            data.get("push_dataset_to_hub")
            and data.get("hf_use_auth_token") is not True
        ):
            raise ValueError(
                "Require cfg.hf_use_auth_token to be True for push_dataset_to_hub"
            )
        return data

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

    @model_validator(mode="after")
    def check_adamw_optimizer_params(self):
        if any([self.adam_beta1, self.adam_beta2, self.adam_epsilon]) and (
            not self.optimizer or "adamw" not in str(self.optimizer).lower()
        ):
            LOG.warning("adamw hyperparameters found, but no adamw optimizer set")
        return self

    @model_validator(mode="before")
    @classmethod
    def check_lr_groups(cls, data):
        if data.get("lr_groups") and data.get("loraplus_lr_ratio"):
            raise ValueError("lr_groups and loraplus_lr_ratio cannot be used together.")
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

    @model_validator(mode="before")
    @classmethod
    def check_warmup(cls, data):
        if data.get("warmup_steps") and data.get("warmup_ratio"):
            raise ValueError("warmup_steps and warmup_ratio are mutually exclusive")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_neftune(cls, data):
        if data.get("noisy_embedding_alpha") and not data.get("neftune_noise_alpha"):
            data["neftune_noise_alpha"] = data["noisy_embedding_alpha"]
            del data["noisy_embedding_alpha"]
        elif data.get("noisy_embedding_alpha") and not data.get("neftune_noise_alpha"):
            raise ValueError(
                "noisy_embedding_alpha is deprecated, use neftune_noise_alpha; both are set, please remove the deprecated noisy_embedding_alpha setting"
            )
        return data

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
        if self.rl == "simpo" and self.warmup_ratio:
            raise ValueError(
                "warmup_ratio is not supported with the simpo trainer. Please use `warmup_steps` instead"
            )
        return self

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
    def check_val_w_test_datasets(cls, data):
        if data.get("test_datasets") and data.get("val_set_size"):
            raise ValueError(
                "non-zero val_set_size should not be used with test_datasets configuration"
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
    def check_dataset_or_pretraining_dataset(cls, data):
        if data.get("datasets") is None and data.get("pretraining_dataset") is None:
            raise ValueError("either datasets or pretraining_dataset is required")
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

    @model_validator(mode="before")
    @classmethod
    def check_kto_config(cls, data):
        if data.get("rl") == "kto":
            if data.get("sample_packing") or data.get("eval_sample_packing"):
                raise ValueError("sample_packing is not supported with kto")

            if data.get("remove_unused_columns") is not False:
                raise ValueError("Set `remove_unused_columns: False` when using kto")

        return data

    @field_validator("sequence_parallel_degree", mode="before")
    @classmethod
    def check_sequence_parallel_degree(cls, value, info):
        if not value:
            value = 1

        if value > 1:
            if not info.data.get("flash_attention"):
                raise ValueError(
                    "flash_attention: true must be set with sequence_parallel_degree > 1"
                )

            if not info.data["micro_batch_size"] == 1:
                raise ValueError(
                    "micro_batch_size must be set to 1 "
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

            # TODO: monkeypatch / callback to average losses correctly across SP ranks
            # / fix gradient scaling across SP ranks. Losses, grads should be scaled
            # according to the proportion of non-padding tokens per rank.
            LOG.warning(
                "Sequence parallelism (SP) is enabled with "
                f"sequence_parallel_degree={value}. Please note that logged losses may "
                "differ slightly to the non-SP losses due to transformers Trainer "
                "implementation details. Please see "
                "https://github.com/axolotl-ai-cloud/axolotl/pull/2495#issuecomment-2784022042 "
                "for more details."
            )

        return value

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


class AxolotlConfigWCapabilities(AxolotlInputConfig):
    """wrapper to valdiate gpu capabilities with the configured options"""

    capabilities: GPUCapabilities
    env_capabilities: EnvCapabilities

    @model_validator(mode="after")
    def check_bf16(self):
        if self.capabilities.bf16:
            if not self.bf16 and not self.bfloat16:
                LOG.info(
                    "bf16 support detected, but not enabled for this configuration."
                )
        else:
            if (
                not self.merge_lora
                and not self.is_preprocess
                and (self.bf16 is True or self.bfloat16 is True)
            ):
                raise ValueError(
                    "bf16 requested, but AMP is not supported on this GPU. Requires Ampere series or above."
                )
        return self

    @model_validator(mode="before")
    @classmethod
    def check_sample_packing_w_sdpa_bf16(cls, data):
        is_sm_90: bool = (
            data["capabilities"]
            and data["capabilities"].get("compute_capability") == "sm_90"
        )
        if (
            data.get("sample_packing")
            and data.get("sdp_attention")
            and (data.get("bfloat16") or data.get("bf16"))
            and not is_sm_90
        ):
            # https://github.com/pytorch/pytorch/blob/1b03423526536b5f3d35bdfa95ccc6197556cf9b/test/test_transformers.py#L2440-L2450
            LOG.warning(
                "sample_packing & torch sdpa with bf16 is unsupported may results in 0.0 loss. "
                "This may work on H100s."
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
    def check_multigpu_unsloth(cls, data):
        if (
            data.get("unsloth_lora_mlp")
            or data.get("unsloth_lora_qkv")
            or data.get("unsloth_lora_o")
        ):
            capabilities = data.get("capabilities")
            if capabilities and capabilities.get("n_gpu", 0) > 1:
                raise ValueError(
                    "unsloth_lora_mlp, unsloth_lora_qkv, and unsloth_lora_o are not compatible with multi-GPU training."
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_multigpu_lora_kernels(cls, data):
        if (
            data.get("lora_mlp_kernel")
            or data.get("lora_qkv_kernel")
            or data.get("lora_o_kernel")
        ):
            capabilities = data.get("capabilities")
            is_fsdp = data.get("fsdp") is not None

            if capabilities and capabilities.get("n_gpu", 0) > 1:
                if is_fsdp:
                    raise ValueError(
                        "lora_mlp_kernel, lora_qkv_kernel, and lora_o_kernel are not compatible with FSDP."
                    )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_adopt_torch_version(cls, data):
        if (data.get("optimizer") is not None) and ("adopt" in data.get("optimizer")):
            env_capabilities = data.get("env_capabilities", {})
            torch_version = env_capabilities.get("torch_version")

            if torch_version is None:
                import torch

                torch_version = str(torch.__version__).split("+", maxsplit=1)[0]

            if version.parse(torch_version) < version.parse("2.5.1"):
                raise ValueError(
                    "ADOPT optimizer is incompatible with torch version < 2.5.1"
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_flex_torch_version(cls, data):
        if (data.get("flex_attention") is not None) and (data.get("flex_attention")):
            env_capabilities = data.get("env_capabilities", {})
            torch_version = env_capabilities.get("torch_version")

            if torch_version is None:
                import torch

                torch_version = str(torch.__version__).split("+", maxsplit=1)[0]

            if version.parse(torch_version) < version.parse("2.6.0"):
                raise ValueError(
                    "Flex attention is not supported on torch version < 2.6.0"
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_torch_compile_auto(cls, data):
        if data.get("torch_compile") == "auto":
            env_capabilities = data.get("env_capabilities", {})
            if env_capabilities.get("torch_version"):
                if version.parse(
                    env_capabilities.get("torch_version")
                ) >= version.parse("2.5.1"):
                    LOG.info(
                        "torch.compile is available, setting torch_compile to True"
                    )
                    data["torch_compile"] = True
                else:
                    data["torch_compile"] = False
            else:
                data["torch_compile"] = False
        return data

    @model_validator(mode="before")
    @classmethod
    def check_beta_and_trl_beta_match(cls, data):
        if data.get("beta") and data.get("trl", {}).get("beta"):
            if data["beta"] != data["trl"]["beta"]:
                raise ValueError("beta and trl.beta must match or one must be removed")
        return data

    @model_validator(mode="after")
    def check_min_torch_version(self):
        if self.env_capabilities and self.env_capabilities.torch_version:
            torch_version = self.env_capabilities.torch_version
            if version.parse(torch_version) < version.parse("2.5.1"):
                LOG.warning(
                    f"torch=={torch_version} may not be supported in future versions. Please consider upgrading to torch>=2.5.1."
                )

        return self
