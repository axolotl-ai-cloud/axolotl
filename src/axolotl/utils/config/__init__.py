"""Module for working with config dicts"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.utils.bench import log_gpu_memory_usage
from axolotl.utils.config.models.input.v0_4_1 import (
    AxolotlConfigWCapabilities,
    AxolotlInputConfig,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model_config

LOG = logging.getLogger("axolotl")


def choose_device(cfg):
    def get_device():
        try:
            if torch.cuda.is_available():
                return f"cuda:{cfg.local_rank}"

            if torch.backends.mps.is_available():
                return "mps"

            raise SystemError("No CUDA/mps device found")
        except Exception:  # pylint: disable=broad-exception-caught
            return "cpu"

    cfg.device = get_device()
    if cfg.world_size == 1:
        cfg.device_map = cfg.device_map or "auto"
    else:
        if cfg.device.startswith("cuda"):
            cfg.device_map = {"": torch.cuda.current_device()}
        else:
            cfg.device_map = {"": cfg.device}

    # in `accelerate launch`, we need to not pass through any device map and let
    # accelerate figure out which parts of the model to put on which gpu
    accelerate_vars = [var for var in os.environ if var.startswith("ACCELERATE_USE_")]
    if accelerate_vars:
        cfg.device_map = None


def normalize_config(cfg):
    # setup some derived config / hyperparams
    cfg.gradient_accumulation_steps = cfg.gradient_accumulation_steps or (
        cfg.batch_size // cfg.micro_batch_size
    )
    cfg.batch_size = (
        cfg.batch_size or cfg.micro_batch_size * cfg.gradient_accumulation_steps
    )
    if cfg.eval_batch_size is None:
        cfg.eval_batch_size = cfg.micro_batch_size
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.eval_table_size = cfg.eval_table_size or 0
    cfg.eval_max_new_tokens = cfg.eval_max_new_tokens or 128
    cfg.eval_causal_lm_metrics = cfg.eval_causal_lm_metrics or [
        "sacrebleu",
        "comet",
        "ter",
        "chrf",
    ]
    choose_device(cfg)
    cfg.ddp = cfg.ddp if cfg.ddp is not None else cfg.world_size != 1
    if cfg.ddp:
        cfg.device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        cfg.batch_size = cfg.batch_size * cfg.world_size

    if cfg.bf16 == "auto":
        if is_torch_bf16_gpu_available():
            LOG.debug("bf16 support detected, enabling for this configuration.")
            cfg.bf16 = True
        else:
            LOG.debug("bf16 support not detected, disabling for this configuration.")
            cfg.bf16 = False
            if cfg.fp16 is None:
                cfg.fp16 = True

    if cfg.device == "mps":
        cfg.load_in_8bit = False
        cfg.tf32 = False
        if cfg.bf16:
            cfg.fp16 = True
        cfg.bf16 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32 or False
        if cfg.bf16:
            cfg.fp16 = False

    if cfg.bf16 or cfg.bfloat16:
        cfg.torch_dtype = torch.bfloat16
    elif cfg.load_in_8bit or cfg.fp16 or cfg.float16:
        cfg.torch_dtype = torch.float16
    else:
        cfg.torch_dtype = torch.float32

    if cfg.saves_per_epoch:
        save_steps = 1.0 / (cfg.saves_per_epoch * cfg.num_epochs)
        if save_steps < 1.0:  # prevent saves on every step
            cfg.save_steps = save_steps
    if (cfg.val_set_size or cfg.test_datasets) and cfg.evals_per_epoch:
        eval_steps = 1.0 / (cfg.evals_per_epoch * cfg.num_epochs)
        if eval_steps < 1.0:  # prevent evals on every step
            cfg.eval_steps = eval_steps

    cfg.dataset_processes = cfg.dataset_processes or os.cpu_count()

    if not cfg.base_model_config:
        cfg.base_model_config = cfg.base_model

    model_config = load_model_config(cfg)
    cfg.model_config_type = model_config.model_type

    # figure out if the model is llama
    cfg.is_llama_derived_model = (
        (hasattr(model_config, "model_type") and model_config.model_type == "llama")
        or cfg.is_llama_derived_model
        or "llama" in cfg.base_model.lower()
        or (cfg.type_of_model and "llama" in cfg.type_of_model.lower())
    )

    # figure out if the model is falcon
    cfg.is_falcon_derived_model = (
        (
            hasattr(model_config, "model_type")
            and model_config.model_type
            in [
                "falcon",
                "RefinedWebModel",
                "RefinedWeb",
            ]
        )
        or cfg.is_falcon_derived_model
        or "falcon" in cfg.base_model.lower()
        or (cfg.type_of_model and "rwforcausallm" in cfg.type_of_model.lower())
    )

    cfg.is_mistral_derived_model = (
        (
            hasattr(model_config, "model_type")
            and model_config.model_type
            in [
                "mistral",
            ]
        )
        or cfg.is_mistral_derived_model
        or "mistral" in cfg.base_model.lower().split("/")[-1]
        or (cfg.type_of_model and "mistral" in cfg.type_of_model.lower())
    )

    cfg.is_qwen_derived_model = (
        hasattr(model_config, "model_type")
        and model_config.model_type
        in [
            "qwen",
        ]
    ) or cfg.is_qwen_derived_model

    if isinstance(cfg.pretraining_dataset, dict):
        cfg.pretraining_dataset = [cfg.pretraining_dataset]

    if (
        cfg.gradient_checkpointing
        and cfg.unfrozen_parameters is None
        and cfg.gradient_checkpointing_kwargs is None
        and cfg.rl is None
    ):
        cfg.gradient_checkpointing_kwargs = {"use_reentrant": True}

    log_gpu_memory_usage(LOG, "baseline", cfg.device)


def normalize_cfg_datasets(cfg):
    """
    helpers for mapping chat_template to various dataset configurations as necessary
    """

    if cfg.chat_template and cfg.chat_template == "chatml":
        if cfg.datasets:
            for idx, ds_cfg in enumerate(cfg.datasets):
                if ds_cfg.type == "sharegpt" and not ds_cfg.conversation:
                    LOG.info(
                        f"updating dataset {ds_cfg.path} with `conversation: chatml` to match your chat_template"
                    )
                    cfg.datasets[idx].conversation = "chatml"


def validate_config(cfg: DictDefault, capabilities: Optional[dict] = None):
    if capabilities:
        return DictDefault(
            dict(
                AxolotlConfigWCapabilities(
                    **cfg.to_dict(), capabilities=capabilities
                ).model_dump(exclude_unset=True)
            )
        )
    return DictDefault(
        dict(AxolotlInputConfig(**cfg.to_dict()).model_dump(exclude_unset=True))
    )


def legacy_validate_config(cfg):
    """
    This is a "pre-validation" step that handles the yaml configuration before we have any
    information about the model architecture
    """
    if is_torch_bf16_gpu_available():
        if not cfg.bf16 and not cfg.bfloat16:
            LOG.info("bf16 support detected, but not enabled for this configuration.")
    else:
        if (
            not cfg.merge_lora
            and not cfg.is_preprocess
            and (cfg.bf16 is True or cfg.bfloat16 is True)
        ):
            raise ValueError(
                "bf16 requested, but AMP is not supported on this GPU. Requires Ampere series or above."
            )
    if (
        # pylint: disable=too-many-boolean-expressions
        not (cfg.bf16 or cfg.bfloat16)
        and (cfg.fp16 or cfg.float16)
        and not cfg.adapter
        and not cfg.flash_attention
        and cfg.sample_packing
    ):
        LOG.warning(
            "Full fine tune w/o FA2 w/ sample packing and fp16/float16 is likely to raise errors. Try LoRA."
        )
        # ValueError: Attempting to unscale FP16 gradients.
        # OR
        # RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half
    if cfg.max_packed_sequence_len:
        raise DeprecationWarning("`max_packed_sequence_len` is no longer supported")

    if cfg.sample_packing and cfg.rl:
        raise ValueError("`sample_packing: true` does not work with RLHF training")

    if cfg.sample_packing and not cfg.pad_to_sequence_len:
        LOG.warning(
            "`pad_to_sequence_len: true` is recommended when using sample_packing"
        )

    if cfg.gradient_accumulation_steps and cfg.batch_size:
        raise ValueError(
            "please set only one of gradient_accumulation_steps or batch_size"
        )
    if cfg.batch_size:
        LOG.warning(
            "%s\n%s",
            "batch_size is not recommended. Please use gradient_accumulation_steps instead.",
            "To calculate the equivalent gradient_accumulation_steps, divide batch_size / micro_batch_size / number of gpus.",
        )
    if (
        cfg.eval_batch_size
        and cfg.micro_batch_size
        and cfg.eval_batch_size != cfg.micro_batch_size
    ):
        LOG.warning(
            "eval_batch_size != micro_batch_size. This can lead to VRAM instability."
        )

    if cfg.adapter == "qlora":
        if cfg.merge_lora:
            # can't merge qlora if loaded in 8bit or 4bit
            if cfg.load_in_8bit:
                raise ValueError("Can't merge qlora if loaded in 8bit")

            if cfg.gptq:
                raise ValueError("Can't merge qlora if gptq")

            if cfg.load_in_4bit:
                raise ValueError("Can't merge qlora if loaded in 4bit")

        else:
            if cfg.load_in_8bit:
                raise ValueError("Can't load qlora in 8bit")

            if cfg.gptq:
                raise ValueError("Can't load qlora if gptq")

            if not cfg.load_in_4bit:
                raise ValueError("Require cfg.load_in_4bit to be True for qlora")

        if cfg.flash_attn_fuse_qkv or cfg.flash_attn_fuse_mlp:
            raise ValueError("Fused modules are not supported with QLoRA")

    loftq = cfg.peft and cfg.peft.loftq_config and cfg.peft.loftq_config.loftq_bits
    if not cfg.load_in_8bit and cfg.adapter == "lora" and not loftq:
        LOG.warning("We recommend setting `load_in_8bit: true` for LORA finetuning")

    if cfg.adapter == "lora" and (cfg.flash_attn_fuse_qkv or cfg.flash_attn_fuse_mlp):
        raise ValueError("Fused modules are not supported with LoRA")

    if cfg.adapter and cfg.peft_layers_to_transform and cfg.unfrozen_parameters:
        raise ValueError(
            "`unfrozen_parameters` used with `peft_layers_to_transform` can have unexpected behavior."
        )

    if cfg.relora_steps:
        if cfg.adapter not in ("lora", "qlora"):
            raise ValueError("cfg.adapter must be lora or qlora to use ReLoRA")

        if cfg.fsdp:
            raise ValueError("fsdp not supported with ReLoRA")

        if cfg.deepspeed:
            raise ValueError("deepspeed not supported with ReLoRA")

        if cfg.lr_scheduler == "one_cycle":
            raise ValueError("ReLoRA is not compatible with the one_cycle scheduler")

        if cfg.flash_attn_fuse_qkv or cfg.flash_attn_fuse_mlp:
            raise ValueError("Fused modules are not supported with ReLoRA")

    if cfg.trust_remote_code:
        LOG.warning(
            "`trust_remote_code` is set to true. Please make sure that you reviewed the remote code/model."
        )

    if cfg.push_dataset_to_hub and cfg.hf_use_auth_token is not True:
        raise ValueError(
            "Require cfg.hf_use_auth_token to be True for push_dataset_to_hub"
        )

    if (cfg.base_model and "falcon" in cfg.base_model.lower()) and cfg.fsdp:
        raise ValueError("FSDP is not supported for falcon models")

    if (
        cfg.base_model and "mpt" in cfg.base_model.lower()
    ) and cfg.gradient_checkpointing:
        raise ValueError("gradient_checkpointing is not supported for MPT models")

    if cfg.flash_optimum is True:
        if cfg.adapter:
            LOG.warning("BetterTransformers probably doesn't work with PEFT adapters")
        if cfg.fp16 or cfg.bf16:
            raise ValueError("AMP is not supported with BetterTransformer")
        if cfg.float16 is not True and cfg.bfloat16 is not True:
            LOG.warning(
                "You should probably set bfloat16 or float16 to true to "
                "load the model in float16 for BetterTransformers"
            )
        if int(torch.__version__.split(".", maxsplit=1)[0]) < 2:
            LOG.warning("torch>=2.0.0 required")
            raise ValueError(
                f"flash_optimum for BetterTransformers may not be used with {torch.__version__}"
            )

    if cfg.pretraining_dataset and cfg.group_by_length:
        LOG.warning(
            "You probably want to disable group_by_length as it will force a streamed dataset to download completely."
        )
    if cfg.pretraining_dataset and not cfg.max_steps:
        raise ValueError(
            "max_steps must be set when using iterable pretraining_dataset, Trainer can't infer length and schedule optimizer/learning rate without it!"
        )

    if any([cfg.adam_beta1, cfg.adam_beta2, cfg.adam_epsilon]) and (
        not cfg.optimizer or "adamw" not in cfg.optimizer
    ):
        LOG.warning("adamw hyperparameters found, but no adamw optimizer set")

    if cfg.push_to_hub_model_id:
        raise ValueError(
            "push_to_hub_model_id is deprecated. Please use hub_model_id instead."
        )

    if cfg.hub_model_id and not (cfg.save_steps or cfg.saves_per_epoch):
        LOG.warning(
            "hub_model_id is set without any models being saved. To save a model, set either save_steps or saves_per_epoch."
        )

    if cfg.gptq and cfg.revision_of_model:
        raise ValueError(
            "revision_of_model is not supported for GPTQ models. "
            + "Please download the model from HuggingFace Hub manually for correct branch, "
            + "point to its path, and remove revision_of_model from the config."
        )

    # if cfg.sample_packing and cfg.sdp_attention:
    #     # incompatible due to bug w/ accelerate causing 0.0 loss when using llama2
    #     raise ValueError(
    #         "sample_packing not compatible with sdp_attention. Use flash_attention"
    #     )

    if cfg.sample_packing and cfg.xformers_attention:
        raise ValueError(
            "sample_packing not compatible with xformers_attention. Use flash_attention"
        )

    if cfg.sample_packing and cfg.sdp_attention and (cfg.bfloat16 or cfg.bf16):
        # https://github.com/pytorch/pytorch/blob/1b03423526536b5f3d35bdfa95ccc6197556cf9b/test/test_transformers.py#L2440-L2450
        LOG.warning(
            "sample_packing & torch sdpa with bf16 is unsupported may results in 0.0 loss. "
            "This may work on H100s."
        )

    if cfg.early_stopping_patience:
        if not cfg.save_steps or not cfg.eval_steps:
            raise ValueError(
                "`early_stopping_patience` requires save_steps and eval_steps to be set. eval_steps should evenly divide save_steps."
            )
        if cfg.save_steps % cfg.eval_steps != 0:
            raise ValueError(
                "`early_stopping_patience` requires that eval_steps should evenly divide save_steps."
            )

    if cfg.datasets:
        for idx, ds_cfg in enumerate(cfg.datasets):
            if not ds_cfg.type:
                continue
            if ds_cfg.type == "sharegpt:chat":
                LOG.warning(
                    PendingDeprecationWarning(
                        "`type: sharegpt:chat` will soon be deprecated. simply use `type: sharegpt` instead."
                    )
                )
                cfg.datasets[idx].type = "sharegpt"
            if "sharegpt_simple" in ds_cfg.type:
                LOG.warning(
                    PendingDeprecationWarning(
                        "`type: sharegpt_simple` will soon be deprecated. simply use `type: sharegpt` instead."
                    )
                )
                cfg.datasets[idx].type = cfg.datasets[idx].type.replace(
                    "sharegpt_simple", "sharegpt"
                )

    if cfg.saves_per_epoch and cfg.save_steps:
        raise ValueError(
            "save_steps and saves_per_epoch are mutually exclusive and cannot be used together."
        )
    if cfg.saves_per_epoch and cfg.save_strategy and cfg.save_strategy != "steps":
        raise ValueError(
            "save_strategy must be empty or set to `steps` when used with saves_per_epoch."
        )
    if cfg.evals_per_epoch and cfg.eval_steps:
        raise ValueError(
            "eval_steps and evals_per_epoch are mutually exclusive and cannot be used together."
        )
    if (
        cfg.evals_per_epoch
        and cfg.evaluation_strategy
        and cfg.evaluation_strategy != "steps"
    ):
        raise ValueError(
            "evaluation_strategy must be empty or set to `steps` when used with evals_per_epoch."
        )
    if cfg.save_strategy and cfg.save_steps and cfg.save_strategy != "steps":
        raise ValueError(
            "save_strategy and save_steps mismatch. Please set save_strategy to 'steps' or remove save_steps."
        )

    if (
        cfg.evaluation_strategy
        and cfg.eval_steps
        and cfg.evaluation_strategy != "steps"
    ):
        raise ValueError(
            "evaluation_strategy and eval_steps mismatch. Please set evaluation_strategy to 'steps' or remove eval_steps."
        )

    if (
        cfg.val_set_size == 0
        and (cfg.eval_steps or cfg.evaluation_strategy)
        and not cfg.test_datasets
    ):
        raise ValueError(
            "eval_steps and evaluation_strategy are not supported with val_set_size == 0"
        )

    if (
        cfg.sample_packing
        and cfg.eval_table_size
        and cfg.eval_sample_packing is not False
    ):
        raise ValueError(
            "eval_table_size and eval_sample_packing are not supported together with sample_packing. Please set 'eval_sample_packing' to false."
        )

    if not cfg.adapter and (cfg.load_in_8bit or cfg.load_in_4bit):
        raise ValueError(
            "load_in_8bit and load_in_4bit are not supported without setting an adapter."
            "If you want to full finetune, please turn off load_in_8bit and load_in_4bit."
        )

    if cfg.rope_scaling:
        LOG.warning("`rope_scaling` should now be be a key under `model_config`")

    if cfg.wandb_run_id and not cfg.wandb_name:
        cfg.wandb_name = cfg.wandb_run_id

        LOG.warning(
            "wandb_run_id sets the ID of the run. If you would like to set the name, please use wandb_name instead."
        )

    if cfg.noisy_embedding_alpha is not None:
        # Deprecated, use neftune_noise_alpha
        LOG.warning("noisy_embedding_alpha is deprecated, use neftune_noise_alpha")
        if cfg.neftune_noise_alpha is None:
            cfg.neftune_noise_alpha = cfg.noisy_embedding_alpha
        else:
            # User is providing both; bail and have them sort out their settings
            raise ValueError(
                "noisy_embedding_alpha is deprecated, use neftune_noise_alpha; both are set, please remove the deprecated noisy_embedding_alpha setting"
            )

    if cfg.neftune_noise_alpha is not None and cfg.neftune_noise_alpha <= 0.0:
        raise ValueError("neftune_noise_alpha must be > 0.0")

    if cfg.max_memory is not None and cfg.gpu_memory_limit is not None:
        raise ValueError(
            "max_memory and gpu_memory_limit are mutually exclusive and cannot be used together."
        )

    if (
        cfg.unfrozen_parameters
        and cfg.gradient_checkpointing_kwargs
        and cfg.gradient_checkpointing_kwargs.use_reentrant is True
    ):
        # https://github.com/huggingface/transformers/issues/21381
        raise ValueError(
            "`use_reentrant` must be false when used with partially frozen model."
        )

    if cfg.deepspeed and Path(cfg.deepspeed).is_file():
        with open(cfg.deepspeed, encoding="utf-8") as file:
            contents = file.read()
            deepspeed_cfg: DictDefault = DictDefault(json.loads(contents))
            if cfg.flash_attention:
                if (
                    deepspeed_cfg.zero_optimization
                    and deepspeed_cfg.zero_optimization.stage == 3
                ):
                    if not (
                        (
                            deepspeed_cfg.bf16
                            and deepspeed_cfg.bf16.enabled  # pylint: disable=no-member
                            is True
                        )
                        or (
                            deepspeed_cfg.fp16
                            and deepspeed_cfg.fp16.enabled  # pylint: disable=no-member
                            is True
                        )
                    ):
                        raise ValueError(
                            "bf16.enabled or fp16.enabled must be set to true when using ZeRO-3 with flash-attention"
                        )
            if "8bit" in cfg.optimizer and deepspeed_cfg.optimizer:
                LOG.warning(
                    f"conflicting optimizer: {cfg.optimizer} used alongside deepspeed optimizer."
                )

    if cfg.test_datasets and cfg.val_set_size:
        raise ValueError(
            "non-zero val_set_size should not be used with test_datasets configuration"
        )

    if cfg.fsdp and "bnb" in cfg.optimizer:
        raise ValueError(f"FSDP not compatible with {cfg.optimizer}")

    if cfg.do_causal_lm_eval and cfg.eval_sample_packing:
        raise ValueError(
            "do_causal_lm_eval is enabled, eval_sample_packing must be set to False"
        )

    if cfg.eval_causal_lm_metrics:
        supported_metrics = ["sacrebleu", "comet", "ter", "chrf"]
        if not isinstance(cfg.eval_causal_lm_metrics, list):
            raise ValueError("eval_causal_lm_metrics must be a list")
        # only ["sacrebleu", "comet", "ter", "chrf"] supported
        if set(cfg.eval_causal_lm_metrics) - set(supported_metrics):
            raise ValueError(
                f"eval_causal_lm_metrics must be one of {supported_metrics}"
            )

    # TODO
    # MPT 7b
    # https://github.com/facebookresearch/bitsandbytes/issues/25
    # no 8bit adaAmw w bf16

    # GPT-NeoX
    # evals broken when extending context len
    # File "/root/miniconda3/envs/py3.9/lib/python3.9/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py", line 162, in forward                        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
    # File "/root/miniconda3/envs/py3.9/lib/python3.9/site-packages/optimum/bettertransformer/models/attention.py", line 74, in gpt2_wrapped_scaled_dot_product
    # attention_mask = causal_mask + attention_mask
    # RuntimeError: The size of tensor a (2048) must match the size of tensor b (8132) at non-singleton dimension 3
