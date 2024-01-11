"""Module for working with config dicts"""

import logging
import os

import torch
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.utils.bench import log_gpu_memory_usage
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
    cfg.eval_table_max_new_tokens = cfg.eval_table_max_new_tokens or 128
    choose_device(cfg)
    cfg.ddp = cfg.ddp if cfg.ddp is not None else cfg.world_size != 1
    if cfg.ddp:
        cfg.device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        cfg.batch_size = cfg.batch_size * cfg.world_size

    if cfg.device == "mps":
        cfg.load_in_8bit = False
        cfg.tf32 = False
        if cfg.bf16:
            cfg.fp16 = True
        cfg.bf16 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32 or False

    if cfg.bf16 or cfg.bfloat16:
        cfg.torch_dtype = torch.bfloat16
    elif cfg.load_in_8bit or cfg.fp16 or cfg.float16:
        cfg.torch_dtype = torch.float16
    else:
        cfg.torch_dtype = torch.float32

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
        or (cfg.model_type and "llama" in cfg.model_type.lower())
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
        or (cfg.model_type and "rwforcausallm" in cfg.model_type.lower())
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
        or "mistral" in cfg.base_model.lower()
        or (cfg.model_type and "mistral" in cfg.model_type.lower())
    )

    cfg.is_qwen_derived_model = (
        (
            hasattr(model_config, "model_type")
            and model_config.model_type
            in [
                "qwen",
            ]
        )
        or cfg.is_qwen_derived_model
        or "qwen" in cfg.base_model.lower()
        or (cfg.model_type and "qwen" in cfg.model_type.lower())
    )

    if isinstance(cfg.learning_rate, str):
        cfg.learning_rate = float(cfg.learning_rate)

    log_gpu_memory_usage(LOG, "baseline", cfg.device)


def validate_config(cfg):
    if is_torch_bf16_gpu_available():
        if not cfg.bf16 and not cfg.bfloat16:
            LOG.info("bf16 support detected, but not enabled for this configuration.")
    else:
        if not cfg.merge_lora and (cfg.bf16 or cfg.bfloat16):
            raise ValueError(
                "bf16 requested, but AMP is not supported on this GPU. Requires Ampere series or above."
            )
    if cfg.max_packed_sequence_len and cfg.sample_packing:
        raise ValueError(
            "please set only one of max_packed_sequence_len (deprecated soon) or sample_packing"
        )
    if cfg.max_packed_sequence_len:
        LOG.warning(
            str(
                PendingDeprecationWarning(
                    "max_packed_sequence_len will be deprecated in favor of sample_packing"
                )
            )
        )

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

    if cfg.load_4bit:
        raise ValueError("cfg.load_4bit parameter has been deprecated")

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

    if not cfg.load_in_8bit and cfg.adapter == "lora":
        LOG.warning("We recommend setting `load_in_8bit: true` for LORA finetuning")

    if cfg.adapter == "lora" and (cfg.flash_attn_fuse_qkv or cfg.flash_attn_fuse_mlp):
        raise ValueError("Fused modules are not supported with LoRA")

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
        if cfg.float16 is not True and cfg.bloat16 is not True:
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

    if cfg.gptq and cfg.model_revision:
        raise ValueError(
            "model_revision is not supported for GPTQ models. "
            + "Please download the model from HuggingFace Hub manually for correct branch, "
            + "point to its path, and remove model_revision from the config."
        )

    if cfg.sample_packing and cfg.sdp_attention:
        # incompatible due to bug w/ accelerate causing 0.0 loss when using llama2
        raise ValueError(
            "sample_packing not compatible with sdp_attention. Use flash_attention"
        )

    if cfg.sample_packing and cfg.xformers_attention:
        raise ValueError(
            "sample_packing not compatible with xformers_attention. Use flash_attention"
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

    if cfg.model_type == "MixFormerSequentialForCausalLM" and cfg.adapter is not None:
        LOG.warning("Use AutoModelForCausalLM for phi/MixFormer models with qLoRA")

    if cfg.model_config_type == "mixformer-sequential":
        if cfg.sample_packing:
            if cfg.adapter is not None:
                LOG.warning(
                    "phi/MixFormer models are not currently compatible with LoRA and sample_packing"
                )
            if cfg.model_type == "AutoModelForCausalLM":
                raise ValueError(
                    "`model_type: MixFormerSequentialForCausalLM` required for sample_packing"
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

    if cfg.val_set_size == 0 and (cfg.eval_steps or cfg.evaluation_strategy):
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

    if cfg.warmup_steps and cfg.warmup_ratio:
        raise ValueError("warmup_steps and warmup_ratio are mutually exclusive")

    if cfg.is_qwen_derived_model and cfg.gradient_checkpointing:
        LOG.warning(
            "Gradient checkpointing is broken for Qwen models for transformers>=4.35.0, except main branch."
        )

    if cfg.wandb_run_id and not cfg.wandb_name:
        cfg.wandb_name = cfg.wandb_run_id

        LOG.warning(
            "wandb_run_id sets the ID of the run. If you would like to set the name, please use wandb_name instead."
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
