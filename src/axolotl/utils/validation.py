"""Module for validating config files"""

import logging

import torch


def validate_config(cfg):
    if cfg.gradient_accumulation_steps and cfg.batch_size:
        raise ValueError(
            "please set only one of gradient_accumulation_steps or batch_size"
        )
    if cfg.batch_size:
        logging.warning(
            "%s\n%s",
            "batch_size is not recommended. Please use gradient_accumulation_steps instead.",
            "To calculate the equivalent gradient_accumulation_steps, divide batch_size / micro_batch_size / number of gpus.",
        )
    if cfg.load_4bit:
        raise ValueError(
            "cfg.load_4bit parameter has been deprecated and replaced by cfg.gptq"
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

    if not cfg.load_in_8bit and cfg.adapter == "lora":
        logging.warning("We recommend setting `load_in_8bit: true` for LORA finetuning")

    if cfg.trust_remote_code:
        logging.warning(
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
            logging.warning(
                "BetterTransformers probably doesn't work with PEFT adapters"
            )
        if cfg.fp16 or cfg.bf16:
            raise ValueError("AMP is not supported with BetterTransformer")
        if cfg.float16 is not True and cfg.bloat16 is not True:
            logging.warning(
                "You should probably set bfloat16 or float16 to true to "
                "load the model in float16 for BetterTransformers"
            )
        if int(torch.__version__.split(".")[0]) < 2:
            logging.warning("torch>=2.0.0 required")
            raise ValueError(
                f"flash_optimum for BetterTransformers may not be used with {torch.__version__}"
            )

    if cfg.pretraining_dataset and cfg.group_by_length:
        logging.warning(
            "You probably want to disable group_by_length as it will force a streamed dataset to download completely."
        )

    if any([cfg.adamw_beta1, cfg.adamw_beta2, cfg.adamw_epsilon]) and (
        not cfg.optimizer or "adamw" not in cfg.optimizer
    ):
        logging.warning("adamw hyperparameters found, but no adamw optimizer set")

    if cfg.push_to_hub_model_id:
        raise ValueError(
            "push_to_hub_model_id is deprecated. Please use hub_model_id instead."
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
