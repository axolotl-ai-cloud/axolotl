"""Module for validating config files"""

import logging


def validate_config(cfg):
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

    # TODO
    # MPT 7b
    # https://github.com/facebookresearch/bitsandbytes/issues/25
    # no 8bit adamw w bf16
