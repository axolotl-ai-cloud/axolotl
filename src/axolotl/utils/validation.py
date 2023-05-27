import logging


def validate_config(cfg):
    if cfg.load_4bit:
        raise ValueError("cfg.load_4bit parameter has been deprecated and replaced by cfg.gptq")

    if cfg.adapter == "qlora":
        if cfg.merge_lora:
            # can't merge qlora if loaded in 8bit or 4bit
            assert cfg.load_in_8bit is not True
            assert cfg.gptq is not True
            assert cfg.load_in_4bit is not True
        else:
            assert cfg.load_in_8bit is not True
            assert cfg.gptq is not True
            assert cfg.load_in_4bit is True

    if not cfg.load_in_8bit and cfg.adapter == "lora":
        logging.warning("We recommend setting `load_in_8bit: true` for LORA finetuning")
        
    if cfg.trust_remote_code:
        logging.warning("`trust_remote_code` is set to true. Please make sure that you reviewed the remote code/model.")

    # TODO
    # MPT 7b
    # https://github.com/facebookresearch/bitsandbytes/issues/25
    # no 8bit adamw w bf16
