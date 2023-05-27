import logging


def validate_config(cfg):
    if cfg.load_4bit:
        raise ValueError("cfg.load_4bit parameter has been deprecated and replaced by cfg.gptq")

    if cfg.adapter == "qlora":
        if cfg.merge_lora:
            # can't merge qlora if loaded in 8bit or 4bit
            assert cfg.load_in_8bit is False
            assert cfg.gptq is False
            assert cfg.load_in_4bit is False
        else:
            assert cfg.load_in_8bit is False
            assert cfg.gptq is False
            assert cfg.load_in_4bit is True

    if not cfg.load_in_8bit and cfg.adapter == "lora":
        logging.warning("We recommend setting `load_in_8bit: true` for LORA finetuning")

    # TODO
    # MPT 7b
    # https://github.com/facebookresearch/bitsandbytes/issues/25
    # no 8bit adamw w bf16
