import logging


def validate_config(cfg):
    if cfg.adapter == "qlora":
        if cfg.merge_lora:
            # can't merge qlora if loaded in 8bit or 4bit
            assert cfg.load_in_8bit is False
            assert cfg.load_4bit is False
            assert cfg.load_in_4bit is False
        else:
            assert cfg.load_in_8bit is False
            assert cfg.load_4bit is False
            assert cfg.load_in_4bit is True
    if cfg.load_in_8bit and cfg.adapter == "lora":
        logging.warning("we recommend setting `load_in_8bit: true`")

    # TODO
    # MPT 7b
    # https://github.com/facebookresearch/bitsandbytes/issues/25
    # no 8bit adamw w bf16
