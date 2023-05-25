def validate_config(cfg):
    if cfg.adapter == "qlora":
        assert cfg.load_in_8bit is False
        assert cfg.load_4bit is False
        assert cfg.load_in_4bit is True
    pass
    # TODO
    # MPT 7b
    # https://github.com/facebookresearch/bitsandbytes/issues/25
    # no 8bit adamw w bf16
