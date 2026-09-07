"""Monkeypatch for TRL trainer FSDP preparation."""


def prepare_fsdp(model, accelerator):
    from axolotl.monkeypatch.accelerate.fsdp2 import fsdp2_prepare_model

    return fsdp2_prepare_model(accelerator, model)


def patch_trl_prepare_fsdp2():
    import trl.models.utils

    trl.models.utils.prepare_fsdp = prepare_fsdp
