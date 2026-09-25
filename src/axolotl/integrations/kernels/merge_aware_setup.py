"""Select merge-aware training for recognized NVFP4 LoRA bases."""

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def configure_modelopt_merge_aware(cfg):
    """Apply the merge-aware default after the checkpoint format is known."""
    if cfg.get("adapter") not in ("lora", "multilora"):
        return
    if cfg.get("nvfp4_merge_aware") is False:
        LOG.warning(
            "NVFP4 MERGE WARNING: nvfp4_merge_aware is explicitly disabled. "
            "Continuing with ordinary LoRA; merging back into NVFP4 can round away "
            "the learned adapter update."
        )
        return
    if not cfg.get("use_sonicmoe"):
        cfg["nvfp4_merge_aware"] = False
        LOG.warning(
            "NVFP4 MERGE WARNING: the selected expert backend does not support "
            "merge-aware training. Continuing with ordinary LoRA; the merged NVFP4 "
            "model may lose the learned adapter update."
        )
        return
    if cfg.get("nvfp4_merge_aware") is None:
        cfg["nvfp4_merge_aware"] = True
        LOG.info("Enabled merge-aware training for the NVFP4 LoRA base")


def configure_native_merge_aware(cfg, model, *, sharded_backend=None):
    """Install native merge-aware forwards where adapter weights remain materialized."""
    if cfg.get("adapter") not in ("lora", "multilora") or not any(
        type(parameter).__name__ == "NVFP4Tensor" for parameter in model.parameters()
    ):
        return
    if cfg.get("adapter") == "multilora":
        sharded_backend = "multi-LoRA"
    if cfg.get("nvfp4_merge_aware") is False or sharded_backend:
        reason = (
            f"native merge-aware integration with {sharded_backend} is not qualified"
            if sharded_backend
            else "merge-aware training is explicitly disabled"
        )
        LOG.warning(
            "NVFP4 MERGE WARNING: %s. Continuing with ordinary LoRA; "
            "merging into NVFP4 may round away the learned adapter update.",
            reason,
        )
        model._axolotl_merge_aware_unsupported = True
        return
    from axolotl.monkeypatch.torchao_nvfp4_merge import (
        install_native_nvfp4_merge_aware_lora_linears,
    )

    installed = install_native_nvfp4_merge_aware_lora_linears(model)
    if installed:
        LOG.info(
            "Enabled native NVFP4 merge-aware training on %d projections", installed
        )
    else:
        model._axolotl_merge_aware_unsupported = True
        LOG.warning(
            "NVFP4 MERGE WARNING: no supported native merge-aware LoRA projections "
            "were installed. Continuing without a merged-NVFP4 parity guarantee."
        )
