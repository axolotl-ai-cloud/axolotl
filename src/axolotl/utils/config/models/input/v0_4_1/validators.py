import logging

from axolotl.utils.config.models.input.v0_4_1 import AxolotlInputConfig
from axolotl.utils.config.models.internals import GPUCapabilities

LOG = logging.getLogger("axolotl.utils.config.models.input.validators")


def hints_against_gpu_support(
    cfg: AxolotlInputConfig, gpu_capabilities: GPUCapabilities
):
    if not cfg.bf16 and not cfg.bfloat16 and gpu_capabilities.bf16:
        LOG.info("bf16 support detected, but not enabled for this configuration.")


def validate_against_gpu_support(
    cfg: AxolotlInputConfig, gpu_capabilities: GPUCapabilities
):
    if not gpu_capabilities.bf16:
        if (
            not cfg.merge_lora
            and not cfg.is_preprocess
            and (cfg.bf16 is True or cfg.bfloat16 is True)
        ):
            raise ValueError(
                "bf16 requested, but AMP is not supported on this GPU. Requires Ampere series or above."
            )
