"""NVFP4Plugin — wires the FP4-GEMM training swap into the model-load lifecycle via plugin hooks (core tree unmodified)."""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class NVFP4Plugin(BasePlugin):
    """Native FP4-GEMM training on Blackwell (sm_100/sm_120) via module-swap."""

    def get_input_args(self):
        return "axolotl.integrations.nvfp4.NVFP4PluginArgs"

    @staticmethod
    def _enabled(cfg) -> bool:
        nvfp4 = getattr(cfg, "nvfp4_training", None)
        if not nvfp4:
            return False
        if isinstance(nvfp4, dict):
            return bool(nvfp4.get("enabled"))
        return bool(getattr(nvfp4, "enabled", False))

    @staticmethod
    def _validate_cfg(cfg):
        if cfg.load_in_4bit or cfg.load_in_8bit:
            raise ValueError(
                "nvfp4_training is incompatible with load_in_4bit/load_in_8bit "
                "(bitsandbytes replaces the Linear modules, so the NVFP4 swap "
                "matches zero layers and silently does nothing). NVFP4-QLoRA is "
                "the 4-bit storage path: drop load_in_*bit and use adapter: lora "
                "with nvfp4_training.base: storage."
            )
        if cfg.deepspeed:
            raise ValueError(
                "nvfp4_training does not support DeepSpeed (there is no ZeRO "
                "path for the FP4-GEMM module swap). Remove `deepspeed` and use "
                "DDP, FSDP2, or single-GPU."
            )
        if cfg.fp16:
            raise ValueError(
                "nvfp4_training requires bf16 — the NVFP4 quantizer does not "
                "support fp16. Set bf16: true instead of fp16."
            )

    def pre_model_load(self, cfg):
        """Config validation + requirements check + dynamo tuning before the
        model is built."""
        if not self._enabled(cfg):
            return
        from .nvfp4_training import nvfp4_supported
        from .patches import configure_dynamo_for_nvfp4, warn_unfilled_gemms

        self._validate_cfg(cfg)
        ok, reason = nvfp4_supported()
        if not ok:
            raise RuntimeError(f"nvfp4_training enabled but unsupported: {reason}")
        warn_unfilled_gemms(cfg)
        configure_dynamo_for_nvfp4(cfg)

    def post_model_load(self, cfg, model):
        """The module swap — runs after weights load and PEFT wrapping."""
        if not self._enabled(cfg):
            return
        from .swap import apply_nvfp4_training, mark_ddp_ignore

        apply_nvfp4_training(cfg, model)
        mark_ddp_ignore(cfg, model)

    def post_trainer_create(self, cfg, trainer):
        """Attach the resume-integrity guard and the FP4-packed save sidecar."""
        if not self._enabled(cfg):
            return
        from .callbacks import NVFP4ResumeIntegrityCallback, NVFP4SaveCallback

        if cfg.resume_from_checkpoint:
            trainer.add_callback(NVFP4ResumeIntegrityCallback(cfg))
        if getattr(cfg.nvfp4_training, "save_packed", False):
            trainer.add_callback(NVFP4SaveCallback(cfg, trainer))

    def post_train(self, cfg, model):
        """Write the final FP4-packed sidecar for a save_packed run."""
        if not self._enabled(cfg) or not getattr(
            cfg.nvfp4_training, "save_packed", False
        ):
            return
        import torch.distributed as dist

        # one shared sidecar -> only the main process writes (avoids a multi-GPU race)
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        from .nvfp4_training import save_nvfp4_packed

        save_nvfp4_packed(model, cfg.output_dir)
