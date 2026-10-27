"""Merge-rounding diagnostic for LoRA weight sync to vLLM."""

from transformers import Trainer

from axolotl.utils.logging import get_logger
from axolotl.utils.lora_merge_precision import (
    VLLM_SYNC_REMEDY,
    MergeRetention,
    log_merge_retention,
    measure_lora_merge_retention,
)

LOG = get_logger(__name__)

DEFAULT_MERGE_CHECK_INTERVAL = 50


class LoraMergeCheckMixin(Trainer):
    """Measure how much of the LoRA update a merged vLLM weight sync keeps.

    Runs on the main process at the first sync after training starts and then every
    ``vllm_merge_check_interval`` optimizer steps, logging ``lora_merge/retained`` and
    ``lora_merge/changed_fraction``.
    """

    _lora_merge_check_step: int | None = None
    _lora_merge_check_unavailable_logged = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wrap_merged_sync_with_check()

    def _lora_merge_check_interval(self) -> int:
        return int(
            getattr(
                self.args, "vllm_merge_check_interval", DEFAULT_MERGE_CHECK_INTERVAL
            )
            or 0
        )

    def _wrap_merged_sync_with_check(self) -> None:
        from accelerate.utils import is_peft_model

        vllm_generation = getattr(self, "vllm_generation", None)
        if vllm_generation is None or not self._lora_merge_check_interval():
            return
        if getattr(self.args, "vllm_lora_sync", False):
            return
        if not is_peft_model(vllm_generation.model):
            return
        # Paths that replace sync_weights (LoRA filesystem sync, async no-merge sync)
        # drop this wrapper; the no-merge sync measures its own tensors.
        sync_weights = vllm_generation.sync_weights

        def sync_weights_with_merge_check():
            if self._lora_merge_check_due():
                self._record_lora_merge_check(
                    measure_lora_merge_retention(vllm_generation.model)
                )
            sync_weights()

        vllm_generation.sync_weights = sync_weights_with_merge_check

    def _lora_merge_check_due(self) -> bool:
        interval = self._lora_merge_check_interval()
        if not interval or not self.accelerator.is_main_process:
            return False
        step = self.state.global_step
        if self._lora_merge_check_step is None:
            return step > 0
        return step - self._lora_merge_check_step >= interval

    def _record_lora_merge_check(self, stats: MergeRetention) -> None:
        step = self.state.global_step
        if stats.retained is None:
            if stats.skipped and not self._lora_merge_check_unavailable_logged:
                LOG.info(
                    "LoRA merge rounding check skipped: %d LoRA linears are sharded or "
                    "quantized on this rank",
                    stats.skipped,
                )
                self._lora_merge_check_unavailable_logged = True
            return
        # At the first sync every run's delta is still far below a BF16 ulp; only
        # later checks are actionable.
        first = self._lora_merge_check_step is None
        self._lora_merge_check_step = step
        log_merge_retention(
            stats,
            f"vLLM merged weight sync at step {step}",
            VLLM_SYNC_REMEDY,
            warn=not first,
        )
        metrics = getattr(self, "_metrics", None)
        if metrics is not None:
            metrics["train"]["lora_merge/retained"].append(stats.retained)
            metrics["train"]["lora_merge/changed_fraction"].append(
                stats.changed_fraction
            )
