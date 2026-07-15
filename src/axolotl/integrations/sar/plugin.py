"""SAR (Subspace-Aligned Rewiring) plugin for Axolotl."""

import os

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process
from axolotl.utils.logging import get_logger

from .args import SARConfig

LOG = get_logger(__name__)


class SARPlugin(BasePlugin):
    """Projects post-training weight deltas onto the base model's spectral subspace.

    The projection runs on the main process only and can take hours at large scale;
    list this plugin last under ``plugins:`` so other plugins' post-train hooks (and
    any collectives they run) are not stuck behind it on the remaining ranks.
    """

    def get_input_args(self) -> str:
        return "axolotl.integrations.sar.args.SARArgs"

    def post_train_unload(self, cfg: DictDefault):
        if not cfg.sar:
            return
        sar_cfg = SARConfig.model_validate(cfg.sar)
        if not sar_cfg.run_after_training:
            return

        if not is_main_process():
            return

        rank_ratios = sar_cfg.rank_ratio
        if not isinstance(rank_ratios, list):
            rank_ratios = [rank_ratios]

        base_revision = sar_cfg.base_model_revision
        if base_revision is None and not sar_cfg.base_model:
            base_revision = cfg.revision_of_model

        try:
            from axolotl.integrations.sar.core import run_sar

            run_sar(
                base_model=sar_cfg.base_model or cfg.base_model,
                trained_model=sar_cfg.trained_model or cfg.output_dir,
                output_dir=sar_cfg.output_dir or os.path.join(cfg.output_dir, "sar"),
                merge_target=sar_cfg.merge_target,
                rank_ratios=rank_ratios,
                delta_rank_ratio=sar_cfg.delta_rank_ratio,
                projection=sar_cfg.projection,
                rewiring=sar_cfg.rewiring,
                scale=sar_cfg.scale,
                target_modules=sar_cfg.target_modules,
                exclude_modules=sar_cfg.exclude_modules,
                svd_device=sar_cfg.svd_device,
                save_dtype=sar_cfg.save_dtype,
                save_rewiring_matrix=sar_cfg.save_rewiring_matrix,
                base_model_revision=base_revision,
                trained_model_revision=sar_cfg.trained_model_revision,
                merge_target_revision=sar_cfg.merge_target_revision,
            )
        except Exception:
            LOG.error("SAR post-training projection failed", exc_info=True)
