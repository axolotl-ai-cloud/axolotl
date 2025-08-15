"""Training utils for checkpoints"""

from pathlib import Path

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def determine_last_checkpoint(cfg: DictDefault, update: bool = True) -> str | None:
    """
    Determine the checkpoint to resume from based on configuration.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        update: Whether to update the config with the determined checkpoint

    Returns:
        Path to the checkpoint to resume from, or `None` if not resuming.
    """
    last_checkpoint = None
    possible_checkpoints = [str(cp) for cp in Path(cfg.output_dir).glob("checkpoint-*")]
    if len(possible_checkpoints) > 0:
        sorted_paths = sorted(
            possible_checkpoints,
            key=lambda path: int(path.split("-")[-1]),
        )
        if not update:
            return sorted_paths[-1]
        last_checkpoint = sorted_paths[-1]

    if cfg.resume_from_checkpoint is None and cfg.auto_resume_from_checkpoints:
        if last_checkpoint is not None:
            cfg.resume_from_checkpoint = sorted_paths[-1]
            LOG.info(
                f"Using Auto-resume functionality to start with checkpoint at {cfg.resume_from_checkpoint}"
            )
    return cfg.resume_from_checkpoint
