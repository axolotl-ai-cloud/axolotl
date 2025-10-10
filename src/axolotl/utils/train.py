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
    checkpoints = sorted(
        (
            p
            for p in Path(cfg.output_dir).glob("checkpoint-*")
            if p.name.split("-")[-1].isdigit()
        ),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if checkpoints:
        last_checkpoint = str(checkpoints[-1])
        if not update:
            LOG.info(f"Resuming from last checkpoint at {last_checkpoint}")
            return last_checkpoint

    if (
        cfg.resume_from_checkpoint is None
        and cfg.auto_resume_from_checkpoints
        and last_checkpoint is not None
    ):
        cfg.resume_from_checkpoint = last_checkpoint
        LOG.info(
            "Using auto-resume functionality to resume from checkpoint at "
            f"{cfg.resume_from_checkpoint}"
        )
    return cfg.resume_from_checkpoint
