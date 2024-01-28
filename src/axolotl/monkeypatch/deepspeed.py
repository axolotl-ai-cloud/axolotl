"""
Integration with Deepspeed
"""

import logging

import transformers.trainer

LOG = logging.getLogger("axolotl.monkeypatch.deepspeed")


def patched_deepspeed_load_checkpoint(deepspeed_engine, checkpoint_path):
    # it's possible that the user is trying to resume from model_path, which doesn't necessarily
    # contain a deepspeed checkpoint. e.g. examples just check if the dir exists and assume it's
    # a resume from a checkpoint and not just a local pretrained weight. So we check here if the
    # path contains what looks like a deepspeed checkpoint
    import glob

    deepspeed_checkpoint_dirs = sorted(glob.glob(f"{checkpoint_path}/global_step*"))

    if len(deepspeed_checkpoint_dirs) > 0:
        LOG.info(f"Attempting to resume from {checkpoint_path}")
        # this magically updates self.optimizer and self.lr_scheduler
        load_path, _ = deepspeed_engine.load_checkpoint(
            checkpoint_path,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_strict=False,
        )
        if load_path is None:
            raise ValueError(
                f"[deepspeed] failed to resume from checkpoint {checkpoint_path}"
            )
    else:
        raise ValueError(f"Can't find a valid checkpoint at {checkpoint_path}")


transformers.trainer.deepspeed_load_checkpoint = patched_deepspeed_load_checkpoint
