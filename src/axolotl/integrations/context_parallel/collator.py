"""Cut each rank's context-parallel shard in the collator instead of the forward pre-hook."""

from functools import partial
from typing import Callable

from axolotl.utils.collators import DataCollatorForSeq2Seq
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

CP_SHARD_KEY = "_cp_shard"


def collator_shard_fn(runtime) -> Callable | None:
    """Sharding callable for the collators, or ``None`` when ringmaster cannot pre-shard."""
    from ringmaster import shard as rm_shard

    shard = getattr(rm_shard, "shard_collated_batch", None)
    if shard is None:
        LOG.info(
            "installed ringmaster has no shard_collated_batch; sharding in the forward pre-hook"
        )
        return None
    load_balance = runtime.shard_load_balance
    if load_balance not in ("contiguous", "head_tail"):
        LOG.info(
            "shard layout %s is not supported in the collator; sharding in the forward pre-hook",
            load_balance,
        )
        return None
    return partial(
        shard,
        cp_rank=runtime.cp_rank,
        cp_size=runtime.cp_size,
        load_balance=load_balance,
    )


def dispatches_batches(cfg) -> bool:
    """Accelerate slices every tensor of a dispatched batch, including a pre-cut shard."""
    if not (
        getattr(cfg, "streaming", None) or getattr(cfg, "pretraining_dataset", None)
    ):
        return False
    return (getattr(cfg, "accelerator_config", None) or {}).get(
        "dispatch_batches"
    ) is not False


def attach_collator_sharding(trainer, runtime) -> Callable[[], None]:
    """Point the trainer's sequence collators at ringmaster; returns a function that detaches them."""
    shard = collator_shard_fn(runtime)
    attached = []
    if shard is not None:
        for collator in (
            getattr(trainer, "data_collator", None),
            getattr(trainer, "eval_data_collator", None),
        ):
            if (
                isinstance(collator, DataCollatorForSeq2Seq)
                and collator.cp_shard is None
            ):
                collator.cp_shard = shard
                attached.append(collator)
    if attached:
        LOG.info(
            "context-parallel shards are cut in the collator (%d collators)",
            len(attached),
        )

    def restore():
        for collator in attached:
            collator.cp_shard = None

    return restore
