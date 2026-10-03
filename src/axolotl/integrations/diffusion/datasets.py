"""Dataset preparation and packed-step accounting for native diffusion."""

from __future__ import annotations

import math
import os

import pyarrow as pa
import pyarrow.compute as pc
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from axolotl.common.datasets import TrainDatasetMeta
from axolotl.loaders import load_processor, load_tokenizer
from axolotl.utils.data.lock import FileLockLoader
from axolotl.utils.data.sft import _load_and_prepare_datasets
from axolotl.utils.distributed import reduce_and_broadcast
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler
from axolotl.utils.trainer import calculate_total_num_steps

from .lm.sampling import (
    filter_native_diffusion_dataset,
    native_packing_lengths,
    native_packing_options,
    resolve_native_packing_budget,
)

LOG = get_logger(__name__)


def _count_packed_steps(cfg, dataset, *, update: bool) -> int:
    if update and not cfg.total_num_tokens:
        cfg.total_num_tokens = int(
            pc.sum(dataset.data.column("length"), min_count=0).as_py()
            if "length" in dataset.column_names
            else pc.sum(
                pc.list_value_length(dataset.data.column("input_ids")), min_count=0
            ).as_py()
        )
    if update and not cfg.total_supervised_tokens:
        cfg.total_supervised_tokens = sum(
            int(
                (
                    batch.column("labels").flatten().to_numpy(zero_copy_only=False)
                    != -100
                ).sum()
            )
            if pa.types.is_list(batch.column("labels").type)
            or pa.types.is_large_list(batch.column("labels").type)
            else int(
                (batch.column("labels").to_numpy(zero_copy_only=False) != -100).sum()
            )
            for batch in dataset.data.to_batches(max_chunksize=1024)
        )
    options = native_packing_options(cfg)
    lengths = native_packing_lengths(dataset, **options)
    budget = resolve_native_packing_budget(cfg).payload_capacity
    sampler_cls = SequentialSampler if cfg.curriculum_sampling else RandomSampler
    sampler = MultipackBatchSampler(
        sampler=sampler_cls(dataset),
        lengths=lengths,
        batch_size=1,
        batch_max_len=budget,
        group_size=cfg.sample_packing_group_size,
        bin_size=cfg.sample_packing_bin_size,
        sequential=cfg.sample_packing_sequentially,
        drop_last=True,
        num_processes=cfg.dataset_num_proc,
        mp_start_method=cfg.sample_packing_mp_start_method or "fork",
    )
    count = len(DataLoader(dataset, batch_sampler=sampler))
    steps_per_epoch = max(1, count * cfg.micro_batch_size // cfg.batch_size)
    steps = math.floor(steps_per_epoch * cfg.num_epochs)
    if cfg.dataloader_drop_last:
        steps -= math.ceil(cfg.num_epochs)
    efficiency = reduce_and_broadcast(lambda: sampler.efficiency(), max)
    if update:
        cfg.sample_packing_eff_est = math.ceil(efficiency * 100) / 100
    return steps


def load_native_datasets(cfg, *, preprocess: bool = False) -> TrainDatasetMeta:
    """Use the standard SFT tokenizer/cache path before native budget checks."""
    if cfg.streaming or cfg.pretraining_dataset:
        raise ValueError("native diffusion requires a finite SFT dataset")
    tokenizer = load_tokenizer(cfg)
    processor = load_processor(cfg, tokenizer=tokenizer) if cfg.processor_type else None

    def load():
        train, evaluation, _ = _load_and_prepare_datasets(
            tokenizer, cfg, split="train", processor=processor
        )
        if cfg.test_datasets:
            _, evaluation, _ = _load_and_prepare_datasets(
                tokenizer, cfg, split="test", processor=processor
            )
        return train, evaluation

    loader = FileLockLoader(cfg)
    try:
        train, evaluation = loader.load(load)
    finally:
        loader.cleanup()

    train = filter_native_diffusion_dataset(cfg, train, split="train")
    evaluation = filter_native_diffusion_dataset(cfg, evaluation, split="eval")
    if preprocess or os.environ.get("AXOLOTL_IS_PREPROCESS") == "1":
        return TrainDatasetMeta(train, evaluation, -1)

    if (
        evaluation is not None
        and cfg.sample_packing
        and cfg.eval_sample_packing is not False
    ):
        if _count_packed_steps(cfg, evaluation, update=False) == 0:
            raise ValueError(
                "eval dataset split is too small for sample_packing. "
                "Set eval_sample_packing: false in the config."
            )
    if cfg.sample_packing:
        total_steps = _count_packed_steps(cfg, train, update=True)
    else:
        total_steps = calculate_total_num_steps(cfg, train)
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    LOG.info("Maximum number of steps set at %s", total_steps)
    return TrainDatasetMeta(train, evaluation, total_steps)
