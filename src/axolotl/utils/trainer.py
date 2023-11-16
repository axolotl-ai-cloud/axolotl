"""Module containing the Trainer class and related functions"""
import math
import os
from contextlib import contextmanager
from functools import partial
from typing import List

import numpy as np
import torch
import torch.cuda
from accelerate.logging import get_logger
from datasets import set_caching_enabled
from torch.utils.data import DataLoader, RandomSampler

from axolotl.core.trainer_builder import HFCausalTrainerBuilder
from axolotl.utils.distributed import is_main_process, reduce_and_broadcast, zero_first
from axolotl.utils.samplers import MultipackBatchSampler

LOG = get_logger("axolotl")


@torch.jit.script
def weighted_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
):
    # Flatten the logits, labels, and weights tensors
    logits = logits.view(
        -1, logits.size(-1)
    )  # logits becomes of shape [batch_size*sequence_length, vocab_size]
    labels = labels.view(-1)  # labels becomes of shape [batch_size*sequence_length]
    weights = weights.view(-1)  # weights becomes of shape [batch_size*sequence_length]

    # Compute the unweighted cross entropy loss
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

    # Apply the weights to the losses and compute their sum
    return (weights * losses).sum()


@torch.jit.script
def create_weighted_mask(labels: torch.Tensor):
    # Check if the tensor is 2D. If not, unsqueeze it to make it 2D
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(0)

    weights = torch.zeros_like(labels).float()
    for i in range(labels.shape[0]):
        mask = labels[i] != -100

        # Create a tensor to track group ids
        group_ids = torch.zeros_like(labels[i]).int()
        curr_group_id = 0

        for j in range(1, len(labels[i])):
            if mask[j] and not mask[j - 1]:  # switch from masked to unmasked label
                curr_group_id += 1  # start new group
            group_ids[j] = (
                curr_group_id if mask[j] else 0
            )  # assign group id if unmasked label

        # Count only unmasked labels in each group
        group_counts = torch.bincount(group_ids[mask])

        mask_weights = torch.zeros_like(labels[i]).float()
        mask_weights[mask] = 1.0 / group_counts[group_ids[mask]]

        weights[i] = mask_weights

    return weights.squeeze()  # squeeze the output to match the input dimension


def trainer_weighted_loss(model_output, labels, shift_labels=True):
    logits = (
        model_output["logits"] if isinstance(model_output, dict) else model_output[0]
    )
    if shift_labels:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    weights = create_weighted_mask(labels)
    return weighted_cross_entropy(logits, labels, weights)


def add_position_ids(sample):
    sample_len = len(sample["input_ids"])
    sample["position_ids"] = torch.arange(len(sample["input_ids"]))
    sample["length"] = sample_len
    return sample


def add_length(sample):
    sample["length"] = len(sample["input_ids"])
    return sample


def drop_long_seq(sample, sequence_len=2048):
    return len(sample["input_ids"]) <= sequence_len and len(sample["input_ids"]) > 0


@contextmanager
def disable_datasets_caching():
    try:
        set_caching_enabled(False)
        yield
    finally:
        set_caching_enabled(True)


def process_datasets_for_packing(cfg, train_dataset, eval_dataset, tokenizer):
    drop_long = partial(drop_long_seq, sequence_len=cfg.sequence_len)
    with zero_first(is_main_process()):
        train_dataset = train_dataset.filter(drop_long, num_proc=cfg.dataset_processes)
        if eval_dataset:
            eval_dataset = eval_dataset.filter(
                drop_long, num_proc=cfg.dataset_processes
            )

        if cfg.group_by_length:
            train_dataset = train_dataset.map(
                add_length, num_proc=cfg.dataset_processes
            )

        if cfg.sample_packing:
            train_dataset = train_dataset.map(
                add_position_ids, num_proc=cfg.dataset_processes
            )
            if cfg.eval_sample_packing is not False:
                if eval_dataset:
                    eval_dataset = eval_dataset.map(
                        add_position_ids, num_proc=cfg.dataset_processes
                    )

        # Phi doesn't want the attention_mask feature when training
        if "CodeGenTokenizer" in tokenizer.__class__.__name__ or (
            cfg.is_mistral_derived_model and cfg.flash_attention
        ):
            train_dataset = train_dataset.remove_columns("attention_mask")
            if eval_dataset:
                eval_dataset = eval_dataset.remove_columns("attention_mask")

    return train_dataset, eval_dataset


def calculate_total_num_steps(cfg, train_dataset, update=True):
    if not cfg.total_num_tokens:
        total_num_tokens = np.sum(
            train_dataset.data.column("input_ids")
            .to_pandas()
            .apply(lambda x: len(x))  # pylint: disable=unnecessary-lambda
            .values
        )
        LOG.debug(f"total_num_tokens: {total_num_tokens}", main_process_only=True)
        if update:
            cfg.total_num_tokens = total_num_tokens

    if not cfg.total_supervised_tokens:
        total_supervised_tokens = (
            train_dataset.data.column("labels")
            .to_pandas()
            .apply(lambda x: np.sum(np.array(x) != -100))
            .sum()
        )
        LOG.debug(
            f"`total_supervised_tokens: {total_supervised_tokens}`",
            main_process_only=True,
        )
        if update:
            cfg.total_supervised_tokens = total_supervised_tokens

    if cfg.sample_packing:
        # we have to drop anything longer then sequence len otherwise
        # flash attention with position ids fails

        if cfg.sample_packing_eff_est:
            total_num_steps = (
                # match count to len est in dataloader
                (
                    math.floor(
                        0.99
                        * cfg.total_num_tokens
                        / cfg.sample_packing_eff_est
                        / cfg.sequence_len
                        // cfg.batch_size
                        // int(os.environ.get("WORLD_SIZE", 1))
                    )
                    - 1
                )
                * cfg.num_epochs
            )
            LOG.debug(
                f"total_num_tokens: {cfg.total_num_tokens}, total_num_steps: {total_num_steps}",
                main_process_only=True,
            )
        else:
            sampler = MultipackBatchSampler(
                sampler=RandomSampler(train_dataset),
                batch_size=cfg.micro_batch_size,
                drop_last=True,
                batch_max_len=cfg.micro_batch_size
                * (cfg.max_packed_sequence_len or cfg.sequence_len),
                lengths=(
                    train_dataset.data.column("position_ids")
                    .to_pandas()
                    .apply(lambda x: x[-1] + 1)
                    .values
                ),
            )

            data_loader = DataLoader(
                train_dataset.remove_columns(["length"]),
                batch_sampler=sampler,
            )
            data_loader_len = len(data_loader)
            actual_eff = sampler.efficiency()
            LOG.debug(f"data_loader_len: {data_loader_len}", main_process_only=True)
            # FIXME: is there a bug here somewhere? the total num steps depends
            # on the agreed on value for sample_packing_eff_est
            total_num_steps = int(
                math.floor(
                    data_loader_len
                    * cfg.num_epochs
                    / int(os.environ.get("WORLD_SIZE", 1))
                )
            )

            def calc_sample_packing_eff_est(estimates: List[float]):
                LOG.info(f"sample_packing_eff_est across ranks: {repr(estimates)}")
                return max(estimates)

            sample_packing_actual_eff_all = reduce_and_broadcast(
                lambda: actual_eff,
                calc_sample_packing_eff_est,
            )
            sample_packing_eff_est = (
                math.ceil(sample_packing_actual_eff_all * 100.0) / 100.0
            )
            if update:
                cfg.sample_packing_eff_est = sample_packing_eff_est
            LOG.debug(
                f"sample_packing_eff_est: {cfg.sample_packing_eff_est}",
                main_process_only=True,
            )
    else:
        total_num_steps = int(
            math.ceil(
                len(train_dataset)
                * cfg.num_epochs
                / int(os.environ.get("WORLD_SIZE", 1))
                / cfg.batch_size
            )
        )
    LOG.debug(f"total_num_steps: {total_num_steps}", main_process_only=True)
    return total_num_steps


def setup_fsdp_envs(cfg):
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    if cfg.fsdp_config.fsdp_offload_params:
        os.environ["FSDP_OFFLOAD_PARAMS"] = "true"
    if cfg.fsdp_config.fsdp_sync_module_states:
        os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    if cfg.fsdp_config.fsdp_state_dict_type:
        os.environ["FSDP_STATE_DICT_TYPE"] = cfg.fsdp_config.fsdp_state_dict_type
    if cfg.fsdp_config.fsdp_transformer_layer_cls_to_wrap:
        os.environ[
            "FSDP_TRANSFORMER_CLS_TO_WRAP"
        ] = cfg.fsdp_config.fsdp_transformer_layer_cls_to_wrap


def setup_trainer(cfg, train_dataset, eval_dataset, model, tokenizer, total_num_steps):
    if cfg.fsdp:
        setup_fsdp_envs(cfg)
    elif cfg.deepspeed:
        os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"

    trainer_builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
    trainer_builder.train_dataset = train_dataset
    trainer_builder.eval_dataset = eval_dataset

    return trainer_builder.build(total_num_steps)
