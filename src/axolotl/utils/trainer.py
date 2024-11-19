"""Module containing the Trainer class and related functions"""
import json
import math
import os
import random
from contextlib import contextmanager
from functools import partial
from typing import List, Optional

import numpy as np
import torch
import torch.cuda
from accelerate.logging import get_logger
from datasets import disable_caching, enable_caching
from torch.utils.data import DataLoader, RandomSampler
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.core.trainer_builder import HFCausalTrainerBuilder, HFRLTrainerBuilder
from axolotl.utils.distributed import reduce_and_broadcast
from axolotl.utils.environment import check_cuda_p2p_ib_support
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths

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


@contextmanager
def disable_datasets_caching():
    try:
        disable_caching()
        yield
    finally:
        enable_caching()


def add_position_ids(sample):
    sample_len = len(sample["input_ids"])
    sample["position_ids"] = torch.arange(len(sample["input_ids"]))
    sample["length"] = sample_len
    return sample


def add_pose_position_ids(
    sample,
    max_context_len=32768,
    split_on_token_ids: Optional[List[int]] = None,
    chunks: int = 2,
):
    """
    use the PoSE technique to extend the context length by randomly skipping
    positions in the context. We only want to skip right before tokens in
    the split_on_token_ids list. We should attempt to randomly distribute
    the skips, but we don't need the final position_ids to be the full
    context_len. There may be multiple turns in the context, so we want to
    make sure we take into account the maximum possible number of skips
    remaining in each sample.
    """

    input_ids = sample["input_ids"]
    sample_len = len(input_ids)
    max_skips = max_context_len - sample_len

    if split_on_token_ids is None:
        split_on_token_ids = []

    if split_on_token_ids:
        split_indices = [
            i for i, token_id in enumerate(input_ids) if token_id in split_on_token_ids
        ]
    else:
        chunk_len = sample_len // chunks
        split_indices = [i * chunk_len for i in range(1, chunks)]
    split_indices.append(len(input_ids))  # make sure we go to the end of the sample
    if split_indices[0] < 2:
        # drop the first split index if it's too close to the beginning
        split_indices = split_indices[1:]

    position_ids = []
    prev_index = 0
    total_skips = 0

    for split_index in split_indices:
        num_skips = (
            random.randint(0, max_skips)  # nosec B311
            if prev_index != 0 and max_skips
            else 0
        )
        max_skips -= num_skips
        total_skips += num_skips

        segment_position_ids = list(
            range(prev_index + total_skips, split_index + total_skips)
        )

        position_ids.extend(segment_position_ids)
        prev_index = split_index

    sample["sequence_len"] = position_ids[-1]
    position_ids = torch.tensor(position_ids)

    sample["position_ids"] = position_ids
    sample["length"] = len(position_ids)
    assert len(position_ids) == len(input_ids)

    return sample


def add_length(sample):
    sample["length"] = len(sample["input_ids"])
    return sample


def drop_long_seq(sample, sequence_len=2048, min_sequence_len=2):
    return (
        len(sample["input_ids"]) <= sequence_len
        and len(sample["input_ids"]) >= min_sequence_len
    )


def process_datasets_for_packing(cfg, train_dataset, eval_dataset):
    drop_long = partial(
        drop_long_seq,
        sequence_len=cfg.sequence_len,
        min_sequence_len=cfg.min_sample_len or 2,
    )

    min_input_len = np.min(get_dataset_lengths(train_dataset))
    LOG.debug(f"min_input_len: {min_input_len}", main_process_only=True)
    max_input_len = np.max(get_dataset_lengths(train_dataset))
    LOG.debug(f"max_input_len: {max_input_len}", main_process_only=True)

    if cfg.model_config_type == "mamba":
        LOG.info("dropping attention_mask column")
        train_dataset = train_dataset.remove_columns("attention_mask")
        if eval_dataset:
            eval_dataset = eval_dataset.remove_columns("attention_mask")

    if cfg.model_config_type == "falcon":
        LOG.info("dropping token_type_ids column if it exists")
        if "token_type_ids" in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns("token_type_ids")
        if eval_dataset and "token_type_ids" in eval_dataset.column_names:
            eval_dataset = eval_dataset.remove_columns("token_type_ids")

    prior_len = len(train_dataset)
    train_dataset = train_dataset.filter(
        drop_long,
        num_proc=cfg.dataset_processes,
        load_from_cache_file=not cfg.is_preprocess,
        desc="Dropping Long Sequences",
    )
    dropped = prior_len - len(train_dataset)
    if dropped:
        LOG.warning(f"Dropped {dropped} long samples from train dataset")

    if eval_dataset:
        prior_len = len(eval_dataset)
        eval_dataset = eval_dataset.filter(
            drop_long,
            num_proc=cfg.dataset_processes,
            load_from_cache_file=not cfg.is_preprocess,
            desc="Dropping Long Sequences",
        )
        dropped = prior_len - len(eval_dataset)
        if dropped:
            LOG.warning(f"Dropped {dropped} long samples from eval dataset")

    # drop samples with where the number of elements with labels not equal to -100 is zero
    def drop_no_trainable_tokens(sample):
        return np.sum(np.array(sample["labels"]) != -100) > 0

    prior_len = len(train_dataset)
    train_dataset = train_dataset.filter(
        drop_no_trainable_tokens,
        num_proc=cfg.dataset_processes,
        load_from_cache_file=not cfg.is_preprocess,
        desc="Drop Samples with Zero Trainable Tokens",
    )
    dropped = prior_len - len(train_dataset)
    if dropped:
        LOG.warning(
            f"Dropped {dropped} samples with no trainable tokens from train dataset"
        )

    if eval_dataset:
        prior_len = len(eval_dataset)
        eval_dataset = eval_dataset.filter(
            drop_no_trainable_tokens,
            num_proc=cfg.dataset_processes,
            load_from_cache_file=not cfg.is_preprocess,
            desc="Drop Samples with Zero Trainable Tokens",
        )
        dropped = prior_len - len(eval_dataset)
        if dropped:
            LOG.warning(
                f"Dropped {dropped} samples with no trainable tokens from eval dataset"
            )

    if cfg.group_by_length:
        train_dataset = train_dataset.map(
            add_length,
            num_proc=cfg.dataset_processes,
            load_from_cache_file=not cfg.is_preprocess,
            desc="Group By Length",
        )

    if cfg.use_pose:
        pose_kwargs = {}
        if cfg.pose_num_chunks is not None:
            pose_kwargs["chunks"] = cfg.pose_num_chunks
        pose_fn = partial(
            add_pose_position_ids,
            max_context_len=cfg.pose_max_context_len,
            split_on_token_ids=cfg.pose_split_on_token_ids,
            **pose_kwargs,
        )
        train_dataset = train_dataset.map(
            pose_fn,
            num_proc=cfg.dataset_processes,
            load_from_cache_file=not cfg.is_preprocess,
            desc="Add position_id column (PoSE)",
        )
        train_dataset = train_dataset.sort("sequence_len")
        if cfg.eval_sample_packing is not False:
            if eval_dataset:
                eval_dataset = eval_dataset.map(
                    pose_fn,
                    num_proc=cfg.dataset_processes,
                    load_from_cache_file=not cfg.is_preprocess,
                    desc="Add position_id column (PoSE)",
                )
    elif cfg.sample_packing:
        train_dataset = train_dataset.map(
            add_position_ids,
            num_proc=cfg.dataset_processes,
            load_from_cache_file=not cfg.is_preprocess,
            desc="Add position_id column (Sample Packing)",
        )
        if cfg.eval_sample_packing is not False:
            if eval_dataset:
                eval_dataset = eval_dataset.map(
                    add_position_ids,
                    num_proc=cfg.dataset_processes,
                    load_from_cache_file=not cfg.is_preprocess,
                    desc="Add position_id column (Sample Packing)",
                )

    return train_dataset, eval_dataset


def process_pretraining_datasets_for_packing(
    train_dataset, sequence_len, skip_position_ids=True
):
    drop_long = partial(drop_long_seq, sequence_len=sequence_len)

    train_dataset = train_dataset.filter(
        drop_long,
        desc="Dropping Long Sequences",
    )
    if skip_position_ids:
        train_dataset = train_dataset.map(
            add_position_ids,
            desc="Add position_id column (Pretraining Sample Packing)",
        )

    return train_dataset


def calculate_total_num_steps(cfg, train_dataset, update=True):
    if (
        not cfg.total_num_tokens
        and not cfg.skip_prepare_dataset
        and not cfg.reward_model
    ):
        total_num_tokens = np.sum(
            train_dataset.data.column("input_ids")
            .to_pandas()
            .apply(lambda x: len(x))  # pylint: disable=unnecessary-lambda
            .values
        )
        LOG.debug(f"total_num_tokens: {total_num_tokens:_}", main_process_only=True)
        if update:
            cfg.total_num_tokens = total_num_tokens

    skip_estimates = cfg.model_config_type == "mamba"

    if (
        not skip_estimates
        and not cfg.total_supervised_tokens
        and not cfg.skip_prepare_dataset
        and not cfg.reward_model
    ):
        total_supervised_tokens = (
            train_dataset.data.column("labels")
            .to_pandas()
            .apply(lambda x: np.sum(np.array(x) != -100))
            .sum()
        )
        LOG.debug(
            f"`total_supervised_tokens: {total_supervised_tokens:_}`",
            main_process_only=True,
        )
        if update:
            cfg.total_supervised_tokens = total_supervised_tokens

    if not skip_estimates and cfg.sample_packing:
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
                    )
                    - 1
                )
                * cfg.num_epochs
            )
            LOG.debug(
                f"total_num_tokens: {cfg.total_num_tokens:_}, total_num_steps: {total_num_steps:_}",
                main_process_only=True,
            )
        else:
            if cfg.flash_attention and not cfg.multipack_real_batches:
                sampler_batch_size = 1
                batch_max_len = cfg.micro_batch_size * cfg.sequence_len
            else:
                sampler_batch_size = cfg.micro_batch_size
                batch_max_len = cfg.sequence_len
            sampler = MultipackBatchSampler(
                sampler=RandomSampler(train_dataset),
                lengths=get_dataset_lengths(train_dataset),
                batch_size=sampler_batch_size,
                batch_max_len=batch_max_len,
                group_size=cfg.sample_packing_group_size,
                bin_size=cfg.sample_packing_bin_size,
                drop_last=True,
            )

            data_loader = DataLoader(
                train_dataset.remove_columns(["length"]),
                batch_sampler=sampler,
            )
            data_loader_len = len(data_loader) * cfg.micro_batch_size // cfg.batch_size
            LOG.debug(f"data_loader_len: {data_loader_len}", main_process_only=True)
            # FIXME: is there a bug here somewhere? the total num steps depends
            # on the agreed on value for sample_packing_eff_est
            total_num_steps = int(math.floor(data_loader_len * cfg.num_epochs))

            def calc_sample_packing_eff_est(estimates: List[float]):
                LOG.info(f"sample_packing_eff_est across ranks: {repr(estimates)}")
                return max(estimates)

            sample_packing_actual_eff_all = reduce_and_broadcast(
                lambda: sampler.efficiency(),  # pylint: disable=unnecessary-lambda
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
            math.ceil(len(train_dataset) * cfg.num_epochs / cfg.batch_size)
        )
    LOG.debug(f"total_num_steps: {total_num_steps}", main_process_only=True)
    return total_num_steps


def setup_torch_compile_env(cfg):
    if cfg.torch_compile:
        if not cfg.torch_compile_backend:
            os.environ["ACCELERATE_DYNAMO_BACKEND"] = "INDUCTOR"
        else:
            os.environ["ACCELERATE_DYNAMO_BACKEND"] = cfg.torch_compile_backend.upper()


def setup_deepspeed_env(cfg, stage=None):
    from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

    os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
    os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] = cfg.deepspeed
    if stage:
        os.environ["ACCELERATE_DEEPSPEED_ZERO_STAGE"] = str(stage)
        if stage == 3:
            os.environ["ACCELERATE_DEEPSPEED_ZERO3_INIT"] = "true"
    # If we don't assign this, it doesn't actually get set in the accelerate weakref
    _ = HfTrainerDeepSpeedConfig(cfg.deepspeed)


def setup_fsdp_envs(cfg):
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    if cfg.fsdp_config.fsdp_activation_checkpointing:
        os.environ["FSDP_ACTIVATION_CHECKPOINTING"] = "true"
    if cfg.fsdp_config.fsdp_offload_params:
        os.environ["FSDP_OFFLOAD_PARAMS"] = "true"
    if cfg.fsdp_config.fsdp_sync_module_states:
        os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    if cfg.fsdp_config.fsdp_cpu_ram_efficient_loading:
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
    if cfg.fsdp_config.fsdp_use_orig_params:
        os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
    if cfg.fsdp_config.fsdp_state_dict_type:
        os.environ["FSDP_STATE_DICT_TYPE"] = cfg.fsdp_config.fsdp_state_dict_type
    if cfg.fsdp_config.fsdp_auto_wrap_policy:
        os.environ["FSDP_AUTO_WRAP_POLICY"] = cfg.fsdp_config.fsdp_auto_wrap_policy
    if cfg.fsdp_config.fsdp_transformer_layer_cls_to_wrap:
        os.environ[
            "FSDP_TRANSFORMER_CLS_TO_WRAP"
        ] = cfg.fsdp_config.fsdp_transformer_layer_cls_to_wrap


def prepare_optim_env(cfg):
    if not check_cuda_p2p_ib_support():
        if os.getenv("NCCL_P2P_DISABLE") is None:
            os.environ["NCCL_P2P_DISABLE"] = "1"
    if cfg.fsdp:
        setup_fsdp_envs(cfg)
    elif cfg.deepspeed:
        stage = None
        # check if the cfg.deepspeed is a file
        if os.path.isfile(cfg.deepspeed):
            # parse with json
            with open(cfg.deepspeed, "r", encoding="utf-8") as fin:
                deepspeed_config = json.load(fin)
            stage = deepspeed_config.get("zero_optimization", {}).get("stage", None)
        setup_deepspeed_env(cfg, stage=stage)

    setup_torch_compile_env(cfg)

    if (cfg.bf16 == "auto" and is_torch_bf16_gpu_available()) or cfg.bf16 is True:
        os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
    elif cfg.fp16:
        os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"


def prepare_opinionated_env(cfg):
    if cfg.qlora_sharded_model_loading:
        # model loading is forked after the tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_trainer(
    cfg, train_dataset, eval_dataset, model, tokenizer, processor, total_num_steps
):
    if cfg.rl in ("dpo", "ipo", "orpo", "kto", "simpo"):
        trainer_builder = HFRLTrainerBuilder(cfg, model[0], tokenizer, processor)
        trainer_builder.model_ref = model[1]
        trainer_builder.peft_config = model[2]
    else:
        trainer_builder = HFCausalTrainerBuilder(cfg, model[0], tokenizer, processor)

    trainer_builder.train_dataset = train_dataset
    trainer_builder.eval_dataset = eval_dataset

    return trainer_builder.build(total_num_steps)
