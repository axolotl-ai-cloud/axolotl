"""
module for customized trainers
"""

from __future__ import annotations

# pylint: disable=too-many-lines
import logging
import os
from collections import defaultdict
from functools import wraps
from typing import Dict, Literal, Optional

import torch
from datasets import Dataset
from peft.optimizers import create_loraplus_optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, seed_worker
from transformers.utils import is_sagemaker_mp_enabled
from trl import CPOTrainer, KTOTrainer, ORPOTrainer, PRMTrainer, RewardTrainer
from trl.trainer.utils import pad_to_length

from axolotl.monkeypatch.relora import ReLoRAScheduler
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.schedulers import (
    get_cosine_schedule_with_min_lr,
    get_cosine_schedule_with_quadratic_warmup,
    get_cosine_schedule_with_warmup_decay_constant,
)

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

LOG = logging.getLogger("axolotl.core.trainer_builder")


def _sanitize_kwargs_for_tagging(tag_names, kwargs=None):
    if isinstance(tag_names, str):
        tag_names = [tag_names]

    if kwargs is not None:
        if "tags" not in kwargs:
            kwargs["tags"] = tag_names
        elif "tags" in kwargs and isinstance(kwargs["tags"], list):
            kwargs["tags"].extend(tag_names)
        elif "tags" in kwargs and isinstance(kwargs["tags"], str):
            tag_names.append(kwargs["tags"])
            kwargs["tags"] = tag_names

    return kwargs


def _sanitize_kwargs_for_ds_tagging(dataset_tags, kwargs=None):
    if isinstance(dataset_tags, str):
        dataset_tags = [dataset_tags]

    if (dataset_tags is not None) and (kwargs is not None):
        if "dataset_tags" not in kwargs:
            kwargs["dataset_tags"] = dataset_tags
        elif "dataset_tags" in kwargs and isinstance(kwargs["dataset_tags"], list):
            kwargs["dataset_tags"].extend(dataset_tags)
        elif "dataset_tags" in kwargs and isinstance(kwargs["dataset_tags"], str):
            dataset_tags.append(kwargs["dataset_tags"])
            kwargs["dataset_tags"] = dataset_tags

    return kwargs


class SchedulerMixin(Trainer):
    """
    Mixin class for scheduler setup in CausalTrainer.
    """

    args = None  # type: "AxolotlTrainingArguments"  # type: ignore[name-defined]

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
            optimizer (torch.optim.Optimizer): The training optimizer
        """
        use_cosine_quadratic = (
            self.args.lr_scheduler_type == "cosine"
            and self.args.lr_quadratic_warmup is True
        )

        use_cosine_min_lr = (
            self.args.lr_scheduler_type == "cosine"
            and self.args.cosine_min_lr_ratio is not None
        )

        # fmt: off
        if self.lr_scheduler is None:  # type: ignore  # pylint: disable=access-member-before-definition
            # fmt: on
            if self.args.alternate_lr_scheduler_type == "one_cycle":
                num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
                pct_start = num_warmup_steps / num_training_steps
                extra_lr_kwargs = {}
                if "pct_start" not in self.args.lr_scheduler_kwargs:
                    extra_lr_kwargs["pct_start"] = pct_start
                if "anneal_strategy" not in self.args.lr_scheduler_kwargs:
                    extra_lr_kwargs["anneal_strategy"] = "cos"

                self.lr_scheduler = OneCycleLR(
                    optimizer,
                    max_lr=self.args.learning_rate,
                    total_steps=num_training_steps,
                    **extra_lr_kwargs,
                    **self.args.lr_scheduler_kwargs,
                )
            elif use_cosine_quadratic:
                if use_cosine_min_lr:
                    LOG.warning("Both cosine quadratic warmup and min lr detected. Using quadratic warmup.")

                self.lr_scheduler = get_cosine_schedule_with_quadratic_warmup(  # pylint: disable=attribute-defined-outside-init
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            elif self.args.cosine_min_lr_ratio and self.args.cosine_constant_lr_ratio and use_cosine_min_lr:
                assert 0 <= self.args.cosine_min_lr_ratio <= 1.0, "cosine_min_lr_ratio must be between 0.0 and 1.0"
                assert 0 <= self.args.cosine_constant_lr_ratio <= 1.0, "cosine_constant_lr_ratio must be between 0.0 and 1.0"
                self.lr_scheduler = get_cosine_schedule_with_warmup_decay_constant(  # pylint: disable=attribute-defined-outside-init
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=self.args.cosine_min_lr_ratio,
                    constant_lr_ratio=self.args.cosine_constant_lr_ratio,
                )
            elif self.args.cosine_min_lr_ratio and use_cosine_min_lr:
                assert 0 <= self.args.cosine_min_lr_ratio <= 1.0, "cosine_min_lr_ratio must be between 0.0 and 1.0"
                self.lr_scheduler = get_cosine_schedule_with_min_lr(  # pylint: disable=attribute-defined-outside-init
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=self.args.cosine_min_lr_ratio,
                )
            else:
                return super().create_scheduler(num_training_steps, optimizer=optimizer)
        else:
            if use_cosine_quadratic:
                LOG.warning("axolotl's cosine scheduler with quadratic warmup not used (e.g., because of deepspeed).")

            if use_cosine_min_lr:
                LOG.warning("axolotl's cosine scheduler with min lr not used (e.g., because of deepspeed).")

        return self.lr_scheduler


class AxolotlTrainer(SchedulerMixin, Trainer):
    """
    Extend the base Trainer for axolotl helpers
    """

    args = None  # type: "AxolotlTrainingArguments"  # type: ignore[name-defined]
    tag_names = ["axolotl"]

    def __init__(
        self,
        *_args,
        bench_data_collator=None,
        eval_data_collator=None,
        dataset_tags=None,
        **kwargs,
    ):
        self.bench_data_collator = bench_data_collator
        self.eval_data_collator = eval_data_collator
        self.dataset_tags = dataset_tags
        self._signature_columns = None  # workaround for pylint
        super().__init__(*_args, **kwargs)
        self.train_data_collator = self.data_collator
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        if self.args.orpo_alpha:
            self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    def _wrap_model(self, model, training=True, dataloader=None):
        if self.args.torch_compile:
            torch._dynamo.config.accumulated_cache_size_limit = (  # pylint: disable=protected-access
                256
            )
            model = torch.compile(
                model,
                backend=self.args.torch_compile_backend,
                mode=self.args.torch_compile_mode,
            )
        return super()._wrap_model(model, training=training, dataloader=dataloader)

    def create_optimizer_grouped_parameters(self, opt_model, optimizer_kwargs):
        decay_parameters = self.get_decay_parameter_names(opt_model)
        params = {
            "to_weight_decay": {},  # LayerNorm and bias
            "embeddings": {},  # lm_head, embed_tokens,
            "no_weight_decay": {},
        }
        lr_groups_lookup = {}
        lr_groups_learning_rates = {}
        if self.args.lr_groups:
            for lr_group in self.args.lr_groups:
                group_name = lr_group["name"]
                group_modules = lr_group["modules"]
                for module in group_modules:
                    lr_groups_lookup[module] = group_name
                lr_groups_learning_rates[group_name] = lr_group["lr"]
                params[f"to_weight_decay_{group_name}"] = {}

        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("modules_to_save.default.weight") or any(
                embed_name in name for embed_name in ["embed_tokens", "lm_head"]
            ):
                params["embeddings"][name] = param
            elif name in decay_parameters:
                lr_group_modules = [
                    group_modules
                    for group_modules in lr_groups_lookup
                    if group_modules in name
                ]
                if lr_groups_lookup and any(lr_group_modules):
                    lr_group_module = lr_group_modules[0]
                    group_name = lr_groups_lookup[lr_group_module]
                    params[f"to_weight_decay_{group_name}"][name] = param
                else:
                    params["to_weight_decay"][name] = param
            else:
                params["no_weight_decay"][name] = param
        optimizer_grouped_parameters = []
        if params["to_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": list(params["to_weight_decay"].values()),
                    "weight_decay": self.args.weight_decay,
                    "lr": optimizer_kwargs["lr"],
                }
            )
        if params["embeddings"]:
            lr = optimizer_kwargs["lr"]  # pylint: disable=invalid-name
            if self.args.embedding_lr_scale:
                lr *= self.args.embedding_lr_scale  # pylint: disable=invalid-name
            elif self.args.embedding_lr:
                lr = self.args.embedding_lr  # pylint: disable=invalid-name
            optimizer_grouped_parameters.append(
                {
                    "params": list(params["embeddings"].values()),
                    "weight_decay": 0.0,
                    "lr": lr,
                }
            )
        if params["no_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": list(params["no_weight_decay"].values()),
                    "weight_decay": 0.0,
                    "lr": optimizer_kwargs["lr"],
                }
            )
        for group_name, group_lr in lr_groups_learning_rates.items():
            if params[f"to_weight_decay_{group_name}"]:
                optimizer_grouped_parameters.append(
                    {
                        "params": list(
                            params[f"to_weight_decay_{group_name}"].values()
                        ),
                        "weight_decay": self.args.weight_decay,
                        "lr": group_lr,
                    }
                )

        return optimizer_grouped_parameters

    def create_optimizer(self):
        if (
            self.args.loraplus_lr_ratio is None
            and self.args.embedding_lr_scale is None
            and self.args.embedding_lr is None
            and self.args.lr_groups is None
            and self.args.alternate_optimizer
            not in [
                "optimi_adamw",
                "ao_adamw_8bit",
                "ao_adamw_4bit",
                "ao_adamw_fp8",
                "adopt_adamw",
            ]
        ):
            return super().create_optimizer()

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:  # pylint: disable=access-member-before-definition
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args,
                opt_model,
            )
            optimizer_grouped_parameters = self.create_optimizer_grouped_parameters(
                opt_model, optimizer_kwargs
            )

            if self.args.loraplus_lr_ratio is not None:
                loraplus_lr_ratio = getattr(self.args, "loraplus_lr_ratio", None)
                loraplus_lr_embedding = getattr(
                    self.args, "loraplus_lr_embedding", 1e-6
                )
                self.optimizer = create_loraplus_optimizer(  # pylint: disable=attribute-defined-outside-init
                    opt_model,
                    optimizer_cls,
                    loraplus_lr_ratio=loraplus_lr_ratio,
                    loraplus_lr_embedding=loraplus_lr_embedding,
                    **optimizer_kwargs,
                )
            elif (
                self.args.embedding_lr_scale is not None
                or self.args.embedding_lr is not None
                or self.args.lr_groups is not None
            ):
                self.optimizer = (  # pylint: disable=attribute-defined-outside-init
                    optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                )
            elif self.args.alternate_optimizer == "optimi_adamw":
                from optimi import AdamW

                self.optimizer = (  # pylint: disable=attribute-defined-outside-init
                    AdamW(
                        optimizer_grouped_parameters, foreach=False, **optimizer_kwargs
                    )
                )
            elif self.args.alternate_optimizer == "ao_adamw_4bit":
                from torchao.prototype.low_bit_optim import AdamW4bit

                self.optimizer = (  # pylint: disable=attribute-defined-outside-init
                    AdamW4bit(optimizer_grouped_parameters, **optimizer_kwargs)
                )
            elif self.args.alternate_optimizer == "ao_adamw_8bit":
                from torchao.prototype.low_bit_optim import AdamW8bit

                self.optimizer = (  # pylint: disable=attribute-defined-outside-init
                    AdamW8bit(optimizer_grouped_parameters, **optimizer_kwargs)
                )
            elif self.args.alternate_optimizer == "ao_adamw_fp8":
                from torchao.prototype.low_bit_optim import AdamWFp8

                self.optimizer = (  # pylint: disable=attribute-defined-outside-init
                    AdamWFp8(optimizer_grouped_parameters, **optimizer_kwargs)
                )
            elif self.args.alternate_optimizer == "adopt_adamw":
                from axolotl.utils.optimizers.adopt import ADOPT

                self.optimizer = (  # pylint: disable=attribute-defined-outside-init
                    ADOPT(
                        optimizer_grouped_parameters,
                        decouple=True,
                        **optimizer_kwargs,
                    )
                )

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(  # pylint: disable=attribute-defined-outside-init
                self.optimizer
            )

        return self.optimizer

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.args.sample_packing and not self.args.pretraining:
            if self.args.multipack_real_batches:
                batch_size = self.args.per_device_train_batch_size
                batch_max_len = self.args.max_seq_length
            else:
                batch_size = 1
                train_batch_size = (
                    self.state.train_batch_size or self.args.per_device_train_batch_size
                )
                batch_max_len = train_batch_size * self.args.max_seq_length

            if self.args.curriculum_sampling:
                sampler = SequentialSampler(self.train_dataset)
            else:
                sampler = RandomSampler(self.train_dataset)

            return MultipackBatchSampler(
                sampler,
                lengths=get_dataset_lengths(self.train_dataset),
                packing_efficiency_estimate=self.args.sample_packing_efficiency,
                batch_max_len=batch_max_len,
                batch_size=batch_size,
                group_size=self.args.sample_packing_group_size,
                bin_size=self.args.sample_packing_bin_size,
                drop_last=True,
            )
        if self.args.curriculum_sampling:
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()

    def _get_eval_sampler(
        self, eval_dataset: Dataset
    ) -> Optional[torch.utils.data.Sampler]:
        if self.args.sample_packing and self.args.eval_sample_packing is not False:
            if self.args.multipack_real_batches:
                batch_size = self.args.per_device_eval_batch_size
                batch_max_len = self.args.max_seq_length
            else:
                batch_size = 1
                batch_max_len = (
                    self.args.per_device_eval_batch_size * self.args.max_seq_length
                )
            return MultipackBatchSampler(
                SequentialSampler(eval_dataset),
                lengths=get_dataset_lengths(self.eval_dataset),
                packing_efficiency_estimate=self.args.sample_packing_efficiency,
                batch_max_len=batch_max_len,
                batch_size=batch_size,
                group_size=self.args.sample_packing_group_size,
                bin_size=self.args.sample_packing_bin_size,
                drop_last=True,
            )
        return super()._get_eval_sampler(eval_dataset)

    def get_train_dataloader(self) -> DataLoader:
        if self.args.sample_packing and not self.args.pretraining:
            train_dataset = self.train_dataset
            if "length" in train_dataset.features.keys():
                train_dataset = train_dataset.remove_columns(["length"])
            data_collator = self.data_collator
            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            }
            if self.args.dataloader_prefetch_factor:
                dataloader_params[
                    "prefetch_factor"
                ] = self.args.dataloader_prefetch_factor

            sampler = self._get_train_sampler()
            if isinstance(sampler, BatchSampler):
                dataloader_params["batch_sampler"] = sampler
                del dataloader_params["batch_size"]
            else:
                dataloader_params["sampler"] = sampler
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

            self.accelerator.even_batches = False
            return self.accelerator.prepare_data_loader(
                DataLoader(train_dataset, **dataloader_params)
            )
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if self.args.sample_packing and self.args.eval_sample_packing is False:
            self.data_collator = (  # pylint: disable=attribute-defined-outside-init
                self.eval_data_collator
            )
            if eval_dataset:
                eval_dataset = eval_dataset.remove_columns(["length"])
            dataloader = super().get_eval_dataloader(eval_dataset)
            self.data_collator = (  # pylint: disable=attribute-defined-outside-init
                self.train_data_collator
            )
            return dataloader

        if self.args.sample_packing and self.args.eval_sample_packing is not False:
            eval_dataset = (
                eval_dataset if eval_dataset is not None else self.eval_dataset
            )

            eval_sampler = self._get_eval_sampler(eval_dataset)
            eval_dataset = eval_dataset.remove_columns(["length"])
            data_collator = self.data_collator
            dataloader_params = {
                "batch_size": self.args.eval_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            }
            if self.args.dataloader_prefetch_factor:
                dataloader_params[
                    "prefetch_factor"
                ] = self.args.dataloader_prefetch_factor

            if isinstance(eval_sampler, BatchSampler):
                dataloader_params["batch_sampler"] = eval_sampler
                del dataloader_params["batch_size"]
            else:
                dataloader_params["sampler"] = eval_sampler
                dataloader_params["drop_last"] = self.args.dataloader_drop_last

            self.accelerator.even_batches = False
            return self.accelerator.prepare_data_loader(
                DataLoader(eval_dataset, **dataloader_params)
            )

        return super().get_eval_dataloader(eval_dataset)

    def _get_bench_sampler(
        self, bench_dataset: Dataset
    ) -> Optional[torch.utils.data.Sampler]:
        if self.args.world_size <= 1:
            return SequentialSampler(bench_dataset)
        return None

    def get_bench_dataloader(
        self,
        bench_dataset: Dataset,
    ) -> DataLoader:
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.bench_data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if self.args.dataloader_prefetch_factor:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        if not isinstance(bench_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_bench_sampler(bench_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return DataLoader(bench_dataset, **dataloader_params)
        # return self.accelerator.prepare(DataLoader(bench_dataset, **dataloader_params))

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # use one's weighted cross entropy loss calc
        # if self.args.sample_packing:
        #     labels = inputs.pop("labels")
        #     outputs = model(**inputs)
        #     loss = trainer_weighted_loss(outputs, labels, shift_labels=True)
        #     return (loss, outputs) if return_outputs else loss
        if self.args.orpo_alpha:
            return self.orpo_compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    @staticmethod
    def orpo_concatenate_inputs(inputs, label_pad_token=-100, pad_token=0, device=None):
        concatenated_batch = {}

        max_length = max(
            inputs["input_ids"].shape[1], inputs["rejected_input_ids"].shape[1]
        )
        # Concatenate positive and negative inputs
        concatenated_batch["input_ids"] = pad_to_length(
            inputs["input_ids"], max_length, pad_token
        )
        concatenated_batch["rejected_input_ids"] = pad_to_length(
            inputs["rejected_input_ids"], max_length, pad_token
        )
        concatenated_batch["labels"] = pad_to_length(
            inputs["labels"], max_length, label_pad_token
        )
        concatenated_batch["rejected_labels"] = pad_to_length(
            inputs["rejected_labels"], max_length, label_pad_token
        )
        concatenated_batch["attention_mask"] = pad_to_length(
            inputs["attention_mask"], max_length, 0
        )
        concatenated_batch["rejected_attention_mask"] = pad_to_length(
            inputs["rejected_attention_mask"], max_length, 0
        )
        concatenated_batch["prompt_attention_mask"] = pad_to_length(
            inputs["prompt_attention_mask"], max_length, 0
        ).to(device=device)

        input_ids = torch.cat(
            [concatenated_batch["input_ids"], concatenated_batch["rejected_input_ids"]],
            dim=0,
        ).to(device=device)
        attention_mask = torch.cat(
            [
                concatenated_batch["attention_mask"],
                concatenated_batch["rejected_attention_mask"],
            ],
            dim=0,
        ).to(device=device)
        labels = torch.cat(
            [concatenated_batch["labels"], concatenated_batch["rejected_labels"]], dim=0
        ).to(device=device)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "prompt_attention_mask": concatenated_batch["prompt_attention_mask"],
        }

    def orpo_compute_custom_loss(self, logits, labels):
        logits = logits.contiguous()
        loss = 0.0

        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss = self.loss_fct(shift_logits.transpose(2, 1), shift_labels).mean(
                dim=-1
            )

        return loss

    def orpo_compute_logps(
        self, prompt_attention_mask, chosen_inputs, chosen_attention_mask, logits
    ):
        # Get the shape of chosen_attention_mask[:, :-1]
        chosen_shape = chosen_attention_mask[:, :-1].shape

        # Calculate the padding size
        pad_length = chosen_shape[1] - (prompt_attention_mask.shape[1] - 1)

        # Pad prompt_attention_mask with zeros to match the desired shape
        prompt_attention_mask_padded = torch.nn.functional.pad(
            prompt_attention_mask[:, 1:], (0, pad_length), mode="constant", value=0
        )

        # Perform the subtraction operation
        mask = chosen_attention_mask[:, :-1] > prompt_attention_mask_padded

        per_token_logps = torch.gather(
            logits[:, :-1, :].log_softmax(-1),
            dim=2,
            index=(mask * chosen_inputs[:, 1:]).unsqueeze(2),
        ).squeeze(2)
        return torch.mul(per_token_logps, mask).sum(dim=1) / mask.sum(dim=1)

    def orpo_compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,  # pylint: disable=unused-argument
    ):
        concat_inputs = AxolotlTrainer.orpo_concatenate_inputs(
            inputs,
            label_pad_token=-100,
            pad_token=self.tokenizer.pad_token_id,
            device=self.accelerator.device,
        )

        # Perform a single forward pass
        outputs = model(
            **{
                "input_ids": concat_inputs["input_ids"],
                "attention_mask": concat_inputs["attention_mask"],
                "labels": concat_inputs["labels"],
            },
            output_hidden_states=True,
        )

        # Split the outputs for positive and negative examples
        outputs_pos, outputs_neg = outputs.logits.chunk(2)

        # Calculate NLL loss
        pos_loss = self.orpo_compute_custom_loss(
            logits=outputs_pos, labels=concat_inputs["input_ids"].chunk(2)[0]
        )

        # Calculate Log Probability
        pos_prob = self.orpo_compute_logps(
            prompt_attention_mask=concat_inputs["prompt_attention_mask"],
            chosen_inputs=concat_inputs["input_ids"].chunk(2)[0],
            chosen_attention_mask=concat_inputs["attention_mask"].chunk(2)[0],
            logits=outputs_pos,
        )
        neg_prob = self.orpo_compute_logps(
            prompt_attention_mask=concat_inputs["prompt_attention_mask"],
            chosen_inputs=concat_inputs["input_ids"].chunk(2)[1],
            chosen_attention_mask=concat_inputs["attention_mask"].chunk(2)[1],
            logits=outputs_neg,
        )

        # Calculate log odds
        log_odds = (pos_prob - neg_prob) - (
            torch.log(1 - torch.exp(pos_prob)) - torch.log(1 - torch.exp(neg_prob))
        )
        sig_ratio = torch.nn.functional.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)

        # Calculate the Final Loss
        loss = torch.mean(pos_loss - self.args.orpo_alpha * ratio).to(
            dtype=torch.bfloat16
        )

        metrics = {}
        metrics["chosen_geometric_mean"] = torch.mean(pos_prob).cpu().item()
        metrics["rejected_geometric_mean"] = torch.mean(neg_prob).cpu().item()
        metrics["log_odds_ratio"] = torch.mean(ratio).cpu().item()
        metrics["log_odds"] = torch.mean(log_odds).cpu().item()
        self.store_metrics(metrics, train_eval="train")

        return (loss, outputs_pos) if return_outputs else loss

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, *args, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tags when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = _sanitize_kwargs_for_ds_tagging(
            dataset_tags=self.dataset_tags, kwargs=kwargs
        )
        kwargs = _sanitize_kwargs_for_tagging(tag_names=self.tag_names, kwargs=kwargs)

        return super().push_to_hub(*args, **kwargs)

    @wraps(Trainer.create_accelerator_and_postprocess)
    def create_accelerator_and_postprocess(self):
        res = super().create_accelerator_and_postprocess()

        if self.is_fsdp_enabled:
            if (
                "limit_all_gathers" in self.args.fsdp_config
                and self.args.fsdp_config["limit_all_gathers"]
            ):
                self.accelerator.state.fsdp_plugin.limit_all_gathers = True

        return res

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        return super().log(logs, start_time)

    def store_metrics(
        self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def _save_checkpoint(self, model, trial, **kwargs):
        # make sure the checkpoint dir exists, since trainer is flakey
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        return super()._save_checkpoint(model, trial, **kwargs)


class AxolotlMambaTrainer(AxolotlTrainer):
    """
    Mamba specific trainer to handle loss calculation
    """

    tag_names = ["axolotl", "mamba"]

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,  # pylint: disable=unused-argument
        num_items_in_batch=None,  # pylint: disable=unused-argument
    ):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
        )

        return lm_loss


class ReLoRATrainer(AxolotlTrainer):
    """
    Trainer subclass that uses the OneCycleLR scheduler
    """

    tag_names = ["axolotl", "relora"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = None

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        optimizer = self.optimizer if optimizer is None else optimizer
        lr_scheduler = super().create_scheduler(num_training_steps, optimizer)

        if self.args.relora_steps:
            warmup_steps = (
                self.args.relora_warmup_steps if self.args.relora_warmup_steps else 10
            )
            anneal_steps = (
                self.args.relora_anneal_steps if self.args.relora_anneal_steps else 1
            )
            self.lr_scheduler = ReLoRAScheduler(
                optimizer,
                lr_scheduler,
                self.args.relora_steps,
                anneal_steps,
                warmup_steps,
            )
        else:
            self.lr_scheduler = lr_scheduler

        return self.lr_scheduler


class AxolotlORPOTrainer(SchedulerMixin, ORPOTrainer):
    """
    Extend the base ORPOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "orpo"]


class AxolotlKTOTrainer(SchedulerMixin, KTOTrainer):
    """
    Extend the base KTOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "kto"]


class AxolotlCPOTrainer(SchedulerMixin, CPOTrainer):
    """
    Extend the base CPOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "cpo"]


class AxolotlRewardTrainer(SchedulerMixin, RewardTrainer):
    """
    Extend the base RewardTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "reward"]


class AxolotlPRMTrainer(SchedulerMixin, PRMTrainer):
    """
    Extend the base trl.PRMTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "prm"]
