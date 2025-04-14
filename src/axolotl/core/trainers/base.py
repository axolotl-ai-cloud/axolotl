"""Module for customized trainers"""

# pylint: disable=too-many-lines

from __future__ import annotations

import logging
import os
from collections import defaultdict
from functools import wraps
from typing import Literal

import datasets
import torch
from datasets import Dataset
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, seed_worker
from trl.trainer.utils import pad_to_length
from typing_extensions import override

from axolotl.core.trainers.mixins import (
    OptimizerMixin,
    RngLoaderMixin,
    SchedulerMixin,
    SequenceParallelMixin,
)
from axolotl.core.trainers.utils import (
    sanitize_kwargs_for_ds_tagging,
    sanitize_kwargs_for_tagging,
)
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths

LOG = logging.getLogger(__name__)


class AxolotlTrainer(
    SchedulerMixin, OptimizerMixin, RngLoaderMixin, SequenceParallelMixin, Trainer
):
    """Extend the base Trainer for axolotl helpers"""

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

        # Initialize sequence parallelism if enabled
        if self.args.sequence_parallel_degree > 1:
            self._setup_sequence_parallel()

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

    def _create_multipack_sampler(
        self, base_sampler: Sampler, dataset: Dataset
    ) -> MultipackBatchSampler:
        """
        Helper method to create a `MultipackBatchSampler` for multipacking sequences
        for training.

        Args:
            base_sampler: Sampler to wrap with `MultipackBatchSampler`.
            dataset: Dataset to sample from.

        Returns:
            Multipack (sample packing) batch sampler.
        """
        if self.args.multipack_real_batches:
            batch_size = self.args.per_device_train_batch_size
            batch_max_len = self.args.max_seq_length
        else:
            batch_size = 1
            train_batch_size = (
                self.state.train_batch_size or self.args.per_device_train_batch_size
            )
            batch_max_len = train_batch_size * self.args.max_seq_length

        return MultipackBatchSampler(
            base_sampler,
            lengths=get_dataset_lengths(dataset),
            packing_efficiency_estimate=self.args.sample_packing_efficiency,
            batch_max_len=batch_max_len,
            batch_size=batch_size,
            sequential=self.args.sample_packing_sequentially,
            drop_last=True,
        )

    def _get_train_sampler(self) -> Sampler | None:
        """
        Helper method to get the sampler for training. Handles cases for sequence
        parallelism, sample packing, and curriculum sampling (sequential).

        Returns:
            If the dataset is non-empty, a sampler is returned, the type of which
                depends on the passed training args.
        """
        use_sample_packing = self.args.sample_packing and not self.args.pretraining

        # Determine the base sampler first
        if self.args.sequence_parallel_degree > 1:
            base_sampler = self._sp_get_train_sampler(self.train_dataset)
        elif self.args.curriculum_sampling:
            base_sampler = SequentialSampler(self.train_dataset)
        elif use_sample_packing:
            base_sampler = RandomSampler(self.train_dataset)
        else:
            # Default to parent class implementation for standard random sampling
            return super()._get_train_sampler()

        # Apply multipack wrapper if needed
        if use_sample_packing:
            return self._create_multipack_sampler(
                base_sampler=base_sampler,
                dataset=self.train_dataset,
            )

        return base_sampler

    def _get_eval_sampler(self, eval_dataset: Dataset | None = None) -> Sampler | None:
        """
        Helper method to get the sampler for evaluation. Handles sequence parallelism
        and sample packing cases.

        Returns:
            If the dataset is non-empty, a sampler is returned, the type of which
                depends on the passed training args.
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        # Multipacking enabled if training is enabled and eval is not explicitly disabled
        use_multipack = (
            self.args.sample_packing and self.args.eval_sample_packing is not False
        )

        # Determine the base sampler
        if self.args.sequence_parallel_degree > 1:
            base_sampler = self._sp_get_eval_sampler(eval_dataset)
        elif use_multipack:
            base_sampler = SequentialSampler(eval_dataset)
        else:
            return super()._get_eval_sampler(eval_dataset)

        # Apply multipack wrapper if needed
        if use_multipack:
            return self._create_multipack_sampler(
                base_sampler=base_sampler,
                dataset=eval_dataset,
            )

        return base_sampler

    def _create_dataloader_params(self, is_eval=False, custom_batch_size=None):
        """Create common dataloader parameters for train or eval."""
        batch_size = custom_batch_size or (
            self.args.eval_batch_size if is_eval else self._train_batch_size
        )

        params = {
            "batch_size": batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        # Add persistent workers only for training
        if not is_eval and hasattr(self.args, "dataloader_persistent_workers"):
            params["persistent_workers"] = self.args.dataloader_persistent_workers

        # Add prefetch factor if specified
        if self.args.dataloader_prefetch_factor:
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return params

    def _prepare_dataloader(
        self, dataset, sampler, is_eval=False, custom_batch_size=None
    ):
        """Prepare a dataloader with the given dataset and sampler."""
        # Get base parameters
        dataloader_params = self._create_dataloader_params(is_eval, custom_batch_size)

        # Add sampler configuration
        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if isinstance(sampler, BatchSampler):
                # batch_size and batch_sampler are mutually exclusive
                dataloader_params["batch_sampler"] = sampler
                del dataloader_params["batch_size"]
            else:
                dataloader_params["sampler"] = sampler
                dataloader_params["drop_last"] = self.args.dataloader_drop_last

            if not is_eval:
                dataloader_params["worker_init_fn"] = seed_worker

        # Create the dataloader
        dataloader = DataLoader(dataset, **dataloader_params)

        if self.args.sample_packing and (
            (not is_eval and not self.args.pretraining)
            or (is_eval and self.args.eval_sample_packing is not False)
        ):
            self.accelerator.even_batches = False

        # Return unprepared dataloader if using sequence parallelism
        # TODO(djsaunde): We might be able to use `accelerate`'s dataloader preparation
        # if we use `dispatch_batches` and `slice_fn_for_dispatch` properly (i.e.,
        # slice each batch along the sequence dimension).
        if self.args.sequence_parallel_degree > 1:
            return dataloader

        # Otherwise prepare with accelerator
        return self.accelerator.prepare_data_loader(dataloader)

    def get_train_dataloader(self) -> DataLoader:
        """Get dataloader for training"""
        train_dataset = self.train_dataset
        data_collator = self.data_collator  # type: ignore

        # Handle dataset preprocessing
        if isinstance(train_dataset, datasets.Dataset):
            if self.args.sample_packing and not self.args.pretraining:
                train_dataset = train_dataset.remove_columns(["length"])
            if not self.args.sample_packing or self.args.pretraining:
                train_dataset = self._remove_unused_columns(
                    train_dataset, description="training"
                )
        else:
            self.data_collator = self._get_collator_with_removed_columns(  # pylint: disable=attribute-defined-outside-init
                data_collator,
                description="training",
            )

        # Get sampler and create dataloader
        sampler = self._get_train_sampler()
        return self._prepare_dataloader(train_dataset, sampler, is_eval=False)

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """Get dataloader for evaluation"""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        # Handle special case: sample packing is enabled but eval_sample_packing is False
        if self.args.sample_packing and self.args.eval_sample_packing is False:
            self.data_collator = (  # pylint: disable=attribute-defined-outside-init
                self.eval_data_collator
            )
            if "length" in eval_dataset.column_names:
                eval_dataset = eval_dataset.remove_columns(["length"])
            dataloader = super().get_eval_dataloader(eval_dataset)
            self.data_collator = (  # pylint: disable=attribute-defined-outside-init
                self.train_data_collator
            )

            return dataloader

        # Handle sample packing or sequence parallelism
        if (
            self.args.sample_packing
            and self.args.eval_sample_packing is not False
            or self.args.sequence_parallel_degree > 1
        ):
            # Get appropriate data collator
            self.data_collator = (  # pylint: disable=attribute-defined-outside-init
                self.eval_data_collator
                if hasattr(self, "eval_data_collator") and self.eval_data_collator
                else self.data_collator
            )
            if "length" in eval_dataset.column_names:
                eval_dataset = eval_dataset.remove_columns(["length"])

            # Handle dataset preprocessing for SP
            if self.args.sequence_parallel_degree > 1:
                if isinstance(eval_dataset, datasets.Dataset):
                    eval_dataset = self._remove_unused_columns(
                        eval_dataset, description="evaluation"
                    )
                else:
                    self.data_collator = self._get_collator_with_removed_columns(  # pylint: disable=attribute-defined-outside-init
                        self.data_collator, description="evaluation"
                    )

            # Use eval_batch_size for sample packing, per_device_eval_batch_size otherwise
            batch_size = (
                self.args.eval_batch_size
                if self.args.sample_packing
                else self.args.per_device_eval_batch_size
            )
            sampler = self._get_eval_sampler(eval_dataset)
            dataloader = self._prepare_dataloader(
                eval_dataset, sampler, is_eval=True, custom_batch_size=batch_size
            )

            return dataloader

        return super().get_eval_dataloader(eval_dataset)

    def _get_bench_sampler(
        self, bench_dataset: Dataset
    ) -> torch.utils.data.Sampler | None:
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

    @override
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
        kwargs = sanitize_kwargs_for_ds_tagging(
            dataset_tags=self.dataset_tags, kwargs=kwargs
        )
        kwargs = sanitize_kwargs_for_tagging(tag_names=self.tag_names, kwargs=kwargs)

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

    def additional_accelerator_args(
        self, fp8=None, **kwargs
    ):  # pylint: disable=unused-argument
        ret_kwargs = {}
        if fp8:
            from accelerate.utils import AORecipeKwargs

            ret_kwargs["mixed_precision"] = "fp8"
            ret_kwargs["kwargs_handlers"] = [AORecipeKwargs()]
            os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"

        return ret_kwargs

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs: The values to log.
            start_time: The start of training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        return super().log(logs, start_time)

    def store_metrics(
        self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train"
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
