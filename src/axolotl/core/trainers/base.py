"""Module for customized trainers"""

from __future__ import annotations

import os
from collections import defaultdict
from functools import partial, wraps
from typing import Any, Callable, Literal, Optional

import datasets
import safetensors
import torch
from accelerate.state import AcceleratorState
from datasets import Dataset
from peft import PeftModel
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from transformers import PreTrainedModel, Trainer
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length, seed_worker
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME, is_peft_available
from trl.trainer.utils import pad_to_length
from typing_extensions import override

from axolotl.core.trainers.mixins import (
    ActivationOffloadingMixin,
    CheckpointSaveMixin,
    DistributedParallelMixin,
    OptimizerMixin,
    PackingMixin,
    RngLoaderMixin,
    SchedulerMixin,
)
from axolotl.core.trainers.utils import (
    sanitize_kwargs_for_ds_tagging,
    sanitize_kwargs_for_tagging,
)
from axolotl.utils import get_not_null
from axolotl.utils.bench import get_gpu_memory_usage
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_distributed, is_main_process
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths

LOG = get_logger(__name__)

REDUCTION_FNS = {
    "mean": torch.mean,
    "min": torch.min,
    "max": torch.max,
    "sum": torch.sum,
}


class AxolotlTrainer(
    PackingMixin,
    SchedulerMixin,
    OptimizerMixin,
    RngLoaderMixin,
    CheckpointSaveMixin,
    ActivationOffloadingMixin,
    DistributedParallelMixin,
    Trainer,
):
    """Extend the base Trainer for axolotl helpers"""

    args = None  # type: "AxolotlTrainingArguments"  # type: ignore[name-defined]
    tag_names = ["axolotl"]
    _axolotl_cfg: DictDefault | None = None

    @property
    def axolotl_cfg(self):
        return self._axolotl_cfg

    @axolotl_cfg.setter
    def axolotl_cfg(self, cfg):
        self._axolotl_cfg = cfg

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
        self._stored_metrics = defaultdict(
            lambda: defaultdict(lambda: {"values": [], "reduction": "mean"})
        )
        if self.args.orpo_alpha:
            self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

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

        sampler = MultipackBatchSampler(
            base_sampler,
            lengths=get_dataset_lengths(dataset),
            packing_efficiency_estimate=self.args.sample_packing_efficiency,
            batch_max_len=batch_max_len,
            batch_size=batch_size,
            group_size=self.args.sample_packing_group_size,
            bin_size=self.args.sample_packing_bin_size,
            sequential=self.args.sample_packing_sequentially,
            drop_last=True,
            num_processes=self.args.dataset_num_proc,
            mp_start_method=self.args.sample_packing_mp_start_method or "fork",
        )

        len(sampler)
        return sampler

    def _get_train_sampler(
        self, train_dataset: Dataset | None = None
    ) -> Sampler | None:
        """
        Helper method to get the sampler for training. Handles cases for sample packing
        and curriculum sampling (sequential).

        Returns:
            If the dataset is non-empty, a sampler is returned, the type of which
                depends on the passed training args.
        """
        # from https://github.com/huggingface/transformers/blob/2166b6b4ff09f6dd3867ab982f262f66482aa968/src/transformers/trainer.py#L969C1-L972C24
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None

        use_sample_packing = self.args.sample_packing and not self.args.pretraining

        # Determine the base sampler first
        if self.args.curriculum_sampling:
            base_sampler = SequentialSampler(train_dataset)
        elif use_sample_packing:
            base_sampler = RandomSampler(train_dataset)
        else:
            # Default to parent class implementation for standard random sampling
            return super()._get_train_sampler(train_dataset)

        # Apply multipack wrapper if needed
        if use_sample_packing:
            return self._create_multipack_sampler(
                base_sampler=base_sampler,
                dataset=train_dataset,
            )

        return base_sampler

    def _get_eval_sampler(self, eval_dataset: Dataset | None = None) -> Sampler | None:
        """
        Helper method to get the sampler for evaluation. Handles sample packing case.

        Returns:
            If the dataset is non-empty, a sampler is returned, the type of which
                depends on the passed training args.
        """
        # from https://github.com/huggingface/transformers/blob/2166b6b4ff09f6dd3867ab982f262f66482aa968/src/transformers/trainer.py#L1065C9-L1066C24
        if eval_dataset is None or not has_length(eval_dataset):
            return None

        # Multipacking enabled if training is enabled and eval is not explicitly disabled
        use_multipack = (
            self.args.sample_packing and self.args.eval_sample_packing is not False
        )

        # Determine the base sampler
        if use_multipack:
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

    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Optional[Callable[[Dataset], torch.utils.data.Sampler]] = None,
        is_training: bool = False,
        dataloader_key: Optional[str] = None,
    ) -> DataLoader:
        """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""

        data_collator = self.data_collator if is_training else self.eval_data_collator

        if isinstance(dataset, datasets.Dataset):
            if is_training:
                if not self.args.sample_packing or self.args.pretraining:
                    dataset = self._remove_unused_columns(
                        dataset, description="training"
                    )
            elif (
                not is_training
                and self.args.sample_packing
                and self.args.eval_sample_packing is not False
            ):
                batch_size = (
                    batch_size
                    if self.args.sample_packing
                    else self.args.per_device_eval_batch_size
                )
            else:
                dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(
                self.data_collator, description=description
            )

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            dataloader_params["drop_last"] = get_not_null(
                self.args.dataloader_drop_last, True
            )
            if sampler_fn is not None:
                sampler = sampler_fn(dataset)
                if isinstance(sampler, BatchSampler):
                    # batch_size and batch_sampler are mutually exclusive
                    dataloader_params["batch_sampler"] = sampler
                    del dataloader_params["batch_size"]
                    del dataloader_params["drop_last"]
                else:
                    dataloader_params["sampler"] = sampler

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.process_index,
                )
        if self.args.sample_packing and (
            (is_training and not self.args.pretraining)
            or (not is_training and self.args.eval_sample_packing is not False)
        ):
            self.accelerator.even_batches = False

        if dataset.column_names and "length" in dataset.column_names:
            dataset = dataset.remove_columns(["length"])

        if (
            dataset.column_names
            and "position_ids" in dataset.column_names
            and "attention_mask" in dataset.column_names
            and self.args.sample_packing
            and self.args.sample_packing_drop_attention_mask
        ):
            dataset = dataset.remove_columns(["attention_mask"])

        dataloader = DataLoader(dataset, **dataloader_params)

        # Accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version for eval dataloaders.
        # fmt: off
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader  # type: ignore
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}
        # fmt: on

        return self.accelerator.prepare(dataloader)

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

        # track number of tokens for tokens per second calculation
        if self.args.include_tkps:
            inputs_key = "labels" if "labels" in inputs else "input_ids"
            num_tokens = (inputs[inputs_key] != -100).sum()
            if is_distributed():
                torch.distributed.all_reduce(
                    num_tokens, op=torch.distributed.ReduceOp.SUM
                )
            if hasattr(self.state, "num_tokens"):
                self.state.num_tokens = (
                    self.state.num_tokens + (inputs[inputs_key] != -100).sum().cpu()
                )
            else:
                self.state.num_tokens = (inputs[inputs_key] != -100).sum().cpu()

            if hasattr(self.state, "total_tokens"):
                self.state.total_tokens += num_tokens
            else:
                self.state.total_tokens = num_tokens

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

    @override
    def evaluate(self, *args, **kwargs):
        LOG.info("Running evaluation step...")
        return super().evaluate(*args, **kwargs)

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
        num_items_in_batch=None,
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
        # cleanup the PartialState states so Accelerate automatically configures everything from the env vars
        accelerator_config = self.args.accelerator_config.to_dict()
        use_configured_state = accelerator_config.get("use_configured_state", False)
        if not use_configured_state:
            AcceleratorState._reset_state(reset_partial_state=True)

        super().create_accelerator_and_postprocess()

    def additional_accelerator_args(
        self, fp8: bool = False, enable_fsdp_float8_all_gather: bool = False, **kwargs
    ) -> dict[str, Any]:
        ret_kwargs = {}
        if fp8:
            from accelerate.utils import AORecipeKwargs
            from torchao.float8 import Float8LinearConfig

            # By default, Float8LinearConfig is instantiated using the "tensorwise"
            # scaling strategy. See more details here:
            # https://github.com/pytorch/ao/tree/main/torchao/float8.
            config = Float8LinearConfig(
                enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
                force_recompute_fp8_weight_in_bwd=enable_fsdp_float8_all_gather is True,
            )

            ret_kwargs["mixed_precision"] = "fp8"
            ret_kwargs["kwargs_handlers"] = [AORecipeKwargs(config=config)]  # type: ignore
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

        for key, metric_data in self._stored_metrics[train_eval].items():
            values = torch.tensor(metric_data["values"])  # type: ignore[arg-type]
            reduction_type = metric_data["reduction"]

            fn = REDUCTION_FNS.get(reduction_type)
            if fn is None:
                raise NotImplementedError(
                    "Metric reduction must be one of [mean, min, max, sum]"
                )
            logs[key] = round(fn(values).item(), 4)

        if is_main_process():
            # Add memory usage
            try:
                active, allocated, reserved = get_gpu_memory_usage()
                logs["memory/max_active (GiB)"] = round(active, 2)
                logs["memory/max_allocated (GiB)"] = round(allocated, 2)
                logs["memory/device_reserved (GiB)"] = round(reserved, 2)
            except (ValueError, TypeError, FileNotFoundError):
                pass

        if self.args.include_tkps and train_eval == "train":
            # each rank will log its own tokens per second
            # for logging_steps > 1 we obtain a moving average of this metric
            logs["tokens_per_second_per_gpu"] = round(
                self.state.last_tokens_per_second.item() / self.args.logging_steps, 2
            )
            logs["total_tokens"] = int(self.state.total_tokens.item())

        del self._stored_metrics[train_eval]

        return super().log(logs, start_time)

    def store_metrics(
        self,
        metrics: dict[str, float] | dict[str, tuple[int | float, str]],
        train_eval: Literal["train", "eval"] = "train",
        reduction: Literal["mean", "min", "max", "sum"] = "mean",
    ) -> None:
        """
        Store metrics with specified reduction type.

        Args:
            metrics: Dictionary of metric names to values, or metric names to (value,
                reduction_type) tuples.
            train_eval: Whether this is for training or evaluation.
        """
        for key, value in metrics.items():
            if isinstance(value, tuple):
                value, _reduction = value  # type: ignore[assignment]
            else:
                value, _reduction = value, reduction

            self._stored_metrics[train_eval][key]["values"].append(value)
            self._stored_metrics[train_eval][key]["reduction"] = _reduction

    def _save_checkpoint(self, model, trial, **kwargs):
        # make sure the checkpoint dir exists, since trainer is flakey
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        return super()._save_checkpoint(model, trial, **kwargs)

    # TODO(wing): remove once https://github.com/huggingface/transformers/pull/39866/files is merged
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        LOG.info(f"Saving model checkpoint to {output_dir}")
        supported_classes = (
            (PreTrainedModel,)
            if not is_peft_available()
            else (PreTrainedModel, PeftModel)
        )
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()
            if isinstance(
                self.accelerator.unwrap_model(self.model, keep_torch_compile=False),
                supported_classes,
            ):
                self.accelerator.unwrap_model(
                    self.model, keep_torch_compile=False
                ).save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                )
            else:
                LOG.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                )
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict,
                        os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                        metadata={"format": "pt"},
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
                is_main_process=self.accelerator.is_main_process,
            )

            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
            elif (
                self.data_collator is not None
                and hasattr(self.data_collator, "tokenizer")
                and self.data_collator.tokenizer is not None
            ):
                LOG.info(
                    "Saving Trainer.data_collator.tokenizer by default as Trainer.processing_class is `None`"
                )
                save_jinja_files = True
                if self.axolotl_cfg:
                    save_jinja_files = self.axolotl_cfg.tokenizer_save_jinja_files
                self.data_collator.tokenizer.save_pretrained(
                    output_dir, save_jinja_files=save_jinja_files
                )
            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
