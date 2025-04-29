"""
DPO trainer for axolotl
"""

import gc
import random
from functools import wraps
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch
import wandb
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from peft.optimizers import create_loraplus_optimizer
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
)
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_sagemaker_mp_enabled
from trl import DPOConfig, DPOTrainer, maybe_apply_chat_template, maybe_extract_prompt
from trl.trainer.utils import log_table_to_comet_experiment

from axolotl.core.trainers.mixins import RngLoaderMixin, SchedulerMixin
from axolotl.core.trainers.utils import (
    sanitize_kwargs_for_ds_tagging,
    sanitize_kwargs_for_tagging,
)

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


class AxolotlDPOTrainer(RngLoaderMixin, SchedulerMixin, DPOTrainer):
    """
    Extend the base DPOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "dpo"]

    def __init__(self, *args, dataset_tags=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_tags = dataset_tags
        self.optimizer = None
        self.model_accepts_loss_kwargs = False

    def create_optimizer(self):
        # pylint: disable=duplicate-code
        if self.args.loraplus_lr_ratio is None:
            return super().create_optimizer()

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:  # pylint: disable=access-member-before-definition
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args,
                opt_model,
            )

            loraplus_lr_ratio = getattr(self.args, "loraplus_lr_ratio", None)
            if loraplus_lr_ratio:
                print("Using lora+")
            loraplus_lr_embedding = getattr(self.args, "loraplus_lr_embedding", None)
            # pylint: disable=duplicate-code
            self.optimizer = create_loraplus_optimizer(  # pylint: disable=attribute-defined-outside-init
                opt_model,
                optimizer_cls,
                loraplus_lr_ratio=loraplus_lr_ratio,
                loraplus_lr_embedding=loraplus_lr_embedding,
                **optimizer_kwargs,
            )

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(  # pylint: disable=attribute-defined-outside-init
                self.optimizer
            )

        return self.optimizer

    @wraps(DPOTrainer.push_to_hub)
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

    # TODO: remove this once https://github.com/huggingface/trl/pull/3377 is in a release
    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[
            PreTrainedTokenizerBase,
            BaseImageProcessor,
            FeatureExtractionMixin,
            ProcessorMixin,
        ],
        args: DPOConfig,
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Build the kwargs for the `map` function
        map_kwargs: Dict[str, Any] = {"writer_batch_size": 10}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Extract prompt if needed
            if isinstance(
                dataset, Dataset
            ):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            # Apply the chat template if needed
            if isinstance(
                dataset, Dataset
            ):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class, "tools": args.tools},
                **map_kwargs,
            )

            # Tokenize the dataset
            if isinstance(
                dataset, Dataset
            ):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            dataset = dataset.map(
                self.tokenize_row if not self.is_vision_model else self.process_row,
                remove_columns=["chosen", "rejected"],
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
                    "add_special_tokens": False,
                },
                **map_kwargs,
            )

        return dataset

    @staticmethod
    def tokenize_row(
        features,
        processing_class,
        max_prompt_length,
        max_completion_length,
        add_special_tokens,
    ) -> Dict:
        res = DPOTrainer.tokenize_row(
            features,
            processing_class,
            max_prompt_length,
            max_completion_length,
            add_special_tokens,
        )
        # fix when the tokenizer doesn't have a bos_token_id, e.g. Qwen
        if processing_class.bos_token is None and res["prompt_input_ids"][0] is None:
            for key in res.keys():
                res[key] = res[key][1:]

        if processing_class.bos_token and processing_class.bos_token_id is not None:
            # dpo trainer may incorrectly prepend the bos_token_id to the dpo outputs
            if res["chosen_input_ids"][0] == processing_class.bos_token_id:
                res["chosen_input_ids"] = res["chosen_input_ids"][1:]
                res["chosen_labels"] = res["chosen_labels"][1:]
                res["chosen_attention_mask"] = res["chosen_attention_mask"][1:]
            if res["rejected_input_ids"][0] == processing_class.bos_token_id:
                res["rejected_input_ids"] = res["rejected_input_ids"][1:]
                res["rejected_labels"] = res["rejected_labels"][1:]
                res["rejected_attention_mask"] = res["rejected_attention_mask"][1:]

        return res

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        loss: torch.Tensor = super().training_step(model, inputs, num_items_in_batch)
        gc.collect()
        torch.cuda.empty_cache()
        return loss

    # TODO: remove this once https://github.com/huggingface/trl/pull/3377 is in a release
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(
                range(num_samples), k=self.args.eval_batch_size
            )

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = (
                self.generate_from_model_and_ref(self.model, random_batch)
            )

            table = pd.DataFrame(
                columns=["Prompt", "Policy", "Ref Model"],
                data=[
                    [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                    for prompt, pol, ref in zip(
                        random_batch_dataset["prompt"],
                        policy_output_decoded,
                        ref_output_decoded,
                    )
                ],
            )
            if "wandb" in self.args.report_to and self.accelerator.is_main_process:
                wandb.log({"game_log": wandb.Table(data=table)})

            if "comet_ml" in self.args.report_to:
                log_table_to_comet_experiment(
                    name="game_log.csv",
                    table=table,
                )

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        return initial_output
