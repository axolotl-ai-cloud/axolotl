"""
DPO trainer for axolotl
"""

import gc
from functools import wraps
from typing import Any, Dict, Union

import torch
from peft.optimizers import create_loraplus_optimizer
from torch import nn
from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled
from trl import DPOTrainer

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
