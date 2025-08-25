"""DPO trainer for axolotl"""

import gc
from functools import wraps
from typing import Any, Dict, Union

import torch
from torch import nn
from trl import DPOTrainer

from axolotl.core.trainers.mixins import (
    DistributedParallelMixin,
    RngLoaderMixin,
    SchedulerMixin,
)
from axolotl.core.trainers.mixins.optimizer import OptimizerInitMixin, OptimizerMixin
from axolotl.core.trainers.utils import (
    sanitize_kwargs_for_ds_tagging,
    sanitize_kwargs_for_tagging,
)


class AxolotlDPOTrainer(
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DPOTrainer,
    DistributedParallelMixin,
):
    """Extend the base DPOTrainer for axolotl helpers."""

    tag_names = ["axolotl", "dpo"]

    def __init__(self, *args, dataset_tags=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_tags = dataset_tags
        self.optimizer = None
        self.model_accepts_loss_kwargs = False

    @wraps(DPOTrainer.push_to_hub)
    def push_to_hub(self, *args, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tags when pushing
        the model on the Hub. Please refer to `~transformers.Trainer.push_to_hub`
        for more details.
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
            if res["rejected_input_ids"][0] == processing_class.bos_token_id:
                res["rejected_input_ids"] = res["rejected_input_ids"][1:]

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

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: dict[str, Union[list, torch.LongTensor]],
        is_ref_model: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.args.dpo_norm_loss:
            # fmt: off
            loss_type: str = self.loss_type  # type: ignore[has-type]
            # fmt: on
            # concatenated_forward handles avg token logprob for ipo case already
            self.loss_type = "ipo"
            res = super().concatenated_forward(model, batch, is_ref_model=is_ref_model)
            self.loss_type = loss_type
            return res
        return super().concatenated_forward(model, batch, is_ref_model=is_ref_model)
