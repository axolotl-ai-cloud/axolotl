"""DPO trainer for axolotl"""

import gc
from functools import wraps
from typing import Any, Dict, Union

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, ProcessorMixin
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
from axolotl.utils.data.utils import remove_double_bos_token


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

    def _tokenize(
        self,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        input: str | list,
        **kwargs,
    ) -> dict[str, list]:
        """
        Override TRL's tokenization in DPO trainer to fix double bos_token bug (eg. llama).
        """
        result = super()._tokenize(
            processing_class=processing_class, input=input, **kwargs
        )

        # Handle multimodal models
        tokenizer = (
            getattr(processing_class, "tokenizer", None)
            if isinstance(processing_class, ProcessorMixin)
            else processing_class
        )

        bos_token_id = getattr(tokenizer, "bos_token_id", None) if tokenizer else None
        if bos_token_id is not None:
            result = remove_double_bos_token(result, bos_token_id)

        return result

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
