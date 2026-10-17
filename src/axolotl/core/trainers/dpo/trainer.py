"""DPO trainer for axolotl"""

import inspect
from functools import wraps

from datasets import Dataset, IterableDataset
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
from axolotl.utils.collators.preference import (
    LOSS_MASK_COLUMNS,
    AxolotlDataCollatorForPreference,
)
from axolotl.utils.data.utils import remove_double_bos_token

TOKENIZED_COLUMNS = ("prompt_ids", "chosen_ids", "rejected_ids")


def _column_names(dataset) -> list[str]:
    if dataset is None:
        return []
    if isinstance(dataset, dict):
        return sorted({c for ds in dataset.values() for c in _column_names(ds)})
    names = getattr(dataset, "column_names", None)
    if names is None and isinstance(dataset, IterableDataset):
        names = list(next(iter(dataset)).keys())
    return list(names or [])


def has_loss_masks(dataset) -> bool:
    return all(c in _column_names(dataset) for c in LOSS_MASK_COLUMNS)


def _build_loss_mask_collator(processing_class, args):
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return AxolotlDataCollatorForPreference(
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.max_length,
        truncation_mode=args.truncation_mode,
        pad_to_multiple_of=args.pad_to_multiple_of,
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
        if has_loss_masks(kwargs.get("train_dataset")):
            collator = kwargs.get("data_collator")
            if collator is None:
                if kwargs.get("processing_class") is None or kwargs.get("args") is None:
                    raise ValueError(
                        "Datasets with loss masks need `args` and `processing_class` "
                        "passed as keyword arguments."
                    )
                kwargs["data_collator"] = _build_loss_mask_collator(
                    kwargs["processing_class"], kwargs["args"]
                )
            elif not isinstance(collator, AxolotlDataCollatorForPreference):
                raise ValueError(
                    "Datasets with `chosen_loss_mask`/`rejected_loss_mask` need "
                    "AxolotlDataCollatorForPreference; "
                    f"got {type(collator).__name__}, which would ignore the masks."
                )
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

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        args,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        if not all(c in _column_names(dataset) for c in TOKENIZED_COLUMNS):
            return super()._prepare_dataset(
                dataset, processing_class, args, dataset_name
            )

        # Already tokenized (e.g. chat_template.multiturn); keep TRL's drop of
        # examples whose prompt alone fills max_length under keep_start truncation.
        if args.max_length is not None and args.truncation_mode == "keep_start":
            max_length = args.max_length
            map_kwargs = {}
            if isinstance(dataset, Dataset):
                map_kwargs = {
                    "num_proc": args.dataset_num_proc,
                    "desc": f"Dropping fully truncated examples from {dataset_name} dataset",
                }
            dataset = dataset.filter(
                lambda example: len(example["prompt_ids"]) < max_length, **map_kwargs
            )
        return dataset

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if "prompt_ids" in self._signature_columns:
            self._signature_columns += [
                c for c in LOSS_MASK_COLUMNS if c not in self._signature_columns
            ]

    def _tokenize(
        self,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        input: str | list,
        is_vlm: bool | None = None,
        **kwargs,
    ) -> dict[str, list]:
        """
        Override TRL's tokenization in DPO trainer to fix double bos_token bug (eg. llama).
        """
        parent_tokenize = super()._tokenize
        try:
            accepts_is_vlm = "is_vlm" in inspect.signature(parent_tokenize).parameters
        except (TypeError, ValueError):
            accepts_is_vlm = False
        if accepts_is_vlm:
            kwargs["is_vlm"] = is_vlm
        result = parent_tokenize(
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
