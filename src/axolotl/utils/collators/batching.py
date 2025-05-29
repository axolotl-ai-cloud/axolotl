"""Data collators for axolotl to pad labels and position_ids for packed sequences"""
import torch
from dataclasses import dataclass
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


def modify_position_ids(ids: torch.Tensor) -> torch.Tensor:
    """
    Modify a tensor of position ids so that for a contiguous block of zeros,
    the filler zeros are replaced with an increasing sequence (0,1,2,...)
    except if a zero is immediately followed by a nonzero (which is taken
    as the start of an already increasing segment) in which case that zero
    remains 0.

    This is to avoid creating too many sub sequences that slows down flash attention computation.
    TODO: making sure that the increasing sequence is not longer than the existing longest increasing sequences
          we can split it into sub sequencces to achieve that.

    Example:
        Input:  tensor([0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 1, 2, 3])
        Output: tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3])
    """
    ids_shape = ids.shape
    ids = ids.view(-1)
    n = ids.shape[0]
    device = ids.device
    indices = torch.arange(n, device=device)
    mask = ids == 0

    # Identify the start of each contiguous group of zeros.
    # A zero starts a group if it is the first element or if the previous element is nonzero.
    new_group = mask.clone()
    new_group[0] = mask[0]
    new_group[1:] = mask[1:] & (~mask[:-1])

    # For the zeros, assign a group id (each contiguous block gets its own id).
    group_ids = torch.cumsum(new_group.to(torch.int64), dim=0) - 1  # valid only where mask is True
    zero_indices = indices[mask]

    # For each group, find the index of its first occurrence.
    num_groups = int(new_group.sum().item())
    if num_groups > 0:
        # Initialize with a large value and then use scatter_reduce to compute the minimum index per group.
        group_first = torch.full((num_groups,), n, dtype=torch.int64, device=device)
        group_first = group_first.scatter_reduce(0, group_ids[mask], zero_indices, reduce="amin")
        # For each zero, its new value is the difference between its index and the group's start.
        new_vals = zero_indices - group_first[group_ids[mask]]
    else:
        new_vals = torch.tensor([], dtype=torch.int64, device=device)

    # Create the output: replace zeros with computed new_vals.
    output = ids.clone()
    output[mask] = new_vals

    # Now “look ahead” one element:
    # If a zero is immediately followed by a nonzero, assume it is the start of a new (increasing) segment.
    # For such boundary zeros, override any filler modification and force the value to 0.
    boundary = torch.zeros(n, dtype=torch.bool, device=device)
    if n > 1:
        # For positions 0..n-2, if current is zero and the next is nonzero, mark it.
        boundary[:-1] = mask[:-1] & (ids[1:] != 0)
    output[boundary] = 0

    return output.view(ids_shape)

@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels and position_ids

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Any | None = None
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    label_pad_token_id: int = -100
    position_pad_token_id: int = 0
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        has_attn_mask = "attention_mask" in features[0].keys()
        labels = None
        if return_tensors is None:
            return_tensors = self.return_tensors

        for feature_name, pad_token_id in [
            ("labels", self.label_pad_token_id),
            ("position_ids", self.position_pad_token_id),
        ]:
            feat = (
                [feature[feature_name] for feature in features]
                if feature_name in features[0].keys()
                else None
            )
            labels = feat if feat and feature_name == "labels" else labels
            # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
            # same length to return tensors.
            if feat is not None:
                max_feature_length = max(len(l) for l in feat)  # noqa: E741
                if self.pad_to_multiple_of is not None:
                    max_feature_length = (
                        (max_feature_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                for feature in features:
                    remainder = [pad_token_id] * (
                        max_feature_length - len(feature[feature_name])
                    )
                    if isinstance(feature[feature_name], list):
                        feature[feature_name] = (
                            feature[feature_name] + remainder
                            if padding_side == "right"
                            else remainder + feature[feature_name]
                        )
                    elif padding_side == "right":
                        feature[feature_name] = np.concatenate(
                            [feature[feature_name], remainder]
                        ).astype(np.int64)
                    else:
                        feature[feature_name] = np.concatenate(
                            [remainder, feature[feature_name]]
                        ).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        if not has_attn_mask:
            del features["attention_mask"]

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        if "position_ids" in features:
            features["position_ids"] = modify_position_ids(features["position_ids"])

        return features


@dataclass
class BatchSamplerDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """
    Collator for multipack specific to the using the BatchSampler
    """

    def __call__(self, features, return_tensors=None):
        if not isinstance(features[0], list):
            features = [features]
        out_features = [{} for _ in features]
        for i, features_ in enumerate(features):
            for feature in features_[0].keys():
                if feature == "length":
                    continue
                if feature == "attention_mask":
                    arrays = [
                        (1) * np.array(item[feature])
                        for i, item in enumerate(features_)
                        if feature in item
                    ]
                    out_features[i][feature] = np.concatenate(arrays)
                else:
                    arrays = [
                        np.array(item[feature]) for item in features_ if feature in item
                    ]
                    out_features[i][feature] = np.concatenate(arrays)

        return super().__call__(out_features, return_tensors=return_tensors)


@dataclass
class V2BatchSamplerDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """
    Collator for multipack specific to the using the BatchSampler
    """

    def __call__(self, features, return_tensors=None):
        if not isinstance(features[0], list):
            features = [features]
        out_features = [{} for _ in features]
        for i, features_ in enumerate(features):
            for feature in features_[0].keys():
                if feature == "length":
                    continue
                if feature == "attention_mask":
                    arrays = [
                        (i + 1) * np.array(item[feature])
                        for i, item in enumerate(features_)
                        if feature in item
                    ]
                    out_features[i][feature] = np.concatenate(arrays)
                else:
                    arrays = [
                        np.array(item[feature]) for item in features_ if feature in item
                    ]
                    out_features[i][feature] = np.concatenate(arrays)

        return super().__call__(out_features, return_tensors=return_tensors)


@dataclass
class PretrainingBatchSamplerDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """
    Collator for multipack specific to the using the BatchSampler
    """

    def __init__(self, *args, multipack_attn=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.multipack_attn = multipack_attn

    def __call__(self, features, return_tensors=None):
        chunked_data = {}
        for feature in features.keys():
            if feature == "length":
                continue
            if feature == "attention_mask":
                if self.multipack_attn:
                    arrays = [
                        (i + 1) * np.array(item)
                        for i, item in enumerate(features[feature])
                    ]
                else:
                    arrays = [(1) * np.array(item) for item in features[feature]]
                chunked_data[feature] = np.concatenate(arrays)
            else:
                arrays = [np.array(item) for item in features[feature]]
                chunked_data[feature] = np.concatenate(arrays)
        features = [chunked_data]
        return super().__call__(features, return_tensors=return_tensors)
