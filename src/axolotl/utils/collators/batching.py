"""
DataCollator for axolotl to pad labels and position_ids for packed sequences
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from axolotl.monkeypatch.flex_attn import (
    create_block_causal_mask,
    packed_block_causal_mask,
)
from axolotl.monkeypatch.utils import get_seqlens_from_pos_ids


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
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    position_pad_token_id: int = 0
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
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
class FlexBatchSamplerDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """
    Collator for multipack specific to Flex Attention using the BatchSampler
    """

    def __call__(self, features, return_tensors=None):
        if not isinstance(features[0], list):
            features = [features]
        out_features = [{} for _ in features]
        for i, features_ in enumerate(features):
            for feature in features_[0].keys():
                if feature in {"length", "attention_mask"}:
                    continue
                else:
                    arrays = [
                        np.array(item[feature]) for item in features_ if feature in item
                    ]
                    out_features[i][feature] = np.concatenate(arrays)
        out = super().__call__(out_features, return_tensors=return_tensors)

        collated_seq_lens, max_seq_len = get_seqlens_from_pos_ids(out["position_ids"])
        out["attention_mask"] = packed_block_causal_mask(collated_seq_lens)
        # out["attention_mask"] = create_block_causal_mask(collated_seq_lens, max_seq_len)
        # raise ValueError(f"{out['attention_mask'].shape}")
        return out


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


def _get_document_ids_from_seq_lens(
    seq_lens: List[torch.Tensor],
) -> torch.Tensor:
    """
    Convert a batch tensor of seq lens into integer IDs denoting sample ownership.
    For example, seq_lens = [2, 3, 1] would return [0, 0, 1, 1, 1, 2].
    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.
    Returns:
        Tensor: Document IDs of shape (batch_size, max_seq_len).
    """
    batch_size = len(seq_lens)
    batch_document_ids = []
    for sample_idx in range(batch_size):
        # We assume seq lens sum to max seq lens, so document_ids should be of
        # shape (max_seq_len, )
        document_ids = torch.cat(
            [
                torch.full((seq_len,), i, dtype=torch.long, device=seq_len.device)
                for i, seq_len in enumerate(seq_lens[sample_idx])
            ]
        )
        batch_document_ids.append(document_ids)
    batch_document_ids = torch.stack(batch_document_ids)
    return batch_document_ids
