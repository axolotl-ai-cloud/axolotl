"""
DataCollator for axolotl to handle KD fields without using -inf for padding,
and with a teacher_mask to identify padded positions.
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from axolotl.utils.collators.batching import DataCollatorForSeq2Seq


@dataclass
class DataCollatorForKD(DataCollatorForSeq2Seq):
    """
    Data collator for KD, including handling KD-specific fields.

    This version avoids using -inf and instead uses a large negative value for padding
    target_logprobs. It also creates a teacher_mask to indicate which entries are valid.
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
        if return_tensors is None:
            return_tensors = self.return_tensors

        padding_side = self.tokenizer.padding_side

        # Pad labels and position_ids first
        for feature_name, pad_token_id in [
            ("labels", self.label_pad_token_id),
            ("position_ids", self.position_pad_token_id),
        ]:
            if feature_name in features[0]:
                feat = [f[feature_name] for f in features]
                max_len = max(len(x) for x in feat)
                if self.pad_to_multiple_of is not None:
                    max_len = (
                        (max_len + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                    ) * self.pad_to_multiple_of

                for f in features:  # pylint: disable=invalid-name
                    remainder = [pad_token_id] * (max_len - len(f[feature_name]))
                    if isinstance(f[feature_name], list):
                        f[feature_name] = (
                            f[feature_name] + remainder
                            if padding_side == "right"
                            else remainder + f[feature_name]
                        )
                    else:
                        # If they are numpy arrays
                        if padding_side == "right":
                            f[feature_name] = np.concatenate(
                                [f[feature_name], remainder]
                            ).astype(np.int64)
                        else:
                            f[feature_name] = np.concatenate(
                                [remainder, f[feature_name]]
                            ).astype(np.int64)

        # Handle target_logprobs and target_token_ids manually
        target_logprobs_list = []
        target_token_ids_list = []
        target_mask_list = []
        has_teacher_data = ("target_logprobs" in features[0]) and (
            "target_token_ids" in features[0]
        )

        if has_teacher_data:
            # Extract and remove from features
            for f in features:  # pylint: disable=invalid-name
                target_logprobs_list.append(f.pop("target_logprobs"))
                target_token_ids_list.append(f.pop("target_token_ids"))
                target_mask_list.append(f.pop("target_mask"))

            # Determine max lengths
            max_teacher_seq_len = max(len(seq) for seq in target_logprobs_list)
            max_k = max(len(seq_k) for seq in target_logprobs_list for seq_k in seq)

            padded_target_logprobs = []
            padded_target_token_ids = []
            padded_teacher_mask_list = []

            for t_logprobs, t_ids, t_mask in zip(
                target_logprobs_list, target_token_ids_list, target_mask_list
            ):
                t_logprobs_padded = []
                t_ids_padded = []
                t_mask_padded = []

                for lp, ids, mask in zip(  # pylint: disable=invalid-name
                    t_logprobs, t_ids, t_mask
                ):
                    lp_len = len(lp)
                    if lp_len < max_k:
                        # Use -1e9 for padding logprobs and 0 for token_ids
                        pad_len = max_k - lp_len
                        lp = lp + [-1e9] * pad_len  # pylint: disable=invalid-name
                        ids = ids + [0] * pad_len
                        mask = mask + [0] * pad_len
                    else:
                        lp = lp[:max_k]  # pylint: disable=invalid-name
                        ids = ids[:max_k]
                        mask = mask[:max_k]

                    t_logprobs_padded.append(lp)
                    t_ids_padded.append(ids)
                    t_mask_padded.append(mask)

                seq_len_diff = max_teacher_seq_len - len(t_logprobs_padded)
                if seq_len_diff > 0:
                    # Pad sequences fully if needed
                    t_logprobs_padded.extend(
                        [[-1e9] * max_k for _ in range(seq_len_diff)]
                    )
                    t_ids_padded.extend([[0] * max_k for _ in range(seq_len_diff)])
                    t_mask_padded.extend([[0] * max_k for _ in range(seq_len_diff)])

                padded_target_logprobs.append(t_logprobs_padded)
                padded_target_token_ids.append(t_ids_padded)
                padded_teacher_mask_list.append(t_mask_padded)

            # Convert to tensors
            padded_target_logprobs = torch.tensor(
                padded_target_logprobs, dtype=torch.float
            )
            padded_target_token_ids = torch.tensor(
                padded_target_token_ids, dtype=torch.long
            )
            padded_teacher_mask_list = torch.tensor(
                padded_teacher_mask_list, dtype=torch.int
            )

        # Pad using tokenizer for regular fields
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # Add back teacher data if present
        if has_teacher_data:
            features["target_logprobs"] = padded_target_logprobs
            features["target_token_ids"] = padded_target_token_ids
            features["target_mask"] = padded_teacher_mask_list

        # Prepare decoder_input_ids if the model supports it
        if (
            "labels" in features
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features
