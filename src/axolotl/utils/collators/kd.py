"""
DataCollator for axolotl to handle KD fields
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

        # Extract labels and position_ids first (as in original code)
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

                padding_side = self.tokenizer.padding_side
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
        has_teacher_data = ("target_logprobs" in features[0]) and (
            "target_token_ids" in features[0]
        )

        if has_teacher_data:
            # Extract these fields
            for f in features:  # pylint: disable=invalid-name
                target_logprobs_list.append(f.pop("target_logprobs"))
                target_token_ids_list.append(f.pop("target_token_ids"))

            # Determine max lengths to pad
            max_teacher_seq_len = max(len(seq) for seq in target_logprobs_list)
            max_k = max(len(seq_k) for seq in target_logprobs_list for seq_k in seq)

            # Pad target_logprobs and target_token_ids
            padded_target_logprobs = []
            padded_target_token_ids = []
            for t_logprobs, t_ids in zip(target_logprobs_list, target_token_ids_list):
                # Pad seq dimension
                t_logprobs_padded = []
                t_ids_padded = []
                for i in range(  # pylint: disable=consider-using-enumerate
                    len(t_logprobs)
                ):
                    lp = t_logprobs[i]  # pylint: disable=invalid-name
                    ids = t_ids[i]
                    # Pad K dimension
                    lp_len = len(lp)
                    if lp_len < max_k:
                        lp = lp + [-float("inf")] * (  # pylint: disable=invalid-name
                            max_k - lp_len
                        )  # or some pad value that won't break exp()
                        ids = ids + [0] * (max_k - lp_len)
                    t_logprobs_padded.append(lp)
                    t_ids_padded.append(ids)

                # If sequence is shorter than max_teacher_seq_len
                seq_len_diff = max_teacher_seq_len - len(t_logprobs_padded)
                if seq_len_diff > 0:
                    t_logprobs_padded.extend(
                        [[-float("inf")] * max_k for _ in range(seq_len_diff)]
                    )
                    t_ids_padded.extend([[0] * max_k for _ in range(seq_len_diff)])

                padded_target_logprobs.append(t_logprobs_padded)
                padded_target_token_ids.append(t_ids_padded)

            # Convert to tensors
            padded_target_logprobs = torch.tensor(
                padded_target_logprobs, dtype=torch.float
            )
            # We can store token_ids as long tensor
            padded_target_token_ids = torch.tensor(
                padded_target_token_ids, dtype=torch.long
            )

        # Now pad using tokenizer for the remaining fields (input_ids, attention_mask, etc.)
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # Add back the teacher data if it exists
        if has_teacher_data:
            features["target_logprobs"] = padded_target_logprobs
            features["target_token_ids"] = padded_target_token_ids

        # Prepare decoder_input_ids if applicable
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
