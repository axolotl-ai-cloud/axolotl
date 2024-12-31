# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    # pylint: disable=duplicate-code
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


class KDBatchSamplerDataCollatorForSeq2Seq(DataCollatorForKD):
    """
    Collator for multipack (batch of sub-batches) specifically for KD.
    Adapts DataCollatorForKD so it can pack multiple sequences in a single batch item.
    """

    def __call__(self, features, return_tensors=None):
        """
        Expects that `features` could be either:
          - a single list of dicts, OR
          - a list of lists of dicts (the "sub-batches" to be packed).
        """
        # 1) If we are *not* dealing with multiple sequences per batch element,
        #    just pass straight to parent.
        if not isinstance(features[0], list):
            return super().__call__(features, return_tensors=return_tensors)

        # 2) Otherwise, we *are* dealing with multiple sequences in each batch item.
        #    We want to produce a single "merged" feature dict for each sub-batch.
        out_features = [{} for _ in features]

        for i, sub_features in enumerate(features):
            # sub_features is a list of dicts, each dict = one sequence’s features
            # We'll merge them into out_features[i].
            #
            # NOTE: You can customize how you combine fields as needed (e.g. summation
            # or offset for attention_mask). Below is a straightforward concatenation/extension.

            for field_name in sub_features[0].keys():
                # Some fields you might want to skip or treat specially:
                if field_name == "length":
                    continue

                # If it’s a KD field that’s a list-of-lists (e.g. target_logprobs),
                # you typically just want to flatten them by extending.
                if field_name in ["target_logprobs", "target_token_ids", "target_mask"]:
                    combined = []
                    for feat in sub_features:
                        combined.extend(feat[field_name])
                    out_features[i][field_name] = combined

                elif field_name == "attention_mask":
                    # Here we apply the (j+1) factor to differentiate each sub-sample
                    # within this merged batch item.
                    arrays = []
                    for j, feat in enumerate(sub_features):
                        if field_name in feat:
                            arrays.append((j + 1) * np.array(feat[field_name]))
                    out_features[i][field_name] = np.concatenate(arrays)
                else:
                    # By default, just concatenate them if they are arrays
                    # or extend them if they are lists.
                    # For example, input_ids or labels are often arrays.
                    arrays = []
                    for feat in sub_features:
                        if field_name in feat:
                            arr = np.array(feat[field_name])
                            arrays.append(arr)
                    out_features[i][field_name] = np.concatenate(arrays)

        # 3) Now call the parent collator, which will do:
        #    - padding of labels/position_ids
        #    - KD-specific padding for target_logprobs, target_token_ids, etc.
        #    - final conversion to return_tensors
        return super().__call__(out_features, return_tensors=return_tensors)
