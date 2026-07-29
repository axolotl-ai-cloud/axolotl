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

from axolotl.integrations.kd.utils import LOGPROB_PAD_VALUE
from axolotl.utils.collators.batching import DataCollatorForSeq2Seq
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

KD_TARGET_FIELDS = ("target_logprobs", "target_token_ids", "target_mask")


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

    # a mis-declared alignment shows up as the alternative hypothesis explaining the
    # labels much better than the declared one; requiring both a ratio and an absolute
    # gap over a minimum number of positions keeps a noisy teacher from tripping it
    ALIGNMENT_WARN_RATIO: float = 1.5
    ALIGNMENT_WARN_GAP: float = 0.1
    ALIGNMENT_MIN_POSITIONS: int = 64
    ALIGNMENT_MAX_BATCHES: int = 8

    def __init__(self, *args, **kwargs):
        alignment = kwargs.pop("kd_prepared_targets_alignment", None)
        super().__init__(*args, **kwargs)
        self.kd_prepared_targets_alignment = alignment or "current"
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self._alignment_checked = False
        self._alignment_check_batches = 0

    def realign_prepared_targets(self, features) -> None:
        """
        Compensate for baked target_* columns that use the legacy convention, where row
        ``j`` holds the distribution over token ``j`` instead of token ``j + 1``.

        Applied per sequence, before packing, so rows never cross a sequence boundary.
        """
        if self.kd_prepared_targets_alignment != "legacy":
            return

        for feature in features:
            if any(
                feature.get(field) is None or len(feature[field]) == 0
                for field in KD_TARGET_FIELDS
            ):
                continue
            for field, pad_value in zip(
                KD_TARGET_FIELDS, (LOGPROB_PAD_VALUE, 0, 0), strict=True
            ):
                rows = feature[field]
                shifted = list(rows[1:])
                shifted.append([pad_value] * len(rows[0]))
                feature[field] = shifted

    @staticmethod
    def _alignment_stats(
        labels, target_token_ids, target_probs, slot_valid, valid, offset, horizon
    ) -> tuple[float, float]:
        """
        How well the teacher rows explain the tokens ``offset`` positions ahead:
        (mean probability mass on that token, rate at which it is in the stored top-k).

        Mass and containment are used instead of top-1 equality because a dataset
        sampled at temperature/top-p (or scored over text the teacher never generated)
        routinely has the actual token outside the teacher's argmax, which would make a
        top-1 statistic look mis-aligned even when it is not.
        """
        target = labels[:, offset : offset + horizon].unsqueeze(-1)
        hits = (target_token_ids[:, :horizon] == target) & slot_valid[:, :horizon]

        mass = (target_probs[:, :horizon] * hits).sum(-1)
        contained = hits.any(-1)

        n_valid = max(1, int(valid.sum().item()))
        return (
            mass[valid].sum().item() / n_valid,
            contained[valid].sum().item() / n_valid,
        )

    def _check_target_alignment(
        self, labels, target_token_ids, target_logprobs, target_mask
    ) -> None:
        """
        Sanity-check the declared alignment against the labels once, cheaply: the teacher
        puts far more mass on the token a row actually describes than on its neighbours,
        so a mis-declared dataset is visible in a single batch.
        """
        if self._alignment_checked or not torch.is_tensor(labels):
            return

        self._alignment_check_batches += 1
        # after compensation every dataset should be in the current convention; an
        # over-shifted (declared legacy, actually current) dataset lands one further out
        alt_offset = 0 if self.kd_prepared_targets_alignment == "current" else 2
        horizon = labels.shape[1] - max(1, alt_offset)
        if horizon <= 0:
            return

        slot_valid = target_mask > 0
        valid = slot_valid[:, :horizon, 0]
        n_valid = int(valid.sum().item())
        if n_valid < self.ALIGNMENT_MIN_POSITIONS:
            if self._alignment_check_batches >= self.ALIGNMENT_MAX_BATCHES:
                self._alignment_checked = True
            return

        self._alignment_checked = True

        # renormalize the stored slice so the statistic is comparable whether or not the
        # producer normalized the top-k, and so padded slots carry no mass
        probs = torch.softmax(
            target_logprobs.float().masked_fill(~slot_valid, float("-inf")), dim=-1
        )
        probs = torch.nan_to_num(probs)

        declared_mass, declared_containment = self._alignment_stats(
            labels, target_token_ids, probs, slot_valid, valid, 1, horizon
        )
        alt_mass, alt_containment = self._alignment_stats(
            labels, target_token_ids, probs, slot_valid, valid, alt_offset, horizon
        )

        if (
            alt_mass > self.ALIGNMENT_WARN_RATIO * declared_mass
            and alt_mass - declared_mass >= self.ALIGNMENT_WARN_GAP
        ):
            suggested = (
                "legacy"
                if self.kd_prepared_targets_alignment == "current"
                else "current"
            )
            LOG.warning(
                "KD teacher targets look mis-aligned: with "
                f"kd_prepared_targets_alignment={self.kd_prepared_targets_alignment!r} the "
                f"teacher puts {declared_mass:.1%} of its mass on the token each row "
                f"should describe (top-k containment {declared_containment:.1%}), but "
                f"{alt_mass:.1%} on the token {alt_offset - 1:+d} positions away "
                f"(containment {alt_containment:.1%}), over {n_valid} positions. "
                "If this dataset was prepared by an older axolotl, set "
                f"kd_prepared_targets_alignment: {suggested}"
            )

    def __call__(self, features, return_tensors=None, targets_realigned=False):
        if return_tensors is None:
            return_tensors = self.return_tensors

        if not targets_realigned:
            self.realign_prepared_targets(features)

        padding_side = self.tokenizer.padding_side
        max_len = 0

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

                for f in features:
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
            if padding_side != "right":
                raise ValueError(
                    "KD requires right padding: teacher targets are right-padded, so "
                    f"padding_side={padding_side!r} would shift them off the tokens "
                    "they describe"
                )

            # Extract and remove from features
            for f in features:
                target_logprobs_list.append(f.pop("target_logprobs"))
                target_token_ids_list.append(f.pop("target_token_ids"))
                target_mask_list.append(f.pop("target_mask"))

            # Determine max lengths
            max_teacher_seq_len = max_len or max(
                len(seq) for seq in target_logprobs_list
            )
            max_k = max(len(seq_k) for seq in target_logprobs_list for seq_k in seq)

            padded_target_logprobs = []
            padded_target_token_ids = []
            padded_teacher_mask_list = []

            for t_logprobs, t_ids, t_mask in zip(
                target_logprobs_list,
                target_token_ids_list,
                target_mask_list,
                strict=True,
            ):
                t_logprobs_padded = []
                t_ids_padded = []
                t_mask_padded = []

                for lp, ids, mask in zip(t_logprobs, t_ids, t_mask, strict=True):
                    lp_len = len(lp)
                    if lp_len < max_k:
                        # Use -1e9 for padding logprobs and 0 for token_ids
                        pad_len = max_k - lp_len
                        lp = lp + [-1e9] * pad_len
                        ids = ids + [0] * pad_len
                        mask = mask + [0] * pad_len
                    else:
                        lp = lp[:max_k]
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
            self._check_target_alignment(
                features.get("labels"),
                padded_target_token_ids,
                padded_target_logprobs,
                padded_teacher_mask_list,
            )

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

    def __call__(self, features, return_tensors=None, targets_realigned=False):
        """
        Expects that `features` could be either:
          - a single list of dicts, OR
          - a list of lists of dicts (the "sub-batches" to be packed).
        """
        # 1) If we are *not* dealing with multiple sequences per batch element,
        #    just pass straight to parent.
        if not isinstance(features[0], list):
            return super().__call__(
                features,
                return_tensors=return_tensors,
                targets_realigned=targets_realigned,
            )

        # realign per sequence, before the sub-batches are concatenated, so a shifted row
        # never crosses into the neighbouring sequence
        if not targets_realigned:
            for sub_features in features:
                self.realign_prepared_targets(sub_features)
            targets_realigned = True

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
                        if field_name in feat and isinstance(
                            feat[field_name], (list, torch.Tensor)
                        ):
                            if isinstance(feat[field_name][0], (dict, str)):
                                continue
                            arr = np.array(feat[field_name])
                            arrays.append(arr)
                    if arrays:
                        out_features[i][field_name] = np.concatenate(arrays)

        # 3) Now call the parent collator, which will do:
        #    - padding of labels/position_ids
        #    - KD-specific padding for target_logprobs, target_token_ids, etc.
        #    - final conversion to return_tensors
        return super().__call__(
            out_features,
            return_tensors=return_tensors,
            targets_realigned=targets_realigned,
        )
