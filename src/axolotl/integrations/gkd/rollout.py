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
Axis B (rollout) seam: recover the prompt (the ``-100``-masked prefix of an
Axolotl-tokenized sample) and left-pad it into the batch shape ``model.generate``
expects, so no bespoke collator is needed.
"""

from __future__ import annotations

import torch


def extract_prompt_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-padded ``(prompt_ids, prompt_attention_mask)`` from ``input_ids`` up to the
    first supervised label; attended tokens within that prefix are kept."""
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    rows = []
    for i in range(batch_size):
        supervised = (labels[i] != -100).nonzero(as_tuple=True)[0]
        cut = int(supervised[0]) if supervised.numel() else seq_len
        row = input_ids[i, :cut]
        if attention_mask is not None:
            row = row[attention_mask[i, :cut].bool()]
        rows.append(row)

    max_len = max((row.numel() for row in rows), default=1) or 1
    prompt_ids = torch.full(
        (batch_size, max_len), pad_token_id, dtype=input_ids.dtype, device=device
    )
    prompt_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        if row.numel():
            prompt_ids[i, max_len - row.numel() :] = row
            prompt_mask[i, max_len - row.numel() :] = 1
    return prompt_ids, prompt_mask
