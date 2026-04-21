# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Convert axolotl batch tensors to Tinker/Hatchery Datum format.

Both Tinker and Hatchery expect the client to apply the causal LM shift:

  Original tokens:  [t0, t1, t2, ..., t_{L-1}]
  model_input:      [t0, t1, ..., t_{L-2}]       (last token dropped)
  target_tokens:    [t1, t2, ..., t_{L-1}]        (first token dropped)
  weights:          [w1, w2, ..., w_{L-1}]        (aligned to targets)

At position i, the model sees t_i and predicts target_tokens[i] = t_{i+1}.
"""

from __future__ import annotations

from typing import Any

import torch


def _tensor_to_wire(t: torch.Tensor) -> dict[str, Any]:
    """Serialize a tensor to the TensorData wire dict."""
    flat = t.detach().cpu().flatten()
    dtype_map = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int64: "int64",
        torch.int32: "int32",
    }
    return {
        "dtype": dtype_map.get(flat.dtype, "float32"),
        "shape": list(t.shape),
        "data": flat.tolist(),
    }


def _make_datum(
    tokens: list[int],
    loss_fn_inputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Build a Datum as a plain dict (wire-compatible with both Tinker and Hatchery)."""
    return {
        "model_input": {
            "chunks": [{"type": "encoded_text", "tokens": tokens}],
        },
        "loss_fn_inputs": {
            key: _tensor_to_wire(tensor) for key, tensor in loss_fn_inputs.items()
        },
    }


def datums_to_tinker(datums: list[dict[str, Any]]):
    """Wrap plain-dict datums into tinker.types.Datum objects.

    Both the Tinker SDK and updated Hatchery client accept these.
    """
    import tinker.types as tt

    result = []
    for d in datums:
        tokens = d["model_input"]["chunks"][0]["tokens"]
        tinker_inputs = {}
        for key, wire in d["loss_fn_inputs"].items():
            tinker_inputs[key] = tt.TensorData(
                data=wire["data"],
                dtype=wire["dtype"],
                shape=wire["shape"],
            )
        result.append(
            tt.Datum(
                model_input=tt.ModelInput.from_ints(tokens),
                loss_fn_inputs=tinker_inputs,
            )
        )
    return result


def batch_to_datums_sft(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Convert an axolotl SFT batch to Datum dicts with causal shift."""
    batch_size = input_ids.size(0)
    datums = []

    for i in range(batch_size):
        ids = input_ids[i]
        lbl = labels[i]

        if attention_mask is not None:
            seq_len = int(attention_mask[i].sum().item())
            ids = ids[:seq_len]
            lbl = lbl[:seq_len]

        model_tokens = ids[:-1].tolist()
        shifted_labels = lbl[1:]

        target_tokens = shifted_labels.clone()
        weights = (shifted_labels != -100).float()
        target_tokens[target_tokens == -100] = 0

        datums.append(
            _make_datum(
                model_tokens,
                {
                    "target_tokens": target_tokens,
                    "weights": weights,
                },
            )
        )

    return datums


def batch_to_datums_rl(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> list[dict[str, Any]]:
    """Convert an RL batch to importance_sampling/ppo Datum dicts with causal shift."""
    batch_size = input_ids.size(0)
    datums = []

    for i in range(batch_size):
        ids = input_ids[i]
        lbl = labels[i]

        nonzero = ids.nonzero()
        if nonzero.numel() > 0:
            seq_len = nonzero[-1].item() + 1
        else:
            seq_len = ids.size(0)
        ids = ids[:seq_len]
        lbl = lbl[:seq_len]
        lp = logprobs[i, :seq_len]
        adv = advantages[i, :seq_len]

        model_tokens = ids[:-1].tolist()

        target_tokens = lbl[1:].clone()
        target_tokens[target_tokens == -100] = 0

        datums.append(
            _make_datum(
                model_tokens,
                {
                    "target_tokens": target_tokens,
                    "logprobs": lp[1:],
                    "advantages": adv[1:],
                },
            )
        )

    return datums
