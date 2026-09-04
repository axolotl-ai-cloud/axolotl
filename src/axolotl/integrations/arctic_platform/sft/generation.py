# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
"""Remote sample generation for Arctic SFT (vLLM sampling job)."""

from __future__ import annotations

from typing import Any

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _prompt_texts_from_dataloader(
    tokenizer: Any,
    dataloader: Any,
    *,
    num_generation_samples: int,
    prompt_ratio: float,
) -> list[str]:
    """Mirror stock ``generate_samples`` prompt slicing; return decoded prompts."""
    prompts: list[str] = []
    for batch in dataloader:
        if len(prompts) >= num_generation_samples:
            break
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        batch_size = input_ids.shape[0]
        indices = torch.randperm(batch_size)[
            : num_generation_samples - len(prompts)
        ]
        for idx in indices:
            if len(prompts) >= num_generation_samples:
                break
            sequence = input_ids[idx]
            if attention_mask is not None:
                seq_len = attention_mask[idx].sum().item()
            else:
                seq_len = sequence.shape[0]
            if seq_len < 5:
                continue
            prompt_len = max(1, int(seq_len * prompt_ratio))
            prompt_ids = sequence[:prompt_len]
            prompts.append(
                tokenizer.decode(prompt_ids, skip_special_tokens=True)
            )
    return prompts


def generate_samples_remote(
    client: Any,
    tokenizer: Any,
    dataloader: Any,
    *,
    num_generation_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    do_sample: bool,
    prompt_ratio: float,
    colocate: bool,
) -> list[dict]:
    """Sync training→sampling, then vLLM-generate. Same return shape as stock SFT."""
    prompts = _prompt_texts_from_dataloader(
        tokenizer,
        dataloader,
        num_generation_samples=num_generation_samples,
        prompt_ratio=prompt_ratio,
    )
    if not prompts:
        LOG.warning("Arctic SFT generate_samples: no prompts collected from dataloader")
        return []

    sampling_params: dict[str, Any] = {
        "max_tokens": max_new_tokens,
        "temperature": temperature if do_sample else 0.0,
    }
    if do_sample:
        if top_p is not None:
            sampling_params["top_p"] = top_p
        if top_k is not None:
            sampling_params["top_k"] = top_k

    # When sharing GPUs with training, sleep the training engine around the
    # weight sync so sampling can use those devices.
    if colocate:
        client.sleep_training(mode="non_lp")
        client.sync_weights(cuda_ipc=True)
        client.sleep_training(mode="lp_params")
    else:
        client.sync_weights(cuda_ipc=False)

    try:
        raw = client.generate(prompts, sampling_params=sampling_params)
    finally:
        if colocate:
            try:
                client.wake_training()
            except Exception:  # noqa: BLE001
                LOG.exception("Arctic SFT: wake_training after generate failed")

    generations: list[dict] = []
    for prompt, result in zip(prompts, raw):
        # vLLM results are typically dicts with "text" / "outputs"; tolerate str.
        if isinstance(result, str):
            generated = result
        elif isinstance(result, dict):
            generated = result.get("text")
            if generated is None and result.get("outputs"):
                out0 = result["outputs"][0]
                generated = out0.get("text") if isinstance(out0, dict) else str(out0)
            if generated is None:
                generated = str(result)
        else:
            generated = str(result)
        generations.append(
            {
                "prompt": prompt,
                "generated": generated,
                "full_text": prompt + generated,
            }
        )
    return generations
