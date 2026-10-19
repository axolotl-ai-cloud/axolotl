"""Spec-driven native diffusion generation without changing legacy samplers."""

from __future__ import annotations

import importlib
from typing import Any, Callable

import torch

from axolotl.model_support import get_model_support, resolve_model_support
from axolotl.model_support.diffusion import GenerationAdapter


def _result(tokenizer: Any, original: torch.Tensor, sequences: torch.Tensor) -> dict:
    ids = sequences[0].detach().cpu().tolist()
    original_ids = original[0].detach().cpu().tolist()
    return {
        "original": tokenizer.decode(original_ids, skip_special_tokens=True),
        "masked": None,
        "generated": tokenizer.decode(ids, skip_special_tokens=True),
        "mask_ratio": None,
        "masked_tokens": 0,
        "total_tokens": len(ids),
        "generated_ids": ids,
        "masked_positions": [],
        "orig_ids": original_ids,
        "formatted": tokenizer.decode(ids, skip_special_tokens=True),
    }


def uses_native_generation(model: torch.nn.Module) -> bool:
    config = getattr(model, "config", None)
    support = resolve_model_support(
        get_model_support(getattr(config, "model_type", None))
    )
    return bool(support and support.diffusion)


def supports_native_infill(model: torch.nn.Module) -> bool:
    """Whether this native model's audited generator accepts masked inputs."""
    config = getattr(model, "config", None)
    support = resolve_model_support(
        get_model_support(getattr(config, "model_type", None))
    )
    return bool(
        support
        and support.diffusion
        and support.diffusion.generation_adapter
        in {
            GenerationAdapter.ENCODER_CANVAS,
            GenerationAdapter.DREAM,
            GenerationAdapter.FULL_SEQUENCE,
        }
    )


def _dream_generation_config(
    model: torch.nn.Module,
    *,
    max_new_tokens: int,
    num_diffusion_steps: int,
    temperature: float,
) -> Any:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        model = get_base_model()
    model_class = type(model)
    source_class = next(
        (
            base
            for base in model_class.__mro__[1:]
            if base.__module__ != __name__
            and base.__module__ != "axolotl.model_support.dream.compat"
        ),
        model_class,
    )
    generation_module = importlib.import_module(
        f"{source_class.__module__.rsplit('.', 1)[0]}.generation_utils"
    )
    config = model.config
    return generation_module.DreamGenerationConfig(
        max_new_tokens=max_new_tokens,
        steps=num_diffusion_steps,
        temperature=temperature,
        mask_token_id=getattr(config, "mask_token_id", None),
        pad_token_id=getattr(config, "pad_token_id", None),
        bos_token_id=getattr(config, "bos_token_id", None),
        eos_token_id=getattr(config, "eos_token_id", None),
    )


def _dream_infill(
    model: torch.nn.Module,
    tokenizer: Any,
    original_sequence: torch.Tensor,
    num_diffusion_steps: int,
    temperature: float,
    target_mask_ratio: float | None,
) -> dict:
    mask_token_id = getattr(model.config, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("Dream native infill requires config.mask_token_id.")
    mask = _infill_mask(
        original_sequence,
        torch.ones_like(original_sequence, dtype=torch.bool),
        target_mask_ratio,
        None,
    )
    masked = original_sequence.masked_fill(mask, int(mask_token_id))
    sequences = model.diffusion_generate(
        inputs=masked,
        generation_config=_dream_generation_config(
            model,
            max_new_tokens=1,
            num_diffusion_steps=num_diffusion_steps,
            temperature=temperature,
        ),
    )
    sequence = getattr(sequences, "sequences", sequences)[
        :, : original_sequence.size(1)
    ]
    result = _result(tokenizer, original_sequence, sequence)
    masked_positions = torch.where(mask[0])[0].detach().cpu().tolist()
    result.update(
        masked=tokenizer.decode(
            masked[0].detach().cpu().tolist(), skip_special_tokens=False
        ),
        mask_ratio=len(masked_positions) / original_sequence.size(1),
        masked_tokens=len(masked_positions),
        masked_positions=masked_positions,
    )
    return result


def _infill_mask(
    sequence: torch.Tensor,
    eligible_mask: torch.Tensor,
    target_mask_ratio: float | None,
    explicit_mask: torch.Tensor | None,
) -> torch.Tensor:
    if explicit_mask is not None:
        if (
            explicit_mask.shape != sequence.shape
            or explicit_mask.dtype is not torch.bool
        ):
            raise ValueError("native infill_mask must be bool and match input_ids.")
        if (explicit_mask & ~eligible_mask).any():
            raise ValueError(
                "native infill_mask selects positions outside the active canvas."
            )
        if not explicit_mask.any():
            raise ValueError(
                "native infill_mask must select at least one canvas position."
            )
        return explicit_mask
    eligible_positions = eligible_mask[0].nonzero().flatten()
    if target_mask_ratio is None:
        target_mask_ratio = torch.rand(1).item() * 0.6 + 0.1
    if not 0.0 <= float(target_mask_ratio) <= 1.0:
        raise ValueError("native infill target_mask_ratio must be in [0, 1].")
    count = max(1, int(eligible_positions.numel() * float(target_mask_ratio)))
    shuffled = eligible_positions[
        torch.randperm(eligible_positions.numel(), device=sequence.device)[:count]
    ]
    selected = torch.zeros_like(sequence, dtype=torch.bool)
    selected[:, shuffled] = True
    return selected


def _native_infill_result(
    tokenizer: Any,
    original: torch.Tensor,
    masked: torch.Tensor,
    generated: torch.Tensor,
    selected: torch.Tensor,
    active_start: int,
) -> dict:
    if not torch.equal(generated[~selected], original[~selected]):
        raise RuntimeError("native infill changed a pinned token.")
    result = _result(tokenizer, original, generated)
    positions = selected[0].nonzero().flatten().detach().cpu().tolist()
    result.update(
        masked=tokenizer.decode(
            masked[0].detach().cpu().tolist(), skip_special_tokens=False
        ),
        mask_ratio=len(positions) / (original.shape[1] - active_start),
        masked_tokens=len(positions),
        masked_positions=positions,
        active_canvas_start=active_start,
        active_canvas_end=original.shape[1],
    )
    return result


def _gemma_infill(
    model: torch.nn.Module,
    tokenizer: Any,
    original: torch.Tensor,
    num_diffusion_steps: int,
    target_mask_ratio: float | None,
    explicit_mask: torch.Tensor | None,
) -> dict:
    if num_diffusion_steps < 1:
        raise ValueError("DiffusionGemma denoising_steps must be at least 1.")
    sequence_length = original.shape[1]
    if sequence_length < 2:
        raise ValueError("DiffusionGemma infill requires a context token and canvas.")
    config = model.config
    max_context = getattr(config.text_config, "max_position_embeddings", None)
    if max_context is not None and sequence_length > int(max_context):
        raise ValueError(
            "DiffusionGemma infill input exceeds its configured context window."
        )
    canvas_limit = min(int(config.canvas_length), 256)
    canvas_length = min(sequence_length - 1, canvas_limit)
    active_start = sequence_length - canvas_length
    eligible = torch.zeros_like(original, dtype=torch.bool)
    eligible[:, active_start:] = True
    selected = _infill_mask(original, eligible, target_mask_ratio, explicit_mask)
    canvas = original[:, active_start:].clone()
    selected_canvas = selected[:, active_start:]
    canvas[selected_canvas] = torch.randint(
        0,
        int(config.text_config.vocab_size),
        (int(selected_canvas.sum()),),
        device=canvas.device,
        dtype=canvas.dtype,
    )
    from axolotl.core.trainers.diffusion_lm.attention import (
        decoder_prefix_canvas_mask,
        encoder_causal_mask,
    )

    prefix = original[:, :active_start]
    encoder_docs = torch.zeros_like(prefix)
    canvas_docs = torch.zeros_like(canvas)
    encoder_valid = torch.ones_like(prefix, dtype=torch.bool)
    canvas_valid = torch.ones_like(canvas, dtype=torch.bool)
    encoder_positions = torch.arange(
        active_start, device=original.device, dtype=torch.long
    )[None]
    decoder_positions = torch.arange(
        active_start, sequence_length, device=original.device, dtype=torch.long
    )[None]
    sliding_window = int(getattr(config.text_config, "sliding_window", 1024))
    prefix_lengths = torch.tensor([active_start], device=original.device)
    encoder_masks = {
        "full_attention": encoder_causal_mask(
            encoder_docs, encoder_valid, encoder_positions
        ),
        "sliding_attention": encoder_causal_mask(
            encoder_docs, encoder_valid, encoder_positions, sliding_window
        ),
    }
    decoder_masks = {
        "full_attention": decoder_prefix_canvas_mask(
            encoder_docs,
            canvas_docs,
            encoder_valid,
            canvas_valid,
            prefix_lengths,
            encoder_positions,
            logical_ids=torch.zeros(1, device=original.device, dtype=torch.long),
        ),
        "sliding_attention": decoder_prefix_canvas_mask(
            encoder_docs,
            canvas_docs,
            encoder_valid,
            canvas_valid,
            prefix_lengths,
            encoder_positions,
            sliding_window,
            logical_ids=torch.zeros(1, device=original.device, dtype=torch.long),
        ),
    }
    output = model(
        encoder_input_ids=prefix,
        encoder_attention_mask=encoder_masks,
        encoder_position_ids=encoder_positions,
        decoder_input_ids=canvas,
        decoder_attention_mask=decoder_masks,
        decoder_position_ids=decoder_positions,
        unroll_steps=num_diffusion_steps,
        grad_through_steps=False,
        k1_conditioning_mask=selected_canvas,
        recurrent_conditioning_mask=selected_canvas,
        update_mask=selected_canvas,
    )
    denoised = getattr(output, "denoised_input_ids", None)
    if denoised is None:
        raise TypeError(
            "DiffusionGemma native infill requires the Axolotl model adapter."
        )
    generated = torch.cat((prefix, denoised), dim=1)
    masked = torch.cat((prefix, canvas), dim=1)
    return _native_infill_result(
        tokenizer, original, masked, generated, selected, active_start
    )


def _nemotron_infill(
    model: torch.nn.Module,
    tokenizer: Any,
    original: torch.Tensor,
    num_diffusion_steps: int,
    temperature: float,
    target_mask_ratio: float | None,
    explicit_mask: torch.Tensor | None,
) -> dict:
    if num_diffusion_steps < 1:
        raise ValueError("Nemotron denoising_steps must be at least 1.")
    maximum = int(getattr(model.config, "max_position_embeddings", 262144))
    if original.shape[1] > maximum:
        raise ValueError("Nemotron infill input exceeds its configured context window.")
    eligible = torch.ones_like(original, dtype=torch.bool)
    selected = _infill_mask(original, eligible, target_mask_ratio, explicit_mask)
    if not hasattr(model, "infill_with_denoising_steps"):
        raise TypeError("Nemotron native infill requires the audited Axolotl adapter.")
    generated, _ = model.infill_with_denoising_steps(
        original, selected, num_diffusion_steps, temperature=temperature
    )
    masked = original.masked_fill(selected, int(model.mask_token_id))
    return _native_infill_result(tokenizer, original, masked, generated, selected, 0)


def generate_native_samples(
    model: torch.nn.Module,
    tokenizer: Any,
    dataloader: Any | None = None,
    num_generation_samples: int = 3,
    max_length: int = 100,
    num_diffusion_steps: int = 128,
    temperature: float = 0.0,
    mask_token_id: int = 0,
    **_: Any,
) -> list[dict]:
    """Generate native completion samples from contiguous SFT label suffixes."""
    if dataloader is None:
        return []
    from axolotl.core.trainers.diffusion_lm.generation import (
        _sample_sequences_from_dataloader,
    )

    unwrapped_model = model.module if hasattr(model, "module") else model
    was_training = unwrapped_model.training
    unwrapped_model.eval()
    try:
        try:
            device = next(unwrapped_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        samples = _sample_sequences_from_dataloader(
            dataloader, num_generation_samples, max_length, device
        )
        generations = []
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("labels") is None:
                continue
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            answer_positions = labels[0] != -100
            if not answer_positions.any():
                continue
            first_answer = int(answer_positions.nonzero()[0].item())
            last_answer = int(answer_positions.nonzero()[-1].item())
            if (
                first_answer == 0
                or not answer_positions[first_answer : last_answer + 1].all()
            ):
                continue
            generations.append(
                generate_for_model(
                    unwrapped_model,
                    tokenizer,
                    input_ids[:, :first_answer],
                    num_diffusion_steps,
                    temperature,
                    mask_token_id,
                    legacy_generate=lambda *args, **kwargs: (_ for _ in ()).throw(
                        AssertionError("native sample generation called legacy sampler")
                    ),
                    mode="completion",
                    completion_tokens=last_answer - first_answer + 1,
                )
            )
        return generations
    finally:
        unwrapped_model.train(was_training)


@torch.inference_mode()
def generate_for_model(
    model: torch.nn.Module,
    tokenizer: Any,
    original_sequence: torch.Tensor,
    num_diffusion_steps: int,
    temperature: float,
    mask_token_id: int,
    *,
    legacy_generate: Callable[..., dict],
    **kwargs: Any,
) -> dict:
    """Use a descriptor's native generator, or delegate unchanged legacy behavior."""
    config = getattr(model, "config", None)
    support = resolve_model_support(
        get_model_support(getattr(config, "model_type", None))
    )
    adapter = (
        support.diffusion.generation_adapter if support and support.diffusion else None
    )
    completion_tokens = int(kwargs.get("completion_tokens", 0))
    max_new_tokens = completion_tokens or int(kwargs.get("max_length", 128))
    mode = kwargs.get("mode", "completion")
    if mode == "random":
        if adapter is GenerationAdapter.DREAM:
            return _dream_infill(
                model,
                tokenizer,
                original_sequence,
                num_diffusion_steps,
                temperature,
                kwargs.get("target_mask_ratio"),
            )
        if adapter is GenerationAdapter.ENCODER_CANVAS:
            return _gemma_infill(
                model,
                tokenizer,
                original_sequence,
                num_diffusion_steps,
                kwargs.get("target_mask_ratio"),
                kwargs.get("infill_mask"),
            )
        if adapter is GenerationAdapter.FULL_SEQUENCE:
            return _nemotron_infill(
                model,
                tokenizer,
                original_sequence,
                num_diffusion_steps,
                temperature,
                kwargs.get("target_mask_ratio"),
                kwargs.get("infill_mask"),
            )
        if adapter is not None:
            raise ValueError(
                "This native diffusion generator supports completion mode only."
            )
    elif adapter is not None and mode != "completion":
        raise ValueError("Native diffusion generation supports completion mode only.")
    if adapter is GenerationAdapter.ENCODER_CANVAS:
        sequences = model.generate(
            input_ids=original_sequence,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_denoising_steps=num_diffusion_steps,
        )
        return _result(
            tokenizer, original_sequence, getattr(sequences, "sequences", sequences)
        )
    if adapter is GenerationAdapter.DREAM:
        sequences = model.diffusion_generate(
            inputs=original_sequence,
            generation_config=_dream_generation_config(
                model,
                max_new_tokens=max_new_tokens,
                num_diffusion_steps=num_diffusion_steps,
                temperature=temperature,
            ),
        )
        return _result(
            tokenizer, original_sequence, getattr(sequences, "sequences", sequences)
        )
    if adapter is GenerationAdapter.FULL_SEQUENCE:
        block_size = int(getattr(config, "block_size", 1))
        native_length = ((max_new_tokens + block_size - 1) // block_size) * block_size
        denoising_steps = (
            block_size if num_diffusion_steps is None else int(num_diffusion_steps)
        )
        if not hasattr(model, "generate_with_denoising_steps"):
            raise ValueError(
                "Nemotron native generation requires the audited configurable-step adapter."
            )
        sequences, _ = model.generate_with_denoising_steps(
            original_sequence,
            max_new_tokens=native_length,
            block_length=block_size,
            denoising_steps=denoising_steps,
            temperature=temperature,
            eos_token_id=getattr(config, "eos_token_id", None),
        )
        return _result(
            tokenizer,
            original_sequence,
            sequences[:, : original_sequence.shape[1] + max_new_tokens],
        )
    return legacy_generate(
        model,
        tokenizer,
        original_sequence=original_sequence,
        num_diffusion_steps=num_diffusion_steps,
        temperature=temperature,
        mask_token_id=mask_token_id,
        **kwargs,
    )
