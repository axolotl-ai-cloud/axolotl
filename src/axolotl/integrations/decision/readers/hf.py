"""Spec-driven in-process structured reads for native diffusion models."""

from __future__ import annotations

import hashlib
import random
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import torch

from axolotl.integrations.diffusion.lm.backends.encoder_canvas import (
    EncoderCanvasBackend,
)
from axolotl.integrations.diffusion.lm.backends.full_sequence import (
    FullSequenceBackend,
)
from axolotl.integrations.diffusion.lm.batch import DiffusionBatch
from axolotl.integrations.diffusion.lm.unroll import run_unroll
from axolotl.model_support.diffusion import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    LogitAlignment,
    MaskTokenPolicy,
)

from ..collator import DecisionCanvasCollator
from ..records import DecisionCanvas
from .base import (
    DecisionRead,
    ReadDiagnostics,
    restricted_probabilities,
    validate_canvas,
)


class HFReader:
    """Read a decision canvas through the diffusion spec's native backend."""

    def __init__(
        self,
        *,
        vocab_size: int | None = None,
        mask_token_id: int | None = None,
        sliding_window: int | None = None,
        attention_backend: str = "flex_attention",
        kernel_options: dict[str, Any] | None = None,
        free_update_policy: str | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        if vocab_size is not None and vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if sliding_window is not None and sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        if attention_backend not in {"dense", "flex_attention", "varlen"}:
            raise ValueError(
                "attention_backend must be dense, flex_attention, or varlen"
            )
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.sliding_window = sliding_window
        self.attention_backend = attention_backend
        self.kernel_options = kernel_options
        if compute_dtype not in (None, torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(
                "compute_dtype must be float32, float16, bfloat16, or None"
            )
        self.compute_dtype = None if compute_dtype is torch.float32 else compute_dtype
        if free_update_policy not in (None, "argmax"):
            raise ValueError("unsupported free_update_policy")
        self.free_update_policy = free_update_policy

    def autocast_context(self, device: torch.device):
        return (
            torch.autocast(device.type, dtype=self.compute_dtype)
            if self.compute_dtype is not None
            else nullcontext()
        )

    def read(
        self,
        model,
        spec: DiffusionSpec,
        canvas: DecisionCanvas,
        *,
        steps: int = 1,
        seed: int | None = None,
        initial_canvas_ids: torch.Tensor | None = None,
        fixed_label_noise: bool = False,
        hold_label_noise: bool = False,
        diagnostics: bool = False,
    ) -> DecisionRead:
        """Score all label positions after exactly the requested denoising reads."""

        if steps < 1:
            raise ValueError("steps must be positive")
        if fixed_label_noise and hold_label_noise:
            raise ValueError(
                "fixed_label_noise and hold_label_noise are mutually exclusive"
            )
        validate_canvas(canvas)
        if self.attention_backend == "varlen":
            self._validate_varlen(model, spec)
        device = self._model_device(model)
        was_training = bool(getattr(model, "training", False))
        model.eval()
        try:
            with torch.inference_mode(), self.autocast_context(device):
                if spec.layout is DiffusionLayout.ENCODER_CANVAS:
                    logits, initial, final, forward_count = self._read_encoder_canvas(
                        model,
                        spec,
                        canvas,
                        steps,
                        seed,
                        initial_canvas_ids,
                        fixed_label_noise,
                        hold_label_noise,
                        device,
                    )
                elif spec.layout is DiffusionLayout.FULL_SEQUENCE:
                    logits, initial, final, forward_count = self._read_full_sequence(
                        model,
                        spec,
                        canvas,
                        steps,
                        seed,
                        initial_canvas_ids,
                        fixed_label_noise,
                        hold_label_noise,
                        device,
                    )
                else:
                    raise ValueError(f"unsupported diffusion layout: {spec.layout}")
        finally:
            model.train(was_training)
        label_positions = torch.as_tensor(
            canvas.label_positions, dtype=torch.long, device=logits.device
        )
        full_vocab_logprobs = torch.log_softmax(logits.float(), dim=-1)
        allowed_ids, candidate_mask, restricted_probs = restricted_probabilities(
            full_vocab_logprobs, canvas.allowed_ids
        )
        return DecisionRead(
            question_ids=tuple(canvas.question_ids),
            label_positions=label_positions,
            allowed_ids=allowed_ids,
            candidate_mask=candidate_mask,
            full_vocab_logprobs=full_vocab_logprobs,
            restricted_probs=restricted_probs,
            diagnostics=(
                ReadDiagnostics(
                    noise_seed=seed,
                    noise_kind=spec.noise.value,
                    update_policy=(
                        "free_slot_argmax_held_labels"
                        if self.free_update_policy == "argmax" and steps > 1
                        else (
                            "fixed_label_noise"
                            if fixed_label_noise
                            else (
                                "held_label_noise_with_sc"
                                if hold_label_noise
                                else "source_serving"
                            )
                        )
                    ),
                    steps=steps,
                    forward_count=forward_count,
                    initial_canvas_ids=initial,
                    final_canvas_ids=final,
                    slot_init_policy=(
                        "fresh_read_seed_v1"
                        if self.free_update_policy == "argmax"
                        and initial_canvas_ids is None
                        and seed is not None
                        else "fresh_read_rng_v1"
                        if self.free_update_policy == "argmax"
                        and initial_canvas_ids is None
                        else "explicit_canvas_v1"
                        if self.free_update_policy == "argmax"
                        else "prepared_canvas_v0"
                    ),
                )
                if diagnostics
                else None
            ),
        )

    def read_batch(
        self,
        model,
        spec: DiffusionSpec,
        canvases: list[DecisionCanvas] | tuple[DecisionCanvas, ...],
        *,
        steps: int = 1,
        seeds: list[int | None] | tuple[int | None, ...] | None = None,
        hold_label_noise: bool = False,
        diagnostics: bool = False,
    ) -> tuple[DecisionRead, ...]:
        """Read independent full-sequence canvases in one packed physical row."""

        if not canvases:
            raise ValueError("read_batch requires at least one canvas")
        if seeds is not None and len(seeds) != len(canvases):
            raise ValueError("read_batch seeds must align with canvases")
        if len(canvases) == 1:
            return (
                self.read(
                    model,
                    spec,
                    canvases[0],
                    steps=steps,
                    seed=None if seeds is None else seeds[0],
                    hold_label_noise=hold_label_noise,
                    diagnostics=diagnostics,
                ),
            )
        if steps < 1:
            raise ValueError("steps must be positive")
        if spec.layout is not DiffusionLayout.FULL_SEQUENCE:
            raise NotImplementedError(
                "batched decision reads support only full-sequence diffusion"
            )
        if spec.self_conditioning:
            raise NotImplementedError(
                "batched decision reads do not support self-conditioning diffusion"
            )
        if self.attention_backend == "varlen":
            self._validate_varlen(model, spec)
        if seeds is None:
            seeds = (None,) * len(canvases)
        for canvas in canvases:
            validate_canvas(canvas)
        device = self._model_device(model)
        was_training = bool(getattr(model, "training", False))
        model.eval()
        try:
            with torch.inference_mode(), self.autocast_context(device):
                results = self._read_full_sequence_batch(
                    model, spec, canvases, steps, seeds, hold_label_noise, device
                )
        finally:
            model.train(was_training)
        reads: list[DecisionRead] = []
        for canvas, seed, (logits, initial, final) in zip(
            canvases, seeds, results, strict=True
        ):
            full_vocab_logprobs = torch.log_softmax(logits.float(), dim=-1)
            allowed_ids, candidate_mask, restricted_probs = restricted_probabilities(
                full_vocab_logprobs, canvas.allowed_ids
            )
            reads.append(
                DecisionRead(
                    question_ids=tuple(canvas.question_ids),
                    label_positions=torch.as_tensor(
                        canvas.label_positions, dtype=torch.long, device=logits.device
                    ),
                    allowed_ids=allowed_ids,
                    candidate_mask=candidate_mask,
                    full_vocab_logprobs=full_vocab_logprobs,
                    restricted_probs=restricted_probs,
                    diagnostics=(
                        ReadDiagnostics(
                            noise_seed=seed,
                            noise_kind=spec.noise.value,
                            update_policy=(
                                "free_slot_argmax_held_labels"
                                if self.free_update_policy == "argmax" and steps > 1
                                else "held_label_noise_with_sc"
                                if hold_label_noise
                                else "source_serving"
                            ),
                            steps=steps,
                            forward_count=steps,
                            initial_canvas_ids=initial,
                            final_canvas_ids=final,
                            slot_init_policy=(
                                "fresh_read_seed_v1"
                                if self.free_update_policy == "argmax"
                                and seed is not None
                                else "fresh_read_rng_v1"
                                if self.free_update_policy == "argmax"
                                else "prepared_canvas_v0"
                            ),
                        )
                        if diagnostics
                        else None
                    ),
                )
            )
        return tuple(reads)

    def _read_full_sequence_batch(
        self,
        model,
        spec: DiffusionSpec,
        canvases,
        steps: int,
        seeds,
        hold_label_noise: bool,
        device: torch.device,
    ):
        vocab_size = self._vocab_size(model)
        pieces: list[torch.Tensor] = []
        documents: list[torch.Tensor] = []
        validity: list[torch.Tensor] = []
        updates: list[torch.Tensor] = []
        offsets: list[tuple[int, int]] = []
        initials: list[torch.Tensor] = []
        cursor = 0
        for document, (canvas, seed) in enumerate(zip(canvases, seeds, strict=True)):
            prompt = torch.as_tensor(canvas.prompt_ids, dtype=torch.long, device=device)
            clean = torch.as_tensor(canvas.canvas_ids, dtype=torch.long, device=device)
            state = self._noise_labels(
                clean[None],
                canvas.label_positions,
                spec,
                vocab_size,
                seed,
                self._mask_token_id(model, spec),
                None,
                canvas.pinned_mask,
            )[0]
            state = self._fresh_free_slots(
                state[None],
                canvas,
                spec,
                vocab_size,
                seed,
                self._mask_token_id(model, spec),
                True,
            )[0]
            initials.append(state.clone())
            length = prompt.numel() + state.numel()
            pieces.append(torch.cat((prompt, state)))
            documents.append(
                torch.full((length,), document, dtype=torch.long, device=device)
            )
            validity.append(
                torch.cat(
                    (
                        torch.ones_like(prompt, dtype=torch.bool),
                        torch.as_tensor(
                            canvas.semantic_mask, dtype=torch.bool, device=device
                        ),
                    )
                )
            )
            update = torch.zeros(length, dtype=torch.bool, device=device)
            slots = torch.as_tensor(canvas.slot_mask, dtype=torch.bool, device=device)
            pinned = torch.as_tensor(
                canvas.pinned_mask, dtype=torch.bool, device=device
            )
            if steps > 1 and self.free_update_policy == "argmax":
                update[prompt.numel() :] = slots & ~pinned
            elif not hold_label_noise:
                labels = torch.as_tensor(
                    canvas.label_positions, dtype=torch.long, device=device
                )
                update[prompt.numel() + labels[~pinned[labels]]] = True
            updates.append(update)
            offsets.append((cursor, prompt.numel()))
            cursor += length
        backend = FullSequenceBackend(
            mask_token_id=self._mask_token_id(model, spec),
            attention_backend=self.attention_backend,
        )
        packed = backend.pack(
            torch.cat(pieces)[None],
            torch.cat(documents)[None],
            torch.cat(validity)[None],
        )
        update_mask = torch.cat(updates)[None]
        if packed["input_ids"].shape[1] != update_mask.shape[1]:
            update_mask = torch.nn.functional.pad(
                update_mask, (0, packed["input_ids"].shape[1] - update_mask.shape[1])
            )
        outputs, aligned, final_state = run_unroll(
            state=packed["input_ids"],
            update_mask=update_mask,
            steps=steps,
            grad_through_steps=False,
            supports_self_conditioning=spec.self_conditioning,
            k1_conditioning_mask=torch.zeros_like(update_mask),
            recurrent_conditioning_mask=torch.zeros_like(update_mask),
            forward_step=lambda state, _conditioning, _conditioning_mask: (
                backend.forward(
                    model, packed, state, kernel_options=self.kernel_options
                )
            ),
            logits_from_outputs=lambda output: backend.canvas_logits(
                output, packed, aligned=spec.logit_alignment is LogitAlignment.ALIGNED
            ),
            update_state=backend.update,
            pilot_for_single_step=False,
        )
        del outputs
        final_state = backend.update(final_state, aligned, update_mask)
        result = []
        for canvas, initial, (offset, prompt_length) in zip(
            canvases, initials, offsets, strict=True
        ):
            labels = torch.as_tensor(
                canvas.label_positions, dtype=torch.long, device=device
            )
            canvas_start = offset + prompt_length
            result.append(
                (
                    aligned[0, canvas_start + labels],
                    initial,
                    final_state[
                        0, canvas_start : canvas_start + len(canvas.canvas_ids)
                    ],
                )
            )
        return result

    def _read_encoder_canvas(
        self,
        model,
        spec: DiffusionSpec,
        canvas: DecisionCanvas,
        steps: int,
        seed: int | None,
        initial_canvas_ids: torch.Tensor | None,
        fixed_label_noise: bool,
        hold_label_noise: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch = self._encoder_canvas_batch(canvas).to(device)
        vocab_size = self._vocab_size(model)
        config = self._model_config(model)
        text_config = getattr(config, "text_config", config)
        backend = EncoderCanvasBackend(
            vocab_size=vocab_size,
            sliding_window=self.sliding_window
            or int(getattr(text_config, "sliding_window", 1024)),
            attention_backend=self.attention_backend,
        )
        packed = backend.pack(batch)
        state = self._noise_labels(
            batch.canvas_clean_ids,
            canvas.label_positions,
            spec,
            vocab_size,
            seed,
            self._mask_token_id(model, spec),
            initial_canvas_ids,
            canvas.pinned_mask,
        )
        state = self._fresh_free_slots(
            state,
            canvas,
            spec,
            vocab_size,
            seed,
            self._mask_token_id(model, spec),
            initial_canvas_ids is None,
        )
        initial = state.clone()
        label_mask = self._canvas_label_mask(
            packed.canvas_clean_ids.shape, canvas.label_positions, packed.device
        )
        update_mask = packed.canvas_update_mask & ~packed.canvas_input_pinned_mask
        slot_mask = (
            torch.as_tensor(canvas.slot_mask, dtype=torch.bool, device=packed.device)[
                None
            ]
            & packed.canvas_semantic_validity
        )
        k1_conditioning_mask = packed.canvas_sc_eligible_mask
        recurrent_conditioning_mask = k1_conditioning_mask
        free_slots = slot_mask & ~packed.canvas_input_pinned_mask
        if self.free_update_policy == "argmax" and steps > 1:
            update_mask = free_slots
            recurrent_conditioning_mask = k1_conditioning_mask | slot_mask
        elif fixed_label_noise:
            update_mask = torch.zeros_like(update_mask)
            k1_conditioning_mask = packed.canvas_sc_eligible_mask & ~label_mask
            recurrent_conditioning_mask = k1_conditioning_mask
        elif hold_label_noise:
            update_mask = torch.zeros_like(update_mask)
            recurrent_conditioning_mask = k1_conditioning_mask | slot_mask
        else:
            recurrent_conditioning_mask = k1_conditioning_mask | slot_mask
        outputs = backend.forward(
            model,
            packed,
            state,
            unroll_steps=steps,
            grad_through_steps=False,
            pilot_for_single_step=False,
            k1_conditioning_mask=k1_conditioning_mask,
            recurrent_conditioning_mask=recurrent_conditioning_mask,
            update_mask=update_mask,
            kernel_options=self.kernel_options,
        )
        label_positions = torch.as_tensor(
            canvas.label_positions, dtype=torch.long, device=packed.device
        )
        logits = outputs.logits[0, label_positions]
        final = getattr(outputs, "denoised_input_ids", None)
        if final is None:
            final = backend.update(state, outputs.logits, update_mask)
        forward_count = steps
        return logits, initial[0], final[0], forward_count

    def _read_full_sequence(
        self,
        model,
        spec: DiffusionSpec,
        canvas: DecisionCanvas,
        steps: int,
        seed: int | None,
        initial_canvas_ids: torch.Tensor | None,
        fixed_label_noise: bool,
        hold_label_noise: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        vocab_size = self._vocab_size(model)
        prompt = torch.as_tensor(canvas.prompt_ids, dtype=torch.long, device=device)
        clean_canvas = torch.as_tensor(
            canvas.canvas_ids, dtype=torch.long, device=device
        )
        state_canvas = self._noise_labels(
            clean_canvas[None],
            canvas.label_positions,
            spec,
            vocab_size,
            seed,
            self._mask_token_id(model, spec),
            initial_canvas_ids,
            canvas.pinned_mask,
        )[0]
        state_canvas = self._fresh_free_slots(
            state_canvas[None],
            canvas,
            spec,
            vocab_size,
            seed,
            self._mask_token_id(model, spec),
            initial_canvas_ids is None,
        )[0]
        input_ids = torch.cat((prompt, state_canvas))[None]
        prompt_length = prompt.numel()
        canvas_semantic = torch.as_tensor(
            canvas.semantic_mask, dtype=torch.bool, device=device
        )
        semantic_validity = torch.cat(
            (torch.ones_like(prompt, dtype=torch.bool), canvas_semantic)
        )[None]
        document_ids = torch.where(
            semantic_validity,
            torch.zeros_like(input_ids),
            torch.full_like(input_ids, -1),
        )
        backend = FullSequenceBackend(
            mask_token_id=self._mask_token_id(model, spec),
            attention_backend=self.attention_backend,
        )
        packed = backend.pack(input_ids, document_ids, semantic_validity)
        update_mask = torch.zeros_like(packed["input_ids"], dtype=torch.bool)
        slot_mask = torch.as_tensor(canvas.slot_mask, device=device, dtype=torch.bool)
        pinned_canvas = torch.as_tensor(
            canvas.pinned_mask, device=device, dtype=torch.bool
        )
        free_slots = slot_mask & ~pinned_canvas
        if steps > 1 and self.free_update_policy == "argmax":
            update_mask[:, prompt_length:] = free_slots[None]
        elif not fixed_label_noise and not hold_label_noise:
            labels = torch.as_tensor(
                canvas.label_positions, device=device, dtype=torch.long
            )
            pinned = torch.as_tensor(
                canvas.pinned_mask, device=device, dtype=torch.bool
            )
            update_mask[
                0,
                prompt_length + labels[~pinned[labels]],
            ] = True
        outputs, aligned, final_state = run_unroll(
            state=packed["input_ids"],
            update_mask=update_mask,
            steps=steps,
            grad_through_steps=False,
            supports_self_conditioning=spec.self_conditioning,
            k1_conditioning_mask=torch.zeros_like(update_mask),
            recurrent_conditioning_mask=torch.zeros_like(update_mask),
            forward_step=lambda state, _conditioning, _conditioning_mask: (
                backend.forward(
                    model, packed, state, kernel_options=self.kernel_options
                )
            ),
            logits_from_outputs=lambda output: backend.canvas_logits(
                output,
                packed,
                aligned=spec.logit_alignment is LogitAlignment.ALIGNED,
            ),
            update_state=backend.update,
            pilot_for_single_step=False,
        )
        del outputs
        label_positions = torch.as_tensor(
            canvas.label_positions, dtype=torch.long, device=device
        )
        logits = aligned[0, prompt_length + label_positions]
        final_state = backend.update(final_state, aligned, update_mask)
        return logits, state_canvas, final_state[0, prompt_length:], steps

    @staticmethod
    def _encoder_canvas_batch(canvas: DecisionCanvas) -> DiffusionBatch:
        batch = DecisionCanvasCollator()([canvas])
        return replace(batch, decoder_prefix_lengths=batch.encoder_lengths)

    @staticmethod
    def _canvas_label_mask(
        shape: torch.Size, positions, device: torch.device
    ) -> torch.Tensor:
        mask = torch.zeros(shape, dtype=torch.bool, device=device)
        mask[0, torch.as_tensor(positions, dtype=torch.long, device=device)] = True
        return mask

    def _noise_labels(
        self,
        clean_ids: torch.Tensor,
        positions,
        spec: DiffusionSpec,
        vocab_size: int,
        seed: int | None,
        mask_token_id: int,
        initial_canvas_ids: torch.Tensor | None,
        pinned_mask,
    ) -> torch.Tensor:
        state = clean_ids.clone()
        indices = torch.as_tensor(positions, dtype=torch.long, device=state.device)
        if initial_canvas_ids is not None:
            supplied = torch.as_tensor(
                initial_canvas_ids, dtype=state.dtype, device=state.device
            )
            if supplied.shape != state.shape[1:]:
                raise ValueError(
                    "initial_canvas_ids must match the decision canvas width"
                )
            pinned = torch.as_tensor(pinned_mask, dtype=torch.bool, device=state.device)
            if not torch.equal(supplied[pinned], state[0, pinned]):
                raise ValueError("initial_canvas_ids cannot alter pinned canvas tokens")
            state[0] = supplied
        elif spec.noise is DiffusionNoise.UNIFORM:
            noise = random.Random(seed)  # nosec B311 - matches the pinned serving sampler.
            state[:, indices] = torch.as_tensor(
                [noise.randrange(vocab_size) for _ in range(indices.numel())],
                dtype=state.dtype,
                device=state.device,
            )
        elif spec.noise is DiffusionNoise.ABSORBING:
            state[:, indices] = mask_token_id
        else:
            raise ValueError(f"unsupported diffusion noise: {spec.noise}")
        return state

    def _fresh_free_slots(
        self,
        state: torch.Tensor,
        canvas: DecisionCanvas,
        spec: DiffusionSpec,
        vocab_size: int,
        seed: int | None,
        mask_token_id: int,
        initialize: bool,
    ) -> torch.Tensor:
        if self.free_update_policy != "argmax" or not initialize:
            return state
        slots = torch.as_tensor(canvas.slot_mask, dtype=torch.bool, device=state.device)
        pinned = torch.as_tensor(
            canvas.pinned_mask, dtype=torch.bool, device=state.device
        )
        mask = slots & ~pinned
        if spec.noise is DiffusionNoise.ABSORBING:
            values = torch.full_like(state, mask_token_id)
        else:
            if seed is None:
                values = torch.randint(
                    vocab_size, state.shape, dtype=state.dtype, device=state.device
                )
                return torch.where(mask[None], values, state)
            digest = hashlib.sha256(
                f"decision-free-slot-read-v1:{seed}".encode("ascii")
            ).digest()
            generator = torch.Generator(device=state.device)
            generator.manual_seed(int.from_bytes(digest[:8], "big"))
            values = torch.randint(
                vocab_size,
                state.shape,
                dtype=state.dtype,
                device=state.device,
                generator=generator,
            )
        return torch.where(mask[None], values, state)

    def _mask_token_id_for_spec(self, spec: DiffusionSpec) -> int:
        if spec.mask_token_policy is MaskTokenPolicy.NONE:
            raise ValueError("uniform diffusion must not request a mask token")
        if self.mask_token_id is None:
            raise ValueError("absorbing HFReader requires a model mask_token_id")
        return self.mask_token_id

    def _mask_token_id(self, model, spec: DiffusionSpec) -> int:
        if spec.mask_token_policy is MaskTokenPolicy.NONE:
            return 0
        if self.mask_token_id is None:
            config_id = getattr(self._model_config(model), "mask_token_id", None)
            if config_id is None:
                raise ValueError("absorbing HFReader requires a model mask_token_id")
            self.mask_token_id = int(config_id)
        return self._mask_token_id_for_spec(spec)

    def _vocab_size(self, model) -> int:
        if self.vocab_size is not None:
            return self.vocab_size
        config = self._model_config(model)
        text_config = getattr(config, "text_config", config)
        vocab_size = getattr(text_config, "vocab_size", None)
        if vocab_size is None:
            raise ValueError("HFReader requires model.config vocab_size")
        return int(vocab_size)

    @staticmethod
    def _model_config(model):
        config = getattr(model, "config", None)
        if config is None:
            config = getattr(getattr(model, "module", None), "config", None)
        if config is None:
            raise TypeError("HFReader model must expose config")
        return config

    @staticmethod
    def _model_device(model) -> torch.device:
        parameter = next(model.parameters(), None)
        if parameter is None:
            return torch.device("cpu")
        return parameter.device

    @staticmethod
    def _validate_varlen(model, spec: DiffusionSpec) -> None:
        if spec.layout is not DiffusionLayout.FULL_SEQUENCE:
            raise ValueError(
                "HFReader varlen attention supports only full-sequence diffusion"
            )
        capability_model = model
        while hasattr(capability_model, "module"):
            capability_model = capability_model.module
        if not getattr(capability_model, "supports_diffusion_varlen", False):
            raise ValueError(
                "HFReader varlen attention requires a native model with "
                "supports_diffusion_varlen"
            )
