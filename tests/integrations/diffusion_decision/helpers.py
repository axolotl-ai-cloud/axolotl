from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from transformers import DiffusionGemmaConfig

from axolotl.core.trainers.diffusion_lm.backends.full_sequence import (
    FullSequenceBackend,
)
from axolotl.integrations.diffusion_decision.loss import decision_example_from_canvas
from axolotl.integrations.diffusion_decision.preprocessing import build_decision_canvas
from axolotl.integrations.diffusion_decision.records import DecisionCanvas
from axolotl.integrations.diffusion_decision.slots import SlotInit, SlotPlan
from axolotl.integrations.diffusion_decision.trainer import DiffusionDecisionTrainer
from axolotl.model_support import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    EosHandling,
    FirstPositionAlignment,
    GenerationAdapter,
    LogitAlignment,
    MaskTokenPolicy,
    ObjectiveReduction,
    ReductionScope,
    TimeWeighting,
)

DJEV_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "djev_template_e5841cf.json").read_text()
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) for character in text]


class ChatCharacterTokenizer(CharacterTokenizer):
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 11
    unk_token_id = 2

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        assert tokenize and add_generation_prompt and not enable_thinking
        assert len(messages) == 2
        return [99]


class RecordedTokenizer:
    def __init__(self, encodings):
        self.encodings = encodings

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return self.encodings[text]


def make_spec(
    *,
    noise: DiffusionNoise = DiffusionNoise.UNIFORM,
    layout: DiffusionLayout = DiffusionLayout.FULL_SEQUENCE,
    self_conditioning: bool | None = None,
    mask_token_policy: MaskTokenPolicy | None = None,
    **fields: Any,
) -> DiffusionSpec:
    values: dict[str, Any] = {
        "noise": noise,
        "layout": layout,
        "logit_alignment": LogitAlignment.ALIGNED,
        "first_position_alignment": FirstPositionAlignment.DUPLICATE_FIRST,
        "self_conditioning": (
            layout is DiffusionLayout.ENCODER_CANVAS
            if self_conditioning is None
            else self_conditioning
        ),
        "max_canvas": 128,
        "max_context": 1024,
        "eos_handling": EosHandling.INDEPENDENT,
        "mask_token_policy": (
            (
                MaskTokenPolicy.MODEL
                if noise is DiffusionNoise.ABSORBING
                else MaskTokenPolicy.NONE
            )
            if mask_token_policy is None
            else mask_token_policy
        ),
        "default_time_weighting": TimeWeighting.NONE,
        "objective_reduction": ObjectiveReduction.MASKED_TOKEN_MEAN,
        "generation_adapter": GenerationAdapter.FULL_SEQUENCE,
        "reduction_scope": ReductionScope.MICROBATCH,
    }
    values.update(fields)
    return DiffusionSpec(**values)


def trainer_spec(
    layout: DiffusionLayout, alignment: LogitAlignment = LogitAlignment.ALIGNED
) -> DiffusionSpec:
    return make_spec(
        noise=DiffusionNoise.ABSORBING,
        layout=layout,
        logit_alignment=alignment,
        max_canvas=16,
        max_context=32,
        objective_reduction=ObjectiveReduction.EXAMPLE_MEAN,
        reduction_scope=ReductionScope.GLOBAL_WINDOW,
    )


def make_record(
    identifier: str | None = "record",
    *,
    source: str = "test",
    group: str = "test",
    state: Any = "state",
    question: str = "q",
    question_type: str = "choice",
    questions: dict[str, Any] | None = None,
    labels: dict[str, Any] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": question_type}
    if question_type == "choice":
        schema.update(instructions="Pick.", options=["one", "two"])
    record: dict[str, Any] = {
        "id": identifier,
        "source": source,
        "group": group,
        "state": state,
        "questions": {question: schema} if questions is None else questions,
        "labels": (
            {question: {"kind": "hard", "gold_idx": 0}} if labels is None else labels
        ),
    }
    record.update(fields)
    return {key: value for key, value in record.items() if value is not None}


def make_canvas(
    prompt_ids=(1, 2),
    canvas_ids=(3, 4, 5, 6),
    label_positions=(1,),
    name: str = "q",
    *,
    allowed=(1, 2, 3),
    allowed_ids=None,
    question_ids=None,
    targets=None,
    **fields: Any,
) -> DecisionCanvas:
    width = len(canvas_ids)
    values: dict[str, Any] = {
        "prompt_ids": tuple(prompt_ids),
        "canvas_ids": tuple(canvas_ids),
        "label_positions": tuple(label_positions),
        "allowed_ids": (
            tuple(tuple(allowed) for _ in label_positions)
            if allowed_ids is None
            else allowed_ids
        ),
        "question_ids": (
            tuple(f"{name}-{index}" for index in range(len(label_positions)))
            if question_ids is None
            else question_ids
        ),
        "targets": (
            tuple({"kind": "hard", "gold_idx": 0} for _ in label_positions)
            if targets is None
            else targets
        ),
        "pinned_mask": (False,) * width,
        "semantic_mask": (True,) * width,
        "slot_mask": (False,) * width,
        "template_length": width,
    }
    values.update(fields)
    return DecisionCanvas(**values)


def make_rows(canvases, *, weights=None) -> list[dict[str, Any]]:
    return [
        {
            "canvas": canvas,
            "source": f"source-{index}",
            "decision_example": decision_example_from_canvas(
                canvas, **({} if weights is None else {"source_weight": weights[index]})
            ),
        }
        for index, canvas in enumerate(canvases)
    ]


def slot_token_ids(mode: str, count: int = 2) -> tuple[int, ...]:
    if mode == "pinned":
        return (7,)
    if mode in {"learned", "prompt"}:
        return (7, 8, 10)[:count]
    return ()


def free_plan_seed(mode: str, noise: DiffusionNoise) -> int | None:
    return 17 if mode == "free" and noise is DiffusionNoise.UNIFORM else None


def make_slot_init(
    mode,
    *,
    count: int = 2,
    ids: tuple[int, ...] = (),
    noise: DiffusionNoise = DiffusionNoise.UNIFORM,
    spec: DiffusionSpec | None = None,
    vocab_size: int = 256,
    **fields: Any,
) -> SlotInit:
    return SlotInit(
        mode,
        token_ids=ids,
        num_slots=count,
        vocab_size=vocab_size,
        pad_id=0,
        spec=make_spec(noise=noise) if spec is None else spec,
        mask_token_id=9,
        **fields,
    )


def make_slot_plan(
    mode,
    *,
    count: int = 2,
    ids: tuple[int, ...] = (),
    noise: DiffusionNoise = DiffusionNoise.UNIFORM,
    spec: DiffusionSpec | None = None,
    seed: int | None = None,
) -> SlotPlan:
    return make_slot_init(mode, count=count, ids=ids, noise=noise, spec=spec).build(
        seed=seed
    )


def build_canvas(
    slot_plan: SlotPlan | None = None,
    *,
    record: dict[str, Any] | None = None,
    tokenizer=None,
    prompt_ids=(99,),
    noise: DiffusionNoise = DiffusionNoise.UNIFORM,
    noise_kind: str | None = None,
    steps: int = 1,
    width: int = 128,
    seed: int = 23,
    thought_open_ids=(70, 71),
    thought_close_ids=(72,),
    **fields: Any,
) -> DecisionCanvas:
    kind = noise.value if noise_kind is None else noise_kind
    arguments: dict[str, Any] = {
        "prompt_ids": prompt_ids,
        "scaffold_ids": (),
        "turn_close_id": 106,
        "pad_id": 0,
        "vocab_size": 256,
        "width": width,
        "seed": seed,
        "steps": steps,
        "noise_kind": kind,
        "mask_token_id": 9 if kind == "absorbing" else None,
        "slot_plan": slot_plan,
        "thought_open_ids": thought_open_ids,
        "thought_close_ids": thought_close_ids,
    }
    arguments.update(fields)
    return build_decision_canvas(
        CharacterTokenizer() if tokenizer is None else tokenizer,
        make_record() if record is None else record,
        **arguments,
    )


def make_cfg(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "seed": 23,
        "model_config": {"vocab_size": 256, "mask_token_id": 9},
        "diffusion_lm": {"canvas_width": 128, "mask_token_id": 9},
        "diffusion_decision": {"max_questions_per_canvas": 20},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def tiny_gemma_config() -> DiffusionGemmaConfig:
    return DiffusionGemmaConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 32,
            "layer_types": ["full_attention"],
            "per_layer_config": {"0": {"head_dim": 8}},
            "sliding_window": 8,
            "use_bidirectional_attention": "vision",
            "num_experts": 2,
            "top_k_experts": 1,
            "moe_intermediate_size": 16,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
        },
        vision_config={
            "model_type": "gemma4_vision",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 32,
            "patch_size": 16,
            "position_embedding_size": 16,
        },
        canvas_length=8,
        boi_token_id=31,
        eoi_token_id=30,
        image_token_id=29,
    )


class EchoModel(nn.Module):
    def __init__(
        self, vocab_size: int = 256, mask_token_id: int = 9, scale: float = 1.0
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            vocab_size=vocab_size, mask_token_id=mask_token_id
        )
        self.scale = scale
        self.calls = 0
        self.seen_input_ids: torch.Tensor | None = None

    def forward(self, input_ids, attention_mask, position_ids, use_cache, **kwargs):
        del attention_mask, position_ids, use_cache, kwargs
        self.calls += 1
        self.seen_input_ids = input_ids.detach().clone()
        logits = torch.nn.functional.one_hot(
            input_ids, num_classes=self.config.vocab_size
        ).float()
        return SimpleNamespace(logits=logits * self.scale + self.anchor)


class TrainerHarness(DiffusionDecisionTrainer):
    def __init__(
        self,
        spec: DiffusionSpec,
        decision: dict[str, object] | None = None,
        *,
        world_size: int = 1,
        native_values: dict[str, object] | None = None,
        mask_token_id: int = 10,
        native_time: float = 0.5,
    ) -> None:
        self._spec = spec
        self.axolotl_cfg = {"diffusion_decision": decision or {}}
        self.args = SimpleNamespace(world_size=world_size)
        self._special_token_ids: set[int] = set()
        self._native_values = native_values or {}
        self._mask_token_id = mask_token_id
        self._native_time = native_time

    @property
    def _native_spec(self) -> DiffusionSpec:
        return self._spec

    def _full_sequence_backend(self) -> FullSequenceBackend:
        return FullSequenceBackend(
            mask_token_id=self._mask_token_id, attention_backend="dense"
        )

    def _native_value(self, name: str, default=None):
        values: dict[str, object] = {"t_eps": 0.5, "self_conditioning": None}
        values.update(self._native_values)
        return values.get(name, default)

    def _native_unroll_settings(self) -> tuple[int, bool]:
        return 1, False

    @staticmethod
    def _sample_native_unroll_steps(k_max: int, device: torch.device) -> int:
        del k_max, device
        return 1

    def _sample_native_times(
        self, count: int, device: torch.device, default_eps: float
    ) -> torch.Tensor:
        del default_eps
        return torch.full((count,), self._native_time, device=device)


class MultistepTrainerHarness(TrainerHarness):
    _sampled_steps_value = 1

    def __init__(
        self,
        spec: DiffusionSpec,
        *,
        k_max: int,
        sampled_steps: int,
        grad_through_steps: bool = False,
        decision: dict[str, object] | None = None,
    ) -> None:
        super().__init__(spec, decision)
        self._k_max = k_max
        MultistepTrainerHarness._sampled_steps_value = sampled_steps
        self._grad_through_steps = grad_through_steps

    def _native_unroll_settings(self) -> tuple[int, bool]:
        return self._k_max, self._grad_through_steps

    @staticmethod
    def _sample_native_unroll_steps(k_max: int, device: torch.device) -> int:
        del k_max, device
        return MultistepTrainerHarness._sampled_steps_value
