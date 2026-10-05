"""Fixed decision slots are resolved once by dataset preparation."""

from __future__ import annotations

from copy import deepcopy

import pytest

from axolotl.integrations.diffusion_decision import datasets
from axolotl.integrations.diffusion_decision.datasets import DecisionDataset
from axolotl.integrations.diffusion_decision.preprocessing import build_decision_canvas
from axolotl.integrations.diffusion_decision.slot_sampling import DecisionDraw
from axolotl.model_support import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
)

from tests.integrations.diffusion_decision.helpers import (
    ChatCharacterTokenizer,
    make_cfg,
    make_record,
    make_spec,
)


def _spec(noise: DiffusionNoise = DiffusionNoise.UNIFORM, *, encoder: bool = False):
    return make_spec(
        noise=noise,
        layout=(
            DiffusionLayout.ENCODER_CANVAS if encoder else DiffusionLayout.FULL_SEQUENCE
        ),
        self_conditioning=False,
    )


def _cfg(mode: str = "none", *, count: int = 0, ids: tuple[int, ...] = ()) -> dict:
    return make_cfg(
        model_config={"thought_open_ids": [70], "thought_close_ids": [71]},
        diffusion_decision={
            "latent": {"mode": mode, "num_slots": count, "token_ids": ids}
        },
    )


def _prepared(cfg: dict, spec: DiffusionSpec):
    rows, dropped = datasets._canvas_rows(
        ChatCharacterTokenizer(), [make_record()], cfg, {}, spec
    )
    assert dropped == 0
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize(
    ("mode", "count", "ids", "noise", "placement", "expected"),
    [
        ("pad", 2, (), DiffusionNoise.UNIFORM, "after_turn", (0, 0)),
        ("pinned", 2, (7,), DiffusionNoise.UNIFORM, "thought", (7, 7)),
        ("learned", 2, (7, 8), DiffusionNoise.UNIFORM, "thought", (7, 8)),
        ("prompt", 2, (7, 8), DiffusionNoise.UNIFORM, "prompt", (7, 8)),
        ("mask", 2, (), DiffusionNoise.ABSORBING, "thought", (9, 9)),
    ],
)
def test_dataset_rows_apply_fixed_slot_plan(
    mode: str,
    count: int,
    ids: tuple[int, ...],
    noise: DiffusionNoise,
    placement: str,
    expected: tuple[int, ...],
):
    row = _prepared(_cfg(mode, count=count, ids=ids), _spec(noise))
    canvas = row["canvas"]
    plan = row["slot_plan"]

    assert plan.placement == placement
    assert plan.ids == expected
    assert all(not canvas.slot_mask[position] for position in canvas.label_positions)
    if placement == "prompt":
        assert canvas.prompt_ids[:count] == expected
        assert not any(canvas.slot_mask)
    else:
        positions = tuple(
            index for index, selected in enumerate(canvas.slot_mask) if selected
        )
        assert len(positions) == count
        assert tuple(canvas.canvas_ids[index] for index in positions) == expected
        assert all(canvas.pinned_mask[index] for index in positions)
    assert plan.trainable_token_ids == ((7, 8) if mode in {"learned", "prompt"} else ())


def test_none_mode_uses_the_legacy_canvas_call_and_row_shape():
    cfg = _cfg()
    record = make_record()
    row = _prepared(cfg, _spec())
    expected = build_decision_canvas(
        ChatCharacterTokenizer(),
        datasets.permute_record(deepcopy(record), seed=23),
        scaffold_ids=(),
        turn_close_id=11,
        pad_id=0,
        vocab_size=256,
        width=128,
        seed=23,
        steps=1,
        noise_kind="uniform",
        mask_token_id=9,
    )

    assert row["canvas"] == expected
    assert set(row) == {
        "canvas",
        "decision_example",
        "source",
        "length",
        "record",
        "max_questions",
    }


def test_slot_width_overflow_is_counted_without_truncation():
    cfg = _cfg("learned", count=2, ids=(7, 8))
    cfg["diffusion_lm"]["canvas_width"] = 8

    rows, dropped = datasets._canvas_rows(
        ChatCharacterTokenizer(), [make_record()], cfg, {}, _spec()
    )

    assert rows == []
    assert dropped == 1


@pytest.mark.parametrize(
    "mode,ids", [("pinned", (7,)), ("learned", (7,)), ("prompt", (7,)), ("pad", ())]
)
def test_dynamic_slots_prepare_maximum_canvas_with_sampling_recipe(
    mode: str, ids: tuple[int, ...]
):
    cfg = _cfg()
    cfg["diffusion_decision"]["latent"] = {
        "mode": mode,
        "num_slots": 1,
        "token_ids": ids,
        "sample_num_slots": True,
    }
    rows, dropped = datasets._canvas_rows(
        ChatCharacterTokenizer(), [make_record()], cfg, {}, _spec()
    )
    assert dropped == 0
    assert rows[0]["slot_sampling"]["max_slots"] == 1
    assert len(rows[0]["slot_plan"].ids) == 1


def test_sampled_prompt_draw_projects_leading_prompt_slots_and_preserves_weight():
    cfg = _cfg("prompt", count=3, ids=(7, 8, 10))
    cfg["diffusion_decision"]["latent"]["sample_num_slots"] = True
    rows, dropped = datasets._canvas_rows(
        ChatCharacterTokenizer(), [make_record()], cfg, {}, _spec()
    )
    assert dropped == 0
    maximum = rows[0]
    maximum["decision_example"] = maximum["decision_example"].__class__(
        questions=maximum["decision_example"].questions, source_weight=1.5
    )
    dataset = DecisionDataset(rows, {})

    projected = [dataset[DecisionDraw(0, 0, ordinal)] for ordinal in range(64)]
    by_count = {row["decision_slot_count"]: row for row in projected}
    assert set(by_count) == {0, 1, 2, 3}
    for count, row in by_count.items():
        canvas = row["canvas"]
        assert canvas.prompt_ids[:count] == (7, 8, 10)[:count]
        assert canvas.prompt_slot_mask == (True,) * count + (False,) * (
            len(canvas.prompt_ids) - count
        )
        assert not any(canvas.slot_mask)
        assert row["decision_example"].source_weight == 1.5


def test_encoder_thought_slots_require_explicit_model_contract_delimiters():
    cfg = _cfg("pinned", count=1, ids=(7,))
    del cfg["model_config"]["thought_open_ids"]
    del cfg["model_config"]["thought_close_ids"]

    with pytest.raises(ValueError, match=r"model_config: \{thought_open_ids"):
        datasets._canvas_rows(
            ChatCharacterTokenizer(), [make_record()], cfg, {}, _spec(encoder=True)
        )


@pytest.mark.parametrize("mode", ("pinned", "learned", "prompt"))
@pytest.mark.parametrize("token_id", (0, 1, 2, 11, 9, 70, 71))
def test_fixed_slot_ids_reject_active_tokenizer_model_and_delimiter_controls(
    mode: str, token_id: int
):
    ids = (token_id,) if mode == "pinned" else (token_id, 8)
    cfg = _cfg(mode, count=2 if mode != "pinned" else 2, ids=ids)

    with pytest.raises(ValueError, match="active control IDs"):
        datasets._canvas_rows(
            ChatCharacterTokenizer(), [make_record()], cfg, {}, _spec()
        )


def test_fixed_slots_reject_tokenizer_eos_without_model_config_metadata():
    cfg = _cfg("pinned", count=1, ids=(11,))
    cfg["model_config"] = {"vocab_size": 256}

    with pytest.raises(ValueError, match="active control IDs"):
        datasets._canvas_rows(
            ChatCharacterTokenizer(), [make_record()], cfg, {}, _spec()
        )


def test_validated_cfg_model_config_overrides_reach_slot_plan(
    min_base_cfg, monkeypatch
):
    from collections import OrderedDict

    from axolotl.integrations.base import PluginManager
    from axolotl.integrations.diffusion_decision._util import model_config_overrides
    from axolotl.integrations.diffusion_decision.plugin import (
        DiffusionDecisionPlugin,
    )
    from axolotl.utils.config import validate_config
    from axolotl.utils.dict import DictDefault

    plugin_path = "axolotl.integrations.diffusion_decision.DiffusionDecisionPlugin"
    monkeypatch.setattr(
        PluginManager.get_instance(),
        "plugins",
        OrderedDict({plugin_path: DiffusionDecisionPlugin()}),
    )
    raw = _cfg("pinned", count=1, ids=(7,))
    cfg = validate_config(
        DictDefault(
            min_base_cfg
            | {
                "plugins": [plugin_path],
                "seed": raw["seed"],
                "model_config": raw["model_config"],
                "diffusion_lm": {"from_causal_lm": False, **raw["diffusion_lm"]},
                "diffusion_decision": raw["diffusion_decision"],
            }
        )
    )

    assert cfg.model_config is None
    assert model_config_overrides(cfg)["thought_open_ids"] == [70]
    plan, thought_open_ids, thought_close_ids = datasets._resolve_slot_plan(
        ChatCharacterTokenizer(), cfg, _spec(encoder=True), 256
    )
    assert plan is not None and plan.placement == "thought"
    assert (thought_open_ids, thought_close_ids) == ((70,), (71,))
    cfg.diffusion_decision["latent"]["token_ids"] = [70]
    with pytest.raises(ValueError, match="active control IDs"):
        datasets._resolve_slot_plan(
            ChatCharacterTokenizer(), cfg, _spec(encoder=True), 256
        )
