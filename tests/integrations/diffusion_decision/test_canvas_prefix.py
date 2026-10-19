"""``diffusion_decision.canvas_prefix_ids`` pins the same clean head on train and read canvases."""

from __future__ import annotations

from copy import deepcopy

import pytest

from axolotl.integrations.diffusion_decision import datasets
from axolotl.integrations.diffusion_decision.args import DiffusionDecisionConfig
from axolotl.integrations.diffusion_decision.manifest import (
    DecisionManifest,
    build_decision_manifest,
    validate_read_manifest,
)
from axolotl.model_support import DiffusionLayout
from axolotl.utils.dict import DictDefault

from tests.integrations.diffusion_decision.helpers import (
    ChatCharacterTokenizer,
    make_cfg,
    make_record,
    make_spec,
)

# Stand-in for DiffusionGemma's empty thought block ``<|channel>thought\n<channel|>``.
PREFIX = (70, 71, 72, 73)


def _cfg(prefix=(), *, k_max: int = 1, latent=None) -> dict:
    cfg = make_cfg(
        model_config={"thought_open_ids": [70], "thought_close_ids": [73]},
        diffusion_decision={
            "canvas_prefix_ids": list(prefix),
            **({"latent": latent} if latent else {}),
        },
    )
    cfg["diffusion_lm"]["unroll"] = {"k_max": k_max}
    return cfg


def _row(cfg: dict, record=None):
    spec = make_spec(layout=DiffusionLayout.ENCODER_CANVAS, self_conditioning=False)
    rows, dropped = datasets._canvas_rows(
        ChatCharacterTokenizer(), [record or make_record()], cfg, {}, spec
    )
    assert dropped == 0
    return rows[0]["canvas"]


@pytest.mark.parametrize("k_max", [1, 2])
def test_prefix_heads_the_canvas_and_shifts_only_the_template(k_max: int):
    plain = _row(_cfg(k_max=k_max))
    pinned = _row(_cfg(PREFIX, k_max=k_max))

    assert pinned.canvas_ids[: len(PREFIX)] == PREFIX
    # The answer template follows unchanged, shifted by the prefix length.
    assert (
        pinned.canvas_ids[len(PREFIX) : len(PREFIX) + plain.template_length]
        == (plain.canvas_ids[: plain.template_length])
    )
    assert pinned.template_length == plain.template_length + len(PREFIX)
    assert pinned.label_positions == tuple(
        position + len(PREFIX) for position in plain.label_positions
    )
    assert pinned.allowed_ids == plain.allowed_ids
    assert pinned.targets == plain.targets
    assert pinned.prompt_ids == plain.prompt_ids
    # The prefix is never noised, and multi-step reads never rewrite it.
    assert not set(range(len(PREFIX))) & set(pinned.label_positions)
    if k_max > 1:
        assert all(pinned.pinned_mask[: len(PREFIX)])


def test_train_and_read_rows_share_the_prefix():
    # Train and eval/read canvases (student_read.py replaces test_datasets) go
    # through the same row builder; the head must not depend on the split.
    cfg = _cfg(PREFIX)
    first = _row(deepcopy(cfg))
    second = _row(deepcopy(cfg))
    assert first == second
    assert first.canvas_ids[: len(PREFIX)] == PREFIX


def test_prefix_rejects_thought_placed_slots():
    cfg = _cfg(PREFIX, latent={"mode": "pinned", "num_slots": 2, "token_ids": [7]})
    with pytest.raises(ValueError, match="canvas_prefix_ids cannot be combined"):
        _row(cfg)


def test_prefix_ids_must_be_nonnegative():
    with pytest.raises(ValueError, match="nonnegative"):
        DiffusionDecisionConfig.model_validate({"canvas_prefix_ids": [-1]})


def _manifest_cfg(prefix, **extra) -> DictDefault:
    return DictDefault(
        {
            "base_model": "nvidia/Nemotron-Labs-Diffusion-3B",
            "model_config_type": "nemotron_labs_diffusion",
            "diffusion_lm": {"canvas_width": 128, "mask_token_id": 100},
            "diffusion_decision": {"canvas_prefix_ids": list(prefix)},
            **extra,
        }
    )


def _manifest(prefix) -> DecisionManifest:
    return build_decision_manifest(_manifest_cfg(prefix))


def test_manifest_records_prefix_and_legacy_manifests_default_to_none():
    assert _manifest(PREFIX).canvas_prefix_ids == list(PREFIX)
    legacy = _manifest(()).model_dump(mode="json")
    legacy.pop("canvas_prefix_ids")
    assert DecisionManifest.model_validate(legacy).canvas_prefix_ids == []


@pytest.mark.parametrize(
    ("trained", "read", "ok"),
    [(PREFIX, PREFIX, True), ((), PREFIX, False), (PREFIX, (), False), ((), (), True)],
)
def test_read_refuses_an_adapter_trained_on_another_canvas_head(
    tmp_path, trained, read, ok
):
    # The read-time manifest check (DiffusionDecisionPlugin.pre_lora_load) compares
    # canvas_prefix_ids along with the rest of the canvas.
    _manifest(trained).write_to(tmp_path)
    cfg = _manifest_cfg(read, lora_model_dir=str(tmp_path))
    if ok:
        assert validate_read_manifest(cfg) is not None
    else:
        with pytest.raises(ValueError, match="canvas_prefix_ids"):
            validate_read_manifest(cfg)


def test_diffusion_gemma_empty_thought_block_ids():
    """The smoke/read configs pin these ids; they must be the model's own block."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/diffusiongemma-26B-A4B-it",
            revision="f7f5b7f5fa82ffc52addd066915886d497f5517b",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("DiffusionGemma tokenizer is not cached")
    ids = tokenizer("<|channel>thought\n<channel|>", add_special_tokens=False).input_ids
    assert ids == [100, 45518, 107, 101]
    assert tokenizer.convert_ids_to_tokens([100, 101]) == ["<|channel>", "<channel|>"]
