"""Image records: validation, grouping, prompt expansion, cache, collation."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from axolotl.integrations.diffusion_decision import datasets, prepared_cache, prompts
from axolotl.integrations.diffusion_decision.adapters.jsonl import normalize_jsonl
from axolotl.integrations.diffusion_decision.args import DiffusionDecisionConfig
from axolotl.integrations.diffusion_decision.grouping import group_records
from axolotl.integrations.diffusion_decision.preprocessing import build_decision_canvas
from axolotl.integrations.diffusion_decision.readers.base import validate_canvas
from axolotl.integrations.diffusion_decision.trainer import _media_model_kwargs
from axolotl.integrations.diffusion_decision.training_collator import (
    DecisionTrainingCollator,
    decision_collator_for_config,
)
from axolotl.model_support.diffusion import DiffusionLayout, FirstPositionAlignment

from tests.integrations.diffusion_decision.helpers import (
    make_canvas,
    make_record,
    make_rows,
    make_spec,
)

_SPECIAL = {
    "<|image_start|>": 18,
    "<|image_pad|>": 19,
    "<|image_break|>": 20,
    "<|image_end|>": 21,
}


class VLMCharacterTokenizer:
    """Character tokenizer whose template emits one placeholder per image block."""

    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert not tokenize and kwargs == {
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        parts = []
        for message in messages:
            content = message["content"]
            if message["role"] == "system":
                if not isinstance(content, str):
                    raise TypeError("the checkpoint template concatenates system text")
                parts.append(content)
                continue
            for block in content:
                parts.append(
                    "<|image_start|>" if block["type"] != "text" else block["text"]
                )
        return "".join(parts) + "\n"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        ids = []
        for piece in re.split(r"(<\|image_[a-z]+\|>)", text):
            ids.extend(
                [_SPECIAL[piece]]
                if piece in _SPECIAL
                else [ord(character) for character in piece]
            )
        return ids


class _CacheTokenizer(SimpleNamespace):
    backend_tokenizer = SimpleNamespace(to_str=lambda: "b")

    def __len__(self) -> int:
        return 1

    def get_vocab(self) -> dict[str, int]:
        return {"x": 0}


def _spec():
    return make_spec(
        layout=DiffusionLayout.FULL_SEQUENCE,
        first_position_alignment=FirstPositionAlignment.REQUIRES_PREDECESSOR,
        max_canvas=None,
        max_context=None,
    )


@pytest.fixture
def images(tmp_path):
    wide = tmp_path / "images" / "wide.png"
    wide.parent.mkdir()
    Image.new("RGB", (56, 28), (255, 0, 0)).save(wide)
    square = tmp_path / "images" / "square.png"
    Image.new("RGB", (28, 28), (0, 0, 255)).save(square)
    return wide, square


def _image_prompt(sizes):
    ids = []
    for h, w in sizes:
        rows = [[19] * (w // 28)] * (h // 28)
        ids.extend((18, *[t for r in rows for t in (*r, 20)][:-1], 21))
    return (*ids, 7)


def _image_canvas(refs, sizes, *, name="q", label=5):
    return make_canvas(
        _image_prompt(sizes),
        (3, 4, label, 6),
        (1,),
        allowed=(1, 2),
        question_ids=(name,),
        image_refs=tuple(refs),
        image_sizes=tuple(sizes),
        template_length=2,
    )


@pytest.mark.parametrize("images", ["x.png", [""], [1], [None]])
def test_jsonl_rejects_malformed_image_lists(images):
    with pytest.raises(ValueError, match="images must be a list"):
        normalize_jsonl(make_record(images=images))


def test_jsonl_accepts_image_reference_lists():
    record = make_record(images=["a.png", "https://example.com/b.png"])
    assert normalize_jsonl(record)["images"] == record["images"]


def test_grouping_never_merges_records_with_different_images():
    shared = dict(source="s", group="g", state="same")
    records = [
        make_record("one", images=["a.png"], **shared),
        make_record("two", images=["b.png"], **shared),
        make_record("three", **shared),
        make_record("four", images=["a.png"], **shared),
    ]
    grouped = group_records(records, max_questions=8)
    assert len(grouped) == 3
    assert grouped[0]["grouped_record_ids"] == ("one", "four")
    assert [record["id"] for record in grouped[1:]] == ["two", "three"]


def test_relative_image_refs_resolve_against_the_jsonl_directory(tmp_path, images):
    wide, _ = images
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(make_record(images=["images/wide.png", "https://x/y.png"])) + "\n"
    )
    entry = {"path": "json", "data_files": str(source), "split": "train"}
    (record,) = datasets._rows(entry)
    assert record["images"] == [str(wide), "https://x/y.png"]
    assert prompts.resolve_image_refs(["/abs/a.png"], tmp_path) == ("/abs/a.png",)
    assert prompts.resolve_image_refs(["rel.png"], tmp_path) == (
        str(tmp_path / "rel.png"),
    )
    assert datasets._prepared_cache_image_paths([source]) is None
    source.write_text(json.dumps(make_record(images=["images/wide.png"])) + "\n")
    assert datasets._prepared_cache_image_paths([source]) == [wide]
    source.write_text(json.dumps(make_record(images=["images/missing.png"])) + "\n")
    with pytest.raises(ValueError, match="image not found"):
        datasets._prepared_cache_image_paths([source])


def test_canvas_prompt_expands_images_before_the_state(images):
    wide, square = images
    record = make_record(state="state", images=[str(wide), str(square)])
    canvas = build_decision_canvas(
        VLMCharacterTokenizer(),
        record,
        scaffold_ids=(),
        turn_close_id=11,
        pad_id=0,
        vocab_size=200000,
        max_image_size=1540,
    )
    ids = list(canvas.prompt_ids)
    assert ids.count(19) == 3
    assert canvas.image_refs == (str(wide), str(square))
    assert canvas.image_sizes == ((28, 56), (28, 28))
    text = ids[ids.index(21, ids.index(21) + 1) + 1 :]
    assert "".join(chr(i) for i in text).startswith("state")
    with pytest.raises(ValueError, match="precomputed prompt_ids"):
        build_decision_canvas(
            VLMCharacterTokenizer(),
            record,
            [1, 2],
            scaffold_ids=(),
            turn_close_id=11,
            pad_id=0,
            vocab_size=200000,
        )


def test_validate_canvas_rejects_pads_in_canvas_and_misaligned_images():
    canvas = _image_canvas(("a.png",), ((28, 56),))
    validate_canvas(canvas)
    with pytest.raises(ValueError, match="image marker tokens cannot occupy"):
        validate_canvas(_image_canvas(("a.png",), ((28, 56),), label=19))
    with pytest.raises(ValueError, match="must align"):
        validate_canvas(make_canvas(image_refs=("a.png",), template_length=2))
    with pytest.raises(ValueError, match="pad tokens must match"):
        validate_canvas(
            make_canvas(
                (18, 19, 21),
                image_refs=("a.png",),
                image_sizes=((28, 56),),
                template_length=2,
            )
        )


def test_prepared_cache_round_trips_image_fields_and_hashes_image_bytes(
    tmp_path, images
):
    wide, square = images
    canvas = _image_canvas((str(wide), str(square)), ((28, 56), (28, 28)))
    restored = prepared_cache._canvas_from_json(
        json.loads(json.dumps(prepared_cache._canvas_to_json(canvas)))
    )
    assert restored == canvas
    assert restored.image_sizes == ((28, 56), (28, 28))
    tokenizer_root = tmp_path / "tok"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer.json").write_text("t")
    tokenizer = _CacheTokenizer(name_or_path=str(tokenizer_root))
    source = tmp_path / "s.jsonl"
    source.write_text("{}\n")

    def identity(cfg):
        return prepared_cache.identity(
            cfg, tokenizer, SimpleNamespace(a=1), [source], image_paths=[wide]
        )[0]

    before = identity({})
    assert identity({"diffusion_decision": {"max_image_size": 784}}) != before
    Image.new("RGB", (56, 28), (0, 255, 0)).save(wide)
    assert identity({}) != before


def test_collator_emits_pixel_values_and_sizes_in_canvas_order(images):
    wide, square = images
    rows = make_rows(
        (
            _image_canvas((str(wide),), ((28, 56),), name="a"),
            make_canvas(
                (4, 5), (6, 7, 8, 9), (1,), allowed=(1, 2), question_ids=("b",)
            ),
            _image_canvas((str(square),), ((28, 28),), name="c"),
        )
    )
    collator = DecisionTrainingCollator(_spec())
    for batch in (collator(rows), collator([[rows[0]], [rows[1], rows[2]]])):
        assert batch["pixel_values"].shape == (2, 3, 28, 56)
        assert batch["image_sizes"].tolist() == [[28, 56], [28, 28]]
        red = collator._media.transform_image(Image.new("RGB", (1, 1), (255, 0, 0)))
        assert torch.allclose(batch["pixel_values"][0, :, 0, 0], red[:, 0, 0])
        assert batch["pixel_values"][1, 2, 0, 0] > batch["pixel_values"][1, 0, 0, 0]
        assert torch.all(batch["pixel_values"][1, :, :, 28:] == 0)
        assert batch["input_ids"].tolist()[0].count(19) == 3
        assert batch["document_ids"].max().item() == 2
    assert len(collator._image_cache) == 2


def test_collator_is_unchanged_without_images():
    rows = make_rows((make_canvas(), make_canvas(name="r")))
    batch = DecisionTrainingCollator(_spec())(rows)
    assert "pixel_values" not in batch and "image_sizes" not in batch


def test_collator_rejects_pad_count_and_size_mismatches(images):
    wide, square = images
    collator = DecisionTrainingCollator(_spec())
    broken = make_canvas(
        (18, 19, 21),
        (3, 4, 5, 6),
        (1,),
        image_refs=(str(wide),),
        image_sizes=((28, 56),),
    )
    with pytest.raises(ValueError, match="image pad tokens for"):
        collator(make_rows((broken,)))
    with pytest.raises(ValueError, match="canvas recorded"):
        collator(make_rows((_image_canvas((str(square),), ((28, 56),)),)))


def test_image_cache_is_bounded(images):
    wide, square = images
    collator = DecisionTrainingCollator(_spec(), image_cache_size=1)
    collator(
        make_rows((_image_canvas((str(wide), str(square)), ((28, 56), (28, 28))),))
    )
    assert list(collator._image_cache) == [str(square)]


def test_config_validates_max_image_size():
    assert DiffusionDecisionConfig().max_image_size == 1400
    assert DiffusionDecisionConfig(max_image_size=784).max_image_size == 784
    with pytest.raises(ValueError, match="multiple of 28"):
        DiffusionDecisionConfig(max_image_size=800)
    with pytest.raises(ValueError):
        DiffusionDecisionConfig(max_image_size=0)


def test_collator_for_config_threads_max_image_size(monkeypatch):
    monkeypatch.setattr(
        "axolotl.integrations.diffusion_decision.training_collator.require_diffusion_spec",
        lambda _cfg: _spec(),
    )
    _, kwargs = decision_collator_for_config(
        {
            "tokenizer": SimpleNamespace(pad_token_id=0),
            "diffusion_decision": {"max_image_size": 784},
        }
    )
    assert kwargs["max_image_size"] == 784
    assert DecisionTrainingCollator(**kwargs)._media.max_image_size == 784


def test_trainer_forwards_only_present_media_keys():
    pixels = torch.zeros(1, 3, 28, 28)
    assert _media_model_kwargs({"input_ids": 1, "pixel_values": pixels}) == {
        "pixel_values": pixels
    }
    assert _media_model_kwargs({"pixel_values": None}) == {}


def test_validate_canvas_counts_image_markers_against_the_grid():
    sizes = ((56, 28), (28, 56))
    canvas = _image_canvas(("a.png", "b.png"), sizes)
    validate_canvas(canvas)
    prompt = list(canvas.prompt_ids)
    for token in (18, 20, 21):
        broken = prompt.copy()
        broken.remove(token)
        with pytest.raises(ValueError, match="start/end/break"):
            validate_canvas(
                make_canvas(
                    broken,
                    image_refs=("a.png", "b.png"),
                    image_sizes=sizes,
                    template_length=2,
                )
            )


@pytest.mark.parametrize("prompt,tokens", [((7, 18, 8), (3, 4)), ((7,), (3, 20))])
def test_text_only_canvas_rejects_reserved_image_markers(prompt, tokens):
    canvas = make_canvas(prompt, (*tokens, 5, 6), (1,), template_length=2)
    validate_canvas(canvas)
    with pytest.raises(ValueError, match="require image_refs"):
        validate_canvas(canvas, image_markers_reserved=True)


def test_build_canvas_reserves_markers_only_for_the_vlm_tokenizer():
    from axolotl.integrations.diffusion_decision.preprocessing import (
        _reserves_image_markers,
    )

    vlm = SimpleNamespace(convert_tokens_to_ids={"<|image_start|>": 18}.get)
    text = SimpleNamespace(convert_tokens_to_ids={"<|image_start|>": 3}.get)
    assert _reserves_image_markers(vlm)
    assert not _reserves_image_markers(text)
    assert not _reserves_image_markers(VLMCharacterTokenizer())


def test_relative_image_refs_resolve_against_a_local_parquet_directory(
    tmp_path, images, monkeypatch
):
    from datasets import Dataset

    wide, _ = images
    source = tmp_path / "train.parquet"
    Dataset.from_list([make_record(images=["images/wide.png"])]).to_parquet(str(source))
    monkeypatch.chdir(wide.parent)
    entry = {"path": "parquet", "data_files": str(source), "split": "train"}
    (record,) = datasets._rows(entry)
    assert record["images"] == [str(wide)]


def test_relative_image_refs_on_hub_sources_are_rejected(monkeypatch):
    rows = [make_record(images=["images/wide.png"])]
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: rows)
    with pytest.raises(ValueError, match="relative decision image paths"):
        datasets._rows({"path": "org/decisions", "split": "train"})
    rows[0] = make_record(images=["https://x/y.png"])
    assert datasets._rows({"path": "org/decisions"})[0]["images"] == ["https://x/y.png"]


def test_prepared_cache_identity_tracks_the_media_adapter_module(tmp_path, monkeypatch):
    from axolotl import processing_strategies

    tokenizer_root = tmp_path / "tok"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer.json").write_text("t")
    tokenizer = _CacheTokenizer(name_or_path=str(tokenizer_root))
    source = tmp_path / "s.jsonl"
    source.write_text("{}\n")

    def identity():
        return prepared_cache.identity(
            {}, tokenizer, SimpleNamespace(a=1), [source], image_paths=[]
        )[0]

    before = identity()
    adapter_file = prepared_cache.Path(processing_strategies.__file__)
    real = prepared_cache.sha256_file
    monkeypatch.setattr(
        prepared_cache,
        "sha256_file",
        lambda path: "edited" if path == adapter_file else real(path),
    )
    assert identity() != before
