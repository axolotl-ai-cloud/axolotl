"""CPU-only tests for the quantizer damage map."""

import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary.modules import iter_ternary_modules
from axolotl.integrations.ternary.ptq import damage
from axolotl.integrations.ternary.ptq.damage import (
    GLOBAL_SCOPES,
    DamageReport,
    DamageRow,
    collect_eval_batches,
    damage_map,
    perplexity,
)
from axolotl.integrations.ternary.swap import convert_model
from axolotl.utils.dict import DictDefault

FAMILIES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _model():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config).to(torch.float32).eval()


def _batches(count: int = 3, sequence_len: int = 16):
    generator = torch.Generator().manual_seed(1)
    return [
        torch.randint(0, 64, (1, sequence_len), generator=generator)
        for _ in range(count)
    ]


@pytest.fixture(name="cfg")
def fixture_cfg():
    return DictDefault(
        {
            "base_model": "stub",
            "output_dir": "/nonexistent",
            "datasets": [{"path": "stub/corpus"}],
            "ternary": {},
        }
    )


@pytest.fixture(name="stubbed")
def fixture_stubbed(monkeypatch):
    """Runs the real damage map against a tiny in-memory model and fixed batches."""
    batches = _batches()
    monkeypatch.setattr(damage, "load_eval_model", lambda _cfg: _model())
    monkeypatch.setattr(
        damage,
        "collect_eval_batches",
        lambda _cfg, _num, _len, _dataset=None: batches,
    )
    return batches


# --------------------------------------------------------------------- the table


def test_rows_cover_the_global_scopes_then_every_family(cfg, stubbed):
    report = damage_map(cfg, num_sequences=3, sequence_len=16)

    assert [row.scope for row in report.rows] == [
        *GLOBAL_SCOPES,
        *(f"family:{family}" for family in FAMILIES),
    ]
    assert report.sequences == 3
    assert report.sequence_len == 16


def test_baseline_is_the_untouched_fp_model(cfg, stubbed):
    reference = perplexity(_model(), stubbed)

    report = damage_map(cfg, num_sequences=3, sequence_len=16)

    assert report.baseline_perplexity == pytest.approx(reference, rel=1e-6)


def test_deltas_are_measured_against_the_baseline(cfg, stubbed):
    report = damage_map(cfg, num_sequences=3, sequence_len=16)

    for row in report.rows:
        assert row.delta == pytest.approx(row.perplexity - report.baseline_perplexity)
        assert math.isfinite(row.perplexity)


def test_quantizing_anything_moves_the_perplexity(cfg, stubbed):
    report = damage_map(cfg, num_sequences=3, sequence_len=16)

    for row in report.rows:
        assert row.perplexity != report.baseline_perplexity


def test_weight_only_quantization_makes_both_the_weights_row(cfg, stubbed):
    cfg["ternary"] = {"activation_bits": None}

    report = damage_map(cfg, num_sequences=3, sequence_len=16)
    rows = {row.scope: row for row in report.rows}

    assert rows["both"].perplexity == rows["weights"].perplexity
    # the activations row measures int8 activations whatever the config asks for
    assert rows["activations"].perplexity != report.baseline_perplexity


def test_a_single_target_family_makes_its_row_the_weights_row(cfg, stubbed):
    """Both are pure weight rows, so with one family they measure the same thing."""
    cfg["ternary"] = {
        "target_modules": [r".*\.mlp\.down_proj"],
        "strict_enumeration": False,
    }

    report = damage_map(cfg, num_sequences=3, sequence_len=16)
    rows = {row.scope: row for row in report.rows}

    assert list(rows) == [*GLOBAL_SCOPES, "family:down_proj"]
    assert rows["family:down_proj"].perplexity == rows["weights"].perplexity
    assert rows["family:down_proj"].perplexity != rows["both"].perplexity


def test_an_unswappable_config_is_rejected(cfg, stubbed):
    cfg["ternary"] = {"target_modules": ["nothing"], "keep_fp_modules": [".*"]}

    with pytest.raises(ValueError, match="at least one swapped linear"):
        damage_map(cfg, num_sequences=3, sequence_len=16)


@pytest.mark.parametrize(
    "ternary",
    [
        {"init": "ternary_fit", "weight_scale": "learnable"},
        {"init": "ternary_fit_calibrated", "weight_scale": "learnable"},
        {"subln": True, "export": {"formats": ["master_bf16"]}},
    ],
)
def test_the_baseline_is_the_fp_model_whatever_the_swap_would_have_done(
    cfg, stubbed, ternary
):
    """A fitted init and subln are not function-preserving at λ=0; the baseline is."""
    reference = perplexity(_model(), stubbed)
    cfg["ternary"] = ternary

    report = damage_map(cfg, num_sequences=3, sequence_len=16)

    assert report.baseline_perplexity == pytest.approx(reference, rel=1e-6)
    rows = {row.scope: row for row in report.rows}
    assert rows["weights"].delta != 0.0


def test_the_swap_config_neutralizes_the_passes_that_would_move_the_baseline():
    swapped = damage._swap_cfg(
        DictDefault(
            {
                "ternary": {
                    "init": "ternary_fit_calibrated",
                    "weight_scale": "learnable",
                    "init_calibration": {"num_sequences": 8},
                    "subln": True,
                    "activation_bits": 8,
                }
            }
        )
    )

    assert swapped["ternary"]["init"] == "absmean"
    assert swapped["ternary"]["subln"] is False
    assert "init_calibration" not in swapped["ternary"]
    # everything that *is* the quantizer survives untouched
    assert swapped["ternary"]["weight_scale"] == "learnable"
    assert swapped["ternary"]["activation_bits"] == 8


def test_the_damage_map_writes_nothing_to_output_dir(cfg, stubbed, tmp_path):
    cfg["output_dir"] = str(tmp_path)

    damage_map(cfg, num_sequences=3, sequence_len=16)

    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------- the scopes


def _swapped():
    model = _model()
    convert_model(model, DictDefault({"ternary": {}}))
    return model, dict(iter_ternary_modules(model))


def test_weights_scope_drops_activation_quantization():
    _, modules = _swapped()

    with damage._scope(modules, "weights", {name: 8 for name in modules}):
        assert all(module.lambda_ == 1.0 for module in modules.values())
        assert all(module.activation_bits is None for module in modules.values())

    assert all(module.activation_bits == 8 for module in modules.values())
    assert all(module.lambda_ == 0.0 for module in modules.values())


def test_activations_scope_leaves_the_weights_alone_and_hooks_the_inputs():
    _, modules = _swapped()

    with damage._scope(modules, "activations", {name: 8 for name in modules}):
        assert all(module.lambda_ == 0.0 for module in modules.values())
        assert all(module._forward_pre_hooks for module in modules.values())

    assert not any(module._forward_pre_hooks for module in modules.values())


def test_activation_hook_quantizes_the_input():
    from axolotl.integrations.ternary import quant

    model, modules = _swapped()
    x = torch.randn(2, 32)
    name, module = next(iter(modules.items()))
    reference = torch.nn.functional.linear(quant.act_quant(x, 1.0), module.weight)

    with damage._scope(modules, "activations", {name: 8 for name in modules}):
        out = module(x)

    assert torch.allclose(out, reference)


def test_family_scope_quantizes_only_that_family():
    _, modules = _swapped()

    with damage._scope(modules, "family:down_proj", {name: 8 for name in modules}):
        quantized = {name for name, module in modules.items() if module.lambda_ == 1.0}

    assert quantized
    assert all(name.endswith("down_proj") for name in quantized)
    assert len(quantized) < len(modules)


def test_configured_activation_bits_are_honored_and_restored():
    _, modules = _swapped()
    configured = {name: None for name in modules}

    with damage._scope(modules, "both", configured):
        assert all(module.activation_bits is None for module in modules.values())

    assert all(module.activation_bits is None for module in modules.values())


# ----------------------------------------------------------------- the collector


class _Tokenizer:
    eos_token_id = 7

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) % 64 for char in text]}


def _stub_corpus(monkeypatch, records):
    calls = {}

    def _load_dataset(path, name, split=None, streaming=False):
        calls.update(path=path, name=name, split=split, streaming=streaming)
        return iter(records)

    monkeypatch.setattr("datasets.load_dataset", _load_dataset)
    monkeypatch.setattr("axolotl.loaders.load_tokenizer", lambda _cfg: _Tokenizer())
    return calls


def test_collect_eval_batches_chunks_the_corpus(cfg, monkeypatch):
    calls = _stub_corpus(monkeypatch, [{"text": "abcdefgh" * 16} for _ in range(8)])

    batches = collect_eval_batches(cfg, num_sequences=4, sequence_len=32)

    assert len(batches) == 4
    assert all(batch.shape == (1, 32) for batch in batches)
    assert all(batch.dtype == torch.long for batch in batches)
    assert calls == {
        "path": "stub/corpus",
        "name": None,
        "split": "train",
        "streaming": True,
    }


def test_collect_eval_batches_prefers_the_explicit_dataset(cfg, monkeypatch):
    calls = _stub_corpus(monkeypatch, [{"text": "abcdefgh" * 16}])

    collect_eval_batches(cfg, num_sequences=1, sequence_len=16, dataset="other/corpus")

    assert calls["path"] == "other/corpus"


def test_collect_eval_batches_skips_records_without_text(cfg, monkeypatch):
    _stub_corpus(
        monkeypatch,
        [{"image": None}, {"content": "abcdefgh" * 8}, {"text": "ijklmnop" * 8}],
    )

    batches = collect_eval_batches(cfg, num_sequences=2, sequence_len=32)

    assert len(batches) == 2


def test_collect_eval_batches_rejects_a_short_corpus(cfg, monkeypatch):
    _stub_corpus(monkeypatch, [{"text": "abc"}])

    with pytest.raises(ValueError, match="too few for one"):
        collect_eval_batches(cfg, num_sequences=2, sequence_len=32)


def test_collect_eval_batches_names_the_missing_text_field(cfg, monkeypatch):
    """An instruction corpus is untokenizable here, and no --sequence-len fixes that."""
    _stub_corpus(
        monkeypatch,
        [{"instruction": "add two numbers", "output": "4"} for _ in range(4)],
    )

    with pytest.raises(ValueError, match="raw-text fields") as excinfo:
        collect_eval_batches(cfg, num_sequences=2, sequence_len=32)

    assert "instruction" in str(excinfo.value)


def test_collect_eval_batches_needs_a_configured_dataset(monkeypatch):
    with pytest.raises(ValueError, match="needs a corpus"):
        collect_eval_batches(DictDefault({}), num_sequences=1, sequence_len=8)


def test_pretraining_dataset_is_used_when_there_is_no_sft_split(monkeypatch):
    cfg = DictDefault(
        {"pretraining_dataset": [{"path": "corpus", "name": "sample", "split": "val"}]}
    )
    calls = _stub_corpus(monkeypatch, [{"text": "abcdefgh" * 16}])

    collect_eval_batches(cfg, num_sequences=1, sequence_len=16)

    assert calls["name"] == "sample"
    assert calls["split"] == "val"


# ------------------------------------------------------------------- the report


def test_perplexity_is_token_weighted():
    model = _model()
    batches = _batches(count=2)

    value = perplexity(model, batches)

    assert math.isfinite(value)
    assert value > 1.0
    # a uniform-random init cannot beat the uniform distribution by much
    assert value < model.config.vocab_size * 2


def test_report_serializes_and_renders():
    report = DamageReport(
        baseline_perplexity=10.0,
        rows=[
            DamageRow("weights", 20.0, 10.0),
            DamageRow("family:down_proj", 11.5, 1.5),
        ],
        sequences=8,
        sequence_len=128,
    )

    payload = report.to_dict()
    table = report.to_table()

    assert payload["baseline_perplexity"] == 10.0
    assert payload["sequences"] == 8
    assert payload["rows"][0] == {
        "scope": "weights",
        "perplexity": 20.0,
        "delta": 10.0,
    }
    assert "8 sequences x 128 tokens" in table
    assert "baseline" in table
    assert "+10.0000" in table
    assert "family:down_proj" in table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="damage map on CUDA")
def test_damage_map_on_cuda(cfg, monkeypatch):
    batches = _batches()
    monkeypatch.setattr(damage, "load_eval_model", lambda _cfg: _model().cuda())
    monkeypatch.setattr(
        damage, "collect_eval_batches", lambda _cfg, _num, _len, _dataset=None: batches
    )

    report = damage_map(cfg, num_sequences=3, sequence_len=16)

    assert len(report.rows) == len(GLOBAL_SCOPES) + len(FAMILIES)
    assert math.isfinite(report.baseline_perplexity)
