"""Projection kernel, density, cache lifecycle, and SFT masking tests."""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from transformers import GPT2Config, GPT2LMHeadModel

from axolotl.integrations.projection_sampling.args import (
    ProjectionSamplingArgs,
    ProjectionSamplingConfig,
)
from axolotl.integrations.projection_sampling.backend import TransformersBackend
from axolotl.integrations.projection_sampling.plugin import (
    ProjectionSamplingPlugin,
    cache_path,
)
from axolotl.integrations.projection_sampling.sampler import ProjectionSampler
from axolotl.integrations.projection_sampling.tokenization import load
from axolotl.utils.dict import DictDefault


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 7
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [1, 2]

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))

    def apply_chat_template(self, messages, **kwargs):
        return [1, 3]


class ScriptedBackend:
    tokenizer = TinyTokenizer()
    eos_token_ids = {7}

    def __init__(self, sequences, target=None, proposal=None):
        self.sequences = iter(sequences)
        self.target = target or (lambda tokens: -float(len(tokens)))
        self.proposal = proposal or (lambda tokens: -float(len(tokens)))
        self.calls = []
        self.closed = False

    def sample(self, context, max_tokens):
        self.calls.append(("sample", context.copy(), max_tokens))
        return next(self.sequences)

    def score(self, context, tokens, *, proposal):
        self.calls.append(
            ("proposal" if proposal else "target", context.copy(), tokens.copy())
        )
        return (self.proposal if proposal else self.target)(tokens)

    def close(self):
        self.closed = True


class FixedRNG:
    def __init__(self, index=0, uniform=0.5):
        self.index = index
        self.uniform = uniform

    def randrange(self, length):
        assert self.index < length
        return self.index

    def random(self):
        return self.uniform


def test_mh_accepts_lower_target_with_proposal_correction():
    backend = ScriptedBackend(
        [[2, 7], [3, 7]],
        target=lambda tokens: -2.0 if tokens[0] == 2 else -4.0,
        proposal=lambda tokens: -1.0 if tokens[0] == 2 else -5.0,
    )
    sampler = ProjectionSampler(
        backend, ProjectionSamplingConfig(block_size=2, max_new_tokens=2, mcmc_steps=1)
    )
    sampler.rng = FixedRNG()
    result = sampler.sample("question", "privileged")
    assert result.token_ids == [3, 7]
    assert result.accepted == result.attempts == 1
    densities = [call for call in backend.calls if call[0] == "proposal"]
    assert densities[0][1] == densities[1][1]
    assert densities[0][2] == [3, 7]
    assert densities[1][2] == [2, 7]


def test_mh_rejects_higher_target_with_unfavorable_proposal():
    backend = ScriptedBackend(
        [[2, 7], [3, 7]],
        target=lambda tokens: -4.0 if tokens[0] == 2 else -2.0,
        proposal=lambda tokens: -5.0 if tokens[0] == 2 else -1.0,
    )
    sampler = ProjectionSampler(
        backend, ProjectionSamplingConfig(block_size=2, max_new_tokens=2, mcmc_steps=1)
    )
    sampler.rng = FixedRNG()
    result = sampler.sample("question", "expert")
    assert result.token_ids == [2, 7]
    assert result.accepted == 0


def test_eos_length_changes_include_cut_probability():
    backend = ScriptedBackend(
        [[2, 7], [7]], target=lambda tokens: 0.0, proposal=lambda tokens: 0.0
    )
    sampler = ProjectionSampler(
        backend, ProjectionSamplingConfig(block_size=2, max_new_tokens=2, mcmc_steps=1)
    )
    sampler.rng = FixedRNG(uniform=0.75)
    assert sampler.sample("question", "expert").token_ids == [7]
    backend = ScriptedBackend(
        [[7], [2, 7]], target=lambda tokens: 0.0, proposal=lambda tokens: 0.0
    )
    sampler = ProjectionSampler(
        backend, ProjectionSamplingConfig(block_size=2, max_new_tokens=2, mcmc_steps=1)
    )
    sampler.rng = FixedRNG(uniform=0.75)
    assert sampler.sample("question", "expert").token_ids == [7]


def test_greedy_uses_mean_likelihood_and_does_not_score_proposals():
    backend = ScriptedBackend(
        [[2, 3], [7]], target=lambda tokens: -3.0 if len(tokens) == 2 else -2.0
    )
    sampler = ProjectionSampler(
        backend,
        ProjectionSamplingConfig(
            acceptance="greedy", block_size=2, max_new_tokens=2, mcmc_steps=1
        ),
    )
    sampler.rng = FixedRNG()
    result = sampler.sample("question", "expert")
    assert result.token_ids == [2, 3]
    assert not result.finished
    assert not any(call[0] == "proposal" for call in backend.calls)


def test_partial_last_block_and_rewrite_baseline():
    backend = ScriptedBackend([[2, 3], [4, 5], [7]])
    sampler = ProjectionSampler(
        backend, ProjectionSamplingConfig(block_size=2, max_new_tokens=5, mcmc_steps=0)
    )
    result = sampler.sample("question", "expert")
    assert result.token_ids == [2, 3, 4, 5, 7]
    assert [call[2] for call in backend.calls if call[0] == "sample"] == [2, 2, 1]
    assert result.attempts == result.accepted == 0


def test_reverse_proposal_rescores_current_suffix_under_new_cut():
    backend = ScriptedBackend([[2, 3, 7], [4, 7]])
    sampler = ProjectionSampler(
        backend, ProjectionSamplingConfig(block_size=3, max_new_tokens=3, mcmc_steps=1)
    )
    sampler.rng = FixedRNG(index=1)
    sampler.sample("question", "expert")
    densities = [call for call in backend.calls if call[0] == "proposal"]
    assert densities[0][1][-1] == 2
    assert densities[0][2] == [4, 7]
    assert densities[1][2] == [3, 7]


@pytest.mark.parametrize("sequence", [[], [2], [2, 7, 3], [2, 3, 4]])
def test_invalid_generation_fails(sequence):
    backend = ScriptedBackend([sequence])
    sampler = ProjectionSampler(
        backend, ProjectionSamplingConfig(block_size=2, max_new_tokens=2)
    )
    with pytest.raises(ValueError, match="Proposal"):
        sampler.sample("question", "expert")


@pytest.fixture
def real_backend():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=8,
                n_positions=128,
                n_embd=16,
                n_layer=1,
                n_head=2,
                bos_token_id=1,
                eos_token_id=7,
                pad_token_id=0,
            )
        )
    return TransformersBackend(
        model,
        TinyTokenizer(),
        ProjectionSamplingConfig(
            device="cpu", dtype="float32", temperature=0.6, repetition_penalty=1.1
        ),
    )


def test_teacher_forcing_matches_generation_log_densities(real_backend):
    from transformers import GenerationConfig

    context = [1, 2, 1]
    with torch.inference_mode():
        output = real_backend.model.generate(
            torch.tensor([context]),
            attention_mask=torch.ones(1, len(context), dtype=torch.long),
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=0.6,
                repetition_penalty=1.1,
                top_k=0,
                top_p=1.0,
                max_new_tokens=3,
                eos_token_id=7,
                pad_token_id=0,
            ),
            return_dict_in_generate=True,
            output_scores=True,
        )
    tokens = output.sequences[0, len(context) :].tolist()
    expected = sum(
        scores[0].log_softmax(-1)[token].item()
        for scores, token in zip(output.scores, tokens, strict=True)
    )
    assert real_backend.score(context, tokens, proposal=True) == pytest.approx(
        expected, abs=1e-6
    )
    ids = torch.tensor([context + tokens])
    with torch.inference_mode():
        logits = real_backend.model(ids).logits[0, len(context) - 1 : -1].float()
    target = (
        logits.log_softmax(-1).gather(-1, torch.tensor(tokens)[:, None]).sum().item()
    )
    assert real_backend.score(context, tokens, proposal=False) == pytest.approx(
        target, abs=1e-6
    )


def test_real_model_sampling_and_context_limit(real_backend):
    sampler = ProjectionSampler(
        real_backend,
        ProjectionSamplingConfig(
            prompt_format="raw", block_size=2, max_new_tokens=3, mcmc_steps=1
        ),
    )
    result = sampler.sample("question", "expert")
    assert 1 <= len(result.token_ids) <= 3
    assert math.isfinite(result.target_logprob)
    with pytest.raises(ValueError, match="exceeds model context"):
        real_backend.sample([1] * 128, 1)


@pytest.mark.parametrize("train_on_inputs", [False, True])
def test_cached_tokens_mask_question_and_preserve_eos(train_on_inputs):
    strategy = load(
        TinyTokenizer(), DictDefault(train_on_inputs=train_on_inputs, sequence_len=64)
    )
    record = {
        "prompt_token_ids": [1, 3],
        "response_token_ids": [2, 7],
        "expert_response": "private",
    }
    tokenized = strategy.tokenize_prompt(record)
    assert tokenized["input_ids"] == [1, 3, 2, 7]
    assert tokenized["labels"] == (
        [1, 3, 2, 7] if train_on_inputs else [-100, -100, 2, 7]
    )


@pytest.fixture
def cfg(tmp_path):
    source = tmp_path / "experts.jsonl"
    source.write_text(json.dumps({"prompt": "question", "response": "expert"}) + "\n")
    return DictDefault(
        base_model="tiny",
        datasets=[{"path": str(source), "ds_type": "json", "split": "train"}],
        projection_sampling={
            "cache_dir": str(tmp_path / "cache"),
            "device": "cpu",
            "block_size": 2,
            "max_new_tokens": 2,
            "mcmc_steps": 0,
        },
        test_datasets=[{"path": "held-out.jsonl", "type": "chat_template"}],
    )


def test_fingerprint_tracks_data_and_sampling_but_not_training(cfg):
    config = ProjectionSamplingConfig.model_validate(cfg.projection_sampling)
    original = cache_path(cfg, config)
    cfg.learning_rate = 0.001
    assert cache_path(cfg, config) == original
    config.temperature = 0.8
    assert cache_path(cfg, config) != original
    config.temperature = 0.6
    Path(cfg.datasets[0].path).write_text(
        '{"prompt": "changed", "response": "expert"}\n'
    )
    assert cache_path(cfg, config) != original


def test_training_requires_cache_without_loading_model(cfg, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Training loaded the sampling model")

    monkeypatch.setattr(TransformersBackend, "from_config", fail)
    with pytest.raises(FileNotFoundError, match="axolotl preprocess"):
        ProjectionSamplingPlugin().load_datasets(cfg)


@pytest.mark.parametrize(
    "sequence,verified,fallback",
    [([2, 7], True, False), ([2, 7], False, True), ([2, 3], True, True)],
)
def test_preprocess_cache_reuse_verification_and_fallback(
    cfg, monkeypatch, sequence, verified, fallback
):
    import axolotl.common.datasets as common

    backend = ScriptedBackend([sequence])
    monkeypatch.setattr(TransformersBackend, "from_config", lambda *args: backend)
    monkeypatch.setitem(
        sys.modules, "my_verifier", SimpleNamespace(check=lambda **kwargs: verified)
    )
    cfg.projection_sampling.verifier = "my_verifier.check"
    loaded = []
    monkeypatch.setattr(
        common,
        "load_datasets",
        lambda **kwargs: loaded.append(kwargs["cfg"]) or "metadata",
    )
    source_datasets = cfg.to_dict()["datasets"]
    torch_state = torch.random.get_rng_state().clone()
    plugin = ProjectionSamplingPlugin()
    assert plugin.load_datasets(cfg, preprocess=True) == "metadata"
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert backend.closed
    config = ProjectionSamplingConfig.model_validate(cfg.projection_sampling)
    path = cache_path(cfg, config)
    record = json.loads(path.read_text())
    assert record["sampling"]["fallback_to_expert"] is fallback
    assert record["response_token_ids"] == ([1, 2, 7] if fallback else sequence)
    assert loaded[0].test_datasets == cfg.test_datasets
    assert cfg.to_dict()["datasets"] == source_datasets
    monkeypatch.setattr(
        TransformersBackend, "from_config", lambda *args: pytest.fail("cache miss")
    )
    assert plugin.load_datasets(cfg) == "metadata"
    assert plugin.load_datasets(cfg, preprocess=True) == "metadata"
    assert not list(path.parent.glob("tmp*.jsonl"))


def test_failed_preprocessing_never_publishes_partial_cache(cfg, monkeypatch):
    backend = ScriptedBackend([[]])
    monkeypatch.setattr(TransformersBackend, "from_config", lambda *args: backend)
    config = ProjectionSamplingConfig.model_validate(cfg.projection_sampling)
    path = cache_path(cfg, config)
    path.parent.mkdir()
    with pytest.raises(ValueError, match="Proposal"):
        ProjectionSamplingPlugin()._generate_cache(cfg, config, path)
    assert not path.exists()
    assert not list(path.parent.glob("*.jsonl"))
    assert backend.closed


def test_disabled_plugin_and_distributed_preprocess(cfg, monkeypatch):
    plugin = ProjectionSamplingPlugin()
    assert plugin.load_datasets(DictDefault()) is None
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single process"):
        plugin.load_datasets(cfg, preprocess=True)


@pytest.mark.parametrize(
    "key,value",
    [
        ("temperature", 0),
        ("temperature", float("nan")),
        ("block_size", 0),
        ("mcmc_steps", -1),
        ("acceptance", "unknown"),
        ("device", "mps"),
        ("proposal_template", "{question} {invalid}"),
    ],
)
def test_config_rejects_invalid_sampling(key, value):
    with pytest.raises(ValidationError):
        ProjectionSamplingConfig(**{key: value})


@pytest.mark.parametrize(
    "key",
    [
        "rl",
        "streaming",
        "pretraining_dataset",
        "skip_prepare_dataset",
        "processor_type",
    ],
)
def test_config_rejects_incompatible_training(key):
    with pytest.raises(ValidationError, match=key):
        ProjectionSamplingArgs.model_validate({"projection_sampling": {}, key: True})


def test_zero_probability_proposal_is_never_accepted():
    backend = ScriptedBackend(
        [[2, 7], [3, 7]],
        target=lambda tokens: 0.0 if tokens[0] == 2 else -math.inf,
        proposal=lambda tokens: 0.0,
    )
    sampler = ProjectionSampler(
        backend, ProjectionSamplingConfig(block_size=2, max_new_tokens=2, mcmc_steps=1)
    )
    sampler.rng = FixedRNG(uniform=0.0)
    assert sampler.sample("question", "expert").token_ids == [2, 7]


@pytest.mark.parametrize("key", ["input_transform", "preprocess_shards"])
def test_unsupported_source_options_fail_before_model_loading(cfg, key):
    cfg.datasets[0][key] = "unsupported"
    with pytest.raises(ValueError, match=key):
        ProjectionSamplingPlugin().load_datasets(cfg, preprocess=True)


def test_cache_accepts_inline_jinja_and_tracks_local_globs(cfg, tmp_path):
    cfg.chat_template_jinja = "{{ messages[0]['content'] }}" * 50
    config = ProjectionSamplingConfig.model_validate(cfg.projection_sampling)
    original = cache_path(cfg, config)
    cfg.chat_template_jinja += "extra"
    assert cache_path(cfg, config) != original
    cfg.datasets[0].data_files = str(tmp_path / "*.jsonl")
    original = cache_path(cfg, config)
    (tmp_path / "additional.jsonl").write_text(
        '{"prompt": "new", "response": "answer"}\n'
    )
    assert cache_path(cfg, config) != original
