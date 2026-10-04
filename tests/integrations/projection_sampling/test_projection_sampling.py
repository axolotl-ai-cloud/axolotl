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
from axolotl.integrations.projection_sampling.backend import SamplingBackend
from axolotl.integrations.projection_sampling.backends.transformers import (
    TransformersBackend,
)
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


class ScriptedBackend(SamplingBackend):
    tokenizer = TinyTokenizer()
    eos_token_ids = {7}

    def __init__(self, sequences, target=None, proposal=None):
        self.sequences = iter(sequences)
        self.target = target or (lambda tokens: -float(len(tokens)))
        self.proposal = proposal or (lambda tokens: -float(len(tokens)))
        self.calls = []
        self.closed = False

    @classmethod
    def from_config(cls, cfg, config):
        raise NotImplementedError

    def sample(self, context, max_tokens):
        self.calls.append(("sample", context.copy(), max_tokens))
        return next(self.sequences)

    def target_logprob(self, context, tokens):
        return self._score(context, tokens, proposal=False)

    def proposal_logprob(self, context, tokens):
        return self._score(context, tokens, proposal=True)

    def _score(self, context, tokens, *, proposal):
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


@pytest.mark.parametrize("proposal_batch_size", [1, 2])
def test_greedy_does_not_regrow_an_eos_shortened_proposal(proposal_batch_size):
    class BudgetBackend(ScriptedBackend):
        def sample(self, context, max_tokens):
            return super().sample(context, max_tokens)[:max_tokens]

    values = {2: -3.0, 7: -3.0, 3: -4.0, 4: -4.0, 5: -0.1, 6: -0.1}
    backend = BudgetBackend(
        [[2, 7]] + [[3, 4, 5, 6]] * proposal_batch_size,
        target=lambda tokens: sum(values[token] for token in tokens),
    )
    sampler = ProjectionSampler(
        backend,
        ProjectionSamplingConfig(
            acceptance="greedy",
            block_size=4,
            max_new_tokens=4,
            mcmc_steps=1,
            proposal_batch_size=proposal_batch_size,
        ),
    )
    sampler.rng = FixedRNG()
    result = sampler.sample("question", "expert")
    assert result.token_ids == [2, 7]
    assert result.finished
    assert result.accepted == 0
    assert [call[2] for call in backend.calls if call[0] == "sample"] == [
        4,
        *([2] * proposal_batch_size),
    ]


def test_greedy_extends_current_length_after_losing_eos():
    backend = ScriptedBackend(
        [[2, 7], [3, 4], [5, 5, 5, 5], [3, 4, 5, 5, 5, 7]],
        target=lambda tokens: -6.0 if tokens[-1] == 7 else -2.0 * len(tokens),
    )
    sampler = ProjectionSampler(
        backend,
        ProjectionSamplingConfig(
            acceptance="greedy", block_size=4, max_new_tokens=8, mcmc_steps=1
        ),
    )
    sampler.rng = FixedRNG()
    result = sampler.sample("question", "expert")
    assert result.token_ids == [3, 4, 5, 5, 5, 7]
    assert result.finished
    assert [call[2] for call in backend.calls if call[0] == "sample"] == [4, 2, 4, 6]


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
    assert real_backend.proposal_logprob(context, tokens) == pytest.approx(
        expected, abs=1e-6
    )
    ids = torch.tensor([context + tokens])
    with torch.inference_mode():
        logits = real_backend.model(ids).logits[0, len(context) - 1 : -1].float()
    target = (
        logits.log_softmax(-1).gather(-1, torch.tensor(tokens)[:, None]).sum().item()
    )
    assert real_backend.target_logprob(context, tokens) == pytest.approx(
        target, abs=1e-6
    )


@pytest.mark.parametrize(
    "defaults",
    [
        {"forced_eos_token_id": 7},
        {"min_new_tokens": 3},
        {"suppress_tokens": [2]},
        {"min_p": 0.9},
    ],
)
def test_model_generation_filters_do_not_change_proposal_density(
    real_backend, monkeypatch, defaults
):
    from copy import deepcopy

    model = real_backend.model
    for name, value in defaults.items():
        setattr(model.generation_config, name, value)
    model.generation_config.eos_token_id = 7
    backend = TransformersBackend(model, TinyTokenizer(), real_backend.config)
    generate = model.generate
    captured = []

    def capture(**kwargs):
        config = deepcopy(kwargs["generation_config"])
        config.return_dict_in_generate = True
        config.output_scores = True
        kwargs["generation_config"] = config
        output = generate(**kwargs)
        captured.append(output)
        return output.sequences

    monkeypatch.setattr(model, "generate", capture)
    context = [1, 2, 1]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        tokens = backend.sample(context, 3)
    expected = sum(
        scores[0].log_softmax(-1)[token].item()
        for scores, token in zip(captured[0].scores, tokens, strict=True)
    )
    assert backend.eos_token_ids == {7}
    assert backend.proposal_logprob(context, tokens) == pytest.approx(
        expected, abs=1e-6
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
        output_dir=str(tmp_path / "output"),
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


@pytest.mark.parametrize("weight", [0.5, 0.01])
def test_small_positive_weight_retains_a_nonempty_source(cfg, monkeypatch, weight):
    cfg.datasets[0].weight = weight
    backend = ScriptedBackend([[2, 7]])
    monkeypatch.setattr(TransformersBackend, "from_config", lambda *args: backend)
    config = ProjectionSamplingConfig.model_validate(cfg.projection_sampling)
    path = cache_path(cfg, config)
    path.parent.mkdir()
    ProjectionSamplingPlugin()._generate_cache(cfg, config, path)
    assert len(path.read_text().splitlines()) == 1
    assert backend.closed


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
    inspection = Path(cfg.output_dir) / "projection-sampling" / "rewritten.jsonl"
    readable = json.loads(inspection.read_text())
    assert readable["response"] == record["response"]
    assert readable["sampling"]["fallback_to_expert"] is fallback
    assert readable["sampling_seed"] == 42
    assert "prompt_token_ids" not in readable
    assert "response_token_ids" not in readable
    assert loaded[0].test_datasets == cfg.test_datasets
    assert cfg.to_dict()["datasets"] == source_datasets
    monkeypatch.setattr(
        TransformersBackend, "from_config", lambda *args: pytest.fail("cache miss")
    )
    assert plugin.load_datasets(cfg) == "metadata"
    inspection.unlink()
    assert plugin.load_datasets(cfg, preprocess=True) == "metadata"
    assert json.loads(inspection.read_text()) == readable
    cfg.output_dir = str(Path(cfg.output_dir).parent / "another-run")
    assert plugin.load_datasets(cfg) == "metadata"
    assert (
        json.loads(
            (
                Path(cfg.output_dir) / "projection-sampling" / "rewritten.jsonl"
            ).read_text()
        )
        == readable
    )
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


def test_sampling_eot_tokens_invalidate_cache(cfg):
    config = ProjectionSamplingConfig.model_validate(cfg.projection_sampling)
    original = cache_path(cfg, config)
    cfg.eot_tokens = ["<turn_end>"]
    assert cache_path(cfg, config) != original


def test_top_level_seed_resamples_and_reuses_same_seed(cfg, monkeypatch):
    import axolotl.common.datasets as common

    seeds = []
    backends = []

    def make_backend(training_cfg, config):
        seeds.append(torch.initial_seed())
        backend = ScriptedBackend([[2, 7]])
        backends.append(backend)
        return backend

    monkeypatch.setattr(TransformersBackend, "from_config", make_backend)
    monkeypatch.setattr(common, "load_datasets", lambda **kwargs: "prepared")
    plugin = ProjectionSamplingPlugin()
    config = ProjectionSamplingConfig.model_validate(cfg.projection_sampling)
    cfg.seed = 0
    path = cache_path(cfg, config)
    plugin.load_datasets(cfg, preprocess=True)
    original = path.read_bytes()
    plugin.load_datasets(cfg, preprocess=True)
    assert seeds == [0]
    cfg.seed = 17
    new_path = cache_path(cfg, config)
    assert new_path != path
    with pytest.raises(FileNotFoundError, match="axolotl preprocess"):
        plugin.load_datasets(cfg)
    plugin.load_datasets(cfg, preprocess=True)
    assert seeds == [0, 17]
    assert path.read_bytes() == original
    assert new_path.exists()
    assert all(backend.closed for backend in backends)
    readable = json.loads(
        (Path(cfg.output_dir) / "projection-sampling" / "rewritten.jsonl").read_text()
    )
    assert readable["sampling_seed"] == 17
    cfg.seed = 0
    plugin.load_datasets(cfg, preprocess=True)
    assert seeds == [0, 17]


def test_plugin_specific_seed_is_rejected():
    with pytest.raises(ValidationError, match="seed"):
        ProjectionSamplingConfig(seed=17)


@pytest.mark.parametrize("seed,accepted", [(0, 0), (2, 1)])
def test_top_level_seed_drives_mh_decisions(cfg, monkeypatch, seed, accepted):
    import axolotl.common.datasets as common

    cfg.seed = seed
    cfg.projection_sampling.mcmc_steps = 1
    backend = ScriptedBackend(
        [[7], [4, 7]], target=lambda tokens: 0.0, proposal=lambda tokens: 0.0
    )
    monkeypatch.setattr(TransformersBackend, "from_config", lambda *args: backend)
    monkeypatch.setattr(common, "load_datasets", lambda **kwargs: "prepared")
    ProjectionSamplingPlugin().load_datasets(cfg, preprocess=True)
    record = json.loads(
        cache_path(
            cfg, ProjectionSamplingConfig.model_validate(cfg.projection_sampling)
        ).read_text()
    )
    assert record["sampling"]["accepted"] == accepted


@pytest.mark.parametrize("ids", [[8], [8, 9]])
def test_configured_chat_eot_stops_sampling_and_invalid_eot_closes(
    cfg, monkeypatch, ids
):
    from axolotl.integrations.projection_sampling.backend import load_backend

    cfg.eot_tokens = ["<turn_end>"]
    backend = ScriptedBackend([])
    backend.tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: ids)
    monkeypatch.setattr(TransformersBackend, "from_config", lambda *args: backend)
    config = ProjectionSamplingConfig.model_validate(cfg.projection_sampling)
    if len(ids) == 1:
        with load_backend(cfg, config) as loaded:
            assert loaded.eos_token_ids == {7, 8}
    else:
        with pytest.raises(ValueError, match="single-token EOT"):
            with load_backend(cfg, config):
                pytest.fail("invalid EOT was accepted")
    assert backend.closed


def test_multiple_try_mh_selects_weighted_candidate_and_reuses_balancing_trials():
    backend = ScriptedBackend(
        [[2, 7], [3, 7], [4, 7]],
        target=lambda tokens: {2: -2.0, 3: -4.0, 4: -3.0, 5: -4.0}[tokens[0]],
        proposal=lambda tokens: -1.0 if tokens[0] == 2 else -5.0,
    )
    config = ProjectionSamplingConfig(
        block_size=2, max_new_tokens=2, mcmc_steps=1, proposal_batch_size=2
    )
    sampler = ProjectionSampler(backend, config)
    sampler.rng = FixedRNG()
    result = sampler.sample("question", "expert")
    assert result.token_ids == [4, 7]
    assert result.attempts == result.accepted == 1
    assert sampler.proposal_statistics(result) == {
        "proposal_batch_size": 2,
        "forward_proposals": 2,
        "balancing_proposals_reused": 1,
    }
    proposals = [call for call in backend.calls if call[0] == "proposal"]
    assert [call[2] for call in proposals] == [[3, 7], [4, 7], [2, 7]]
    assert all(call[1] == proposals[0][1] for call in proposals)


def test_parallel_greedy_selects_best_mean_and_avoids_reverse_proposals():
    backend = ScriptedBackend(
        [[2, 7], [3, 7], [4, 7]], target=lambda tokens: -float(6 - tokens[0])
    )
    config = ProjectionSamplingConfig(
        block_size=2,
        max_new_tokens=2,
        mcmc_steps=1,
        proposal_batch_size=2,
        acceptance="greedy",
    )
    sampler = ProjectionSampler(backend, config)
    sampler.rng = FixedRNG()
    result = sampler.sample("question", "expert")
    assert result.token_ids == [4, 7]
    assert result.accepted == 1
    assert not any(call[0] == "proposal" for call in backend.calls)
    assert sampler.proposal_statistics(result)["balancing_proposals_reused"] == 0


def test_multiple_try_rejects_misaligned_generation_batch():
    backend = ScriptedBackend([[2, 7]])
    backend.sample_batch = lambda *args: []
    sampler = ProjectionSampler(
        backend,
        ProjectionSamplingConfig(
            block_size=2, max_new_tokens=2, mcmc_steps=1, proposal_batch_size=2
        ),
    )
    with pytest.raises(ValueError, match="misaligned"):
        sampler.sample("question", "expert")


def test_multiple_try_rejects_nonfinite_weights():
    backend = ScriptedBackend(
        [[2, 7], [3, 7], [4, 7]], proposal=lambda tokens: math.nan
    )
    sampler = ProjectionSampler(
        backend,
        ProjectionSamplingConfig(
            block_size=2, max_new_tokens=2, mcmc_steps=1, proposal_batch_size=2
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        sampler.sample("question", "expert")


def test_multiple_try_preserves_target_with_variable_eos_lengths():
    import random

    class IndependenceBackend(ScriptedBackend):
        def __init__(self):
            super().__init__([])
            self.rng = random.Random(5)

        def sample(self, context, max_tokens):
            if len(context) > 2:
                return [7]
            return [7] if self.rng.random() < 0.7 else [2, 7]

        def target_logprob(self, context, tokens):
            return math.log(0.2 if len(tokens) == 1 else 0.8)

        def proposal_logprob(self, context, tokens):
            if len(context) > 2:
                return 0.0
            return math.log(0.7 if len(tokens) == 1 else 0.3)

    backend = IndependenceBackend()
    sampler = ProjectionSampler(
        backend,
        ProjectionSamplingConfig(
            block_size=2, max_new_tokens=2, proposal_batch_size=3, prompt_format="raw"
        ),
        seed=21,
    )
    current = [7]
    target = backend.target_logprob([1, 2], current)
    long_states = 0
    for step in range(20500):
        cut = sampler.rng.randrange(len(current))
        prefix = current[:cut]
        context = sampler.proposal_ids("question", "expert", prefix)
        current, target, _ = sampler._batched_step(
            [1, 2], current, target, context, prefix, 2
        )
        if step >= 500:
            long_states += int(len(current) == 2)
    assert long_states / 20000 == pytest.approx(0.8, abs=0.02)
