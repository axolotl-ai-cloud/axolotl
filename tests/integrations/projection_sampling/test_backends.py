"""Backend density semantics, pluggability, lazy imports, and resource ownership."""

import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pydantic import ValidationError

from axolotl.integrations.projection_sampling.args import ProjectionSamplingConfig
from axolotl.integrations.projection_sampling.backend import (
    SamplingBackend,
    load_backend,
    resolve_backend,
)
from axolotl.integrations.projection_sampling.backends.vllm import VLLMBackend
from axolotl.integrations.projection_sampling.plugin import cache_path
from axolotl.utils.dict import DictDefault


class FakeSamplingParams:
    __struct_fields__ = ("logprob_token_ids",)

    def __init__(self, **kwargs):
        self.prompt_logprobs = None
        self.logprobs = None
        self.logprob_token_ids = None
        self.stop_token_ids = []
        self.__dict__.update(kwargs)


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 7

    def __len__(self):
        return 8


class FakeEngine:
    """A minimal runtime whose prompt scores deliberately stay unprocessed."""

    def __init__(self):
        self.shutdown = Mock()
        self.llm_engine = SimpleNamespace(
            model_config=SimpleNamespace(
                max_model_len=128,
                hf_config=SimpleNamespace(eos_token_id=[6, 7]),
                get_vocab_size=lambda: 8,
            ),
            engine_core=SimpleNamespace(shutdown=self.shutdown),
        )
        self.calls = []

    @staticmethod
    def logits(context):
        return torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 0.1 * len(context), -0.5, 0.5])

    def distribution(self, context, params):
        logits = self.logits(context)
        indices = torch.tensor(sorted(set(context)))
        scores = logits[indices]
        logits[indices] = torch.where(
            scores < 0,
            scores * params.repetition_penalty,
            scores / params.repetition_penalty,
        )
        return (logits / params.temperature).log_softmax(-1)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.calls.append((prompts, sampling_params))
        parameters = (
            sampling_params
            if isinstance(sampling_params, list)
            else [sampling_params] * len(prompts)
        )
        outputs = []
        for prompt, params in zip(prompts, parameters, strict=True):
            context = prompt["prompt_token_ids"]
            prompt_scores = None
            if params.prompt_logprobs is not None:
                prompt_scores = [None] + [
                    {
                        token: SimpleNamespace(
                            logprob=self.logits(context[:position])
                            .log_softmax(-1)[token]
                            .item()
                        )
                    }
                    for position, token in enumerate(context[1:], 1)
                ]
            tokens, scores = [], []
            generator = torch.Generator().manual_seed(params.seed)
            for _ in range(params.max_tokens):
                values = self.distribution(context + tokens, params)
                chosen = torch.multinomial(values.exp(), 1, generator=generator).item()
                tokens.append(chosen)
                wanted = params.logprob_token_ids or (
                    range(8) if params.logprobs == -1 else [chosen]
                )
                scores.append(
                    {
                        token: SimpleNamespace(logprob=values[token].item())
                        for token in wanted
                    }
                )
                if chosen in params.stop_token_ids:
                    break
            outputs.append(
                SimpleNamespace(
                    prompt_logprobs=prompt_scores,
                    outputs=[SimpleNamespace(token_ids=tokens, logprobs=scores)],
                )
            )
        return outputs


@pytest.fixture
def vllm_backend(monkeypatch):
    module = ModuleType("vllm")
    module.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    engine = FakeEngine()
    config = ProjectionSamplingConfig(
        backend="vllm",
        temperature=0.6,
        repetition_penalty=1.1,
        backend_kwargs={"score_batch_size": 2},
    )
    backend = VLLMBackend(engine, FakeTokenizer(), config)
    try:
        yield backend
    finally:
        backend.close()


def test_vllm_target_excludes_prompt_and_ignores_proposal_controls(vllm_backend):
    context, tokens = [1, 3], [3, 4, 7]
    expected = sum(
        FakeEngine.logits(context + tokens[:index]).log_softmax(-1)[token].item()
        for index, token in enumerate(tokens)
    )
    assert vllm_backend.target_logprob(context, tokens) == pytest.approx(expected)
    prompts, params = vllm_backend.engine.calls[0]
    assert prompts[0]["prompt_token_ids"] == context + tokens
    assert params.prompt_logprobs == 0
    assert params.temperature == params.repetition_penalty == 1.0


def test_vllm_nested_text_config_eos(vllm_backend):
    engine = vllm_backend.engine
    engine.llm_engine.model_config.hf_config = SimpleNamespace(
        text_config=SimpleNamespace(eos_token_id=6)
    )
    backend = VLLMBackend(engine, FakeTokenizer(), vllm_backend.config)
    assert backend.eos_token_ids == {6}


@pytest.mark.parametrize("specific_logprobs", [True, False])
def test_vllm_proposal_matches_processed_distribution_in_bounded_batches(
    vllm_backend, specific_logprobs
):
    vllm_backend.specific_logprobs = specific_logprobs
    context, tokens = [1, 3], [3, 4, 7]
    params = FakeSamplingParams(temperature=0.6, repetition_penalty=1.1)
    expected = sum(
        vllm_backend.engine.distribution(context + tokens[:index], params)[token].item()
        for index, token in enumerate(tokens)
    )
    assert vllm_backend.proposal_logprob(context, tokens) == pytest.approx(expected)
    assert [len(prompts) for prompts, _ in vllm_backend.engine.calls] == [2, 1]
    prefixes = [
        prompt["prompt_token_ids"]
        for prompts, _ in vllm_backend.engine.calls
        for prompt in prompts
    ]
    assert prefixes == [context, context + [3], context + [3, 4]]
    for _, parameters in vllm_backend.engine.calls:
        for parameter in parameters:
            assert parameter.prompt_logprobs is None
            assert parameter.temperature == 0.6
            assert parameter.repetition_penalty == 1.1
            assert (
                parameter.logprobs is None
                if specific_logprobs
                else parameter.logprobs == -1
            )


def test_vllm_rescoring_matches_actual_generation_scores(vllm_backend):
    context = [1, 3]
    params = vllm_backend._params(
        4, proposal=True, logprobs=0, seed=17, stop_token_ids=[6, 7]
    )
    completion = vllm_backend.engine.generate([{"prompt_token_ids": context}], params)[
        0
    ].outputs[0]
    observed = sum(
        row[token].logprob
        for row, token in zip(completion.logprobs, completion.token_ids, strict=True)
    )
    assert vllm_backend.proposal_logprob(
        context, completion.token_ids
    ) == pytest.approx(observed)


def test_scoring_does_not_advance_generation_seeds(vllm_backend):
    vllm_backend.sample([1, 3], 3)
    vllm_backend.proposal_logprob([1, 3], [4, 7])
    vllm_backend.target_logprob([1, 3], [4, 7])
    vllm_backend.sample([1, 3], 3)
    calls = vllm_backend.engine.calls
    assert calls[0][1].seed == 42
    assert calls[-1][1].seed == 43
    assert calls[0][1].ignore_eos
    assert calls[0][1].stop_token_ids == [6, 7]
    assert calls[0][1].top_k == -1
    assert calls[0][1].top_p == 1.0


def test_target_handles_exact_context_limit(vllm_backend):
    context, tokens = [1, 3], [4, 7]
    expected = vllm_backend.target_logprob(context, tokens)
    vllm_backend.engine.calls.clear()
    vllm_backend.max_model_len = 4
    assert vllm_backend.target_logprob(context, tokens) == pytest.approx(expected)
    assert len(vllm_backend.engine.calls) == 2
    for prompts, _ in vllm_backend.engine.calls:
        assert all(len(prompt["prompt_token_ids"]) < 4 for prompt in prompts)
    with pytest.raises(ValueError, match="context length"):
        vllm_backend.sample(context, 3)


def test_identity_proposal_uses_fast_prompt_scoring(vllm_backend):
    vllm_backend.config.temperature = vllm_backend.config.repetition_penalty = 1.0
    assert vllm_backend.proposal_logprob([1], [3, 7]) == pytest.approx(
        vllm_backend.target_logprob([1], [3, 7])
    )
    assert all(not isinstance(params, list) for _, params in vllm_backend.engine.calls)


def test_missing_or_nonfinite_token_scores_fail_loudly(vllm_backend):
    with pytest.raises(ValueError, match="requested token"):
        vllm_backend._logprob({}, 3)
    with pytest.raises(ValueError, match="non-finite"):
        vllm_backend._logprob({3: SimpleNamespace(logprob=math.nan)}, 3)
    vllm_backend.engine.generate = lambda *args, **kwargs: [
        SimpleNamespace(prompt_logprobs=None)
    ]
    with pytest.raises(ValueError, match="missing or misaligned"):
        vllm_backend.target_logprob([1], [3])


def test_backend_close_is_idempotent(vllm_backend):
    shutdown = vllm_backend.engine.shutdown
    vllm_backend.close()
    vllm_backend.close()
    shutdown.assert_called_once()


class ExternalBackend(SamplingBackend):
    @classmethod
    def from_config(cls, cfg, config):
        backend = cls()
        backend.close = Mock()
        backend.options = config.backend_kwargs
        return backend

    def sample(self, context, max_tokens):
        return [7]

    def target_logprob(self, context, tokens):
        return -1.0

    def proposal_logprob(self, context, tokens):
        return -1.0

    def close(self):
        pass


@pytest.mark.parametrize("fails", [False, True])
def test_external_backend_plugs_in_and_closes_on_success_or_failure(monkeypatch, fails):
    module = ModuleType("external_projection_backend")
    module.ExternalBackend = ExternalBackend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = ProjectionSamplingConfig(
        backend="external_projection_backend.ExternalBackend",
        backend_kwargs={"server_url": "https://example.com"},
    )
    assert resolve_backend(config.backend) is ExternalBackend
    try:
        with load_backend(DictDefault(), config) as backend:
            assert backend.options == config.backend_kwargs
            if fails:
                raise RuntimeError("sampling failed")
    except RuntimeError:
        assert fails
    backend.close.assert_called_once()


def test_invalid_backend_contract_and_unknown_alias_fail():
    with pytest.raises(TypeError, match="subclass SamplingBackend"):
        resolve_backend("types.SimpleNamespace")
    with pytest.raises(ValueError, match="Unknown"):
        resolve_backend("missing")
    with pytest.raises(TypeError, match="abstract"):
        SamplingBackend()


def test_backend_imports_are_lazy():
    root = Path(__file__).resolve().parents[3]
    script = """
import sys
from axolotl.integrations.projection_sampling.backend import resolve_backend
assert 'vllm' not in sys.modules
assert 'torch' not in sys.modules
resolve_backend('vllm')
assert 'vllm' not in sys.modules
assert 'torch' not in sys.modules
assert 'axolotl.integrations.projection_sampling.plugin' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], cwd=root, check=True)


def test_cache_identity_changes_with_backend_and_options(tmp_path):
    cfg = DictDefault(base_model="tiny", datasets=[{"path": "experts"}])
    config = ProjectionSamplingConfig(cache_dir=str(tmp_path))
    default = cache_path(cfg, config)
    config.backend = "vllm"
    assert cache_path(cfg, config) != default
    selected = cache_path(cfg, config)
    config.backend_kwargs = {"tensor_parallel_size": 2}
    assert cache_path(cfg, config) != selected


@pytest.mark.parametrize(
    "settings",
    [
        {"backend": "vllm", "device": "cpu"},
        {"backend": "vllm", "device": "cuda:1"},
        {"backend": "vllm", "temperature": 0.001},
        {"backend": "vllm", "backend_kwargs": {"tensor_parallel_size": 0}},
        {"backend": "vllm", "backend_kwargs": {"gpu_memory_utilization": 1.5}},
        {"backend": "vllm", "backend_kwargs": {"logprobs_mode": "raw_logprobs"}},
        {"backend": "transformers", "backend_kwargs": {"unrecognized": True}},
        {"backend": "sglang"},
    ],
)
def test_backend_config_rejects_unsupported_or_density_changing_settings(settings):
    with pytest.raises(ValidationError):
        ProjectionSamplingConfig(**settings)


def test_vllm_factory_uses_top_level_seed(vllm_backend, monkeypatch):
    from transformers import GenerationConfig

    import axolotl.loaders as loaders

    engine = FakeEngine()
    module = sys.modules["vllm"]
    module.LLM = Mock(return_value=engine)
    monkeypatch.setattr(loaders, "load_tokenizer", lambda cfg: FakeTokenizer())
    monkeypatch.setattr(
        GenerationConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(eos_token_id=7),
    )
    cfg = DictDefault(base_model="tiny", seed=0)
    backend = VLLMBackend.from_config(cfg, vllm_backend.config)
    try:
        assert module.LLM.call_args.kwargs["seed"] == 0
        backend.sample([1, 3], 2)
        backend.proposal_logprob([1, 3], [4, 7])
        backend.sample([1, 3], 2)
        assert engine.calls[0][1].seed == 0
        assert engine.calls[-1][1].seed == 1
    finally:
        backend.close()
