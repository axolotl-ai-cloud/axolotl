"""Unit tests for forwarding `vllm:` engine args to a colocated vLLM engine."""

import pytest

from axolotl.utils.dict import DictDefault

# The monkeypatch package imports triton at package level (GPU-only dependency).
trl_vllm = pytest.importorskip("axolotl.monkeypatch.trainer.trl_vllm")
colocate_vllm_engine_kwargs = trl_vllm.colocate_vllm_engine_kwargs
patch_vllm_colocate_engine_kwargs = trl_vllm.patch_vllm_colocate_engine_kwargs


class TestColocateEngineKwargs:
    """Engine options TRL has no config field for reach LLM() via the monkeypatch."""

    def test_engine_kwargs_from_vllm_block(self):
        assert colocate_vllm_engine_kwargs(
            DictDefault({"enforce_eager": True, "dtype": "auto"})
        ) == {"enforce_eager": True}
        assert colocate_vllm_engine_kwargs(
            DictDefault({"enable_prefix_caching": False, "dtype": "bfloat16"})
        ) == {"enable_prefix_caching": False, "dtype": "bfloat16"}
        assert colocate_vllm_engine_kwargs(None) == {}
        assert colocate_vllm_engine_kwargs(DictDefault({})) == {}

    def test_patch_merges_engine_kwargs_into_llm_call(self, monkeypatch):
        vllm_generation = pytest.importorskip("trl.generation.vllm_generation")

        calls = []

        class FakeLLM:
            """Stand-in for vllm.LLM that records constructor kwargs."""

            def __init__(self, **kwargs):
                calls.append(kwargs)

        monkeypatch.setattr(vllm_generation, "LLM", FakeLLM, raising=False)

        patch_vllm_colocate_engine_kwargs({"enforce_eager": True})
        vllm_generation.LLM(model="m", gpu_memory_utilization=0.3)
        assert calls[-1] == {
            "model": "m",
            "gpu_memory_utilization": 0.3,
            "enforce_eager": True,
        }

        # Re-patching replaces rather than stacks the previous engine kwargs.
        patch_vllm_colocate_engine_kwargs({"dtype": "bfloat16"})
        vllm_generation.LLM(model="m")
        assert calls[-1] == {"model": "m", "dtype": "bfloat16"}

    def test_patch_noop_without_engine_kwargs(self, monkeypatch):
        vllm_generation = pytest.importorskip("trl.generation.vllm_generation")
        sentinel = object()
        monkeypatch.setattr(vllm_generation, "LLM", sentinel, raising=False)
        patch_vllm_colocate_engine_kwargs({})
        assert vllm_generation.LLM is sentinel
