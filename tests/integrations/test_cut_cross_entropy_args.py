"""Config validation and cce_patch wiring for the Cut Cross Entropy plugin."""

from types import SimpleNamespace
from unittest.mock import create_autospec, patch

import pytest

from axolotl.integrations.cut_cross_entropy import CutCrossEntropyPlugin
from axolotl.integrations.cut_cross_entropy.args import CutCrossEntropyArgs
from axolotl.utils.dict import DictDefault


def _args(**overrides):
    return CutCrossEntropyArgs(bf16=True, **overrides)


class TestCutCrossEntropyArgs:
    def test_defaults(self):
        args = _args()
        assert args.cut_cross_entropy_accum_c_fp32 is False
        assert args.cut_cross_entropy_c_grad_chunk_size is None

    @pytest.mark.parametrize("chunk", ["auto", 128, 32768])
    def test_chunk_size_with_fp32_accumulation(self, chunk):
        args = _args(
            cut_cross_entropy_accum_c_fp32=True,
            cut_cross_entropy_c_grad_chunk_size=chunk,
        )
        assert args.cut_cross_entropy_c_grad_chunk_size == chunk

    @pytest.mark.parametrize("chunk", ["auto", 256])
    def test_chunk_size_requires_fp32_accumulation(self, chunk):
        with pytest.raises(ValueError, match="cut_cross_entropy_accum_c_fp32"):
            _args(cut_cross_entropy_c_grad_chunk_size=chunk)

    @pytest.mark.parametrize("chunk", [-128, 100, 129])
    def test_chunk_size_must_be_multiple_of_128(self, chunk):
        with pytest.raises(ValueError, match="multiple of 128"):
            _args(
                cut_cross_entropy_accum_c_fp32=True,
                cut_cross_entropy_c_grad_chunk_size=chunk,
            )

    def test_chunk_size_rejects_other_strings(self):
        with pytest.raises(ValueError):
            _args(
                cut_cross_entropy_accum_c_fp32=True,
                cut_cross_entropy_c_grad_chunk_size="big",
            )


def _cfg(**overrides):
    return DictDefault(
        {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "model_config_type": "llama",
            "cut_cross_entropy": True,
            **overrides,
        }
    )


def _new_cce_patch(
    model_type_or_model,
    *,
    remote_model_id=None,
    accum_c_fp32=False,
    c_grad_chunk_size=0,
):
    pass


def _old_cce_patch(model_type_or_model, *, remote_model_id=None, accum_c_fp32=False):
    pass


def _cce_patch_stub(with_chunk: bool):
    return _new_cce_patch if with_chunk else _old_cce_patch


class TestCutCrossEntropyPluginPatchKwargs:
    @pytest.fixture(autouse=True)
    def _skip_requirement_checks(self):
        with (
            patch.object(CutCrossEntropyPlugin, "_check_requirements"),
            patch.object(CutCrossEntropyPlugin, "_check_generic_patch_applies"),
            patch.object(CutCrossEntropyPlugin, "patch_llama_like"),
        ):
            yield

    def _run(self, cfg, stub):
        mocked = create_autospec(stub)
        with patch("cut_cross_entropy.transformers.patch.cce_patch", new=mocked):
            CutCrossEntropyPlugin().pre_model_load(cfg)
        return mocked

    def test_defaults_do_not_pass_chunk_size(self):
        mocked = self._run(_cfg(), _cce_patch_stub(with_chunk=True))
        mocked.assert_called_once_with(
            "llama", remote_model_id=None, accum_c_fp32=False
        )

    def test_options_are_forwarded(self):
        cfg = _cfg(
            cut_cross_entropy_accum_c_fp32=True,
            cut_cross_entropy_c_grad_chunk_size=32768,
        )
        mocked = self._run(cfg, _cce_patch_stub(with_chunk=True))
        mocked.assert_called_once_with(
            "llama", remote_model_id=None, accum_c_fp32=True, c_grad_chunk_size=32768
        )

    def test_auto_resolves_once_from_config(self):
        cfg = _cfg(
            cut_cross_entropy_accum_c_fp32=True,
            cut_cross_entropy_c_grad_chunk_size="auto",
            micro_batch_size=4,
            sequence_len=2048,
            context_parallel_size=2,
        )
        model_config = SimpleNamespace(vocab_size=151936, hidden_size=4096)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("axolotl.loaders.utils.load_model_config", return_value=model_config),
            patch(
                "cut_cross_entropy.recommend_c_grad_chunk_size",
                create=True,
                return_value=16384,
            ) as recommend,
        ):
            mocked = self._run(cfg, _cce_patch_stub(with_chunk=True))
        recommend.assert_called_once_with(
            num_tokens=4 * 1024, vocab_size=151936, hidden_size=4096, device=0
        )
        mocked.assert_called_once_with(
            "llama", remote_model_id=None, accum_c_fp32=True, c_grad_chunk_size=16384
        )

    def test_old_cce_still_works_without_chunking(self):
        cfg = _cfg(cut_cross_entropy_accum_c_fp32=True)
        mocked = self._run(cfg, _cce_patch_stub(with_chunk=False))
        mocked.assert_called_once_with("llama", remote_model_id=None, accum_c_fp32=True)

    def test_old_cce_rejects_chunking_with_install_hint(self):
        cfg = _cfg(
            cut_cross_entropy_accum_c_fp32=True,
            cut_cross_entropy_c_grad_chunk_size=256,
        )
        with pytest.raises(ImportError, match="ml-cross-entropy.git@v0.2.1"):
            self._run(cfg, _cce_patch_stub(with_chunk=False))


class TestGenericPatchGuard:
    def _check(self, model_type, model_config, patch_fns=None):
        cfg = _cfg(model_config_type=model_type)
        with (
            patch.dict(
                "cut_cross_entropy.transformers.patch.PATCH_FNS",
                patch_fns or {},
                clear=True,
            ),
            patch(
                "axolotl.loaders.utils.load_model_config", return_value=model_config
            ) as load,
        ):
            CutCrossEntropyPlugin()._check_generic_patch_applies(cfg)
        return load

    def test_rejects_softcapped_model_without_dedicated_patch(self):
        config = SimpleNamespace(final_logit_softcapping=30.0)
        with pytest.raises(ValueError, match="final_logit_softcapping"):
            self._check("beam", config)

    def test_rejects_logit_scale(self):
        config = SimpleNamespace(logit_scale=0.0625)
        with pytest.raises(ValueError, match="logit_scale"):
            self._check("cohere_like", config)

    def test_checks_text_config_of_multimodal_models(self):
        text = SimpleNamespace(final_logit_softcapping=30.0)
        config = SimpleNamespace(get_text_config=lambda: text)
        with pytest.raises(ValueError, match="final_logit_softcapping"):
            self._check("beam_vl", config)

    @pytest.mark.parametrize(
        "config",
        [
            SimpleNamespace(),
            SimpleNamespace(final_logit_softcapping=None, logit_scale=1.0),
            SimpleNamespace(logits_scaling=1),
        ],
    )
    def test_allows_plain_llama_like_models(self, config):
        self._check("my_llama", config)

    def test_dedicated_patch_skips_the_check(self):
        config = SimpleNamespace(final_logit_softcapping=30.0)
        load = self._check("beam", config, patch_fns={"beam": object()})
        load.assert_not_called()

    def test_pre_model_load_refuses_before_patching(self):
        config = SimpleNamespace(final_logit_softcapping=30.0)
        with (
            patch.object(CutCrossEntropyPlugin, "_check_requirements"),
            patch.dict(
                "cut_cross_entropy.transformers.patch.PATCH_FNS", {}, clear=True
            ),
            patch("axolotl.loaders.utils.load_model_config", return_value=config),
            patch("cut_cross_entropy.transformers.patch.cce_patch") as cce_patch,
            pytest.raises(ValueError, match="final_logit_softcapping"),
        ):
            CutCrossEntropyPlugin().pre_model_load(_cfg(model_config_type="beam"))
        cce_patch.assert_not_called()
