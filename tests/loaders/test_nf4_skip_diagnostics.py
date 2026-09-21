"""Staged NF4 loading reports which exclusion keys actually matched."""

import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _stage(cfg, model, params, skip_modules):
    import transformers.core_model_loading as loading
    from transformers import BitsAndBytesConfig

    from axolotl.loaders.nf4 import staged_nf4_loading

    with staged_nf4_loading(
        cfg,
        device="cpu",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, llm_int8_skip_modules=skip_modules
        ),
    ):
        for name, value in params:
            loading.set_param_for_module(
                model,
                name,
                value,
                SimpleNamespace(
                    missing_keys=set(), unexpected_keys=set(), mismatched_keys=set()
                ),
                None,
            )


def _model():
    from axolotl.utils.dict import DictDefault

    model = nn.Module()
    model.q_proj = nn.Linear(64, 64, bias=False)
    model.k_proj = nn.Linear(64, 64, bias=False)
    model.embed_tokens = nn.Embedding(8, 64)
    params = [
        ("q_proj.weight", torch.randn(64, 64)),
        ("k_proj.weight", torch.randn(64, 64)),
        ("embed_tokens.weight", torch.randn(8, 64)),
    ]
    return DictDefault(nf4_backend="bitsandbytes"), model, params


def test_unmatched_user_skip_key_warns(caplog):
    cfg, model, params = _model()
    with caplog.at_level(logging.INFO, logger="axolotl.loaders.nf4"):
        _stage(cfg, model, params, ["v_proj"])
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "'v_proj'" in warnings[0].getMessage()
    assert "name" in warnings[0].getMessage()


def test_matched_user_skip_key_logs_count_without_warning(caplog):
    cfg, model, params = _model()
    with caplog.at_level(logging.INFO, logger="axolotl.loaders.nf4"):
        _stage(cfg, model, params, ["_proj."])
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    messages = [r.getMessage() for r in caplog.records]
    assert any("'_proj.'" in m and "2" in m and "substring" in m for m in messages)


def test_builtin_defaults_never_warn(caplog):
    cfg, model, params = _model()
    cfg.model_config_type = "falcon_h1"
    with caplog.at_level(logging.INFO, logger="axolotl.loaders.nf4"):
        _stage(cfg, model, params, [])
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    messages = [r.getMessage() for r in caplog.records]
    assert any("'lm_head'" in m for m in messages)
    assert any("'out_proj'" in m for m in messages)


def test_key_matching_only_a_non_candidate_warns(caplog):
    cfg, model, params = _model()
    with caplog.at_level(logging.INFO, logger="axolotl.loaders.nf4"):
        _stage(cfg, model, params, ["embed_tokens"])
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "'embed_tokens'" in warnings[0].getMessage()


def test_diagnostics_survive_a_failed_stage(caplog):
    import transformers.core_model_loading as loading
    from transformers import BitsAndBytesConfig

    from axolotl.loaders.nf4 import staged_nf4_loading

    cfg, model, params = _model()
    with caplog.at_level(logging.INFO, logger="axolotl.loaders.nf4"):
        with pytest.raises(RuntimeError, match="conversion failed"):
            with staged_nf4_loading(
                cfg,
                device="cpu",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, llm_int8_skip_modules=["v_proj", "q_proj"]
                ),
            ):
                name, value = params[0]
                loading.set_param_for_module(
                    model,
                    name,
                    value,
                    SimpleNamespace(
                        missing_keys=set(), unexpected_keys=set(), mismatched_keys=set()
                    ),
                    None,
                )
                raise RuntimeError("conversion failed")
    messages = [r.getMessage() for r in caplog.records]
    assert any("'q_proj'" in m and "1" in m for m in messages)
    assert any("partial" in m for m in messages)
    # counts from an aborted stage cannot say a key matched nothing
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
