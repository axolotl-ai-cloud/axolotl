"""`get_linear_embedding_layers` against the real transformers model classes."""

import pytest
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from axolotl.loaders.utils import get_linear_embedding_layers

MODEL_KWARGS = {
    "gpt_neox": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "vocab_size": 64,
    },
    "falcon": {
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "vocab_size": 64,
    },
    "nemotron_h": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "vocab_size": 64,
        "hybrid_override_pattern": "M",
    },
    "bailing_hybrid": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "vocab_size": 64,
    },
    "llama": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 64,
    },
}


def _module_path(model, target):
    if target is None:
        return None
    return next(name for name, module in model.named_modules() if module is target)


@pytest.fixture(name="model")
def fixture_model(model_type):
    if model_type not in CONFIG_MAPPING_NAMES:
        pytest.skip(f"{model_type} is not available in this transformers")
    config = AutoConfig.for_model(model_type, **MODEL_KWARGS[model_type])
    return AutoModelForCausalLM.from_config(config)


@pytest.mark.parametrize("model_type", sorted(MODEL_KWARGS))
def test_every_name_matches_an_embedding_or_head(model, model_type):
    names = get_linear_embedding_layers(model_type)
    embedding_paths = {
        _module_path(model, model.get_input_embeddings()),
        _module_path(model, model.get_output_embeddings()),
    } - {None}
    param_paths = [name for name, _ in model.named_parameters()]

    for name in names:
        matched = [path for path in embedding_paths if name in path]
        assert matched, (
            f"{model_type}: {name!r} matches no embedding or output head "
            f"among {sorted(embedding_paths)}"
        )
        assert (
            any(name in path for path in param_paths)
            or model.config.tie_word_embeddings
        ), f"{model_type}: {name!r} matches no parameter path"


@pytest.mark.parametrize("model_type", sorted(MODEL_KWARGS))
def test_input_embedding_and_output_head_are_covered(model, model_type):
    names = get_linear_embedding_layers(model_type)
    for role, module in (
        ("input embedding", model.get_input_embeddings()),
        ("output head", model.get_output_embeddings()),
    ):
        path = _module_path(model, module)
        assert any(name in path for name in names), (
            f"{model_type}: {role} {path!r} is covered by none of {names}"
        )
