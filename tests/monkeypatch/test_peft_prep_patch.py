"""PEFT kbit-preparation patch in `axolotl.monkeypatch.peft.utils`."""

import peft
import pytest
import torch

import axolotl.loaders.model
from axolotl.monkeypatch.peft.utils import (
    check_peft_prep_code_is_patchable,
    patch_peft_prep_code,
)


@pytest.fixture(name="unpatched_peft")
def fixture_unpatched_peft():
    original_peft = peft.utils.other.prepare_model_for_kbit_training
    original_loader = axolotl.loaders.model.prepare_model_for_kbit_training
    yield
    peft.utils.other.prepare_model_for_kbit_training = original_peft
    axolotl.loaders.model.prepare_model_for_kbit_training = original_loader


def test_prep_code_still_matches_installed_peft():
    assert check_peft_prep_code_is_patchable()


def test_patch_replaces_prepare_function(unpatched_peft):
    patch_peft_prep_code()

    assert (
        peft.utils.other.prepare_model_for_kbit_training.__name__
        == "fixed_prepare_model_for_kbit_training"
    )
    assert (
        axolotl.loaders.model.prepare_model_for_kbit_training
        is peft.utils.other.prepare_model_for_kbit_training
    )


def test_patched_prepare_skips_embeddings(unpatched_peft):
    from transformers import LlamaConfig, LlamaForCausalLM

    patch_peft_prep_code()

    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
        )
    ).to(torch.bfloat16)

    peft.utils.other.prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=False
    )

    assert model.get_input_embeddings().weight.dtype == torch.bfloat16
    assert model.get_output_embeddings().weight.dtype == torch.bfloat16
    norms = [
        module
        for module in model.modules()
        if type(module).__name__.endswith("RMSNorm")
    ]
    assert norms
    assert all(module.weight.dtype == torch.float32 for module in norms)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        pytest.param({"adapter": "qlora", "load_in_4bit": True}, True, id="qlora"),
        pytest.param(
            {
                "adapter": "qlora",
                "load_in_4bit": True,
                "fsdp_config": {"fsdp_version": 2},
            },
            False,
            id="fsdp2_qlora",
        ),
    ],
)
def test_patch_manager_applies_patch(unpatched_peft, overrides, expected):
    from axolotl.loaders.patch_manager import PatchManager
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(torch_dtype=torch.bfloat16, **overrides)
    PatchManager(cfg, model_config=DictDefault())._apply_adapter_patches()  # pylint: disable=protected-access

    patched = (
        peft.utils.other.prepare_model_for_kbit_training.__name__
        == "fixed_prepare_model_for_kbit_training"
    )
    assert patched is expected
