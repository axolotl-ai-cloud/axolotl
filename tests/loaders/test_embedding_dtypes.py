"""Embedding dtype configuration in `ModelLoader._configure_embedding_dtypes`."""

import pytest
import torch
from torch import nn

from axolotl.utils.dict import DictDefault


@pytest.fixture(scope="module")
def base_model_path(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM

    path = tmp_path_factory.mktemp("llama")
    LlamaForCausalLM(
        LlamaConfig(
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
        )
    ).save_pretrained(path)
    return path


def _loader(base_model_path, **overrides):
    from transformers import LlamaForCausalLM

    from axolotl.loaders.model import ModelLoader

    cfg = DictDefault(
        base_model=str(base_model_path),
        model_config_type="llama",
        torch_dtype=torch.bfloat16,
        **overrides,
    )
    loader = ModelLoader(cfg, tokenizer=None)
    loader.model = LlamaForCausalLM.from_pretrained(
        str(base_model_path), dtype=torch.bfloat16
    )
    return loader


class _ToCounter:
    """Record which modules `nn.Module.to` is asked to cast, and to which dtype."""

    def __init__(self, monkeypatch):
        self.casts = []
        original = nn.Module.to

        def counting_to(module, *args, **kwargs):
            dtype = kwargs.get("dtype")
            if dtype is None:
                dtype = next((a for a in args if isinstance(a, torch.dtype)), None)
            if dtype is not None:
                self.casts.append((id(module), dtype))
            return original(module, *args, **kwargs)

        monkeypatch.setattr(nn.Module, "to", counting_to)

    def upcast_to_fp32(self, module):
        return (id(module), torch.float32) in self.casts


def _embedding_dtypes(model):
    return {
        model.get_input_embeddings().weight.dtype,
        model.get_output_embeddings().weight.dtype,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {"adapter": "qlora", "load_in_4bit": True},
            id="single_gpu_qlora",
        ),
        pytest.param({"cut_cross_entropy": True}, id="cce_full_finetune"),
    ],
)
def test_no_transient_fp32_upcast(base_model_path, monkeypatch, overrides):
    loader = _loader(base_model_path, **overrides)
    counter = _ToCounter(monkeypatch)

    loader._configure_embedding_dtypes()  # pylint: disable=protected-access

    assert not counter.upcast_to_fp32(loader.model.get_input_embeddings())
    assert not counter.upcast_to_fp32(loader.model.get_output_embeddings())
    assert _embedding_dtypes(loader.model) == {torch.bfloat16}


def test_fsdp2_qlora_upcasts_embeddings(base_model_path):
    loader = _loader(
        base_model_path,
        adapter="qlora",
        load_in_4bit=True,
        fsdp_version=2,
        fsdp_config={"fsdp_version": 2},
    )

    loader._configure_embedding_dtypes()  # pylint: disable=protected-access

    assert _embedding_dtypes(loader.model) == {torch.float32}


def test_fsdp2_qlora_skip_upcast(base_model_path, monkeypatch):
    loader = _loader(
        base_model_path,
        adapter="qlora",
        load_in_4bit=True,
        embeddings_skip_upcast=True,
        fsdp_version=2,
        fsdp_config={"fsdp_version": 2},
    )
    counter = _ToCounter(monkeypatch)

    loader._configure_embedding_dtypes()  # pylint: disable=protected-access

    assert not counter.upcast_to_fp32(loader.model.get_input_embeddings())
    assert not counter.upcast_to_fp32(loader.model.get_output_embeddings())
    assert _embedding_dtypes(loader.model) == {torch.bfloat16}


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"adapter": "qlora", "load_in_4bit": True}, id="qlora"),
        pytest.param({}, id="full_finetune"),
    ],
)
def test_fp32_norms_survive(base_model_path, overrides):
    loader = _loader(
        base_model_path, fp32_norms=True, fp32_norm_classes=["RMSNorm"], **overrides
    )

    loader._configure_embedding_dtypes()  # pylint: disable=protected-access

    norm_dtypes = {
        module.weight.dtype
        for name, module in loader.model.named_modules()
        if type(module).__name__.endswith("RMSNorm")
    }
    assert norm_dtypes == {torch.float32}


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"adapter": "qlora", "load_in_4bit": True}, id="qlora"),
        pytest.param({"cut_cross_entropy": True}, id="cce_full_finetune"),
    ],
)
def test_moe_gate_stays_fp32(base_model_path, overrides):
    loader = _loader(base_model_path, **overrides)
    mlp = loader.model.model.layers[0].mlp
    mlp.gate = nn.Linear(32, 2, bias=False, dtype=torch.bfloat16)

    loader._configure_embedding_dtypes()  # pylint: disable=protected-access

    assert mlp.gate.weight.dtype == torch.float32


class _TensorToCounter:
    """Record the storages `torch.Tensor.to` is asked to cast, and to which dtype."""

    def __init__(self, monkeypatch):
        self.casts = set()
        original = torch.Tensor.to

        def counting_to(tensor, *args, **kwargs):
            dtype = kwargs.get("dtype")
            if dtype is None:
                dtype = next((a for a in args if isinstance(a, torch.dtype)), None)
            if dtype is not None:
                self.casts.add((tensor.data_ptr(), dtype))
            return original(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", counting_to)

    def upcast_to_fp32(self, data_ptr):
        return (data_ptr, torch.float32) in self.casts


@pytest.fixture(name="restore_peft_prep")
def fixture_restore_peft_prep():
    import peft

    import axolotl.loaders.model as model_module

    original_peft = peft.utils.other.prepare_model_for_kbit_training
    original_loader = model_module.prepare_model_for_kbit_training
    yield
    peft.utils.other.prepare_model_for_kbit_training = original_peft
    model_module.prepare_model_for_kbit_training = original_loader


def test_kbit_prep_leaves_embeddings_untouched(
    base_model_path, monkeypatch, restore_peft_prep
):
    from axolotl.loaders.model import should_skip_peft_embedding_upcast
    from axolotl.monkeypatch.peft.utils import patch_peft_prep_code

    loader = _loader(base_model_path, adapter="qlora", load_in_4bit=True)
    assert should_skip_peft_embedding_upcast(loader.cfg)
    patch_peft_prep_code()

    embedding_ptrs = [
        loader.model.get_input_embeddings().weight.data_ptr(),
        loader.model.get_output_embeddings().weight.data_ptr(),
    ]
    counter = _TensorToCounter(monkeypatch)

    loader._configure_embedding_dtypes()  # pylint: disable=protected-access

    assert not any(counter.upcast_to_fp32(ptr) for ptr in embedding_ptrs)
    assert _embedding_dtypes(loader.model) == {torch.bfloat16}


def test_explicit_skip_upcast_override_is_honored(base_model_path):
    from axolotl.loaders.model import should_skip_peft_embedding_upcast

    cfg = _loader(
        base_model_path,
        adapter="qlora",
        load_in_4bit=True,
        embeddings_skip_upcast=False,
    ).cfg
    assert not should_skip_peft_embedding_upcast(cfg)

    cfg = _loader(
        base_model_path,
        adapter="qlora",
        load_in_4bit=True,
        embeddings_skip_upcast=True,
        fsdp_version=2,
        fsdp_config={"fsdp_version": 2},
    ).cfg
    assert should_skip_peft_embedding_upcast(cfg)


def test_fsdp2_qlora_does_not_skip_peft_upcast(base_model_path):
    from axolotl.loaders.model import should_skip_peft_embedding_upcast

    cfg = _loader(
        base_model_path,
        adapter="qlora",
        load_in_4bit=True,
        fsdp_version=2,
        fsdp_config={"fsdp_version": 2},
    ).cfg

    assert not should_skip_peft_embedding_upcast(cfg)
