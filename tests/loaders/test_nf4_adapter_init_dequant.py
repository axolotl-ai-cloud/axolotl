"""Adapter creation on a CPU-staged NF4 model must not dequantize the base weights."""

import json

import pytest
import torch

from axolotl.utils.dict import DictDefault
from axolotl.utils.nf4 import BnbNF4Parametrization, TorchaoNF4Parametrization


def _save_llama(path):
    from transformers import LlamaConfig, LlamaForCausalLM

    LlamaForCausalLM(
        LlamaConfig(
            hidden_size=128,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=128,
        )
    ).save_pretrained(path)
    return LlamaForCausalLM


def _save_qwen3_moe(path):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            hidden_size=64,
            intermediate_size=64,
            moe_intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            vocab_size=128,
        )
    ).save_pretrained(path)
    return Qwen3MoeForCausalLM


def _staged_loader(path, model_class, backend, **overrides):
    from axolotl.loaders.model import ModelLoader

    cfg = DictDefault(
        base_model=str(path),
        nf4_backend=backend,
        load_in_4bit=True,
        adapter="qlora",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "v_proj"],
        fsdp_version=2,
        qlora_sharded_model_loading=True,
        tensor_parallel_size=1,
        context_parallel_size=1,
        torch_dtype=torch.float32,
        **overrides,
    )
    loader = ModelLoader(cfg, tokenizer=None)
    loader.auto_model_loader = model_class
    loader.model_kwargs = {"dtype": cfg.torch_dtype}
    loader._set_quantization_config()
    loader._build_model()
    assert getattr(loader.model, "_axolotl_staged_nf4", False)
    return loader


class _DequantCounter:
    """Count dequantizations and how many dense results are alive at once."""

    def __init__(self, monkeypatch):
        self.count = 0
        self.peak_alive = 0
        self.peak_modules = 0
        self._alive = []
        for cls in (BnbNF4Parametrization, TorchaoNF4Parametrization):
            monkeypatch.setattr(cls, "forward", self._wrap(cls.forward))

    def _wrap(self, original):
        import weakref

        counter = self

        def forward(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            counter.count += 1
            counter._alive = [
                entry for entry in counter._alive if entry[0]() is not None
            ]
            counter._alive.append((weakref.ref(result), id(self)))
            counter.peak_alive = max(counter.peak_alive, len(counter._alive))
            counter.peak_modules = max(
                counter.peak_modules, len({entry[1] for entry in counter._alive})
            )
            return result

        return forward


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_staged_adapter_init_does_not_dequantize(backend, tmp_path, monkeypatch):
    model_class = _save_llama(tmp_path)
    loader = _staged_loader(tmp_path, model_class, backend)
    targets = [
        module
        for _, module in loader.model.named_modules()
        if getattr(module, "parametrizations", None)
    ]
    assert targets

    counter = _DequantCounter(monkeypatch)
    loader._load_adapters()

    assert counter.count == 0
    assert counter.peak_alive == 0
    trainable = [name for name, p in loader.model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)
    for name, param in loader.model.named_parameters():
        if "lora_" in name:
            assert param.device.type == "cpu"
            assert param.dtype == torch.float32


def test_staged_expert_adapter_init_does_not_dequantize(tmp_path, monkeypatch):
    model_class = _save_qwen3_moe(tmp_path)
    loader = _staged_loader(
        tmp_path,
        model_class,
        "bitsandbytes",
        quantize_moe_experts=True,
        lora_target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )
    experts = loader.model.model.layers[0].mlp.experts
    assert getattr(experts, "parametrizations", None)

    counter = _DequantCounter(monkeypatch)
    loader._load_adapters()

    assert counter.count == 0
    assert counter.peak_alive == 0
    trainable = [name for name, p in loader.model.named_parameters() if p.requires_grad]
    assert any("experts" in name for name in trainable)


def test_staged_model_still_dequantizes_after_adapter_init(tmp_path, monkeypatch):
    model_class = _save_llama(tmp_path)
    loader = _staged_loader(tmp_path, model_class, "bitsandbytes")
    q_proj = loader.model.model.layers[0].self_attn.q_proj
    expected = q_proj.weight.clone()
    loader._load_adapters()

    counter = _DequantCounter(monkeypatch)
    tokens = torch.randint(0, 128, (1, 4))
    loader.model(input_ids=tokens)
    assert counter.count > 0
    torch.testing.assert_close(q_proj.weight, expected, rtol=0, atol=0)


def test_non_staged_model_is_untouched(tmp_path, monkeypatch):
    model_class = _save_llama(tmp_path)
    loader = _staged_loader(tmp_path, model_class, "bitsandbytes")
    del loader.model._axolotl_staged_nf4

    counter = _DequantCounter(monkeypatch)
    loader._load_adapters()

    assert counter.count > 0


def test_adapter_init_matches_dequantized_path(tmp_path):
    model_class = _save_llama(tmp_path)
    adapters = []
    for staged in (True, False):
        loader = _staged_loader(tmp_path, model_class, "bitsandbytes")
        if not staged:
            del loader.model._axolotl_staged_nf4
        torch.manual_seed(0)
        loader._load_adapters()
        adapters.append(
            {
                name: param.detach().clone()
                for name, param in loader.model.named_parameters()
                if "lora_" in name
            }
        )
    assert adapters[0].keys() == adapters[1].keys()
    for name, param in adapters[0].items():
        torch.testing.assert_close(param, adapters[1][name], rtol=0, atol=0)


def test_staged_layer_replication_keeps_real_weights(tmp_path, monkeypatch):
    model_class = _save_llama(tmp_path)
    loader = _staged_loader(
        tmp_path,
        model_class,
        "bitsandbytes",
        peft_layer_replication=[[0, 2], [1, 2]],
    )

    counter = _DequantCounter(monkeypatch)
    loader._load_adapters()
    assert counter.count == 0

    weights = [
        module.base_layer.weight
        for name, module in loader.model.named_modules()
        if name.endswith("self_attn.q_proj") and hasattr(module, "base_layer")
    ]
    assert len(weights) == 3
    for weight in weights:
        assert weight.abs().sum().item() > 0


def test_dequantizing_adapter_init_bounds_peak_dense_tensors(tmp_path, monkeypatch):
    model_class = _save_llama(tmp_path)
    loader = _staged_loader(tmp_path, model_class, "bitsandbytes")
    del loader.model._axolotl_staged_nf4

    counter = _DequantCounter(monkeypatch)
    loader._load_adapters()

    assert counter.count > 0
    assert counter.peak_modules == 1


def test_plain_model_adapter_init_is_untouched(tmp_path, monkeypatch):
    from axolotl.loaders.model import ModelLoader

    model_class = _save_llama(tmp_path)
    cfg = DictDefault(
        base_model=str(tmp_path),
        adapter="lora",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "v_proj"],
        tensor_parallel_size=1,
        context_parallel_size=1,
        torch_dtype=torch.float32,
    )
    loader = ModelLoader(cfg, tokenizer=None)
    loader.auto_model_loader = model_class
    loader.model_kwargs = {"torch_dtype": cfg.torch_dtype}
    loader._set_quantization_config()
    loader._build_model()
    assert not getattr(loader.model, "_axolotl_staged_nf4", False)
    assert not any(
        getattr(module, "parametrizations", None)
        for _, module in loader.model.named_modules()
    )

    expected = {
        name: module.weight.clone()
        for name, module in loader.model.named_modules()
        if name.endswith("self_attn.q_proj")
    }
    loader._load_adapters()
    found = {
        name.split("base_model.model.")[-1]: module.base_layer.weight
        for name, module in loader.model.named_modules()
        if name.endswith("self_attn.q_proj") and hasattr(module, "base_layer")
    }
    assert found.keys() == expected.keys()
    for name, weight in found.items():
        torch.testing.assert_close(weight, expected[name], rtol=0, atol=0)


def test_saved_value_dependent_adapter_is_rejected(tmp_path):
    """A resumed OLoRA adapter re-runs its init from adapter_config.json, not from cfg."""
    from peft import LoraConfig, get_peft_model

    model_class = _save_llama(tmp_path)
    adapter_dir = tmp_path / "adapter"
    get_peft_model(
        model_class.from_pretrained(tmp_path, dtype=torch.float32),
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            init_lora_weights="olora",
        ),
    ).save_pretrained(adapter_dir)

    loader = _staged_loader(
        tmp_path, model_class, "bitsandbytes", lora_model_dir=str(adapter_dir)
    )
    with pytest.raises(ValueError, match="residual write-back"):
        loader._load_adapters()


def test_value_dependent_saved_init_does_not_force_real_base_weights(tmp_path):
    """The predicate answers whether the stand-ins suffice; only the check rejects a saved init."""
    model_class = _save_llama(tmp_path)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8, "init_lora_weights": "olora"})
    )

    loader = _staged_loader(
        tmp_path, model_class, "bitsandbytes", lora_model_dir=str(adapter_dir)
    )

    assert loader._staged_nf4_needs_real_base_weights() is False
    with pytest.raises(ValueError, match="residual write-back"):
        loader._reject_value_dependent_saved_adapter_init()


def test_unreadable_saved_adapter_config_falls_back_to_real_weights(
    tmp_path, monkeypatch
):
    """A hub id is a valid lora_model_dir for PEFT but has no local adapter_config.json."""
    model_class = _save_llama(tmp_path)
    loader = _staged_loader(
        tmp_path, model_class, "bitsandbytes", lora_model_dir="some-org/some-adapter"
    )

    assert loader._staged_nf4_needs_real_base_weights() is True
