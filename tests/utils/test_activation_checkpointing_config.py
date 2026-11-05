"""Each activation checkpointing config, and the deprecated keys it replaces, resolves to
the same TrainingArguments kwargs and model checkpointing state the deprecated keys did.

A case runs config validation and normalization, the trainer builder's checkpointing
args, ModelLoader's checkpointing setup and one HF Trainer step on a tiny Llama, and
records what each stage leaves behind.
"""

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from axolotl.monkeypatch.selective_checkpointing import (
    _SUPPORTS_RESPECT_SAVED_TENSORS_HOOKS,
)
from axolotl.utils.activation_checkpointing import (
    CHECKPOINTING_ARG,
    CHECKPOINTING_KWARGS_ARG,
)
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

NUM_LAYERS = 2
LORA = {
    "adapter": "lora",
    "lora_r": 4,
    "lora_alpha": 8,
    "lora_target_modules": ["q_proj", "v_proj"],
}
ZERO3 = {"deepspeed": {"zero_optimization": {"stage": 3}}}
FSDP = {"fsdp_version": 2, "fsdp_config": {"reshard_after_forward": True}}
DPO = {
    "rl": "dpo",
    "datasets": [
        {"path": "argilla/distilabel-intel-orca-dpo-pairs", "type": "chatml.intel"}
    ],
}


@pytest.fixture(scope="module")
def tiny_llama(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM

    path = tmp_path_factory.mktemp("llama")
    LlamaForCausalLM(
        LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=NUM_LAYERS,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
        )
    ).save_pretrained(path)
    return path


def _layer_state(model):
    from peft import PeftModel

    base = model.get_base_model() if isinstance(model, PeftModel) else model
    layers = base.model.layers
    func = getattr(layers[0], "_gradient_checkpointing_func", None)
    if not any(layer.gradient_checkpointing for layer in layers) or func is None:
        return {"layers": [False] * len(layers)}
    return {
        "layers": [bool(layer.gradient_checkpointing) for layer in layers],
        "fn": getattr(func, "func", func).__qualname__,
        "keywords": {
            key: value.__qualname__ if callable(value) else value
            for key, value in getattr(func, "keywords", {}).items()
        },
    }


def _resolve(tiny_llama, min_base_cfg, overrides, model_type="llama"):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaForCausalLM

    from axolotl.core.builders.base import TrainerBuilderBase
    from axolotl.loaders.model import ModelLoader
    from axolotl.utils.trainer import setup_fsdp_envs

    cfg = validate_config(min_base_cfg | DictDefault(overrides))
    with (
        mock.patch(
            "axolotl.utils.config.load_model_config",
            return_value=SimpleNamespace(model_type=model_type),
        ),
        mock.patch.dict(os.environ, {"ACCELERATE_DEEPSPEED_ZERO_STAGE": ""}),
    ):
        normalize_config(cfg)
    cfg.base_model = str(tiny_llama)
    cfg.torch_dtype = torch.float32
    observed = {}

    fsdp_env = None
    if cfg.fsdp_config:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FSDP_ACTIVATION_CHECKPOINTING", None)
            setup_fsdp_envs(cfg)
            fsdp_env = os.environ.get("FSDP_ACTIVATION_CHECKPOINTING")
    observed["fsdp_env"] = fsdp_env

    args: dict = {}
    TrainerBuilderBase._configure_gradient_checkpointing(SimpleNamespace(cfg=cfg), args)
    names = {CHECKPOINTING_ARG: "checkpointing", CHECKPOINTING_KWARGS_ARG: "kwargs"}
    observed["args"] = {names.get(key, key): value for key, value in args.items()}

    torch.manual_seed(0)
    kbit = []
    with (
        mock.patch(
            "axolotl.monkeypatch.activation_offload_checkpoint.patch_hidden_states_offload"
        ) as alst,
        mock.patch(
            "axolotl.loaders.model.prepare_model_for_kbit_training",
            side_effect=lambda model, **kw: kbit.append(kw) or model,
        ),
    ):
        loader = ModelLoader(cfg, tokenizer=None)
        loader.model = LlamaForCausalLM.from_pretrained(str(tiny_llama))
        loader._apply_selective_checkpointing()
        loader._apply_activation_checkpointing()
        loader._configure_embedding_dtypes()
    observed["alst"] = alst.called
    observed["kbit"] = [kw.get("use_gradient_checkpointing") for kw in kbit]
    model = loader.model
    observed["after_load"] = _layer_state(model)

    if cfg.adapter:
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=cfg.lora_target_modules,
            ),
        )
    observed["after_train"] = _train_one_step(model, args)
    return observed


def _train_one_step(model, args):
    import tempfile

    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    with tempfile.TemporaryDirectory() as output_dir:
        training_args = TrainingArguments(
            output_dir=output_dir,
            max_steps=1,
            per_device_train_batch_size=2,
            use_cpu=True,
            report_to="none",
            save_strategy="no",
            **{
                key: args[key]
                for key in (CHECKPOINTING_ARG, CHECKPOINTING_KWARGS_ARG)
                if key in args
            },
        )
        ids = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(1))
        dataset = Dataset.from_list(
            [{"input_ids": x, "labels": x} for x in ids.tolist()]
        )
        Trainer(model=model, args=training_args, train_dataset=dataset).train()
    return _layer_state(model)


OFF = {"layers": [False, False]}


def _on(reentrant=False, **keywords):
    return {
        "layers": [True, True],
        "fn": "checkpoint",
        "keywords": {"use_reentrant": reentrant, **keywords},
    }


SAC = "build_sac_context_fn.<locals>.context_fn"
SAC_OFFLOAD = "build_sac_offload_context_fn.<locals>.context_fn"
SAC_KEYWORDS = (
    {"respect_saved_tensors_hooks": False}
    if _SUPPORTS_RESPECT_SAVED_TENSORS_HOOKS
    else {}
)


def _expected(
    checkpointing=False,
    kwargs=None,
    activation_offloading=None,
    after_load=OFF,
    after_train=OFF,
    alst=False,
    kbit=(),
    fsdp_env=None,
):
    args = {
        "checkpointing": checkpointing,
        "kwargs": kwargs or {"use_reentrant": False},
    }
    if activation_offloading is not None:
        args = {"activation_offloading": activation_offloading, **args}
    return {
        "fsdp_env": fsdp_env,
        "args": args,
        "alst": alst,
        "kbit": list(kbit),
        "after_load": after_load,
        "after_train": after_train,
    }


CASES = [
    pytest.param({}, "llama", _expected(), id="absent"),
    pytest.param(
        {"activation_checkpointing": False}, "llama", _expected(), id="disabled"
    ),
    pytest.param(
        {"activation_checkpointing": True},
        "llama",
        _expected(checkpointing=True, after_train=_on()),
        id="enabled",
    ),
    pytest.param(
        {"activation_checkpointing": {"enabled": True, "reentrant": False}},
        "llama",
        _expected(checkpointing=True, after_train=_on()),
        id="non_reentrant",
    ),
    pytest.param(
        {"activation_checkpointing": {"enabled": True, "reentrant": True}},
        "llama",
        _expected(
            checkpointing=True,
            kwargs={"use_reentrant": True},
            after_train=_on(reentrant=True),
        ),
        id="reentrant",
    ),
    pytest.param(
        {
            "activation_checkpointing": {
                "enabled": True,
                "reentrant": False,
                "kwargs": {"preserve_rng_state": False},
            }
        },
        "llama",
        _expected(
            checkpointing=True,
            kwargs={"use_reentrant": False, "preserve_rng_state": False},
            after_train=_on(preserve_rng_state=False),
        ),
        id="torch_kwargs",
    ),
    pytest.param(
        {"activation_checkpointing": {"enabled": True, "offload": {"all": "cpu"}}},
        "llama",
        _expected(checkpointing=True, activation_offloading=True, after_train=_on()),
        id="offload_string",
    ),
    pytest.param(
        {"activation_checkpointing": {"enabled": True, "offload": {"all": True}}},
        "llama",
        _expected(checkpointing=True, activation_offloading=True, after_train=_on()),
        id="offload_all",
    ),
    pytest.param(
        {"activation_checkpointing": {"offload": {"all": "cpu"}}},
        "llama",
        _expected(checkpointing=True, activation_offloading=True, after_train=_on()),
        id="offload_legacy",
    ),
    pytest.param(
        {
            "activation_checkpointing": {"enabled": False, "offload": {"all": "cpu"}},
            **LORA,
        },
        "llama",
        _expected(activation_offloading=True),
        id="lora_offload_without_recompute",
    ),
    pytest.param(
        {
            "activation_checkpointing": {
                "enabled": True,
                "offload": {"hidden_states": "cpu"},
            }
        },
        "llama",
        _expected(
            checkpointing=True,
            activation_offloading="hidden_states",
            after_train=_on(),
        ),
        id="hidden_states",
    ),
    pytest.param(
        {
            "activation_checkpointing": {
                "enabled": True,
                "reentrant": True,
                "offload": {"hidden_states": "cpu"},
            }
        },
        "llama",
        _expected(
            checkpointing=True,
            kwargs={"use_reentrant": True},
            after_train=_on(reentrant=True),
            alst=True,
        ),
        id="hidden_states_reentrant",
    ),
    pytest.param(
        {"activation_checkpointing": {"enabled": True, "selective": True}},
        "llama",
        _expected(
            checkpointing=True,
            after_train=_on(context_fn=SAC, **SAC_KEYWORDS),
        ),
        id="selective",
    ),
    pytest.param(
        {
            "activation_checkpointing": {
                "enabled": True,
                "selective": {"save": ["attention"]},
                "offload": {"saves": "cpu"},
            }
        },
        "llama",
        _expected(
            checkpointing=True,
            after_train=_on(context_fn=SAC_OFFLOAD, **SAC_KEYWORDS),
        ),
        id="selective_offload",
    ),
    pytest.param(
        {
            "activation_checkpointing": {
                "enabled": True,
                "selective": {"save": ["attention"]},
                "offload": {"hidden_states": "cpu", "saves": "cpu"},
            }
        },
        "llama",
        _expected(
            checkpointing=True,
            activation_offloading="hidden_states",
            after_train=_on(context_fn=SAC_OFFLOAD, **SAC_KEYWORDS),
        ),
        id="selective_offload_hidden_states",
    ),
    pytest.param(
        {"activation_checkpointing": True, **LORA},
        "llama",
        _expected(checkpointing=True, after_load=_on(), after_train=_on()),
        id="lora",
    ),
    pytest.param(
        {
            "activation_checkpointing": {
                "enabled": True,
                "reentrant": False,
                "kwargs": {"every_n_layers": 2},
            },
            **LORA,
        },
        "llama",
        _expected(
            checkpointing=True,
            kwargs={"use_reentrant": False, "every_n_layers": 2},
            # every_n_layers used to reach the torch partial on the loader path
            after_load={**_on(), "layers": [True, False]},
            after_train={**_on(), "layers": [True, False]},
        ),
        id="lora_every_n_layers",
    ),
    pytest.param(
        {
            "activation_checkpointing": True,
            "adapter": "qlora",
            "load_in_4bit": True,
            **{k: v for k, v in LORA.items() if k != "adapter"},
        },
        "llama",
        _expected(checkpointing=True, after_load=_on(), after_train=_on(), kbit=[True]),
        id="qlora",
    ),
    pytest.param(
        {**FSDP, "activation_checkpointing": {"implementation": "fsdp"}},
        "llama",
        _expected(fsdp_env="true"),
        id="fsdp",
    ),
    pytest.param(
        {
            **FSDP,
            "activation_checkpointing": {"enabled": True, "implementation": "fsdp"},
        },
        "llama",
        _expected(fsdp_env="true"),
        id="fsdp_prefixed",
    ),
    pytest.param(
        {"activation_checkpointing": True, **FSDP},
        "llama",
        # the legacy config also requested FSDP's wrapper, which Trainer rejects
        # alongside transformers' checkpointing
        _expected(checkpointing=True, after_train=_on()),
        id="hf_and_fsdp",
    ),
    pytest.param(
        {"activation_checkpointing": True, **ZERO3},
        "llama",
        _expected(
            checkpointing=True,
            kwargs={"use_reentrant": True},
            after_train=_on(reentrant=True),
        ),
        id="zero3",
    ),
    pytest.param(
        {"activation_checkpointing": {"offload": {"all": "cpu"}}, **ZERO3},
        "llama",
        _expected(
            checkpointing=True,
            kwargs={"use_reentrant": True},
            activation_offloading=True,
            after_train=_on(reentrant=True),
        ),
        id="zero3_offload_all",
    ),
    pytest.param(
        {"activation_checkpointing": True, **ZERO3, **DPO},
        "llama",
        _expected(checkpointing=True, after_train=_on()),
        id="zero3_rl",
    ),
    pytest.param(
        {
            "activation_checkpointing": True,
            "unfrozen_parameters": ["model.layers.0.*"],
            **ZERO3,
        },
        "llama",
        _expected(checkpointing=True, after_train=_on()),
        id="zero3_unfrozen",
    ),
    pytest.param(
        {"activation_checkpointing": {"enabled": True, "reentrant": True}},
        "mistral_large4",
        _expected(checkpointing=True, after_train=_on()),
        id="mistral_large4_forces_non_reentrant",
    ),
    pytest.param(
        {"activation_checkpointing": {"enabled": True, "reentrant": True}},
        "gemma4",
        _expected(checkpointing=True, after_train=_on()),
        id="gemma4_forces_non_reentrant",
    ),
]


@pytest.mark.parametrize(("overrides", "model_type", "expected"), CASES)
def test_resolves_to_the_same_checkpointing(
    tiny_llama, min_base_cfg, overrides, model_type, expected
):
    observed = _resolve(tiny_llama, min_base_cfg, overrides, model_type)
    assert observed == expected


_KWARGS = "gradient_checkpointing_kwargs"
# the deprecated form of each case; hf_and_fsdp has none, since setting both
# deprecated flags is rejected
DEPRECATED_INPUTS = {
    "disabled": {"gradient_checkpointing": False},
    "enabled": {"gradient_checkpointing": True},
    "non_reentrant": {
        "gradient_checkpointing": True,
        _KWARGS: {"use_reentrant": False},
    },
    "reentrant": {"gradient_checkpointing": True, _KWARGS: {"use_reentrant": True}},
    "torch_kwargs": {
        "gradient_checkpointing": True,
        _KWARGS: {"use_reentrant": False, "preserve_rng_state": False},
    },
    "offload_string": {"gradient_checkpointing": "offload"},
    "offload_all": {"gradient_checkpointing": True, "activation_offloading": True},
    "offload_legacy": {
        "gradient_checkpointing": True,
        "activation_offloading": "legacy",
    },
    "lora_offload_without_recompute": {"activation_offloading": True, **LORA},
    "hidden_states": {
        "gradient_checkpointing": True,
        "activation_offloading": "hidden_states",
    },
    "hidden_states_reentrant": {
        "gradient_checkpointing": True,
        "activation_offloading": "hidden_states",
        _KWARGS: {"use_reentrant": True},
    },
    "selective": {"gradient_checkpointing": True, "selective_checkpointing": True},
    "selective_offload": {
        "gradient_checkpointing": True,
        "selective_checkpointing": {"save": ["attention"], "offload": True},
    },
    "selective_offload_hidden_states": {
        "gradient_checkpointing": True,
        "activation_offloading": "hidden_states",
        "selective_checkpointing": {"save": ["attention"], "offload": True},
    },
    "lora": {"gradient_checkpointing": True, **LORA},
    "lora_every_n_layers": {
        "gradient_checkpointing": True,
        _KWARGS: {"use_reentrant": False, "every_n_layers": 2},
        **LORA,
    },
    "qlora": {
        "gradient_checkpointing": True,
        "adapter": "qlora",
        "load_in_4bit": True,
        **{k: v for k, v in LORA.items() if k != "adapter"},
    },
    "fsdp": {**FSDP, "fsdp_config": {"activation_checkpointing": True}},
    "fsdp_prefixed": {**FSDP, "fsdp_config": {"fsdp_activation_checkpointing": True}},
    "zero3": {"gradient_checkpointing": True, **ZERO3},
    "zero3_offload_all": {
        "gradient_checkpointing": True,
        "activation_offloading": True,
        **ZERO3,
    },
    "zero3_rl": {"gradient_checkpointing": True, **ZERO3, **DPO},
    "zero3_unfrozen": {
        "gradient_checkpointing": True,
        "unfrozen_parameters": ["model.layers.0.*"],
        **ZERO3,
    },
    "mistral_large4_forces_non_reentrant": {
        "gradient_checkpointing": True,
        _KWARGS: {"use_reentrant": True},
    },
    "gemma4_forces_non_reentrant": {
        "gradient_checkpointing": True,
        _KWARGS: {"use_reentrant": True},
    },
}
DEPRECATED_CASES = [
    pytest.param(DEPRECATED_INPUTS[case.id], *case.values[1:], id=case.id)
    for case in CASES
    if case.id in DEPRECATED_INPUTS
]


@pytest.mark.parametrize(("overrides", "model_type", "expected"), DEPRECATED_CASES)
def test_deprecated_keys_resolve_to_the_same_checkpointing(
    tiny_llama, min_base_cfg, overrides, model_type, expected
):
    observed = _resolve(tiny_llama, min_base_cfg, overrides, model_type)
    assert observed == expected
