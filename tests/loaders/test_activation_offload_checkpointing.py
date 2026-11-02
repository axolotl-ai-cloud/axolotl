"""Activation offloading vs gradient checkpointing, end to end on CPU.

Each config runs ModelLoader's checkpointing setup, the trainer builder's
gradient-checkpointing args and the HF Trainer's enable call, then one
forward/backward inside the offload context ``ActivationOffloadingMixin`` would
build. The counts show what the offload hooks see and whether layers recompute.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault

NUM_LAYERS = 2
LORA = {
    "adapter": "lora",
    "lora_r": 4,
    "lora_alpha": 8,
    "lora_target_modules": ["q_proj", "v_proj"],
}
DISK = pytest.param(
    "disk", marks=pytest.mark.skip(reason="activation_offloading: disk is disabled")
)


@pytest.fixture(scope="module")
def base_model_path(tmp_path_factory):
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


@pytest.fixture(autouse=True)
def cpu_accelerator():
    # TRL's OffloadActivations reads the accelerator type at init; on a CPU-only
    # host there is none. Newer TRL then copies the CPU tensors CPU-to-CPU and
    # calls the device module's empty_cache, which torch.cpu lacks.
    with (
        mock.patch.object(
            torch.accelerator, "current_accelerator", return_value=torch.device("cpu")
        ),
        mock.patch.object(torch.cpu, "empty_cache", lambda: None, create=True),
    ):
        yield


def _train_step(base_model_path, min_base_cfg, **overrides):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import LlamaForCausalLM
    from trl.models.activation_offloading import get_act_offloading_ctx_manager

    from axolotl.core.builders.base import TrainerBuilderBase
    from axolotl.core.trainers.mixins.activation_checkpointing import (
        get_lora_act_offloading_ctx_manager,
    )
    from axolotl.loaders.model import ModelLoader
    from axolotl.monkeypatch.checkpoint_activation_offload import (
        get_checkpoint_hidden_states_offloading_ctx_manager,
    )

    cfg = validate_config(DictDefault(overrides) | min_base_cfg)
    cfg.base_model = str(base_model_path)
    cfg.model_config_type = "llama"
    cfg.torch_dtype = torch.float32

    torch.manual_seed(0)
    loader = ModelLoader(cfg, tokenizer=None)
    loader.model = LlamaForCausalLM.from_pretrained(str(base_model_path))
    loader._apply_selective_checkpointing()
    loader._apply_activation_checkpointing()
    loader._configure_embedding_dtypes()
    model = loader.model
    if cfg.adapter:
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=cfg.lora_target_modules,
            ),
        )

    args: dict = {}
    TrainerBuilderBase._configure_gradient_checkpointing(SimpleNamespace(cfg=cfg), args)
    if args.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=args.get("gradient_checkpointing_kwargs")
        )

    mode = args.get("activation_offloading")
    if mode == "hidden_states":
        offload = get_checkpoint_hidden_states_offloading_ctx_manager(
            use_streams=False, min_offload_size=0
        )
    elif isinstance(model, PeftModel):
        offload = get_lora_act_offloading_ctx_manager(
            model, use_pin_memory=False, use_streams=False, min_offload_size=0
        )
    else:
        offload = get_act_offloading_ctx_manager(
            model, use_pin_memory=False, use_streams=False, min_offload_size=0
        )

    mlp_calls = []
    for name, module in model.named_modules():
        if name.endswith("mlp"):
            module.register_forward_pre_hook(lambda *_: mlp_calls.append(1))

    model.train()
    ids = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(1))
    with offload:
        model(input_ids=ids, labels=ids).loss.backward()
    return SimpleNamespace(
        model=model, args=args, offload=offload, mlp_calls=len(mlp_calls)
    )


def _saved_tensors_without_checkpointing(base_model_path, adapter):
    """Pack-hook calls an outer hook sees with no checkpointing at all.

    Like TRL, the output head runs under a no-op hook and is not counted.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaForCausalLM
    from trl.models.activation_offloading import NoOpManager

    torch.manual_seed(0)
    model = LlamaForCausalLM.from_pretrained(str(base_model_path))
    if adapter:
        model = get_peft_model(
            model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
        )
    model.train()
    head_noop = NoOpManager()
    head = model.get_output_embeddings()
    head.register_forward_pre_hook(lambda *_: head_noop.__enter__())
    head.register_forward_hook(lambda *_: head_noop.__exit__(), always_call=True)
    seen = []

    def pack(tensor):
        seen.append(1)
        return tensor

    ids = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(1))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model(input_ids=ids, labels=ids).loss.backward()
    return len(seen)


def _checkpoint_wrapped(model):
    return any(
        type(module).__name__ == "CheckpointWrapper" for module in model.modules()
    )


@pytest.mark.parametrize("mode", [True, DISK])
@pytest.mark.parametrize("adapter", [False, True], ids=["full", "lora"])
def test_offload_without_gc_offloads_everything(
    base_model_path, min_base_cfg, mode, adapter
):
    step = _train_step(
        base_model_path,
        min_base_cfg,
        gradient_checkpointing=False,
        activation_offloading=mode,
        **(LORA if adapter else {}),
    )

    assert step.args["gradient_checkpointing"] is False
    assert step.args["activation_offloading"] == mode
    assert not step.model.is_gradient_checkpointing
    assert not _checkpoint_wrapped(step.model)
    assert step.mlp_calls == NUM_LAYERS
    # TRL's pack hook runs once per saved tensor: it sees every one of them.
    assert step.offload.tensor_id == _saved_tensors_without_checkpointing(
        base_model_path, adapter
    )
    # Weight views saved for backward must be filtered, not offloaded.
    assert {
        p.untyped_storage().data_ptr() for p in step.model.parameters()
    } <= step.offload.param_storages


@pytest.mark.parametrize("mode", [True, DISK])
@pytest.mark.parametrize("adapter", [False, True], ids=["full", "lora"])
def test_offload_with_gc_recomputes_and_offloads_saved_tensors(
    base_model_path, min_base_cfg, mode, adapter
):
    step = _train_step(
        base_model_path,
        min_base_cfg,
        gradient_checkpointing=True,
        activation_offloading=mode,
        **(LORA if adapter else {}),
    )

    # adapters and full finetunes both checkpoint through HF, as without offloading
    assert step.args["gradient_checkpointing"] is True
    assert step.model.is_gradient_checkpointing
    assert not _checkpoint_wrapped(step.model)
    assert step.args["activation_offloading"] == mode
    assert step.mlp_calls == 2 * NUM_LAYERS
    # The checkpointed layers keep their saves to themselves; the offloader still
    # sees, and offloads, what is saved outside them.
    assert (
        0
        < step.offload.tensor_id
        < _saved_tensors_without_checkpointing(base_model_path, adapter)
    )


@pytest.mark.parametrize("adapter", [False, True], ids=["full", "lora"])
@pytest.mark.parametrize("selective", [False, True], ids=["gc", "sac"])
def test_hidden_states_offload_keeps_checkpointing(
    base_model_path, min_base_cfg, adapter, selective
):
    step = _train_step(
        base_model_path,
        min_base_cfg,
        gradient_checkpointing=True,
        activation_offloading="hidden_states",
        selective_checkpointing=selective,
        **(LORA if adapter else {}),
    )

    assert step.args["gradient_checkpointing"] is True
    assert step.args["activation_offloading"] == "hidden_states"
    assert step.model.is_gradient_checkpointing
    assert step.mlp_calls == 2 * NUM_LAYERS
    assert step.offload.stats.marked_tensors == NUM_LAYERS
    assert step.offload.stats.saved_tensors_seen < _saved_tensors_without_checkpointing(
        base_model_path, adapter
    )
    base = step.model.get_base_model() if adapter else step.model
    assert (
        getattr(base.gradient_checkpointing_enable, "_axolotl_sac", False) is selective
    )
