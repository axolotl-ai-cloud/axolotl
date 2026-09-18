"""NF4 conversion parity, serialization, and Transformers/PEFT integration."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from axolotl.utils.nf4 import (
    BnbNF4Parametrization,
    dequantize_bnb_4bit,
    quantize_bnb_4bit,
    quantize_torchao_nf4,
)


@pytest.mark.parametrize("double_quant", [False, True])
@pytest.mark.parametrize("size", [16384, 16448])
def test_bnb_chunked_parity(double_quant, size):
    import bitsandbytes.functional as F

    value = torch.randn(size)
    actual, state = quantize_bnb_4bit(
        value, compress_statistics=double_quant, chunk_size=4096
    )
    expected, reference = F.quantize_4bit(
        value, compress_statistics=double_quant, quant_type="nf4"
    )
    assert torch.equal(actual, expected)
    assert torch.equal(state.absmax, reference.absmax)
    if double_quant:
        assert torch.equal(state.state2.absmax, reference.state2.absmax)
        assert torch.equal(state.offset, reference.offset)
    torch.testing.assert_close(
        dequantize_bnb_4bit(actual, state, chunk_size=4096).flatten(),
        F.dequantize_4bit(expected, reference).flatten(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_parametrization_state_and_gradients(backend):
    weight = torch.randn(128, 128)
    if backend == "torchao":
        data, transform = quantize_torchao_nf4(weight, chunk_size=16384)
    else:
        data, state = quantize_bnb_4bit(weight)
        transform = BnbNF4Parametrization(state)
    model = nn.Linear(128, 128, bias=False)
    model.weight = nn.Parameter(data, requires_grad=False)
    parametrize.register_parametrization(model, "weight", transform, unsafe=True)
    expected = model.weight.clone()
    meta = copy.deepcopy(model).to("meta")
    assert meta.weight.shape == weight.shape
    meta.load_state_dict(model.state_dict(), assign=True)
    torch.testing.assert_close(meta.weight, expected, rtol=0, atol=0)
    x = torch.randn(2, 128, requires_grad=True)
    meta(x).sum().backward()
    torch.testing.assert_close(x.grad, expected.sum(0).expand_as(x))


def test_torchao_chunk_parity_and_expert_selection():
    from axolotl.utils.nf4 import torchao_nf4_module

    to_nf4 = torchao_nf4_module().to_nf4

    value = torch.randn(3, 128, 128)
    data, transform = quantize_torchao_nf4(value, chunk_size=16384)
    expected = torch.stack([to_nf4(expert).get_original_weight() for expert in value])
    torch.testing.assert_close(transform(data), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        transform(data, torch.tensor([2, 0])), expected[[2, 0]], rtol=0, atol=0
    )


def test_large_bnb_dispatch(monkeypatch):
    import bitsandbytes.functional as F

    from axolotl.monkeypatch import bnb_large_tensors

    value = torch.randn(256, 128)
    quantize, dequantize = F.quantize_4bit, F.dequantize_4bit
    expected, expected_state = quantize(
        value, quant_type="nf4", compress_statistics=True
    )
    expected_dense = dequantize(expected, expected_state)
    monkeypatch.setattr(bnb_large_tensors, "_MAX_ELEMENTS", 4095)
    try:
        bnb_large_tensors.patch_bnb_large_tensors()
        actual, state = F.quantize_4bit(
            value, quant_type="nf4", compress_statistics=True
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            F.dequantize_4bit(actual, state), expected_dense, rtol=0, atol=0
        )
    finally:
        F.quantize_4bit, F.dequantize_4bit = quantize, dequantize


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_transformers_load_and_peft(backend, tmp_path, monkeypatch):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from axolotl.loaders.nf4 import staged_nf4_loading
    from axolotl.monkeypatch.moe_quant import patch_peft_target_parameters_matching
    from axolotl.monkeypatch.peft.nf4 import patch_nf4_merge
    from axolotl.utils.dict import DictDefault

    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
    )
    original = LlamaForCausalLM(config)
    original.save_pretrained(tmp_path)
    cfg = DictDefault(nf4_backend=backend, quantize_moe_experts=True)
    import transformers.core_model_loading as loading

    def reject_prefetch(*args, **kwargs):
        pytest.fail("Staged NF4 must not prefetch unquantized weights")

    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "false")
    with monkeypatch.context() as scoped:
        scoped.setattr(loading, "ThreadPoolExecutor", reject_prefetch)
        with staged_nf4_loading(cfg, device="cpu"):
            model = LlamaForCausalLM.from_pretrained(tmp_path, device_map={"": "cpu"})
    assert parametrize.is_parametrized(model.model.layers[0].self_attn.q_proj, "weight")
    assert not parametrize.is_parametrized(model.lm_head, "weight")
    model.requires_grad_(False)
    patch_peft_target_parameters_matching()
    model = get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    tokens = torch.randint(0, 128, (2, 8))
    model(input_ids=tokens, labels=tokens).loss.backward()
    gradients = [p.grad for name, p in model.named_parameters() if "lora_B" in name]
    assert gradients and all(g is not None and g.abs().sum() > 0 for g in gradients)
    with torch.no_grad():
        for name, weight in model.named_parameters():
            if "lora_B" in name:
                weight.normal_(std=0.01)
    model.eval()
    expected = model(input_ids=tokens).logits.detach()
    patch_nf4_merge()
    merged = model.merge_and_unload()
    torch.testing.assert_close(
        merged(input_ids=tokens).logits, expected, atol=1e-5, rtol=1e-5
    )

    merged.save_pretrained(tmp_path / "merged")
    reloaded = LlamaForCausalLM.from_pretrained(tmp_path / "merged")
    torch.testing.assert_close(
        reloaded(input_ids=tokens).logits, expected, atol=1e-5, rtol=1e-5
    )


def _distributed_nf4_worker(
    rank,
    backend,
    checkpoint,
    rendezvous,
    phase=None,
    device_type="cpu",
    dtype=torch.float32,
    activation_checkpointing=False,
    offload=False,
    sharding_case=None,
    prepare_optimizer=False,
    mixed_precision=False,
):
    from unittest.mock import patch

    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import (
        AutoConfig,
        LlamaForCausalLM,
        PreTrainedModel,
        Qwen3MoeForCausalLM,
    )

    from axolotl.loaders.model import ModelLoader
    from axolotl.monkeypatch.accelerate.fsdp2 import fsdp2_prepare_model
    from axolotl.utils.dict import DictDefault

    sharding_case = sharding_case or {}
    world_size = sharding_case.get("world_size", 2)
    if device_type == "cuda":
        import os

        os.environ.update(
            LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE=str(world_size)
        )
        torch.cuda.set_device(rank)
    device = (
        torch.device(device_type, rank)
        if device_type == "cuda"
        else torch.device("cpu")
    )
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    from accelerate import FullyShardedDataParallelPlugin, PartialState
    from torch.distributed.fsdp import MixedPrecisionPolicy

    PartialState(cpu=device_type == "cpu")
    try:
        expert_model = AutoConfig.from_pretrained(checkpoint).model_type == "qwen3_moe"
        model_class = Qwen3MoeForCausalLM if expert_model else LlamaForCausalLM
        cfg = DictDefault(
            quantize_moe_experts=expert_model,
            base_model=checkpoint,
            nf4_backend=backend,
            load_in_4bit=True,
            adapter="qlora",
            fsdp_version=2,
            qlora_sharded_model_loading=True,
            fsdp_config={"cpu_ram_efficient_loading": True},
            torch_dtype=dtype,
        )
        loader = ModelLoader(cfg, tokenizer=None)
        loader.auto_model_loader = model_class
        loader.model_kwargs = {"dtype": cfg.torch_dtype}
        loader._set_quantization_config()
        reinitialized = []
        original_init_weights = PreTrainedModel._init_weights

        def record_init(self, module, *init_args, **init_kwargs):
            if getattr(module, "parametrizations", None):
                reinitialized.append(type(module).__name__)
            return original_init_weights(self, module, *init_args, **init_kwargs)

        PreTrainedModel._init_weights = record_init
        if rank:
            with patch.object(
                model_class,
                "from_pretrained",
                side_effect=AssertionError("peer read checkpoint"),
            ):
                loader._build_model()
                model = loader.model
            if rank:
                assert all(p.is_meta for p in model.parameters())
        else:
            loader._build_model()
            model = loader.model
        PreTrainedModel._init_weights = original_init_weights
        # rank zero is the one that stages, so it is the rank Transformers would
        # re-initialize, and the FSDP2 guard only covers ranks that are not local
        # rank zero. Checking _is_hf_initialized would pass either way: the flag is
        # also set *after* a module is initialized.
        assert not reinitialized, reinitialized
        model = get_peft_model(
            model,
            LoraConfig(
                r=sharding_case.get("lora_rank", 4),
                target_modules=["q_proj", "v_proj"],
                target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
                if expert_model
                else None,
            ),
        )
        shape_snapshot = (
            _nf4_shape_snapshot(model, model_class, loader.model_config)
            if sharding_case
            else None
        )
        model.eval()
        tokens = torch.arange(8, device=device).reshape(1, 8)
        expected = torch.empty(1, 8, 128, dtype=dtype, device=device)
        if rank == 0:
            reference_model = copy.deepcopy(model).to(device)
            expected.copy_(reference_model(input_ids=tokens).logits.detach())
            del reference_model
        dist.broadcast(expected, src=0)
        mesh = init_device_mesh(
            device_type, (world_size,), mesh_dim_names=("dp_shard",)
        )
        plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            cpu_ram_efficient_loading=True,
            auto_wrap_policy=sharding_case.get("wrap_policy", "TRANSFORMER_BASED_WRAP"),
            min_num_params=1000
            if sharding_case.get("wrap_policy") == "SIZE_BASED_WRAP"
            else None,
            transformer_cls_names_to_wrap=[
                "Qwen3MoeDecoderLayer" if expert_model else "LlamaDecoderLayer"
            ],
            reshard_after_forward=sharding_case.get("reshard_after_forward", True),
            activation_checkpointing=activation_checkpointing,
            cpu_offload=offload,
            # what accelerate builds for `bf16: true`: FSDP2 casts every sharded
            # parameter to param_dtype, packed NF4 bytes included
            mixed_precision_policy=MixedPrecisionPolicy(
                param_dtype=dtype, reduce_dtype=dtype, output_dtype=dtype
            )
            if mixed_precision
            else None,
        )
        accelerator = SimpleNamespace(
            device=device,
            is_main_process=rank == 0,
            state=SimpleNamespace(
                fsdp_plugin=plugin,
                device_mesh=mesh,
                parallelism_config=SimpleNamespace(fsdp_dim_names=("dp_shard",)),
            ),
        )
        state_dict_calls = []
        state_dicts_before_sharding = []
        unprepared = model
        unsharded_state_dict = model.state_dict

        def counting_state_dict(*sd_args, **sd_kwargs):
            state_dict_calls.append(1)
            return unsharded_state_dict(*sd_args, **sd_kwargs)

        model.state_dict = counting_state_dict

        import torch.distributed.fsdp as torch_fsdp

        real_fully_shard = torch_fsdp.fully_shard

        def counting_fully_shard(*fs_args, **fs_kwargs):
            if not state_dicts_before_sharding:
                state_dicts_before_sharding.append(len(state_dict_calls))
            return real_fully_shard(*fs_args, **fs_kwargs)

        torch_fsdp.fully_shard = counting_fully_shard

        optimizer = None
        if prepare_optimizer:
            from accelerate import Accelerator

            from axolotl.loaders.utils import materialize_trainable_meta_params
            from axolotl.monkeypatch.accelerate.fsdp2 import patch_accelerate_fsdp2

            patch_accelerate_fsdp2()
            partial_state = PartialState()
            partial_state.process_index = rank
            partial_state.local_process_index = rank
            partial_state.num_processes = world_size
            real_accelerator = Accelerator(cpu=device_type == "cpu")
            real_accelerator.state.fsdp_plugin = plugin
            real_accelerator.state.device_mesh = mesh
            real_accelerator.state.parallelism_config = SimpleNamespace(
                fsdp_dim_names=("dp_shard",)
            )
            # ModelLoader.load() does this before prepare; this harness builds the
            # model by hand, so it has to do it too or the data_ptr remap collapses
            materialize_trainable_meta_params(model)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=0.001
            )
            model, optimizer = real_accelerator._prepare_fsdp2(model, optimizer)
            assert {
                id(p) for group in optimizer.param_groups for p in group["params"]
            } == {id(p) for p in model.parameters() if p.requires_grad}
        else:
            model = fsdp2_prepare_model(accelerator, model)
        torch_fsdp.fully_shard = real_fully_shard
        del unprepared.state_dict
        # only rank zero holds the staged weights, so only it builds the state dict
        # that gets broadcast; a peer building its own is the materialization this
        # whole path exists to avoid
        assert state_dicts_before_sharding == [1 if rank == 0 else 0], (
            rank,
            state_dicts_before_sharding,
        )
        if offload:
            assert all(p.device.type == "cpu" for p in model.parameters())
        if shape_snapshot:
            _assert_sharded_shapes(model, shape_snapshot, rank, world_size)
            _install_logical_shape_checks(model, shape_snapshot)
        actual = model(input_ids=tokens, labels=tokens)
        if shape_snapshot:
            for name, tensor in model.state_dict().items():
                assert (tuple(tensor.shape), tensor.dtype) == shape_snapshot[0][name], (
                    name
                )
        torch.testing.assert_close(actual.logits, expected, rtol=1e-5, atol=1e-5)
        actual.loss.backward()
        missing_gradients = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not missing_gradients, missing_gradients
        if shape_snapshot:
            _assert_sharded_shapes(model, shape_snapshot, rank, world_size)
        if phase is None:
            _check_fsdp_resume(model, rank, tokens, checkpoint, optimizer)
        else:
            _check_fresh_trainer_resume(model, rank, tokens, checkpoint, phase)
        if shape_snapshot:
            _assert_sharded_shapes(model, shape_snapshot, rank, world_size)
    finally:
        dist.destroy_process_group()


def _check_fsdp_resume(model, rank, tokens, checkpoint, optimizer=None):
    from pathlib import Path

    import torch.distributed as dist
    from accelerate.utils import fsdp_utils
    from torch.distributed.fsdp import StateDictType

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.001
        )
    optimizer.step()
    optimizer.zero_grad()
    accelerator = SimpleNamespace(
        process_index=rank,
        num_processes=dist.get_world_size(),
        is_fsdp2=True,
        is_main_process=rank == 0,
        wait_for_everyone=dist.barrier,
    )
    for kind in (StateDictType.SHARDED_STATE_DICT, StateDictType.FULL_STATE_DICT):
        plugin = SimpleNamespace(
            fsdp_version=2,
            state_dict_type=kind,
            state_dict_config=SimpleNamespace(
                offload_to_cpu=True, rank0_only=kind == StateDictType.FULL_STATE_DICT
            ),
            optim_state_dict_config=SimpleNamespace(
                rank0_only=kind == StateDictType.FULL_STATE_DICT
            ),
        )
        directory = str(Path(checkpoint).parent / kind.name)
        saved_weights = {
            name: p.to_local().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        fsdp_utils.save_fsdp_model(
            plugin, accelerator, model, directory, adapter_only=True
        )
        fsdp_utils.save_fsdp_optimizer(plugin, accelerator, optimizer, model, directory)
        expected_loss = model(input_ids=tokens, labels=tokens).loss
        expected_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        expected_weights = {
            name: p.to_local().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        probe = None
        if tokens.device.type == "cpu" and kind == StateDictType.FULL_STATE_DICT:
            from torch._subclasses.fake_tensor import FakeTensorMode

            with FakeTensorMode():
                probe = torch.empty(4, device="cuda")
            model.register_buffer("_nf4_restore_device_probe", probe)
        parameter_ids = {name: id(p) for name, p in model.named_parameters()}
        fsdp_utils.load_fsdp_model(
            plugin, accelerator, model, directory, adapter_only=True
        )
        assert parameter_ids == {name: id(p) for name, p in model.named_parameters()}
        if probe is not None:
            assert model._nf4_restore_device_probe is probe
            del model._nf4_restore_device_probe
        for name, p in model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(
                    p.to_local(),
                    saved_weights[name],
                    msg=f"{kind} {name}",
                )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.2
        )
        fsdp_utils.load_fsdp_optimizer(plugin, accelerator, optimizer, model, directory)
        from torch.distributed.tensor import DTensor

        for parameter, state in optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                assert state[key].shape == parameter.shape
                if isinstance(parameter, DTensor):
                    assert isinstance(state[key], DTensor)
                    assert state[key].to_local().shape == parameter.to_local().shape
        resumed_loss = model(input_ids=tokens, labels=tokens).loss
        torch.testing.assert_close(resumed_loss, expected_loss)
        resumed_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        for name, p in model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(
                    p.to_local(), expected_weights[name], rtol=0, atol=0
                )


def _mixed_precision_case(backend, tmp_path, device_type):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
    )
    checkpoint = tmp_path / "base"
    LlamaForCausalLM(config).save_pretrained(checkpoint)
    torch.multiprocessing.spawn(
        _distributed_nf4_worker,
        args=(
            backend,
            str(checkpoint),
            str(tmp_path / "rendezvous"),
            None,
            device_type,
            torch.bfloat16,
            False,
            False,
            None,
            False,
            True,
        ),
        nprocs=2,
    )


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
@pytest.mark.nf4_distributed
def test_nf4_mixed_precision_matches_unsharded(backend, tmp_path):
    """A bf16 param_dtype policy must not reinterpret the packed NF4 storage."""
    _mixed_precision_case(backend, tmp_path, "cpu")


@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_cuda_nf4_mixed_precision_matches_unsharded(backend, tmp_path):
    _mixed_precision_case(backend, tmp_path, "cuda")


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.nf4_distributed
def test_rank_zero_load_shard_and_backward(backend, dtype, tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
    )
    checkpoint = tmp_path / "base"
    LlamaForCausalLM(config).save_pretrained(checkpoint)
    torch.multiprocessing.spawn(
        _distributed_nf4_worker,
        args=(
            backend,
            str(checkpoint),
            str(tmp_path / "rendezvous"),
            None,
            "cpu",
            dtype,
            False,
            False,
            None,
            True,
        ),
        nprocs=2,
    )


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_fused_expert_merge(backend):
    from peft import LoraConfig, get_peft_model

    from axolotl.monkeypatch.moe_quant import patch_peft_target_parameters_matching
    from axolotl.monkeypatch.peft.nf4 import patch_nf4_merge

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(2, 128, 128))

        def forward(self, x):
            return torch.einsum("bi,eoi->beo", x, self.gate_up_proj)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Experts()

        def forward(self, x):
            return self.experts(x)

    model = Model()
    value = model.experts.gate_up_proj.detach()
    if backend == "torchao":
        packed, transform = quantize_torchao_nf4(value)
    else:
        packed, state = quantize_bnb_4bit(value)
        transform = BnbNF4Parametrization(state)
    model.experts.gate_up_proj = nn.Parameter(packed, requires_grad=False)
    parametrize.register_parametrization(
        model.experts, "gate_up_proj", transform, unsafe=True
    )
    patch_peft_target_parameters_matching()
    patch_nf4_merge()
    model = get_peft_model(
        model,
        LoraConfig(r=4, target_modules=[], target_parameters=["experts.gate_up_proj"]),
    )
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_B" in name:
                value.normal_(std=0.01)
    x = torch.randn(3, 128)
    expected = model(x).detach()
    merged = model.merge_and_unload()
    torch.testing.assert_close(merged(x), expected, atol=2e-5, rtol=1e-5)
    assert merged.experts.gate_up_proj.shape == (2, 128, 128)


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_efficient_merge_quantization_matches_training(backend):
    from axolotl.cli.utils.lora_merge import _simulate_nf4_roundtrip

    value = torch.randn(3, 128, 128)
    if backend == "torchao":
        data, transform = quantize_torchao_nf4(value)
        expected = transform(data)
    else:
        data, state = quantize_bnb_4bit(value)
        expected = dequantize_bnb_4bit(data, state)
    actual = _simulate_nf4_roundtrip(value, backend=backend, device="cpu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
@pytest.mark.parametrize("setting", ["omitted", None, True, False])
def test_sharded_nf4_loading_default(backend, setting):
    from axolotl.utils.schemas.config import AxolotlInputConfig

    config = dict(
        base_model="test",
        learning_rate=1e-4,
        datasets=[{"path": "test", "type": "alpaca"}],
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        adapter="qlora",
        load_in_4bit=True,
        nf4_backend=backend,
        fsdp_version=2,
        fsdp_config={"cpu_ram_efficient_loading": True},
    )
    if setting != "omitted":
        config["qlora_sharded_model_loading"] = setting
    if setting is False:
        with pytest.raises(ValueError, match="cpu_ram_efficient_loading"):
            AxolotlInputConfig(**config)
    else:
        validated = AxolotlInputConfig(**config)
        assert validated.qlora_sharded_model_loading is True
        assert AxolotlInputConfig(
            **validated.model_dump(exclude_none=True)
        ).qlora_sharded_model_loading


@pytest.mark.parametrize("setting", ["omitted", None, False])
@pytest.mark.parametrize(
    "overrides",
    [
        {"fsdp_version": 1},
        {"fsdp_config": None, "fsdp_version": None},
        {"fsdp_config": {"cpu_ram_efficient_loading": False}},
        {"adapter": "lora", "load_in_4bit": False},
    ],
)
def test_sharded_nf4_loading_default_outside_fsdp2_qlora(setting, overrides):
    from axolotl.utils.schemas.config import AxolotlInputConfig

    config = dict(
        base_model="test",
        learning_rate=1e-4,
        datasets=[{"path": "test", "type": "alpaca"}],
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        adapter="qlora",
        load_in_4bit=True,
        fsdp_version=2,
        fsdp_config={"cpu_ram_efficient_loading": True},
    )
    config.update(overrides)
    if setting != "omitted":
        config["qlora_sharded_model_loading"] = setting
    assert AxolotlInputConfig(**config).qlora_sharded_model_loading is False


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_linear_backward_does_not_retain_dense_weights(backend, dtype):
    from axolotl.utils.nf4 import checkpoint_nf4_linear

    value = torch.randn(128, 128, dtype=dtype)
    if backend == "torchao":
        data, transform = quantize_torchao_nf4(value)
    else:
        data, state = quantize_bnb_4bit(value)
        transform = BnbNF4Parametrization(state)
    model = nn.Linear(128, 128, bias=False)
    model.weight = nn.Parameter(data, requires_grad=False)
    checkpoint_nf4_linear(model)
    parametrize.register_parametrization(model, "weight", transform, unsafe=True)
    dense = model.weight.detach()
    x = torch.randn(2, 128, dtype=dtype, requires_grad=True)
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = model(x)
        result.sum().backward()
    torch.testing.assert_close(result, torch.nn.functional.linear(x, dense))
    torch.testing.assert_close(x.grad, dense.sum(0).expand_as(x))
    assert all(tensor.numel() < value.numel() for tensor in saved)


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_merge_matches_loaded_model(backend, dtype, tmp_path):
    from peft import LoraConfig, get_peft_model
    from transformers import BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM

    from axolotl.cli.utils.lora_merge import merge_lora_sharded_efficient
    from axolotl.loaders.nf4 import load_nf4_model
    from axolotl.utils.dict import DictDefault
    from axolotl.utils.nf4 import nf4_skip_modules

    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
    )
    base = tmp_path / "base"
    LlamaForCausalLM(config).save_pretrained(base)
    quantization = {"llm_int8_skip_modules": ["q_proj"]}
    cfg = DictDefault(
        base_model=str(base),
        nf4_backend=backend,
        torch_dtype=dtype,
        bnb_config_kwargs=quantization,
    )
    model = load_nf4_model(
        LlamaForCausalLM,
        config,
        {
            "dtype": dtype,
            "device_map": {"": "cpu"},
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                **quantization,
            ),
        },
        cfg,
        "cpu",
    )
    assert not parametrize.is_parametrized(
        model.model.layers[0].self_attn.q_proj, "weight"
    )
    model = get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    model.eval()
    adapter = tmp_path / "adapter"
    model.save_pretrained(adapter)
    merged = tmp_path / "merged"
    merge_lora_sharded_efficient(
        base,
        adapter,
        merged,
        simulate_nf4=True,
        nf4_backend=backend,
        nf4_skips=nf4_skip_modules("llama", quantization),
        nf4_dtype=dtype,
    )
    actual = LlamaForCausalLM.from_pretrained(merged, dtype=dtype)
    tokens = torch.arange(8).reshape(1, 8)
    torch.testing.assert_close(
        actual(input_ids=tokens).logits, model(input_ids=tokens).logits, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "name, key, expected",
    [
        ("lm_head.weight", "lm_head", False),
        ("model.lm_head.weight", "lm_head", False),
        ("model.layers.0.self_attn.q_proj.weight", "q_proj", False),
        ("model.layers.0.self_attn.q_proj.weight", "proj", True),
        ("model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp", False),
        ("model.layers.10.mlp.down_proj.weight", "model.layers.1", True),
        ("model.layers.0.self_attn.q_proj.weight", "_proj.", False),
        ("model.layers.0.mlp.experts.gate_up_proj", ".experts", False),
        ("model.layers.0.self_attn.q_proj.weight", ".*_proj$", True),
        ("model.layers.0.self_attn.q_proj.weight", r"q_proj\.weight$", False),
        ("model.layers.3.mlp.down_proj.weight", r"layers\.[0-3]\.", False),
        ("model.layers.4.mlp.down_proj.weight", r"layers\.[0-3]\.", True),
    ],
    ids=[
        "name-root",
        "name-nested",
        "name-component",
        "name-not-substring",
        "dotted-prefix",
        "dotted-prefix-not-textual",
        "substring-trailing-dot",
        "substring-leading-dot",
        "regex-unanchored-suffix",
        "regex-anchored",
        "regex-range-hit",
        "regex-range-miss",
    ],
)
def test_nf4_skip_matching_rules(name, key, expected):
    from axolotl.utils.nf4 import nf4_should_quantize

    assert nf4_should_quantize(name, linear=True, expert=False, skips={key}) is expected


def test_nf4_skip_rejects_invalid_regex():
    from axolotl.utils.nf4 import nf4_skip_modules

    with pytest.raises(ValueError, match="Invalid regex"):
        nf4_skip_modules(None, {"llm_int8_skip_modules": ["layers[.weight"]})


def test_resolved_architecture_exclusions_reach_staged_loader(tmp_path):
    import transformers.core_model_loading as loading
    from transformers import BitsAndBytesConfig

    from axolotl.loaders.nf4 import staged_nf4_loading
    from axolotl.utils.dict import DictDefault

    model = nn.Module()
    model.out_proj = nn.Linear(128, 128, bias=False)
    cfg = DictDefault(nf4_backend="bitsandbytes", model_config_type="falcon_h1")
    value = torch.randn(128, 128)
    with staged_nf4_loading(
        cfg,
        device="cpu",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, llm_int8_skip_modules=["out_proj"]
        ),
    ):
        loading.set_param_for_module(
            model,
            "out_proj.weight",
            value,
            SimpleNamespace(
                missing_keys=set(), unexpected_keys=set(), mismatched_keys=set()
            ),
            None,
        )
    assert not parametrize.is_parametrized(model.out_proj)
    torch.testing.assert_close(model.out_proj.weight, value, rtol=0, atol=0)


def _check_fresh_trainer_resume(model, rank, tokens, checkpoint, phase):
    from pathlib import Path

    import torch.distributed as dist
    from accelerate import FullyShardedDataParallelPlugin
    from transformers import Trainer, TrainingArguments

    directory = Path(checkpoint).parent / "fresh-resume"
    directory.mkdir(exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(directory), use_cpu=True, report_to=[]),
    )
    trainer.is_fsdp_enabled = True
    trainer.accelerator.state.fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2, state_dict_type="SHARDED_STATE_DICT"
    )
    trainer.optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.001
    )
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        trainer.optimizer, step_size=1, gamma=0.8
    )
    if phase == "save":
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        trainer.optimizer.zero_grad()
        trainer._save_optimizer_and_scheduler(str(directory))
    else:
        trainer.optimizer.zero_grad()
        trainer._load_from_checkpoint(str(directory), model)
        trainer._load_optimizer_and_scheduler(str(directory))
    loss = model(input_ids=tokens, labels=tokens).loss
    loss.backward()
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    state = {
        name: p.to_local().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    state["loss"] = loss.detach()
    state["lr"] = trainer.lr_scheduler.get_last_lr()
    reference = directory / f"expected-{rank}.pt"
    if phase == "save":
        torch.save(state, reference)
    else:
        torch.testing.assert_close(
            state, torch.load(reference, weights_only=True), rtol=0, atol=0
        )
    dist.barrier()


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
@pytest.mark.nf4_distributed
def test_fresh_process_trainer_resume(backend, tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    base = tmp_path / "base"
    LlamaForCausalLM(
        LlamaConfig(
            hidden_size=128,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=128,
        )
    ).save_pretrained(base)
    for phase in ("save", "resume"):
        torch.multiprocessing.spawn(
            _distributed_nf4_worker,
            args=(backend, str(base), str(tmp_path / phase), phase),
            nprocs=2,
        )


@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_cuda_nccl_bf16_loading_and_resume(backend, tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    base = tmp_path / "base"
    LlamaForCausalLM(
        LlamaConfig(
            hidden_size=128,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=128,
        )
    ).save_pretrained(base)
    torch.multiprocessing.spawn(
        _distributed_nf4_worker,
        args=(backend, str(base), str(tmp_path / "nccl"), None, "cuda", torch.bfloat16),
        nprocs=2,
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA GPU with at least 16 GiB free",
)
@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_cuda_tensor_exceeding_int32_limit(backend):
    import bitsandbytes.functional as F

    from axolotl.monkeypatch.bnb_large_tensors import patch_bnb_large_tensors

    if torch.cuda.mem_get_info()[0] < 16 * 1024**3:
        pytest.skip("requires at least 16 GiB free GPU memory")
    value = torch.empty(2**31 + 128, device="cuda", dtype=torch.bfloat16).uniform_(
        -1, 1
    )
    if backend == "bitsandbytes":
        patch_bnb_large_tensors()
        data, state = F.quantize_4bit(value, quant_type="nf4", compress_statistics=True)
        actual = F.dequantize_4bit(data, state)
    else:
        data, transform = quantize_torchao_nf4(value)
        actual = transform(data)
    assert actual.shape == value.shape
    for start in (0, 2**30, value.numel() - 128):
        reference = value[start : start + 128]
        reconstructed = actual[start : start + 128]
        assert torch.isfinite(reconstructed).all()
        torch.testing.assert_close(reconstructed, reference, rtol=0, atol=0.18)


def _cuda_trainer_worker(rank, backend, base, rendezvous, output, phase):
    import os
    from pathlib import Path

    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    from axolotl.loaders.model import ModelLoader
    from axolotl.monkeypatch.accelerate.fsdp2 import patch_accelerate_fsdp2
    from axolotl.utils.dict import DictDefault

    os.environ.update(LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE="2")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        patch_accelerate_fsdp2()
        args = TrainingArguments(
            output_dir=str(Path(output) / phase),
            max_steps=2,
            per_device_train_batch_size=1,
            save_steps=1,
            learning_rate=0.001,
            bf16=True,
            report_to=[],
            disable_tqdm=True,
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "version": 2,
                "cpu_ram_efficient_loading": True,
                "state_dict_type": "SHARDED_STATE_DICT",
                "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
            },
        )
        cfg = DictDefault(
            base_model=base,
            nf4_backend=backend,
            adapter="qlora",
            load_in_4bit=True,
            fsdp_version=2,
            qlora_sharded_model_loading=True,
            fsdp_config=DictDefault(cpu_ram_efficient_loading=True),
            torch_dtype=torch.bfloat16,
        )
        loader = ModelLoader(cfg, tokenizer=None)
        loader.model_kwargs = {"dtype": torch.bfloat16}
        loader._set_quantization_config()
        loader._build_model()
        model = get_peft_model(
            loader.model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"])
        )
        # ModelLoader.load() does this before prepare; this harness builds the
        # adapter by hand, so it has to as well or the data_ptr remap collapses
        # the optimizer on the meta ranks and the sharded optimizer save fails
        loader.model = model
        loader._materialize_trainable_meta_params()
        tokens = torch.arange(8)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=[{"input_ids": tokens, "labels": tokens}] * 8,
        )
        resume = (
            str(Path(output) / "reference" / "checkpoint-1")
            if phase == "resume"
            else None
        )
        trainer.train(resume_from_checkpoint=resume)
        state = {
            name: p.to_local().cpu()
            for name, p in trainer.model.named_parameters()
            if p.requires_grad
        }
        path = Path(output) / f"reference-{rank}.pt"
        if phase == "reference":
            torch.save(state, path)
        else:
            torch.testing.assert_close(
                state, torch.load(path, weights_only=True), rtol=0, atol=0
            )
        assert trainer.state.global_step == 2
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_cuda_trainer_train_and_fresh_resume(backend, tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    base = tmp_path / "base"
    LlamaForCausalLM(
        LlamaConfig(
            hidden_size=128,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=128,
        )
    ).save_pretrained(base)
    for phase in ("reference", "resume"):
        torch.multiprocessing.spawn(
            _cuda_trainer_worker,
            args=(
                backend,
                str(base),
                str(tmp_path / phase),
                str(tmp_path / "training"),
                phase,
            ),
            nprocs=2,
        )


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
@pytest.mark.nf4_distributed
def test_real_moe_fsdp_resume(backend, tmp_path):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    base = tmp_path / "base"
    config = Qwen3MoeConfig(
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
        num_experts=2,
        num_experts_per_tok=1,
    )
    original = Qwen3MoeForCausalLM(config)
    original.save_pretrained(base)
    from safetensors.torch import save_file

    state = dict(original.state_dict())
    for name in list(state):
        if name.endswith(".experts.gate_up_proj"):
            tensor = state.pop(name)
            prefix = name.removesuffix(".gate_up_proj")
            for expert, weight in enumerate(tensor):
                gate, up = weight.chunk(2, dim=0)
                state[f"{prefix}.{expert}.gate_proj.weight"] = gate.contiguous()
                state[f"{prefix}.{expert}.up_proj.weight"] = up.contiguous()
        elif name.endswith(".experts.down_proj"):
            tensor = state.pop(name)
            prefix = name.removesuffix(".down_proj")
            for expert, weight in enumerate(tensor):
                state[f"{prefix}.{expert}.down_proj.weight"] = weight.contiguous()
    save_file(state, base / "model.safetensors", metadata={"format": "pt"})
    torch.multiprocessing.spawn(
        _distributed_nf4_worker,
        args=(
            backend,
            str(base),
            str(tmp_path / "moe"),
            None,
            "cpu",
            torch.float32,
            False,
            False,
            None,
        ),
        nprocs=2,
    )


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_legacy_merge_rejects_staged_nf4(backend):
    from axolotl.cli.merge_lora import do_merge_lora
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(
        merge_method="legacy", _original_nf4_backend=backend, _original_staged_nf4=True
    )
    with pytest.raises(ValueError, match="memory_efficient"):
        do_merge_lora.__wrapped__(cfg=cfg)


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
@pytest.mark.nf4_distributed
def test_outer_activation_checkpointing(backend, tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    base = tmp_path / "base"
    LlamaForCausalLM(
        LlamaConfig(
            hidden_size=128,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=128,
        )
    ).save_pretrained(base)
    torch.multiprocessing.spawn(
        _distributed_nf4_worker,
        args=(
            backend,
            str(base),
            str(tmp_path / "activation"),
            None,
            "cpu",
            torch.bfloat16,
            True,
        ),
        nprocs=2,
    )


@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_cuda_cpu_offload(backend, tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    base = tmp_path / "base"
    LlamaForCausalLM(
        LlamaConfig(
            hidden_size=128,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=128,
        )
    ).save_pretrained(base)
    torch.multiprocessing.spawn(
        _distributed_nf4_worker,
        args=(
            backend,
            str(base),
            str(tmp_path / "offload"),
            None,
            "cuda",
            torch.bfloat16,
            True,
            True,
        ),
        nprocs=2,
    )


def _nf4_shape_snapshot(model, model_class, config):
    import math

    from accelerate import init_empty_weights

    from axolotl.utils.nf4 import TorchaoNF4Parametrization

    with init_empty_weights():
        dense_reference = model_class(config)
    dense_shapes = {
        name: tuple(p.shape) for name, p in dense_reference.named_parameters()
    }
    logical = {}
    for path, module in model.named_modules():
        for name, chain in getattr(module, "parametrizations", {}).items():
            if not isinstance(
                chain[0], (BnbNF4Parametrization, TorchaoNF4Parametrization)
            ):
                continue
            reference_name = f"{path}.{name}".replace(".base_layer", "").removeprefix(
                "base_model.model."
            )
            shape = dense_shapes[reference_name]
            assert tuple(chain[0].shape) == shape
            numel = math.prod(shape)
            if isinstance(chain[0], TorchaoNF4Parametrization):
                numel = math.ceil(numel / 16384) * 16384
            assert chain.original.shape == ((numel + 1) // 2, 1)
            assert chain.original.dtype == torch.uint8
            logical[(path, name)] = shape
    assert logical
    return (
        {
            name: (tuple(value.shape), value.dtype)
            for name, value in model.state_dict().items()
        },
        logical,
    )


def _assert_sharded_shapes(model, snapshot, rank, world_size):
    from torch.distributed.tensor import DTensor, Shard

    expected, _ = snapshot
    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    empty_lora_shards = 0
    for name, tensor in actual.items():
        shape, dtype = expected[name]
        assert tuple(tensor.shape) == shape, name
        assert tensor.dtype == dtype, name
        if isinstance(tensor, DTensor):
            assert tensor.placements == (Shard(0),), name
            rows = (shape[0] + world_size - 1) // world_size
            local_rows = max(0, min(rows, shape[0] - rank * rows))
            assert tensor.to_local().shape == (local_rows, *shape[1:]), name
            if "lora_A" in name and local_rows == 0:
                empty_lora_shards += 1
    if rank == world_size - 1 and world_size > 2:
        assert empty_lora_shards > 0


def _install_logical_shape_checks(model, snapshot):
    _, logical = snapshot
    for (path, name), expected in logical.items():
        module = model.get_submodule(path)

        def check_shape(current, _inputs, parameter_name=name, expected_shape=expected):
            assert tuple(getattr(current, parameter_name).shape) == expected_shape

        module.register_forward_pre_hook(check_shape)


@pytest.mark.parametrize("architecture", ["dense", "moe"])
@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
@pytest.mark.parametrize("reshard", [True, False], ids=["full_shard", "shard_grad_op"])
@pytest.mark.parametrize(
    "wrap_policy",
    ["TRANSFORMER_BASED_WRAP", "SIZE_BASED_WRAP", pytest.param(None, id="root_wrap")],
)
@pytest.mark.nf4_distributed
def test_nf4_fsdp2_shape_matrix(architecture, backend, reshard, wrap_policy, tmp_path):
    _run_nf4_shape_case(architecture, backend, reshard, wrap_policy, tmp_path, "cpu")


def _run_nf4_shape_case(
    architecture, backend, reshard, wrap_policy, tmp_path, device_type
):
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
    )

    options = dict(
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
    )
    if architecture == "moe":
        model = Qwen3MoeForCausalLM(
            Qwen3MoeConfig(
                **options,
                moe_intermediate_size=192,
                num_experts=3,
                num_experts_per_tok=2,
            )
        )
    else:
        model = LlamaForCausalLM(LlamaConfig(**options))
    checkpoint = tmp_path / "base"
    model.save_pretrained(checkpoint)
    case = {
        "world_size": 3,
        "lora_rank": 2,
        "reshard_after_forward": reshard,
        "wrap_policy": wrap_policy,
    }
    torch.multiprocessing.spawn(
        _distributed_nf4_worker,
        args=(
            backend,
            str(checkpoint),
            str(tmp_path / "shapes"),
            None,
            device_type,
            torch.bfloat16,
            False,
            False,
            case,
        ),
        nprocs=3,
    )


def _hybrid_nf4_rejection_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from axolotl.monkeypatch.accelerate.fsdp2_nf4 import load_staged_nf4_state

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4
    )
    try:
        mesh = init_device_mesh(
            "cpu", (2, 2), mesh_dim_names=("dp_replicate", "dp_shard")
        )
        model = nn.Linear(128, 128, bias=False, device="meta")
        fully_shard(model, mesh=mesh)
        state = {"weight": torch.ones(128, 128)} if rank == 0 else {}
        with pytest.raises(ValueError, match="one-dimensional Shard"):
            load_staged_nf4_state(
                SimpleNamespace(device=torch.device("cpu"), is_main_process=rank == 0),
                model,
                state,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.nf4_distributed
def test_hybrid_nf4_mesh_is_explicitly_rejected(tmp_path):
    torch.multiprocessing.spawn(
        _hybrid_nf4_rejection_worker, args=(str(tmp_path / "hybrid"),), nprocs=4
    )


@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 3, reason="requires three CUDA GPUs")
@pytest.mark.parametrize("architecture", ["dense", "moe"])
@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
@pytest.mark.parametrize("reshard", [True, False], ids=["full_shard", "shard_grad_op"])
@pytest.mark.parametrize(
    "wrap_policy",
    ["TRANSFORMER_BASED_WRAP", "SIZE_BASED_WRAP", pytest.param(None, id="root_wrap")],
)
def test_cuda_nf4_fsdp2_shape_matrix(
    architecture, backend, reshard, wrap_policy, tmp_path
):
    _run_nf4_shape_case(architecture, backend, reshard, wrap_policy, tmp_path, "cuda")


def _nf4_delayed_loading_worker(
    rank, checkpoint, rendezvous, timeout_source, device_type
):
    import os
    import time
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist
    from transformers import LlamaConfig, LlamaForCausalLM

    from axolotl.loaders.nf4 import load_nf4_model
    from axolotl.utils.dict import DictDefault

    if device_type == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=3),
    )
    cfg = DictDefault(
        base_model=checkpoint,
        fsdp_config={"cpu_ram_efficient_loading": True},
        nf4_backend="bitsandbytes",
        torch_dtype=torch.float32,
        ddp_timeout=1 if timeout_source == "env" else 30,
    )
    original = LlamaForCausalLM.from_pretrained

    def delayed(*args, **kwargs):
        assert rank == 0
        time.sleep(4)
        return original(*args, **kwargs)

    try:
        with patch.dict(
            os.environ,
            {"AXOLOTL_NCCL_TIMEOUT": "30"} if timeout_source == "env" else {},
        ):
            with patch.object(LlamaForCausalLM, "from_pretrained", delayed):
                model = load_nf4_model(
                    LlamaForCausalLM,
                    LlamaConfig.from_pretrained(checkpoint),
                    {"dtype": torch.float32, "device_map": {"": "cpu"}},
                    cfg,
                    torch.device(device_type, rank) if device_type == "cuda" else "cpu",
                )
        assert all(parameter.is_meta == (rank != 0) for parameter in model.parameters())
    finally:
        dist.destroy_process_group()


def _run_nf4_delayed_loading(tmp_path, timeout_source, device_type):
    from transformers import LlamaConfig, LlamaForCausalLM

    base = tmp_path / "base"
    LlamaForCausalLM(
        LlamaConfig(
            hidden_size=128,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=128,
        )
    ).save_pretrained(base)
    torch.multiprocessing.spawn(
        _nf4_delayed_loading_worker,
        args=(str(base), str(tmp_path / "delay"), timeout_source, device_type),
        nprocs=2,
    )


@pytest.mark.nf4_distributed
@pytest.mark.parametrize("timeout_source", ["config", "env"])
def test_nf4_staging_outlives_default_group_timeout(
    tmp_path, timeout_source, monkeypatch
):
    monkeypatch.delenv("AXOLOTL_NCCL_TIMEOUT", raising=False)
    _run_nf4_delayed_loading(tmp_path, timeout_source, "cpu")


@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
@pytest.mark.parametrize("timeout_source", ["config", "env"])
def test_cuda_nf4_staging_outlives_default_group_timeout(
    tmp_path, timeout_source, monkeypatch
):
    monkeypatch.delenv("AXOLOTL_NCCL_TIMEOUT", raising=False)
    _run_nf4_delayed_loading(tmp_path, timeout_source, "cuda")


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_nf4_peer_metadata_does_not_copy_cpu_buffers(backend, monkeypatch):
    from axolotl.loaders.nf4 import _collect_nf4_structures

    value = torch.randn(128, 128)
    if backend == "torchao":
        data, transform = quantize_torchao_nf4(value)
    else:
        data, state = quantize_bnb_4bit(value)
        transform = BnbNF4Parametrization(state)
    model = nn.Linear(128, 128, bias=False)
    model.weight = nn.Parameter(data, requires_grad=False)
    parametrize.register_parametrization(model, "weight", transform, unsafe=True)

    def reject_copy(*args, **kwargs):
        pytest.fail("Peer metadata must not copy tensor data")

    monkeypatch.setattr(torch.Tensor, "__deepcopy__", reject_copy)
    structures = []
    _collect_nf4_structures(model, structures)
    assert structures and all(buffer.is_meta for buffer in structures[0][2].buffers())
    assert all(buffer.device.type == "cpu" for buffer in transform.buffers())


def test_nf4_phase_reports_progress_and_failure(monkeypatch):
    import re
    import threading

    from axolotl.utils import nf4_loading

    messages = []
    heartbeat = threading.Event()

    def record(message, *args):
        messages.append(message % args)
        if "still running" in message:
            heartbeat.set()

    monkeypatch.setattr(nf4_loading.LOG, "info", record)
    with pytest.raises(ValueError, match="failed conversion"):
        with nf4_loading.nf4_phase("Test phase", interval=0.01):
            # a heartbeat that only reports elapsed time cannot tell slow from hung
            nf4_loading.record_progress(4096)
            assert heartbeat.wait(5)
            raise ValueError("failed conversion")
    assert any(
        re.search(r"\b[1-9]\d* tensors", message)
        for message in messages
        if "still running" in message
    ), messages
    assert any("starting" in message for message in messages)
    assert any("failed after" in message for message in messages)
    assert not any(thread.name == "nf4-progress" for thread in threading.enumerate())


@pytest.mark.parametrize(
    "environment,configured,expected",
    [(None, None, 1800), (None, 21600, 21600), ("36000", 21600, 36000)],
)
def test_nf4_loading_group_timeout_and_cleanup(
    environment, configured, expected, monkeypatch
):
    from datetime import timedelta
    from unittest.mock import Mock

    from axolotl.utils import nf4_loading
    from axolotl.utils.dict import DictDefault

    if environment is None:
        monkeypatch.delenv("AXOLOTL_NCCL_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("AXOLOTL_NCCL_TIMEOUT", environment)
    group = object()
    create = Mock(return_value=group)
    destroy = Mock()
    monkeypatch.setattr(nf4_loading.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(nf4_loading.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(nf4_loading.dist, "new_group", create)
    monkeypatch.setattr(nf4_loading.dist, "barrier", Mock())
    monkeypatch.setattr(nf4_loading.dist, "destroy_process_group", destroy)
    cfg = DictDefault(
        fsdp_config={"cpu_ram_efficient_loading": True}, ddp_timeout=configured
    )
    with pytest.raises(RuntimeError, match="loading failed"):
        with nf4_loading.nf4_loading_group(cfg) as actual:
            assert actual is group
            raise RuntimeError("loading failed")
    create.assert_called_once_with(backend="gloo", timeout=timedelta(seconds=expected))
    destroy.assert_called_once_with(group)


def _divergent_plan_worker(rank, rendezvous):
    import torch.distributed as dist

    from axolotl.monkeypatch.accelerate.fsdp2_nf4 import _agree_distribution_plan

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        accelerator = SimpleNamespace(
            device=torch.device("cpu"), is_main_process=rank == 0
        )
        shape = (4, 2) if rank == 0 else (8, 2)
        targets = {"weight": torch.zeros(shape)}
        if rank == 0:
            plan = _agree_distribution_plan(accelerator, targets, accelerator.device)
            assert [entry[0] for entry in plan] == ["weight"]
        else:
            with pytest.raises(ValueError, match="disagrees on weight"):
                _agree_distribution_plan(accelerator, targets, accelerator.device)

        targets = {} if rank else {"weight": torch.zeros(4, 2)}
        if rank == 0:
            _agree_distribution_plan(accelerator, targets, accelerator.device)
        else:
            with pytest.raises(ValueError, match="missing weight"):
                _agree_distribution_plan(accelerator, targets, accelerator.device)
    finally:
        dist.destroy_process_group()


@pytest.mark.nf4_distributed
def test_nf4_distribution_plan_mismatch_raises(tmp_path):
    """A per-rank state_dict disagreement must raise, not desynchronize the group."""
    torch.multiprocessing.spawn(
        _divergent_plan_worker, args=(str(tmp_path / "plan"),), nprocs=2
    )


def _shard_traffic_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from axolotl.monkeypatch.accelerate.fsdp2_nf4 import load_staged_nf4_state

    world_size = 3
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(0)
        full = {"0.weight": torch.randn(128, 64), "1.weight": torch.randn(2, 4)}
        model = nn.Sequential(
            nn.Linear(64, 128, bias=False, device="meta"),
            nn.Linear(4, 2, bias=False, device="meta"),
        )
        fully_shard(
            model,
            mesh=init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp_shard",)),
        )
        traffic = []
        original = {name: getattr(dist, name) for name in ("broadcast", "scatter")}

        def counted(name):
            def call(tensor, *args, **kwargs):
                traffic.append((name, tensor.numel() * tensor.element_size()))
                return original[name](tensor, *args, **kwargs)

            return call

        for name in original:
            setattr(dist, name, counted(name))
        try:
            load_staged_nf4_state(
                SimpleNamespace(device=torch.device("cpu"), is_main_process=rank == 0),
                model,
                dict(full) if rank == 0 else {},
            )
        finally:
            for name, value in original.items():
                setattr(dist, name, value)

        state = model.state_dict()
        budget = 0
        for name, value in full.items():
            rows = -(-value.shape[0] // world_size)
            start = min(rank * rows, value.shape[0])
            end = min(start + rows, value.shape[0])
            torch.testing.assert_close(
                state[name].to_local(), value[start:end], rtol=0, atol=0
            )
            budget += rows * value[0].numel() * value.element_size()
        moved = sum(size for _, size in traffic)
        assert moved <= budget, (
            f"rank {rank} moved {moved} bytes through {traffic}, but its own "
            f"shards are only {budget} bytes"
        )
        assert len(traffic) == len(full), (
            f"rank {rank} issued {len(traffic)} data collectives for "
            f"{len(full)} sharded parameters: {traffic}"
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.nf4_distributed
def test_nf4_shard_distribution_traffic(tmp_path):
    """Every rank receives its own shard only, not one copy of every rank's shard."""
    torch.multiprocessing.spawn(
        _shard_traffic_worker, args=(str(tmp_path / "traffic"),), nprocs=3
    )


@pytest.mark.parametrize("rows", list(range(40)) + [128, 1000, 151936])
def test_nf4_shard_bounds_match_torch_chunk(rows):
    """Shard boundaries must stay bit-identical to what FSDP2's chunking produces."""
    from axolotl.monkeypatch.accelerate.fsdp2_nf4 import _shard_bounds

    for size in range(1, 9):
        offset = 0
        expected = []
        for chunk in torch.empty(rows, 1).chunk(size):
            expected.append((offset, offset + chunk.shape[0]))
            offset += chunk.shape[0]
        expected += [(rows, rows)] * (size - len(expected))
        bounds = _shard_bounds(rows, size)
        assert bounds == expected, f"{rows} rows over {size} ranks"
        assert all(0 <= start <= end <= rows for start, end in bounds)


def test_init_distributed_state_surfaces_partialstate_errors(monkeypatch):
    """A PartialState construction failure must not be swallowed silently."""
    import logging

    from axolotl.utils import distributed as axolotl_distributed

    records = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    class Exploding:
        _shared_state: dict = {}

        def __init__(self, *args, **kwargs):
            raise ValueError("backend mismatch")

    handler = Collector(level=logging.WARNING)
    logger = logging.getLogger("axolotl.utils.distributed")
    logger.addHandler(handler)
    monkeypatch.setattr(axolotl_distributed, "PartialState", Exploding)
    monkeypatch.setattr(axolotl_distributed, "distributed_state", None)
    try:
        axolotl_distributed.init_distributed_state()
        assert axolotl_distributed.distributed_state is None
        assert any("backend mismatch" in message for message in records), records
    finally:
        logger.removeHandler(handler)


def _nf4_failure_propagation_worker(rank, checkpoint, rendezvous, results, failing):
    import json
    import time
    from datetime import timedelta
    from pathlib import Path
    from unittest.mock import patch

    import torch.distributed as dist
    from transformers import LlamaConfig, LlamaForCausalLM

    import axolotl.loaders.nf4 as nf4_loader
    from axolotl.utils.dict import DictDefault

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=5),
    )
    cfg = DictDefault(
        base_model=checkpoint,
        fsdp_config={"cpu_ram_efficient_loading": True},
        nf4_backend="bitsandbytes",
        torch_dtype=torch.float32,
        ddp_timeout=5,
    )

    def peer_boom(*args, **kwargs):
        raise RuntimeError("peer boom")

    def meta_boom(*args, **kwargs):
        raise RuntimeError("meta boom")

    def payload_boom(*args, **kwargs):
        raise RuntimeError("payload boom")

    patches = []
    if failing == "peer" and rank == 1:
        patches.append(patch.object(LlamaForCausalLM, "_from_config", peer_boom))
    if failing == "rank_zero_metadata" and rank == 0:
        patches.append(patch.object(nf4_loader, "_collect_nf4_structures", meta_boom))
    if failing == "peer_payload" and rank == 1:
        patches.append(patch.object(nf4_loader, "checkpoint_nf4_linear", payload_boom))

    error = None
    start = time.monotonic()
    try:
        for entry in patches:
            entry.start()
        try:
            nf4_loader.load_nf4_model(
                LlamaForCausalLM,
                LlamaConfig.from_pretrained(checkpoint),
                {"dtype": torch.float32, "device_map": {"": "cpu"}},
                cfg,
                "cpu",
            )
        finally:
            for entry in reversed(patches):
                entry.stop()
    except BaseException as exc:  # pylint: disable=broad-except
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - start
    Path(results, f"rank{rank}.json").write_text(
        json.dumps({"error": error, "elapsed": elapsed})
    )
    try:
        dist.destroy_process_group()
    except BaseException:  # pylint: disable=broad-except
        pass


def _run_nf4_failure_propagation(tmp_path, failing):
    import json

    from transformers import LlamaConfig, LlamaForCausalLM

    base = tmp_path / "base"
    LlamaForCausalLM(
        LlamaConfig(
            hidden_size=128,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=128,
        )
    ).save_pretrained(base)
    results = tmp_path / "results"
    results.mkdir()
    torch.multiprocessing.spawn(
        _nf4_failure_propagation_worker,
        args=(str(base), str(tmp_path / "fail"), str(results), failing),
        nprocs=2,
    )
    return [json.loads((results / f"rank{rank}.json").read_text()) for rank in range(2)]


@pytest.mark.nf4_distributed
@pytest.mark.parametrize("failing", ["peer", "peer_payload"])
def test_nf4_peer_load_failure_propagates_to_rank_zero(failing, tmp_path, monkeypatch):
    """A peer that fails before or after the metadata broadcast must abort rank zero."""
    monkeypatch.delenv("AXOLOTL_NCCL_TIMEOUT", raising=False)
    cause = "peer boom" if failing == "peer" else "payload boom"
    outcomes = _run_nf4_failure_propagation(tmp_path, failing)
    assert outcomes[1]["error"] and cause in outcomes[1]["error"]
    assert outcomes[0]["error"], "rank zero completed while a peer failed"
    assert "rank 1" in outcomes[0]["error"], outcomes[0]["error"]
    assert cause in outcomes[0]["error"], outcomes[0]["error"]
    assert outcomes[0]["elapsed"] < 5, outcomes[0]


@pytest.mark.nf4_distributed
def test_nf4_rank_zero_metadata_failure_propagates_to_peers(tmp_path, monkeypatch):
    """A rank-zero metadata failure must abort peers with the real cause, not a timeout."""
    monkeypatch.delenv("AXOLOTL_NCCL_TIMEOUT", raising=False)
    outcomes = _run_nf4_failure_propagation(tmp_path, "rank_zero_metadata")
    assert outcomes[0]["error"] and "meta boom" in outcomes[0]["error"]
    assert outcomes[1]["error"], "peer completed while rank zero failed"
    assert "rank 0" in outcomes[1]["error"], outcomes[1]["error"]
    assert "meta boom" in outcomes[1]["error"], outcomes[1]["error"]
    assert outcomes[1]["elapsed"] < 5, outcomes[1]


_UNSET = object()

_STAGED_NF4_BASE = dict(
    base_model="test",
    learning_rate=1e-4,
    datasets=[{"path": "test", "type": "alpaca"}],
    micro_batch_size=1,
    gradient_accumulation_steps=1,
    adapter="qlora",
    load_in_4bit=True,
    fsdp_version=2,
    fsdp_config={"cpu_ram_efficient_loading": True},
    qlora_sharded_model_loading=True,
)


def _staged_nf4_config(**overrides):
    config = dict(_STAGED_NF4_BASE)
    config.update(overrides)
    return {key: value for key, value in config.items() if value is not _UNSET}


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, None),
        ({"adapter": "lora"}, "adapter: qlora and load_in_4bit: true"),
        ({"load_in_8bit": True}, "adapter: qlora and load_in_4bit: true"),
        ({"peft_use_dora": True}, "without DoRA or modules_to_save"),
        ({"lora_modules_to_save": ["embed_tokens"]}, "without DoRA or modules_to_save"),
        ({"dp_replicate_size": 2}, "dp_replicate_size must be 1"),
        (
            {"bnb_config_kwargs": {"bnb_4bit_quant_type": "fp4"}},
            "requires bnb_4bit_quant_type: nf4",
        ),
        ({"peft_init_lora_weights": "pissa"}, "value-dependent LoRA init"),
        ({"peft_init_lora_weights": "olora"}, "value-dependent LoRA init"),
        ({"peft_init_lora_weights": "loftq"}, "value-dependent LoRA init"),
        ({"peft_init_lora_weights": "corda"}, "value-dependent LoRA init"),
        ({"peft_init_lora_weights": "eva"}, "value-dependent LoRA init"),
        ({"peft_init_lora_weights": "gaussian"}, None),
        (
            {"peft": {"loftq_config": {"loftq_bits": 4}}},
            "value-dependent LoRA init",
        ),
        ({"tensor_parallel_size": 2}, "tensor, expert or context parallelism"),
        (
            {"deepspeed": "deepspeed_configs/zero3.json"},
            "tensor, expert or context parallelism",
        ),
    ],
    ids=[
        "staged-ok",
        "adapter-not-qlora",
        "load_in_8bit",
        "peft_use_dora",
        "lora_modules_to_save",
        "dp_replicate_size",
        "bnb_4bit_quant_type",
        "init-pissa",
        "init-olora",
        "init-loftq",
        "init-corda",
        "init-eva",
        "init-gaussian",
        "loftq_config",
        "tensor_parallel_size",
        "deepspeed",
    ],
)
def test_staged_nf4_validation_via_config(backend, overrides, expected):
    """check_staged_nf4 through the real pydantic model, not as an unbound call."""
    from axolotl.utils.schemas.config import AxolotlInputConfig

    config = _staged_nf4_config(nf4_backend=backend, **overrides)
    if expected is None:
        validated = AxolotlInputConfig(**config)
        assert validated.qlora_sharded_model_loading is True
    else:
        with pytest.raises(ValueError, match=expected):
            AxolotlInputConfig(**config)


@pytest.mark.parametrize(
    "backend, expected", [("bitsandbytes", None), ("torchao", "load_in_4bit")]
)
def test_stale_sharded_flag_without_4bit(backend, expected):
    """The flag only stages with load_in_4bit, as uses_staged_nf4 reads it; torchao has no other path."""
    from axolotl.loaders.nf4 import uses_staged_nf4
    from axolotl.utils.schemas.config import AxolotlInputConfig

    config = _staged_nf4_config(nf4_backend=backend, adapter="lora", load_in_4bit=False)
    if expected:
        with pytest.raises(ValueError, match=expected):
            AxolotlInputConfig(**config)
        return
    validated = AxolotlInputConfig(**config)
    assert not uses_staged_nf4(validated)


def _timeout_warnings(monkeypatch, tmp_path, requested, group_timeout):
    import logging
    from datetime import timedelta

    import torch.distributed as dist
    from accelerate import PartialState

    from axolotl.utils import distributed as axolotl_distributed

    records = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Collector(level=logging.WARNING)
    logger = logging.getLogger("axolotl.utils.distributed")
    logger.addHandler(handler)
    if requested is None:
        monkeypatch.delenv("AXOLOTL_NCCL_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("AXOLOTL_NCCL_TIMEOUT", str(requested))
    axolotl_distributed.distributed_state = None
    PartialState._reset_state()
    if group_timeout is not None:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{tmp_path / 'rendezvous'}",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=group_timeout),
        )
    try:
        axolotl_distributed.init_distributed_state()
        return records
    finally:
        logger.removeHandler(handler)
        if dist.is_initialized():
            dist.destroy_process_group()
        PartialState._reset_state()
        axolotl_distributed.distributed_state = None


def test_init_distributed_state_is_silent_when_the_group_timeout_suffices(
    tmp_path, monkeypatch
):
    assert _timeout_warnings(monkeypatch, tmp_path, requested=5, group_timeout=60) == []


def test_init_distributed_state_is_silent_without_a_preexisting_group(
    tmp_path, monkeypatch
):
    assert (
        _timeout_warnings(monkeypatch, tmp_path, requested=5, group_timeout=None) == []
    )


def test_inert_sharded_flag_does_not_reject_lora():
    """adapter: lora with the inert flag combination validated on main and must still."""
    from axolotl.utils.schemas.config import AxolotlInputConfig

    config = _staged_nf4_config(
        nf4_backend="bitsandbytes",
        adapter="lora",
        fsdp_config={"cpu_ram_efficient_loading": False},
    )
    assert AxolotlInputConfig(**config).qlora_sharded_model_loading is False


def test_sharded_loading_without_cpu_ram_efficient_loading_warns(caplog):
    """The flag used to be silently inert here; it must stay inert but say so."""
    from axolotl.utils.schemas.config import AxolotlInputConfig

    config = _staged_nf4_config(
        nf4_backend="bitsandbytes",
        fsdp_config={"cpu_ram_efficient_loading": False},
    )
    with caplog.at_level("WARNING", logger="axolotl.utils.schemas.validation"):
        validated = AxolotlInputConfig(**config)
    assert validated.qlora_sharded_model_loading is False
    assert any("has no effect" in record.getMessage() for record in caplog.records)


def test_torchao_without_cpu_ram_efficient_loading_is_rejected():
    """torchao has no non-staged loader to fall back to."""
    from axolotl.utils.schemas.config import AxolotlInputConfig

    config = _staged_nf4_config(
        nf4_backend="torchao",
        fsdp_config={"cpu_ram_efficient_loading": False},
    )
    with pytest.raises(ValueError, match="requires FSDP2, cpu_ram_efficient_loading"):
        AxolotlInputConfig(**config)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"adapter": "lora"}, "adapter: qlora and load_in_4bit"),
        ({"load_in_4bit": False}, "adapter: qlora and load_in_4bit"),
        ({"load_in_8bit": True}, "adapter: qlora and load_in_4bit"),
        (
            {"bnb_config_kwargs": {"blocksize": 128}},
            "torchao NF4 requires blocksize 64",
        ),
        (
            {"bnb_config_kwargs": {"bnb_4bit_use_double_quant": False}},
            "torchao NF4 requires blocksize 64",
        ),
    ],
    ids=[
        "adapter-not-qlora",
        "load_in_4bit-false",
        "load_in_8bit",
        "torchao-blocksize",
        "torchao-double-quant",
    ],
)
def test_staged_nf4_validation_torchao_without_fsdp(overrides, expected):
    """nf4_backend: torchao stages with no fsdp_config at all, a second route in."""
    from axolotl.utils.schemas.config import AxolotlInputConfig

    config = _staged_nf4_config(
        nf4_backend="torchao",
        fsdp_version=_UNSET,
        fsdp_config=_UNSET,
        qlora_sharded_model_loading=_UNSET,
        **overrides,
    )
    if expected is None:
        AxolotlInputConfig(**config)
    else:
        with pytest.raises(ValueError, match=expected):
            AxolotlInputConfig(**config)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"load_in_4bit": False}, "Require cfg.load_in_4bit to be True for qlora"),
        ({"adapter": "lora"}, "FSDP2 does not support `cpu_ram_efficient_loading`"),
        (
            {"qlora_sharded_model_loading": False},
            "FSDP2 does not support `cpu_ram_efficient_loading`",
        ),
    ],
    ids=["load_in_4bit-false", "adapter-not-qlora", "sharded-false"],
)
def test_staged_nf4_validation_shadowed_when_sharding_is_defaulted(overrides, expected):
    """Without an explicit qlora_sharded_model_loading, earlier validators reject first."""
    from axolotl.utils.schemas.config import AxolotlInputConfig

    config = _staged_nf4_config(
        **{
            "nf4_backend": "bitsandbytes",
            "qlora_sharded_model_loading": _UNSET,
            **overrides,
        }
    )
    with pytest.raises(ValueError, match=expected):
        AxolotlInputConfig(**config)


def test_staged_nf4_validation_expert_parallel():
    """expert_parallel_size only exists once its plugin args are merged in."""
    from axolotl.integrations.expert_parallel.args import ExpertParallelArgs
    from axolotl.utils.schemas.config import AxolotlInputConfig

    class _EPConfig(AxolotlInputConfig, ExpertParallelArgs):
        pass

    config = _staged_nf4_config(nf4_backend="bitsandbytes")
    assert _EPConfig(**config).expert_parallel_size == 1
    with pytest.raises(ValueError, match="tensor, expert or context parallelism"):
        _EPConfig(**dict(config, expert_parallel_size=2))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("double_quant", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_merge_roundtrip_matches_training(double_quant, dtype):
    """The merge roundtrip must reproduce the training-time parametrization exactly.

    Nothing asserted this on CUDA before: the only parity coverage ran CPU against CPU.
    """
    from axolotl.cli.utils.lora_merge import _simulate_nf4_roundtrip

    torch.manual_seed(0)
    value = torch.randn(512, 512, dtype=dtype)
    merged = _simulate_nf4_roundtrip(
        value, device="cuda", compress_statistics=double_quant
    )
    assert merged.device.type == "cpu"
    assert merged.dtype == dtype

    data, state = quantize_bnb_4bit(
        value,
        device=torch.device("cuda"),
        storage_device=torch.device("cuda"),
        compress_statistics=double_quant,
    )
    trained = BnbNF4Parametrization(state)(data).reshape(value.shape).cpu()
    torch.testing.assert_close(merged, trained, rtol=0, atol=0)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("double_quant", [False, True])
def test_cuda_nf4_dequant_is_device_independent(double_quant):
    """One quantization dequantized on either device must agree bit-for-bit.

    Quantization itself is not device-independent: bnb rounds the blockwise double-quant of
    the scale vector differently on CPU and CUDA (~18 of 4096 codes on a 512x512 draw), so a
    CPU merge and a GPU merge of the same weight legitimately differ. Dequantization must not
    add to that, since it is what the merge and the training forward pass share.
    """
    torch.manual_seed(0)
    value = torch.randn(512, 512, dtype=torch.bfloat16)
    data, state = quantize_bnb_4bit(
        value,
        device=torch.device("cuda"),
        storage_device=torch.device("cuda"),
        compress_statistics=double_quant,
    )
    on_gpu = dequantize_bnb_4bit(data, state).reshape(value.shape).cpu()
    on_cpu = dequantize_bnb_4bit(
        data, state, out=torch.empty(state.shape, dtype=state.dtype, device="cpu")
    ).reshape(value.shape)
    torch.testing.assert_close(on_gpu, on_cpu, rtol=0, atol=0)


def test_init_distributed_state_warns_without_explicit_timeout(tmp_path, monkeypatch):
    """A launcher-managed group must not be hard-failed when the user asked for nothing.

    Ray and torchrun create the process group themselves; only an explicit
    AXOLOTL_NCCL_TIMEOUT (which prepare_optim_env derives from ddp_timeout) is intent.
    """
    import logging
    from datetime import timedelta

    import torch.distributed as dist
    from accelerate import PartialState

    from axolotl.utils import distributed as axolotl_distributed

    records = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Collector(level=logging.WARNING)
    logger = logging.getLogger("axolotl.utils.distributed")
    logger.addHandler(handler)
    monkeypatch.delenv("AXOLOTL_NCCL_TIMEOUT", raising=False)
    # assigned directly, not via monkeypatch: teardown would restore a PartialState
    # object that _reset_state() has already emptied, breaking later tests
    axolotl_distributed.distributed_state = None
    PartialState._reset_state()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'rendezvous'}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=5),
    )
    try:
        axolotl_distributed.init_distributed_state()
        assert any("0:00:05" in message for message in records), records
    finally:
        logger.removeHandler(handler)
        dist.destroy_process_group()
        PartialState._reset_state()
        axolotl_distributed.distributed_state = None


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_staged_loading_does_not_reinitialize_quantized_weights(backend, tmp_path):
    """Staged loading must not let Transformers re-initialize the weights it just packed.

    Popping ``quantization_config`` means Transformers registers no quantizer, so the
    parametrized weights read as missing keys and ``_init_weights`` draws a full
    ``normal_`` over every one of them, single-threaded on CPU, before step one.
    """
    from transformers import BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM

    from axolotl.loaders.nf4 import load_nf4_model
    from axolotl.utils.dict import DictDefault

    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=512,
    )
    checkpoint = tmp_path / "base"
    LlamaForCausalLM(config).save_pretrained(checkpoint)

    original = torch.Tensor.normal_
    drawn = []

    def counting_normal(self, *args, **kwargs):
        drawn.append(self.numel())
        return original(self, *args, **kwargs)

    def measure(load):
        drawn.clear()
        torch.Tensor.normal_ = counting_normal
        try:
            load()
        finally:
            torch.Tensor.normal_ = original
        return sum(drawn)

    baseline = measure(
        lambda: LlamaForCausalLM.from_pretrained(checkpoint, dtype=torch.float32)
    )
    staged = measure(
        lambda: load_nf4_model(
            LlamaForCausalLM,
            config,
            {
                "dtype": torch.float32,
                "device_map": {"": "cpu"},
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4"
                ),
            },
            DictDefault(
                base_model=str(checkpoint),
                nf4_backend=backend,
                torch_dtype=torch.float32,
            ),
            "cpu",
        )
    )
    assert staged == baseline, (
        f"staged loading drew {staged} elements against a plain load's {baseline}; "
        "quantized modules are being re-initialized"
    )
