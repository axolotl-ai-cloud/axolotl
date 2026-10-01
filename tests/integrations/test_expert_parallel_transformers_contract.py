"""Contract tests between the Expert-Parallel integration and transformers' MoE seam."""

import copy
import inspect
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from transformers.distributed.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    EpRouterParallel,
    MoEParamShard,
)
from transformers.integrations.moe import (
    ALL_EXPERTS_FUNCTIONS,
    batched_mm_experts_forward,
    grouped_mm_experts_forward,
    use_experts_implementation,
)
from transformers.modeling_utils import PreTrainedModel

from axolotl.integrations.expert_parallel import experts_fn
from axolotl.integrations.expert_parallel.experts_fn import (
    EXPERT_PARALLEL,
    REGISTRY,
    _decorator_experts_interface,
    _normalize_sentinels,
    register_all,
)
from axolotl.integrations.expert_parallel.shard import (
    _detect_experts_modules,
    _is_param_wrapper,
    _real_experts_base,
)

E, K, H, INTER = 4, 2, 8, 16
KINDS = ("qwen3_moe", "mixtral")


def _qwen3moe_config():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    return Qwen3MoeConfig(
        hidden_size=H,
        moe_intermediate_size=INTER,
        num_experts=E,
        num_experts_per_tok=K,
        norm_topk_prob=True,
    )


def _mixtral_config():
    from transformers.models.mixtral.configuration_mixtral import MixtralConfig

    return MixtralConfig(
        hidden_size=H,
        intermediate_size=INTER,
        num_local_experts=E,
        num_experts_per_tok=K,
    )


def _build_experts(kind):
    torch.manual_seed(0)
    if kind == "qwen3_moe":
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeExperts as cls,
        )

        cfg = _qwen3moe_config()
    else:
        from transformers.models.mixtral.modeling_mixtral import MixtralExperts as cls

        cfg = _mixtral_config()
    m = cls(cfg)
    with torch.no_grad():
        m.gate_up_proj.normal_(0, 0.2)
        m.down_proj.normal_(0, 0.2)
    return m


def _build_qwen3moe_block():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    torch.manual_seed(0)
    block = Qwen3MoeSparseMoeBlock(_qwen3moe_config())
    with torch.no_grad():
        block.experts.gate_up_proj.normal_(0, 0.2)
        block.experts.down_proj.normal_(0, 0.2)
        block.gate.weight.normal_(0, 1.0)
    return block


def _unwrap(y):
    return y[0] if isinstance(y, tuple) else y


class TestDecoratedExpertsAttributes:
    @pytest.mark.parametrize("kind", KINDS)
    def test_module_attributes(self, kind):
        m = _build_experts(kind)
        assert m.has_gate is True
        assert m.has_bias is False
        assert m.is_transposed is False
        assert m.is_concatenated is True
        assert m._is_expert_parallel is False
        assert m.num_experts == E
        assert callable(m.act_fn)
        assert isinstance(m.gate_up_proj, torch.nn.Parameter)
        assert tuple(m.gate_up_proj.shape) == (E, 2 * INTER, H)
        assert isinstance(m.down_proj, torch.nn.Parameter)
        assert tuple(m.down_proj.shape) == (E, H, INTER)
        assert hasattr(m, "_apply_gate")
        assert hasattr(m, "config")
        assert getattr(m.config, "_experts_implementation", None) is None
        assert not hasattr(m, "gate_up_proj_bias")

    def test_decorator_signature(self):
        p = inspect.signature(use_experts_implementation).parameters
        assert p["experts_interface"].default is ALL_EXPERTS_FUNCTIONS
        assert p["is_concatenated"].default is True
        assert p["is_transposed"].default is False
        assert p["has_bias"].default is False
        assert p["has_gate"].default is True
        assert _decorator_experts_interface() is ALL_EXPERTS_FUNCTIONS

    def test_kernel_call_convention(self):
        for fn in (grouped_mm_experts_forward, batched_mm_experts_forward):
            assert list(inspect.signature(fn).parameters) == [
                "self",
                "hidden_states",
                "top_k_index",
                "top_k_weights",
            ]
        assert ALL_EXPERTS_FUNCTIONS["grouped_mm"] is grouped_mm_experts_forward
        assert ALL_EXPERTS_FUNCTIONS["batched_mm"] is batched_mm_experts_forward


class TestShardExpertWeightsOnDecoratedModule:
    @pytest.mark.parametrize("rank", [0, 1])
    @pytest.mark.parametrize("kind", KINDS)
    def test_sharded_module_matches_kernel_contract(self, fake_ep_sharder, kind, rank):
        model = torch.nn.Module()
        model.experts = _build_experts(kind)
        model.dense = torch.nn.Linear(4, 4)
        orig_gu = model.experts.gate_up_proj.detach().clone()
        orig_dn = model.experts.down_proj.detach().clone()
        assert fake_ep_sharder(model, rank) == 1

        m = model.experts
        e_local = E // 2
        assert m.num_experts == m.num_local_experts == e_local
        assert m.num_experts_global == E
        assert m.local_expert_offset == rank * e_local
        assert m._is_expert_parallel is True
        assert model._ep_num_experts_global == E
        lo, hi = rank * e_local, (rank + 1) * e_local
        torch.testing.assert_close(m.gate_up_proj, orig_gu[lo:hi])
        torch.testing.assert_close(m.down_proj, orig_dn[lo:hi])
        assert isinstance(m.gate_up_proj, torch.nn.Parameter)
        assert isinstance(m.down_proj, torch.nn.Parameter)
        assert m.gate_up_proj.requires_grad and m.down_proj.requires_grad
        assert m.has_gate is True
        assert m.has_bias is False
        assert m.is_transposed is False
        assert m.is_concatenated is True
        assert model._ddp_params_and_buffers_to_ignore == [
            "experts.gate_up_proj",
            "experts.down_proj",
        ]
        assert not getattr(model.dense, "_is_expert_parallel", False)

        idx, w = _normalize_sentinels(
            torch.tensor([[0, 1], [-1, 1], [-1, -1]]),
            torch.rand(3, 2),
            m.num_local_experts,
        )
        x = torch.randn(3, H)
        y = grouped_mm_experts_forward(m, x, idx, w)
        assert torch.isfinite(y).all()
        assert torch.allclose(y, experts_fn._eager_local(m, x, idx, w), atol=1e-6)


class TestRegistrationAndDetection:
    def test_registry_resolves_through_decorator_interface(self):
        register_all()
        assert set(REGISTRY) == {EXPERT_PARALLEL}
        stub = PreTrainedModel.__new__(PreTrainedModel)
        for name, fn in REGISTRY.items():
            assert ALL_EXPERTS_FUNCTIONS.get_interface(name, None) is fn
            assert _decorator_experts_interface().get_interface(name, None) is fn
            stub.config = SimpleNamespace(_experts_implementation=name)
            assert (
                PreTrainedModel.get_correct_experts_implementation(stub, name) == name
            )
        assert (
            experts_fn.resolve_local_implementation("eager") is experts_fn._eager_local
        )
        assert (
            experts_fn.resolve_local_implementation("grouped_mm")
            is grouped_mm_experts_forward
        )
        assert (
            experts_fn.resolve_local_implementation("batched_mm")
            is batched_mm_experts_forward
        )
        assert "sonicmoe" in ALL_EXPERTS_FUNCTIONS

    @pytest.mark.parametrize("kind", KINDS)
    def test_decorated_forward_dispatches_to_expert_parallel(self, kind):
        register_all()
        if kind == "qwen3_moe":
            block = _build_qwen3moe_block()
            inputs = (torch.randn(1, 5, H),)
        else:
            block = _build_experts("mixtral")
            idx = torch.tensor([[0, 1], [2, 3], [1, 2], [3, 0], [0, 2]])
            inputs = (torch.randn(5, H), idx, torch.rand(5, K))

        ref = copy.deepcopy(block)
        ref_experts = ref.experts if hasattr(ref, "experts") else ref
        ref_experts.config = copy.copy(ref_experts.config)
        ref_experts.config._experts_implementation = "eager"
        ep = copy.deepcopy(block)
        ep_experts = ep.experts if hasattr(ep, "experts") else ep
        ep_experts.config = copy.copy(ep_experts.config)
        ep_experts.config._experts_implementation = EXPERT_PARALLEL

        prev = experts_fn.get_backend()
        experts_fn.set_backend("torch")
        try:
            y_ref = _unwrap(ref(*inputs))
            y_ep = _unwrap(ep(*inputs))
        finally:
            experts_fn.set_backend(prev)

        assert torch.allclose(y_ep, y_ref, atol=1e-6)
        assert ep_experts._is_expert_parallel is True
        assert ref_experts._is_expert_parallel is False

    def test_detects_decorated_experts_in_real_block(self):
        block = _build_qwen3moe_block()
        assert list(_detect_experts_modules(block)) == [("experts", block.experts)]
        holder = torch.nn.Module()
        holder.experts = _build_experts("mixtral")
        assert list(_detect_experts_modules(holder)) == [("experts", holder.experts)]

    @pytest.mark.parametrize("kind", KINDS)
    def test_detection_skips_real_peft_param_wrapper(self, kind):
        from peft import LoraConfig
        from peft.tuners.lora.layer import ParamWrapper

        holder = torch.nn.Module()
        base = _build_experts(kind)
        holder.experts = ParamWrapper(
            base,
            "default",
            parameter_name="gate_up_proj",
            config=LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0),
            r=2,
            lora_alpha=4,
        )
        assert _is_param_wrapper(holder.experts)
        assert _real_experts_base(holder.experts) is base
        assert list(_detect_experts_modules(holder)) == [("experts.base_layer", base)]
        assert base.gate_up_proj.dim() == 3


class TestTransformersExpertParallelPlan:
    def test_plan_styles_exist(self):
        assert "grouped_gemm" in ALL_PARALLEL_STYLES
        assert "ep_router" in ALL_PARALLEL_STYLES
        assert isinstance(ALL_PARALLEL_STYLES["grouped_gemm"], MoEParamShard)
        assert ALL_PARALLEL_STYLES["grouped_gemm"].shards_expert_dim is True
        assert isinstance(ALL_PARALLEL_STYLES["ep_router"], EpRouterParallel)

    def test_ep_router_sentinel_matches_normalize_sentinels(self):
        mesh = SimpleNamespace(
            get_local_rank=lambda: 1,
            size=lambda: 2,
            ndim=1,
            get_group=lambda *a: None,
        )
        router = SimpleNamespace(num_experts=E)
        logits = torch.zeros(3, E)
        scores = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.2, 0.8]])
        ind = torch.tensor([[0, 3], [1, 2], [2, 3]])
        _, s_tf, i_tf = EpRouterParallel().transform_output_post_forward(
            router, (logits, scores, ind), mesh
        )
        i_ax, s_ax = _normalize_sentinels(
            torch.tensor([[-1, 1], [-1, 0], [0, 1]]), scores, E // 2
        )
        assert torch.equal(i_tf, i_ax)
        assert torch.equal(i_tf, torch.tensor([[2, 1], [2, 0], [0, 1]]))
        assert torch.equal(s_tf, s_ax)
        assert int(i_tf.max()) == E // 2


class TestMoEParamShardMarksExpertParallel:
    def setup_method(self):
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29571")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

    def teardown_method(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    @pytest.mark.parametrize("kind", KINDS)
    def test_shard_param_sets_flag_and_local_count(self, kind):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("ep",))
        m = _build_experts(kind)
        ALL_PARALLEL_STYLES["grouped_gemm"].shard_param(m, "gate_up_proj", mesh)
        assert m._is_expert_parallel is True
        assert m.num_experts == E
        assert isinstance(m.gate_up_proj, torch.nn.Parameter)
        assert isinstance(m.gate_up_proj.data, DTensor)
        assert tuple(m.gate_up_proj.shape) == (E, 2 * INTER, H)
