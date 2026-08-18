"""Regression tests: the Qwen3-Next decoder loop must compile under torch.compile, reusing the shared GatedDeltaNet ops (no aten.nonzero graph break in the loop body)."""

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

pytest.importorskip("transformers.models.qwen3_next")
pytest.importorskip("fla")


@pytest.fixture
def packing_patched():
    """Apply the qwen3_next packing patch (torch_compile on) and restore globals after."""
    from fla.modules import FusedRMSNormGated
    from transformers.models.qwen3_next import modeling_qwen3_next as hf

    from axolotl.monkeypatch.models.qwen3_next import modeling as qm

    saved = {
        "decoder_forward": hf.Qwen3NextDecoderLayer.forward,
        "gdn_forward": hf.Qwen3NextGatedDeltaNet.forward,
        "norm_forward": FusedRMSNormGated.forward,
        "norm_present": hasattr(FusedRMSNormGated, "_axolotl_compile_boundary"),
        "norm_flag": getattr(FusedRMSNormGated, "_axolotl_compile_boundary", None),
        "chunk": getattr(hf, "chunk_gated_delta_rule", None),
        "recurrent": getattr(hf, "fused_recurrent_gated_delta_rule", None),
        "norm_cls": getattr(hf, "FusedRMSNormGated", None),
        "fast_path": getattr(hf, "is_fast_path_available", None),
        "fla_ops_flag": qm._FLA_COMPILED_OPS,
    }
    qm.patch_qwen3_next_modeling_packing(torch_compile=True)
    yield qm
    hf.Qwen3NextDecoderLayer.forward = saved["decoder_forward"]
    hf.Qwen3NextGatedDeltaNet.forward = saved["gdn_forward"]
    FusedRMSNormGated.forward = saved["norm_forward"]
    if saved["norm_present"]:
        FusedRMSNormGated._axolotl_compile_boundary = saved["norm_flag"]
    else:
        try:
            delattr(FusedRMSNormGated, "_axolotl_compile_boundary")
        except AttributeError:
            pass
    hf.chunk_gated_delta_rule = saved["chunk"]
    hf.fused_recurrent_gated_delta_rule = saved["recurrent"]
    hf.FusedRMSNormGated = saved["norm_cls"]
    hf.is_fast_path_available = saved["fast_path"]
    qm._FLA_COMPILED_OPS = saved["fla_ops_flag"]


def _build_model(seed: int = 0, attn: str = "sdpa"):
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextModel

    torch.manual_seed(seed)
    cfg = Qwen3NextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    cfg._attn_implementation = attn
    return Qwen3NextModel(cfg).cuda().to(torch.bfloat16)


def _packed_inputs(T: int = 64):
    torch.manual_seed(123)
    input_ids = torch.randint(0, 128, (1, T), device="cuda")
    pos = torch.cat(
        [torch.arange(40, device="cuda"), torch.arange(T - 40, device="cuda")]
    ).view(1, T)
    return input_ids, pos


def _fwd_bwd(model, fn=None):
    input_ids, position_ids = _packed_inputs()
    if fn is None:
        fn = model
    out = fn(
        input_ids=input_ids, position_ids=position_ids, use_cache=False
    ).last_hidden_state
    loss = out.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    grads = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return loss.detach(), grads


class TestQwen3NextDecoderLoopCompiles:
    def test_no_graph_breaks_with_grad_checkpointing(self, packing_patched):
        """The qwen3_next decoder loop must trace with ZERO breaks via the shared GatedDeltaNet ops."""
        torch._dynamo.reset()
        from torch._dynamo.utils import counters

        counters.clear()
        model = _build_model()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        _fwd_bwd(model, torch.compile(model))

        breaks = dict(counters["graph_break"])
        assert not breaks, f"qwen3_next decoder loop graph-broke: {list(breaks)}"
        assert counters["stats"]["unique_graphs"] >= 1

    def test_eager_parity_ops_vs_legacy_bitwise(self, packing_patched):
        """The shared opaque-op path must be bitwise-identical to qwen3_next's legacy eager (cast_g=False, g stays f32)."""
        qm = packing_patched
        assert qm._FLA_COMPILED_OPS
        model = _build_model()
        model.train()
        state = {k: v.clone() for k, v in model.state_dict().items()}

        qm._FLA_COMPILED_OPS = False
        loss_a, grads_a = _fwd_bwd(model)
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        qm._FLA_COMPILED_OPS = True
        loss_b, grads_b = _fwd_bwd(model)

        assert torch.equal(loss_a, loss_b)
        assert set(grads_a) == set(grads_b)
        for n in grads_a:
            assert torch.equal(grads_a[n], grads_b[n]), f"grad {n} not bitwise"
