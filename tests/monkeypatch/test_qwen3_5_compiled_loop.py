"""Regression tests: the Qwen3.5 decoder loop must compile under torch.compile (no aten.nonzero graph break in the loop body)."""

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

pytest.importorskip("transformers.models.qwen3_5")
pytest.importorskip("fla")


@pytest.fixture
def packing_patched():
    """Apply the packing patch (torch_compile on) and restore globals after; op registrations can't be undone, so _OPS_BUILT stays True by design (re-registering raises)."""
    import fla.modules.fused_norm_gate  # noqa: F401  (ensure importable)
    from fla.modules import FusedRMSNormGated
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    from axolotl.monkeypatch.models.qwen3_5 import modeling as qm

    saved = {
        "decoder_forward": hf.Qwen3_5DecoderLayer.forward,
        "gdn_forward": hf.Qwen3_5GatedDeltaNet.forward,
        "norm_forward": FusedRMSNormGated.forward,
        "norm_present": hasattr(FusedRMSNormGated, "_axolotl_compile_boundary"),
        "norm_flag": getattr(FusedRMSNormGated, "_axolotl_compile_boundary", None),
        "chunk": getattr(hf, "chunk_gated_delta_rule", None),
        "recurrent": getattr(hf, "fused_recurrent_gated_delta_rule", None),
        "norm_cls": getattr(hf, "FusedRMSNormGated", None),
        "fast_path": getattr(hf, "is_fast_path_available", None),
        "fla_ops_flag": qm._FLA_COMPILED_OPS,
    }
    qm.patch_qwen3_5_modeling_packing(torch_compile=True)
    yield qm
    hf.Qwen3_5DecoderLayer.forward = saved["decoder_forward"]
    hf.Qwen3_5GatedDeltaNet.forward = saved["gdn_forward"]
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
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    cfg._attn_implementation = attn
    return Qwen3_5TextModel(cfg).cuda().to(torch.bfloat16)


def _build_vl_model(seed: int = 0):
    """Full multimodal Qwen3_5ForConditionalGeneration — the only Qwen3.5 checkpoint shape that ships."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    torch.manual_seed(seed)
    text = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    cfg = Qwen3_5Config(
        text_config=text.to_dict(), vision_config=Qwen3_5VisionConfig().to_dict()
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3_5ForConditionalGeneration(cfg).cuda().to(torch.bfloat16)


def _packed_inputs(T: int = 64):
    torch.manual_seed(123)
    input_ids = torch.randint(0, 128, (1, T), device="cuda")
    pos = torch.cat(
        [torch.arange(40, device="cuda"), torch.arange(T - 40, device="cuda")]
    ).view(1, T)
    position_ids = pos[None, ...].expand(4, 1, T).contiguous()
    return input_ids, position_ids


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


class TestDecoderLoopCompiles:
    def test_no_graph_breaks_with_grad_checkpointing(self, packing_patched):
        """Production shape (GC on): the loop must trace with ZERO breaks."""
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
        assert not breaks, f"decoder loop graph-broke: {list(breaks)}"
        assert counters["stats"]["unique_graphs"] >= 1

    def test_fa2_loop_compiles_with_varlen_kwargs(self, packing_patched):
        """FA2: collator-precomputed cu_seq_lens/max_length let the loop compile; without them transformers' per-layer varlen derivation graph-breaks it."""
        from transformers.modeling_flash_attention_utils import (
            prepare_fa_kwargs_from_position_ids,
        )

        torch._dynamo.reset()
        from torch._dynamo.utils import counters

        counters.clear()
        model = _build_model(attn="flash_attention_2")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        input_ids, position_ids = _packed_inputs()
        (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(
            position_ids[0]
        )
        out = torch.compile(model)(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            cu_seq_lens_q=cu_q,
            cu_seq_lens_k=cu_k,
            max_length_q=int(max_q),
            max_length_k=int(max_k),
        ).last_hidden_state
        out.float().pow(2).mean().backward()
        torch.cuda.synchronize()

        breaks = dict(counters["graph_break"])
        assert not breaks, (
            f"FA2 loop graph-broke even with varlen kwargs: {list(breaks)}"
        )
        assert counters["stats"]["unique_graphs"] >= 1

    def test_no_graph_breaks_vl_conditional_generation(self, packing_patched):
        """The VL ForConditionalGeneration must compile its loop with zero breaks on 3-D MRoPE position_ids (the layout VL produces)."""
        torch._dynamo.reset()
        from torch._dynamo.utils import counters

        counters.clear()
        model = _build_vl_model()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        input_ids, position_ids = _packed_inputs()
        out = torch.compile(model)(
            input_ids=input_ids, position_ids=position_ids, use_cache=False
        )
        out.logits.float().pow(2).mean().backward()
        torch.cuda.synchronize()

        breaks = dict(counters["graph_break"])
        assert not breaks, f"VL decoder loop graph-broke: {list(breaks)}"
        assert counters["stats"]["unique_graphs"] >= 1

    def test_no_nonzero_break_without_grad_checkpointing(self, packing_patched):
        """GC off: only the intentional eager self-attn boundary may break; the aten.nonzero break must never return."""
        torch._dynamo.reset()
        from torch._dynamo.utils import counters

        counters.clear()
        model = _build_model()
        model.train()
        _fwd_bwd(model, torch.compile(model))

        breaks = dict(counters["graph_break"])
        offenders = [k for k in breaks if "nonzero" in k.lower()]
        assert not offenders, f"aten.nonzero graph break is back: {offenders}"
        # only the intentional dynamo.disable boundary around self-attention
        for k in breaks:
            assert "disable" in k.lower(), f"unexpected graph break: {k}"

    def test_eager_parity_ops_vs_legacy_bitwise(self, packing_patched):
        """The opaque-op path must be bitwise-identical to the legacy eager path (same kernels, same saved tensors)."""
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

    def test_compiled_matches_eager(self, packing_patched):
        """Compiled (GC on) vs eager: agree within bf16/Inductor reordering noise."""
        torch._dynamo.reset()
        model = _build_model()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        state = {k: v.clone() for k, v in model.state_dict().items()}

        loss_e, grads_e = _fwd_bwd(model)
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        loss_c, grads_c = _fwd_bwd(model, torch.compile(model))

        assert torch.allclose(loss_e, loss_c, rtol=1e-3, atol=1e-3)
        for n in grads_e:
            assert torch.allclose(
                grads_e[n].float(), grads_c[n].float(), rtol=5e-2, atol=1e-3
            ), f"grad {n}: {(grads_e[n].float() - grads_c[n].float()).abs().max()}"

    def test_fa2_compiled_matches_eager_grads(self, packing_patched):
        """FA2 + GC (the production path): compiled grads must match eager — guards the Inductor fusion hazard the non-GC dynamo.disable boundary exists for."""
        from transformers.modeling_flash_attention_utils import (
            prepare_fa_kwargs_from_position_ids,
        )

        torch._dynamo.reset()
        model = _build_model(attn="flash_attention_2")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        input_ids, position_ids = _packed_inputs()
        (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(
            position_ids[0]
        )
        kwargs = dict(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            cu_seq_lens_q=cu_q,
            cu_seq_lens_k=cu_k,
            max_length_q=int(max_q),
            max_length_k=int(max_k),
        )

        def fwd_bwd(fn):
            out = fn(**kwargs).last_hidden_state
            loss = out.float().pow(2).mean()
            loss.backward()
            torch.cuda.synchronize()
            grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            return loss.detach(), grads

        loss_e, grads_e = fwd_bwd(model)
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        loss_c, grads_c = fwd_bwd(torch.compile(model))

        assert torch.allclose(loss_e, loss_c, rtol=1e-3, atol=1e-3)
        assert set(grads_e) == set(grads_c)
        for n in grads_e:
            assert torch.allclose(
                grads_e[n].float(), grads_c[n].float(), rtol=5e-2, atol=1e-3
            ), f"grad {n}: {(grads_e[n].float() - grads_c[n].float()).abs().max()}"

    def test_moe_loop_compiles(self, packing_patched):
        """qwen3_5_moe (claimed scope): expert routing must trace with zero breaks."""
        pytest.importorskip("transformers.models.qwen3_5_moe")
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf_moe
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeTextConfig,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeTextModel,
        )

        from axolotl.monkeypatch.models.qwen3_5.modeling import (
            patch_qwen3_5_moe_modeling_packing,
        )

        saved_moe = {
            "decoder_forward": hf_moe.Qwen3_5MoeDecoderLayer.forward,
            "gdn_forward": hf_moe.Qwen3_5MoeGatedDeltaNet.forward,
        }
        patch_qwen3_5_moe_modeling_packing(torch_compile=True)
        torch._dynamo.reset()
        from torch._dynamo.utils import counters

        counters.clear()
        torch.manual_seed(0)
        cfg = Qwen3_5MoeTextConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=512,
            num_experts=4,
            num_experts_per_tok=2,
            shared_expert_intermediate_size=64,
            decoder_sparse_step=1,
            layer_types=[
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ],
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
        )
        cfg._attn_implementation = "sdpa"
        model = Qwen3_5MoeTextModel(cfg).cuda().to(torch.bfloat16)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        try:
            _fwd_bwd(model, torch.compile(model))

            breaks = dict(counters["graph_break"])
            assert not breaks, f"MoE decoder loop graph-broke: {list(breaks)}"
            assert counters["stats"]["unique_graphs"] >= 1
        finally:
            hf_moe.Qwen3_5MoeDecoderLayer.forward = saved_moe["decoder_forward"]
            hf_moe.Qwen3_5MoeGatedDeltaNet.forward = saved_moe["gdn_forward"]

    def test_batch_gt1_packed_raises(self, packing_patched):
        """B>1 with packed position_ids must fail loudly on the op path (FLA's own inner assert is stripped under -O)."""
        qm = packing_patched
        assert qm._FLA_COMPILED_OPS
        model = _build_model()
        model.train()
        torch.manual_seed(123)
        T = 64
        input_ids = torch.randint(0, 128, (2, T), device="cuda")
        pos = torch.cat(
            [torch.arange(40, device="cuda"), torch.arange(T - 40, device="cuda")]
        )
        position_ids = pos.view(1, T).expand(2, T).contiguous()

        with pytest.raises(ValueError, match="batch size is expected to be 1"):
            model(input_ids=input_ids, position_ids=position_ids, use_cache=False)

    def test_opcheck_custom_ops(self, packing_patched):
        """opcheck the fakes' hardcoded fla 0.4.1 shapes/dtypes (chunk-64 A, f32 g_cum/rstd) so a silent FLA drift turns into a red CI, not a miscompile."""
        torch.manual_seed(1)
        B, T, H, K, V = 1, 64, 4, 16, 16
        q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(B, T, H, device="cuda", dtype=torch.float32).sigmoid().log()
        beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16)
        for t in (q, k, v, g, beta):
            t.requires_grad_(True)
        pos2d = torch.cat(
            [torch.arange(40, device="cuda"), torch.arange(T - 40, device="cuda")]
        ).view(1, T)
        pos3d = pos2d[None].expand(4, 1, T).contiguous()  # MRoPE layout

        for pos in (pos2d, pos3d, None):
            for cast_g in (True, False):  # qwen3_5 casts g, qwen3_next doesn't
                torch.library.opcheck(
                    torch.ops.axolotl_gdn.gdn_chunk,
                    (q, k, v, g, beta, K**-0.5, pos, cast_g),
                )

        x = torch.randn(
            1, T, 96, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w = torch.randn(96, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        b = torch.randn(96, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for pos in (pos2d, None):
            for bias in (b, None):
                torch.library.opcheck(
                    torch.ops.axolotl_gdn.gdn_conv,
                    (x, w, bias, "silu", pos),
                )
