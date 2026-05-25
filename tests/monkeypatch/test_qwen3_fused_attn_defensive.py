"""Defensive regression tests for edge cases of the fused-attn patches."""

import inspect
import logging

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

pytest.importorskip("transformers.models.qwen3")


def _clear_patched_flag(cls):
    try:
        delattr(cls, "_axolotl_fused_attn_patched")
    except AttributeError:
        pass


@pytest.fixture
def restore_qwen3_attention():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    saved = Qwen3Attention.forward
    saved_flag = getattr(Qwen3Attention, "_axolotl_fused_attn_patched", False)
    yield Qwen3Attention
    Qwen3Attention.forward = saved
    if saved_flag:
        Qwen3Attention._axolotl_fused_attn_patched = saved_flag
    else:
        _clear_patched_flag(Qwen3Attention)


class TestFusedAttnKernelUnsupportedWarning:
    """The warning lives in ``PatchManager`` (not the schema validator) so it
    runs after ``normalize_config()`` has derived ``model_config_type``. A
    normal CLI flow with ``fused_attn_kernel: true`` on e.g. a Llama config
    must warn loudly instead of silently no-op'ing."""

    def test_warns_on_unsupported_model_type(self, caplog):
        from types import SimpleNamespace

        from axolotl.loaders.patch_manager import PatchManager

        cfg = SimpleNamespace(fused_attn_kernel=True, model_config_type="llama")
        with caplog.at_level(logging.WARNING, logger="axolotl"):
            PatchManager._warn_if_fused_attn_unsupported(cfg)
        assert any(
            "fused_attn_kernel" in r.message and "llama" in r.message
            for r in caplog.records
        ), f"expected warning about llama; got {[r.message for r in caplog.records]}"

    @pytest.mark.parametrize(
        "model_type",
        [
            "qwen3",
            "qwen3_moe",
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            "gemma4",
            "gemma4_text",
        ],
    )
    def test_no_warn_on_supported_model_type(self, caplog, model_type):
        from types import SimpleNamespace

        from axolotl.loaders.patch_manager import PatchManager

        cfg = SimpleNamespace(fused_attn_kernel=True, model_config_type=model_type)
        with caplog.at_level(logging.WARNING, logger="axolotl"):
            PatchManager._warn_if_fused_attn_unsupported(cfg)
        assert not any(
            "fused_attn_kernel" in r.message and model_type in r.message
            for r in caplog.records
        ), f"unexpected warning for supported {model_type}"

    def test_no_warn_when_fused_attn_kernel_false(self, caplog):
        from types import SimpleNamespace

        from axolotl.loaders.patch_manager import PatchManager

        cfg = SimpleNamespace(fused_attn_kernel=False, model_config_type="llama")
        with caplog.at_level(logging.WARNING, logger="axolotl"):
            PatchManager._warn_if_fused_attn_unsupported(cfg)
        assert not any("fused_attn_kernel" in r.message for r in caplog.records), (
            "no warning expected when fused_attn_kernel is False"
        )

    def test_warning_is_invoked_by_apply_model_specific_patches(self):
        """Source-line check that ``_apply_model_specific_patches`` actually
        calls ``_warn_if_fused_attn_unsupported``. Without this, the standalone
        helper passes its unit tests but never runs in practice."""
        import inspect

        from axolotl.loaders.patch_manager import PatchManager

        src = inspect.getsource(PatchManager._apply_model_specific_patches)
        assert "_warn_if_fused_attn_unsupported" in src, (
            "_apply_model_specific_patches no longer invokes "
            "_warn_if_fused_attn_unsupported — the warning will be dead code"
        )


class TestPeftModulesToSaveWrapper:
    """``modules_to_save=["q_norm","k_norm"]`` wraps the norms in ``ModulesToSaveWrapper``; the patched forward must resolve through it."""

    def _make_wrapper(self, original):
        def _make_clone(m):
            clone = torch.nn.Module()
            clone.weight = torch.nn.Parameter(m.weight.detach().clone())
            clone.variance_epsilon = m.variance_epsilon
            clone.eps = getattr(m, "eps", m.variance_epsilon)
            return clone

        class _StubWrapper(torch.nn.Module):
            """Mirrors PEFT ``ModulesToSaveWrapper``: ``_active_adapter`` is a
            ``list[str]`` and ``active_adapter`` / ``active_adapters`` are
            properties returning that list."""

            def __init__(self, orig):
                super().__init__()
                self.original_module = orig
                self.modules_to_save = torch.nn.ModuleDict(
                    {"default": _make_clone(orig)}
                )
                self._active_adapter = ["default"]

            @property
            def active_adapter(self):
                return self._active_adapter

            @property
            def active_adapters(self):
                return self._active_adapter

            def forward(self, x):
                return self.modules_to_save[self._active_adapter[0]](x)

        return _StubWrapper(original)

    def test_resolve_norm_module_returns_active_adapter_not_original(self):
        """Direct unit-test of ``_resolve_norm_module``: with PEFT's actual
        ``active_adapter = ["default"]`` (list), the helper must return the
        wrapped adapter module, not the frozen ``original_module``. The earlier
        ``isinstance(adapter, str)`` check silently failed this case."""
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        from axolotl.monkeypatch.models.qwen3.fused_attn import (
            _resolve_norm_module,
        )

        orig = Qwen3RMSNorm(16, eps=1e-6)
        wrapper = self._make_wrapper(orig)
        resolved = _resolve_norm_module(wrapper)
        assert resolved is wrapper.modules_to_save["default"], (
            "_resolve_norm_module returned the frozen original instead of the "
            "active adapter — PEFT stores active_adapter as a list, not a str"
        )

    def test_resolve_through_real_peft_modules_to_save(self):
        """End-to-end: build a Qwen3 model, wrap ``q_norm`` / ``k_norm`` with
        ``peft.get_peft_model(..., modules_to_save=[...])``, set the active
        adapter's weight to a value distinct from the frozen original, and
        confirm ``_resolve_norm_module`` returns the active-adapter module
        (so the fused kernel reads the trainable weight, not the frozen one).
        This exercises the real PEFT object shape, not a stub."""
        from peft import LoraConfig, get_peft_model
        from peft.utils.other import ModulesToSaveWrapper
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        from axolotl.monkeypatch.models.qwen3.fused_attn import (
            _resolve_norm_module,
        )

        cfg = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=256,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
        )
        base = Qwen3ForCausalLM(cfg)
        lora = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=["q_norm", "k_norm"],
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(base, lora)

        attn = peft_model.base_model.model.model.layers[0].self_attn
        assert isinstance(attn.q_norm, ModulesToSaveWrapper), (
            "PEFT did not wrap q_norm — test premise is invalid"
        )
        with torch.no_grad():
            attn.q_norm.modules_to_save["default"].weight.fill_(7.0)
            attn.q_norm.original_module.weight.fill_(0.0)

        resolved = _resolve_norm_module(attn.q_norm)
        assert resolved is attn.q_norm.modules_to_save["default"], (
            "_resolve_norm_module did not return the active-adapter module — "
            "real PEFT exposes active_adapter as a list, but the helper "
            "treated only the str case"
        )
        assert torch.equal(
            resolved.weight.detach(),
            torch.full_like(resolved.weight, 7.0),
        ), "resolved module is not the trainable adapter weight"

    def test_qwen3_forward_under_modules_to_save_wrapper(self, restore_qwen3_attention):
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

        from axolotl.monkeypatch.models.qwen3.fused_attn import (
            patch_qwen3_fused_attn,
        )

        cfg = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=256,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
        )
        cfg._attn_implementation = "sdpa"
        m = Qwen3Model(cfg).cuda().to(torch.bfloat16)

        for layer in m.layers:
            attn = layer.self_attn
            attn.q_norm = self._make_wrapper(attn.q_norm).cuda().to(torch.bfloat16)
            attn.k_norm = self._make_wrapper(attn.k_norm).cuda().to(torch.bfloat16)

        patch_qwen3_fused_attn()
        ids = torch.randint(0, 128, (1, 16), device="cuda")
        mask = torch.ones(1, 16, dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=mask).last_hidden_state
        assert torch.isfinite(out).all(), (
            "fused forward through ModulesToSaveWrapper produced non-finite output"
        )


class TestKernelProductionHeadDim:
    """Kernel parity at ``head_dim=256`` (Qwen3.5 production); unit tests only cover 32/64."""

    @pytest.mark.parametrize("head_dim", [128, 256])
    @pytest.mark.parametrize("unit_offset", [False, True])
    def test_fused_rms_norm_rope_parity(self, head_dim, unit_offset):
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        B, S, H, D = 2, 32, 4, head_dim
        torch.manual_seed(11)
        x = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        cos = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        eps = 1e-6

        x32 = x.to(torch.float32)
        rms = x32.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        scale = (w.to(torch.float32) + 1.0) if unit_offset else w.to(torch.float32)
        x_norm = x32 * rms * scale
        half = D // 2
        rot = torch.cat([-x_norm[..., half:], x_norm[..., :half]], dim=-1)
        ref = (
            x_norm * cos.to(torch.float32).unsqueeze(2)
            + rot * sin.to(torch.float32).unsqueeze(2)
        ).to(torch.bfloat16)

        got = fused_rms_norm_rope(x, w, cos, sin, eps=eps, unit_offset=unit_offset)
        assert got.shape == ref.shape
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, ref, rtol=5e-2, atol=5e-2)


class TestAttentionMaskPassThrough:
    """Sample-packing masks must flow through the patched forward verbatim."""

    def test_qwen3_padding_mask_runs_clean(self, restore_qwen3_attention):
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

        from axolotl.monkeypatch.models.qwen3.fused_attn import (
            patch_qwen3_fused_attn,
        )

        cfg = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=256,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
        )
        cfg._attn_implementation = "sdpa"
        m = Qwen3Model(cfg).cuda().to(torch.bfloat16)
        patch_qwen3_fused_attn()

        ids = torch.randint(0, 128, (2, 16), device="cuda")
        mask = torch.cat([torch.ones(2, 8), torch.zeros(2, 8)], dim=1).long().cuda()
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=mask).last_hidden_state
        assert torch.isfinite(out).all()


class TestSlidingWindowKwarg:
    """The fused forward must preserve ``sliding_window`` on the attention-interface call."""

    def test_fused_forward_passes_sliding_window(self):
        from axolotl.monkeypatch.models.qwen3 import fused_attn

        src = inspect.getsource(fused_attn._make_fused_forward)
        assert "sliding_window=self.sliding_window" in src, (
            "Qwen3 fused_forward must pass sliding_window to attention_interface "
            "to preserve sliding-attention layer behavior"
        )


class TestGetTextConfigDispatch:
    """A multimodal Qwen3-VL text branch surfaces as ``model_config_type='qwen3'``; the patch must still fire."""

    def test_qwen3_text_branch_dispatch(self, restore_qwen3_attention):
        from axolotl.loaders.patch_manager import PatchManager
        from axolotl.utils.dict import DictDefault

        cfg = DictDefault(
            {
                "base_model": "fake/qwen3-vl-text-branch",
                "model_config_type": "qwen3",
                "fused_attn_kernel": True,
                "lora_qkv_kernel": False,
                "lora_o_kernel": False,
                "context_parallel_size": 1,
            }
        )
        mc = type("MC", (), {"model_type": "qwen3"})()
        pm = PatchManager(cfg=cfg, model_config=mc, inference=False)
        pm._apply_model_specific_patches()

        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        assert getattr(Qwen3Attention, "_axolotl_fused_attn_patched", False)
