"""Tests for NVFP4-GEMM training (real FP4 forward + backward).

Gated on Blackwell (sm >= 10.0) since the FP4 tensor-core GEMM has no CPU/
non-Blackwell path.
"""

import pytest
import torch
from torch import nn

from tests.e2e.utils import require_torch_2_8_0, requires_sm_ge_100


@require_torch_2_8_0
@requires_sm_ge_100
class TestNVFP4Training:
    def test_supported(self):
        from axolotl.integrations.nvfp4.nvfp4_training import nvfp4_supported

        ok, reason = nvfp4_supported()
        assert ok, reason

    def test_linear_forward_backward_finite(self):
        """Module runs with a non-32-aligned token count (padding path)."""
        from axolotl.integrations.nvfp4.nvfp4_training import NVFP4Linear, NVFP4Recipe

        torch.manual_seed(0)
        linear = nn.Linear(512, 256, bias=True).cuda().bfloat16()
        mod = NVFP4Linear.from_linear(linear, NVFP4Recipe())
        x = torch.randn(
            37, 7, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )  # 259 tokens, not a multiple of 32
        out = mod(x)
        assert out.shape == (37, 7, 256)
        out.sum().backward()
        for t in (out, x.grad, mod.weight.grad, mod.bias.grad):
            assert torch.isfinite(t).all().item()
        assert x.grad.shape == x.shape

    def test_learns(self):
        """Loss decreases when fitting a teacher weight through FP4 GEMMs."""
        from axolotl.integrations.nvfp4.nvfp4_training import NVFP4Linear, NVFP4Recipe

        torch.manual_seed(1)
        w_star = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) * 0.05
        linear = nn.Linear(512, 256, bias=False).cuda().bfloat16()
        with torch.no_grad():
            linear.weight.mul_(0.02)
        mod = NVFP4Linear.from_linear(linear, NVFP4Recipe())
        opt = torch.optim.SGD(mod.parameters(), lr=0.5)
        x = torch.randn(259, 512, device="cuda", dtype=torch.bfloat16)
        tgt = (x @ w_star.t()).detach()
        losses = []
        for _ in range(80):
            opt.zero_grad()
            loss = (mod(x) - tgt).float().pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        assert losses[-1] < 0.5 * losses[0]
        assert all(torch.isfinite(torch.tensor(losses)))

    def test_gemm_wiring_matches_dequant_matmul(self):
        """The FP4 GEMM must equal dequant(a) @ dequant(b)."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            QuantPolicy,
            _fp4_mm,
            _quantize,
        )

        torch.manual_seed(0)
        a = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16)
        got = _fp4_mm(a, b, QuantPolicy(), QuantPolicy())
        a_q = _quantize(a, QuantPolicy())
        b_q = _quantize(b.t().contiguous(), QuantPolicy()).t()
        ref = a_q.dequantize(torch.bfloat16) @ b_q.dequantize(torch.bfloat16)
        rel = ((got - ref).norm() / ref.norm()).item()
        assert rel < 2e-2, f"GEMM wiring rel-err {rel}"

    @pytest.mark.parametrize("stochastic", [False, True])
    @pytest.mark.parametrize("hadamard", [False, True])
    def test_cached_weight_is_bit_exact(self, stochastic, hadamard):
        """The per-step weight-quant cache must be bit-identical to re-quantizing.

        Weight quant is deterministic RTN, so the cached b-operands reproduce the
        old per-call quant exactly — out, grad_x, and grad_w must match with zero
        difference. A fixed seed around fwd/bwd aligns any SR draws (which are on
        the activation/gradient operands, not the cached weight)."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Linear,
            NVFP4Recipe,
            QuantPolicy,
            _fp4_mm,
            _pad_to_block,
        )

        class OldFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight, bias, recipe):
                orig_shape = x.shape
                x2d = x.reshape(-1, orig_shape[-1])
                x2d_p, m = _pad_to_block(x2d, 0)
                out = _fp4_mm(x2d_p, weight.t(), QuantPolicy(), QuantPolicy())[:m]
                if bias is not None:
                    out = out + bias
                ctx.save_for_backward(x2d, weight)
                ctx.recipe = recipe
                ctx.has_bias = bias is not None
                ctx.x_shape = orig_shape
                return out.reshape(*orig_shape[:-1], weight.shape[0])

            @staticmethod
            def backward(ctx, grad_out):
                x2d, weight = ctx.saved_tensors
                recipe = ctx.recipe
                g = grad_out.reshape(-1, weight.shape[0])
                grad_x = grad_w = grad_bias = None
                g_pol = QuantPolicy(stochastic=recipe.stochastic_rounding)
                rht_pol = QuantPolicy(
                    stochastic=recipe.stochastic_rounding, hadamard=recipe.hadamard
                )
                if ctx.needs_input_grad[0]:
                    g_p, m = _pad_to_block(g, 0)
                    grad_x = _fp4_mm(g_p, weight, g_pol, QuantPolicy())[:m]
                    grad_x = grad_x.reshape(ctx.x_shape)
                if ctx.needs_input_grad[1]:
                    gt, _ = _pad_to_block(g.t().contiguous(), 1)
                    xp, _ = _pad_to_block(x2d, 0)
                    grad_w = _fp4_mm(gt, xp, rht_pol, rht_pol)
                if ctx.has_bias and ctx.needs_input_grad[2]:
                    grad_bias = g.sum(dim=0)
                return grad_x, grad_w, grad_bias, None

        recipe = NVFP4Recipe(stochastic_rounding=stochastic, hadamard=hadamard)
        torch.manual_seed(0)
        N, K, M = 1024, 1024, 512
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        x_data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        gout = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        new_mod = NVFP4Linear(nn.Parameter(w.clone()), nn.Parameter(b.clone()), recipe)
        xn = x_data.clone().requires_grad_(True)
        torch.manual_seed(1234)
        out_new = new_mod(xn)
        torch.manual_seed(1234)
        out_new.backward(gout.clone())

        w_old, b_old = nn.Parameter(w.clone()), nn.Parameter(b.clone())
        xo = x_data.clone().requires_grad_(True)
        torch.manual_seed(1234)
        out_old = OldFn.apply(xo, w_old, b_old, recipe)
        torch.manual_seed(1234)
        out_old.backward(gout.clone())

        assert torch.equal(out_new, out_old)
        assert torch.equal(xn.grad, xo.grad)
        assert torch.equal(new_mod.weight.grad, w_old.grad)
        assert torch.equal(new_mod.bias.grad, b_old.grad)

    def test_weight_quant_cached_per_version(self):
        """For a FROZEN weight, the b-operands are quantized twice (fprop+dgrad
        layout) and then cached: across N passes with no update the count stays at
        2, and an in-place weight update (bumping ``_version``) re-quantizes once.

        The cache is keyed on ``weight._version`` and only safely helps frozen
        weights — a TRAINABLE master is covered by
        ``test_trainable_weight_not_stale_under_fused_optimizer`` (it must rebuild
        every forward because fused optimizers don't bump ``_version``)."""
        import axolotl.integrations.nvfp4.nvfp4_training as nvmod
        from axolotl.integrations.nvfp4.nvfp4_training import NVFP4Linear, NVFP4Recipe

        recipe = NVFP4Recipe(stochastic_rounding=False, hadamard=False)
        torch.manual_seed(0)
        w = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        mod = NVFP4Linear(nn.Parameter(w.clone()), None, recipe)
        mod.weight.requires_grad_(False)  # frozen: the version-keyed cache applies

        calls = {"n": 0}
        real = nvmod._quantize

        def counting(t, policy):
            calls["n"] += 1
            return real(t, policy)

        # Per fwd+bwd the non-weight quant calls for a FROZEN weight (no wgrad):
        # fprop activation (1) + dgrad gradient (1) = 2.
        non_weight = 2
        n_passes = 4
        nvmod._quantize = counting
        try:
            base = calls["n"]
            for _ in range(n_passes):
                x = torch.randn(
                    128, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True
                )
                mod(x).sum().backward()
            weight_quant = (calls["n"] - base) - non_weight * n_passes
            assert weight_quant == 2, weight_quant  # cached across all passes

            before = calls["n"]
            with torch.no_grad():
                mod.weight.add_(0.01)  # _version bump
            x = torch.randn(
                128, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            mod(x).sum().backward()
            weight_quant_after = (calls["n"] - before) - non_weight
            assert weight_quant_after == 2, weight_quant_after  # re-quant once
        finally:
            nvmod._quantize = real

        # Fresh forward after an update must use the NEW weight, not a stale cache.
        torch.manual_seed(0)
        w2 = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        m2 = NVFP4Linear(nn.Parameter(w2.clone()), None, recipe)
        m2.weight.requires_grad_(False)  # exercise the version-keyed cache path
        xt = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
        out_a = m2(xt).detach().clone()
        with torch.no_grad():
            m2.weight.mul_(2.0)
        out_b = m2(xt).detach().clone()
        ref = NVFP4Linear(nn.Parameter((w2 * 2.0).clone()), None, recipe)
        assert not torch.equal(out_a, out_b)
        assert torch.equal(out_b, ref(xt))

    def test_trainable_weight_not_stale_under_fused_optimizer(self):
        """Full-FT NVFP4Linear must serve FRESH weights every step under a fused
        optimizer.

        Regression: ``torch._fused_adamw_`` mutates params WITHOUT bumping
        ``param._version`` (torch 2.12; foreach/single-tensor DO bump). Keying
        the weight-quant cache on ``_version`` froze it at the step-0 quant, so a
        TRAINABLE master under ``adamw_torch_fused`` served stale forward weights
        while wgrad flowed to the live weight -> fprop/bwd mismatch -> divergence.
        A single step passes even when broken (the cache only goes stale on the
        SECOND fused step, since the first step's mutation is what fails to bump),
        so this must run N>1 steps. The fix rebuilds the cache whenever the weight
        is trainable; this asserts the module forward stays bitwise-identical to a
        fresh-built NVFP4Linear over the live weight after each fused step.
        """
        from axolotl.integrations.nvfp4.nvfp4_training import NVFP4Linear, NVFP4Recipe

        recipe = NVFP4Recipe(stochastic_rounding=False, hadamard=False)
        torch.manual_seed(0)
        w = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16) * 0.05
        mod = NVFP4Linear(nn.Parameter(w.clone()), None, recipe)
        assert mod.weight.requires_grad  # full-FT: trainable master
        opt = torch.optim.AdamW(mod.parameters(), lr=1e-2, fused=True)

        # fused mutation must NOT bump _version (otherwise this test can't catch
        # the bug); guards against a torch version that starts bumping.
        x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
        mod(x).sum().backward()
        v_before = mod.weight._version
        opt.step()
        assert mod.weight._version == v_before, (
            "fused AdamW bumped _version; this torch no longer exposes the bug"
        )
        opt.zero_grad()

        for step in range(4):
            xx = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
            out = mod(xx).detach().clone()
            # fresh module built from the CURRENT live weight = ground truth
            fresh = NVFP4Linear(nn.Parameter(mod.weight.detach().clone()), None, recipe)
            ref = fresh(xx).detach().clone()
            assert torch.equal(out, ref), (
                f"stale forward at step {step}: cache not rebuilt for trainable "
                "weight under fused optimizer"
            )
            mod(xx).sum().backward()
            opt.step()
            opt.zero_grad()

    def test_frozen_weight_cache_preserved(self):
        """A FROZEN weight (LoRA base / frozen FFT layer) must keep caching across
        passes: the rebuild-if-trainable fix must NOT regress the frozen fast
        path. The cached b-operand objects must be identical across passes (no
        re-quant), and they must match a one-shot quant of the frozen weight."""
        from axolotl.integrations.nvfp4.nvfp4_training import NVFP4Linear, NVFP4Recipe

        recipe = NVFP4Recipe(stochastic_rounding=False, hadamard=False)
        torch.manual_seed(0)
        w = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        mod = NVFP4Linear(nn.Parameter(w.clone()), None, recipe)
        mod.weight.requires_grad_(False)

        wf0, wd0 = mod._quantized_weights()
        for _ in range(4):
            x = torch.randn(128, 1024, device="cuda", dtype=torch.bfloat16)
            mod(x).sum()
            wf, wd = mod._quantized_weights()
            # identity (not just equality): the cache was reused, not rebuilt
            assert wf is wf0 and wd is wd0

    def test_convert_excludes_sensitive_and_odd_dims(self):
        """Swap eligible linears; keep lm_head/embeddings high-precision."""
        from transformers import AutoModelForCausalLM

        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Linear,
            convert_to_nvfp4_training,
        )

        model = AutoModelForCausalLM.from_pretrained(
            "axolotl-ai-co/tiny-qwen2-129m", device_map="auto", dtype=torch.bfloat16
        )
        n = convert_to_nvfp4_training(model)
        assert n > 0
        assert not isinstance(model.lm_head, NVFP4Linear)
        ids = torch.randint(0, 1000, (2, 48), device=model.device)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        assert torch.isfinite(out.loss).item()

    def test_swap_frozen_lm_head_compute_and_storage(self):
        """A bare frozen lm_head swaps to the matching NVFP4 base and stays finite."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4ComputeBaseLinear,
            NVFP4FrozenBaseLinear,
            NVFP4Recipe,
            swap_frozen_linear_to_nvfp4,
        )

        torch.manual_seed(0)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(2048, 4096, bias=False)

        for mode, _cls in (
            ("compute", NVFP4ComputeBaseLinear),
            ("storage", NVFP4FrozenBaseLinear),
        ):
            m = M().cuda().bfloat16()
            ok = swap_frozen_linear_to_nvfp4(
                m, "lm_head", NVFP4Recipe(), base_mode=mode
            )
            assert ok
            # mslk-fast variants are subclasses; both modes must not stay nn.Linear
            assert not isinstance(m.lm_head, nn.Linear)
            x = torch.randn(2, 16, 2048, device="cuda", dtype=torch.bfloat16)
            out = m.lm_head(x)
            assert out.shape == (2, 16, 4096)
            assert torch.isfinite(out).all().item()

    def test_lm_head_forward_dynamo_disabled(self):
        """The swapped FP4 lm_head runs eager under torch.compile.

        Regression: gc-off + quantize_lm_head + torch.compile + flash_attention_2
        fused the flash-attn backward with the FP4 lm_head dgrad into a NaN-
        producing graph (loss collapses to ln(vocab) at step ~4). Breaking the
        graph around the lm_head forward avoids that fused region. Assert the
        marker so the disable can't silently regress.
        """
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Recipe,
            swap_frozen_linear_to_nvfp4,
            swap_tied_embedding_and_lm_head_to_nvfp4,
        )

        for mode in ("compute", "storage", "hp"):

            class Untied(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lm_head = nn.Linear(2048, 4096, bias=False)
                    self.lm_head.weight.requires_grad_(False)

            m = Untied().cuda().bfloat16()
            assert swap_frozen_linear_to_nvfp4(
                m, "lm_head", NVFP4Recipe(), base_mode=mode
            )
            assert getattr(m.lm_head.forward, "_torchdynamo_disable", False), mode
            # the eager wrapper must still compute the head
            out = m.lm_head(
                torch.randn(2, 16, 2048, device="cuda", dtype=torch.bfloat16)
            )
            assert out.shape == (2, 16, 4096) and torch.isfinite(out).all().item()

        class Tied(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(2048, 512)
                self.lm_head = nn.Linear(512, 2048, bias=False)
                self.lm_head.weight = self.embed.weight
                for p in self.parameters():
                    p.requires_grad_(False)

        m = Tied().cuda().bfloat16()
        assert swap_tied_embedding_and_lm_head_to_nvfp4(
            m, "embed", "lm_head", NVFP4Recipe()
        )
        assert getattr(m.lm_head.forward, "_torchdynamo_disable", False)

    def test_swap_frozen_lm_head_skips_odd_dims(self):
        """A non-%32 output dim is left in high precision (no crash)."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Recipe,
            swap_frozen_linear_to_nvfp4,
        )

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(2048, 4095, bias=False)  # 4095 % 32 != 0

        m = M().cuda().bfloat16()
        assert not swap_frozen_linear_to_nvfp4(m, "lm_head", NVFP4Recipe())
        assert isinstance(m.lm_head, nn.Linear)

    def test_quantize_lm_head_guard_tied_embeddings(self):
        """Tied + FROZEN now succeeds (quantize-once); tied + TRAINABLE still
        raises; untied passes the lm_head-quantizable guard."""
        from axolotl.integrations.nvfp4 import swap
        from axolotl.integrations.nvfp4.args import NVFP4Args
        from axolotl.utils.dict import DictDefault

        class Cfg:
            tie_word_embeddings = False

        class TinyModel(nn.Module):
            def __init__(self, tie, frozen=True):
                super().__init__()
                self.config = Cfg()
                self.config.tie_word_embeddings = tie
                self.embed = nn.Embedding(64, 32)
                self.lm_head = nn.Linear(32, 64, bias=False)
                if tie:
                    self.lm_head.weight = self.embed.weight
                if frozen:
                    for p in self.parameters():
                        p.requires_grad_(False)

            def get_output_embeddings(self):
                return self.lm_head

            def get_input_embeddings(self):
                return self.embed

        no_cce = DictDefault({"cut_cross_entropy": False})

        assert swap._model_ties_embeddings(TinyModel(tie=True))
        assert not swap._model_ties_embeddings(TinyModel(tie=False))

        nvfp4 = NVFP4Args(enabled=True, quantize=["lm_head"])

        # tied + frozen: no longer raises — the shared weight is quantized once.
        swap._check_lm_head_quantizable(no_cce, nvfp4, TinyModel(tie=True, frozen=True))

        # tied + trainable shared weight: still raises (FP4-storing it corrupts
        # training).
        with pytest.raises(RuntimeError, match="FROZEN"):
            swap._check_lm_head_quantizable(
                no_cce, nvfp4, TinyModel(tie=True, frozen=False)
            )

        # untied: lm_head is independently quantizable — no raise.
        swap._check_lm_head_quantizable(no_cce, nvfp4, TinyModel(tie=False))

        # untied + cut_cross_entropy: raises (CCE reads the head weight directly).
        cce = DictDefault({"cut_cross_entropy": True})
        with pytest.raises(RuntimeError, match="cut_cross_entropy"):
            swap._check_lm_head_quantizable(cce, nvfp4, TinyModel(tie=False))

        # untied + cut_cross_entropy + FP4-aware fused CE (cross_entropy.mode=fp4):
        # the guard is relaxed because the fused CE reads the packed lm_head directly.
        nvfp4_fused = NVFP4Args(
            enabled=True,
            quantize=["lm_head"],
            cross_entropy={"mode": "fp4"},
        )
        swap._check_lm_head_quantizable(cce, nvfp4_fused, TinyModel(tie=False))

    def test_fused_fp4_cross_entropy_matches_materialized(self):
        """Fused FP4 lm_head + CE loss & dL/dhidden match the same-weight
        materialized reference; the [M, V] logits are never built."""
        from axolotl.integrations.nvfp4.kernels.nvfp4_fused_ce import (
            _nvfp4_lm_head_store,
            fused_fp4_cross_entropy,
        )
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4ComputeBaseLinear,
            NVFP4FrozenBaseLinear,
            NVFP4Recipe,
        )

        torch.manual_seed(0)
        M, H, V = 192, 256, 4096 + 512  # crosses a vocab-tile boundary
        lin = nn.Linear(H, V, bias=False).cuda().bfloat16()

        for cls in (NVFP4FrozenBaseLinear, NVFP4ComputeBaseLinear):
            mod = cls.from_linear(lin, NVFP4Recipe())
            store = _nvfp4_lm_head_store(mod)
            assert store is not None  # row-sliceable (non-swizzled torchao store)
            for num_items in (None, 137.0):
                hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
                labels = torch.randint(0, V, (M,), device="cuda")
                labels[::7] = -100  # mask some tokens

                # reference: dequant the SAME store once, bf16 logits, standard CE
                weight = store.dequantize(torch.bfloat16)
                h_ref = hidden.clone().requires_grad_(True)
                logits = (h_ref @ weight.t()).float()
                reduction = "sum" if num_items is not None else "mean"
                ref = torch.nn.functional.cross_entropy(
                    logits, labels, ignore_index=-100, reduction=reduction
                )
                if num_items is not None:
                    ref = ref / num_items
                ref.backward()

                # fused (shift=False to align with the un-shifted reference)
                h_fused = hidden.clone().requires_grad_(True)
                fused = fused_fp4_cross_entropy(
                    h_fused, mod, labels, num_items_in_batch=num_items, shift=False
                )
                fused.backward()

                loss_rel = (fused - ref).abs() / (ref.abs() + 1e-9)
                grad_rel = (h_fused.grad - h_ref.grad).float().norm() / (
                    h_ref.grad.float().norm() + 1e-9
                )
                assert loss_rel < 1e-3, (cls.__name__, num_items, loss_rel.item())
                assert grad_rel < 2e-2, (cls.__name__, num_items, grad_rel.item())

    def test_fused_fp4_cross_entropy_skips_materialization(self):
        """The fused path's peak memory stays near a single tile, far below the
        full [M, V] logit tensor of the materialized path."""
        from axolotl.integrations.nvfp4.kernels.nvfp4_fused_ce import (
            _nvfp4_lm_head_store,
            fused_fp4_cross_entropy,
        )
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4FrozenBaseLinear,
            NVFP4Recipe,
        )

        torch.manual_seed(0)
        M, H, V = 4096, 2048, 128256
        lin = nn.Linear(H, V, bias=False).cuda().bfloat16()
        mod = NVFP4FrozenBaseLinear.from_linear(lin, NVFP4Recipe())
        store = _nvfp4_lm_head_store(mod)
        del lin
        torch.cuda.empty_cache()
        labels = torch.randint(0, V, (M,), device="cuda")

        def peak(fn):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated()

        def materialized():
            h = torch.randn(
                M, H, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            weight = store.dequantize(torch.bfloat16)
            logits = (h @ weight.t()).float()
            torch.nn.functional.cross_entropy(
                logits.view(-1, V), labels, ignore_index=-100
            ).backward()

        def fused():
            h = torch.randn(
                M, H, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            fused_fp4_cross_entropy(h, mod, labels, shift=False).backward()

        mat_peak = peak(materialized)
        fused_peak = peak(fused)
        # The [M, V] fp32 logits alone are M*V*4 bytes; the fused path must stay
        # well under that (it only ever holds one [M, V_BLOCK] tile).
        assert fused_peak < 0.25 * mat_peak, (fused_peak, mat_peak)

    def test_embedding_quant_forward_and_memory(self):
        """NVFP4Embedding lookup matches bf16 F.embedding within FP4 tolerance and
        the weight memory drops ~3.5x."""
        import torch.nn.functional as F

        from axolotl.integrations.nvfp4.nvfp4_training import NVFP4Embedding

        torch.manual_seed(0)
        emb = nn.Embedding(2048, 512).cuda().bfloat16()
        emb.weight.requires_grad_(False)
        ne = NVFP4Embedding.from_embedding(emb)
        idx = torch.randint(0, 2048, (4, 32), device="cuda")
        ref = F.embedding(idx, emb.weight)
        got = ne(idx)
        cos = F.cosine_similarity(
            ref.flatten().float(), got.flatten().float(), dim=0
        ).item()
        assert cos > 0.99, cos

        bf16_bytes = emb.weight.numel() * 2
        fp4_bytes = (
            ne.w_q.qdata.numel()
            + ne.w_q.scale.numel()
            + ne.w_q.per_tensor_scale.numel() * 4
        )
        assert bf16_bytes / fp4_bytes > 3.0, bf16_bytes / fp4_bytes

    def test_embedding_quant_skips_trainable(self):
        """A trainable embedding is left in high precision (no FP4 master)."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Embedding,
            swap_frozen_embedding_to_nvfp4,
        )

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(256, 64)

        m = M().cuda().bfloat16()  # weight trainable by default
        assert swap_frozen_embedding_to_nvfp4(m, "embed") is None
        assert isinstance(m.embed, nn.Embedding)
        assert not isinstance(m.embed, NVFP4Embedding)

    def test_tied_quantize_once_shared_store(self):
        """Tied frozen weight: the embedding lookup and the lm_head GEMM read the
        SAME dequantized FP4 store (quantize-once)."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Embedding,
            NVFP4Recipe,
            NVFP4TiedLMHead,
            swap_tied_embedding_and_lm_head_to_nvfp4,
        )

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(2048, 512)
                self.lm_head = nn.Linear(512, 2048, bias=False)
                self.lm_head.weight = self.embed.weight
                for p in self.parameters():
                    p.requires_grad_(False)

            def get_input_embeddings(self):
                return self.embed

            def get_output_embeddings(self):
                return self.lm_head

        m = M().cuda().bfloat16()
        ok = swap_tied_embedding_and_lm_head_to_nvfp4(
            m, "embed", "lm_head", NVFP4Recipe()
        )
        assert ok
        assert isinstance(m.embed, NVFP4Embedding)
        assert isinstance(m.lm_head, NVFP4TiedLMHead)
        # one store, two roles
        assert m.lm_head.w_q is m.embed.w_q
        assert torch.equal(m.embed.weight, m.lm_head.weight)

        x = torch.randn(2, 16, 512, device="cuda", dtype=torch.bfloat16)
        logits = m.lm_head(x)
        ref = x @ m.embed.weight.t()  # same dequantized weight as the lookup
        rel = ((logits - ref).norm() / ref.norm()).item()
        assert rel < 0.2 and torch.isfinite(logits).all().item(), rel

    def test_vision_tower_targets_only_vision_linears(self):
        """The vision-tower swap touches ONLY eligible frozen linears under the
        vision module — merger/patch-embed/non-%32 dims and the language head are
        left untouched."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Recipe,
            _find_vision_tower,
            convert_vision_tower_to_nvfp4,
            is_nvfp4_base,
        )

        class VisionBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Module()
                self.attn.qkv = nn.Linear(1152, 3456, bias=True)
                self.attn.proj = nn.Linear(1152, 1152)
                self.mlp = nn.Module()
                self.mlp.fc1 = nn.Linear(1152, 4304)  # 4304 not %32 -> skip
                self.mlp.fc2 = nn.Linear(4304, 1152)

        class VisionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Module()
                self.patch_embed.proj = nn.Conv2d(3, 1152, 16, 16)
                self.blocks = nn.ModuleList([VisionBlock() for _ in range(2)])
                self.merger = nn.Module()
                self.merger.linear_fc1 = nn.Linear(1152, 2048)

        class FakeVL(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.visual = VisionModel()
                self.lm_head = nn.Linear(2048, 1024)

        m = FakeVL().cuda().bfloat16()
        for p in m.parameters():
            p.requires_grad_(False)

        name, _vt = _find_vision_tower(m)
        assert name == "model.visual"
        n = convert_vision_tower_to_nvfp4(m, NVFP4Recipe(), base_mode="storage")
        # qkv + proj per block (fc1/fc2 skipped for %32), 2 blocks => 4
        assert n == 4, n
        assert is_nvfp4_base(m.model.visual.blocks[0].attn.qkv)
        assert is_nvfp4_base(m.model.visual.blocks[0].attn.proj)
        assert isinstance(m.model.visual.blocks[0].mlp.fc1, nn.Linear)
        assert isinstance(m.model.visual.merger.linear_fc1, nn.Linear)
        assert isinstance(m.lm_head, nn.Linear)

    def test_vision_tower_warns_on_text_model(self):
        """No vision tower => no swap, count 0 (text-only model)."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Recipe,
            convert_vision_tower_to_nvfp4,
        )

        class TextModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Linear(64, 64)])
                self.lm_head = nn.Linear(64, 64)

        m = TextModel().cuda().bfloat16()
        assert convert_vision_tower_to_nvfp4(m, NVFP4Recipe()) == 0
        assert isinstance(m.lm_head, nn.Linear)

    def test_compile_no_graph_breaks(self):
        import torch._dynamo as dyn

        from axolotl.integrations.nvfp4.nvfp4_training import NVFP4Linear, NVFP4Recipe

        torch.manual_seed(0)
        linear = nn.Linear(512, 256).cuda().bfloat16()
        # recipe on: SR + RHT live at the quant boundary, must not break the graph
        mod = NVFP4Linear.from_linear(
            linear, NVFP4Recipe(stochastic_rounding=True, hadamard=True)
        )
        x = torch.randn(
            128, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        dyn.reset()
        explanation = dyn.explain(lambda z: mod(z).sum())(x)
        assert explanation.graph_break_count == 0
        # fullgraph fwd+bwd must trace cleanly (RHT's Hadamard build must not
        # reach a torch.Generator inside the backward graph)
        compiled = torch.compile(lambda z: mod(z).sum(), fullgraph=True)
        compiled(x).backward()
        assert torch.isfinite(mod.weight.grad).all().item()

    def test_mslk_compute_compile_opaque(self):
        """The MSLK fused-quant compute base compiles fullgraph: its Triton quant
        kernels stay opaque custom ops (no decompose crash, no graph break, no
        eager fallback). suppress_errors=False so a fallback surfaces as a failure.
        """
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4FastComputeBaseLinear,
            NVFP4Recipe,
            _mslk_available,
        )

        if not _mslk_available():
            pytest.skip("MSLK not available")

        import torch._dynamo as dyn

        prev = dyn.config.suppress_errors
        dyn.config.suppress_errors = False
        try:
            torch.manual_seed(0)
            linear = nn.Linear(256, 512, bias=False).cuda().bfloat16()
            mod = NVFP4FastComputeBaseLinear.from_linear(
                linear, NVFP4Recipe(stochastic_rounding=False, hadamard=False)
            )
            x = torch.randn(
                64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            dyn.reset()
            explanation = dyn.explain(lambda z: mod(z).sum())(x)
            assert explanation.graph_break_count == 0
            compiled = torch.compile(lambda z: mod(z).sum(), fullgraph=True)
            compiled(x).backward()
            assert torch.isfinite(x.grad).all().item()
        finally:
            dyn.config.suppress_errors = prev

    def test_shared_base_fprop_many_matches_individual(self):
        """Shared activation packing must match calling each NVFP4 base fprop."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4ComputeBaseLinear,
            NVFP4Recipe,
            nvfp4_base_fprop,
            nvfp4_base_fprop_many,
        )

        torch.manual_seed(0)
        recipe = NVFP4Recipe(stochastic_rounding=False, hadamard=False)
        bases = [
            NVFP4ComputeBaseLinear.from_linear(
                nn.Linear(256, out, bias=False).cuda().bfloat16(), recipe
            )
            for out in (512, 384, 640)
        ]
        x = torch.randn(2, 19, 256, device="cuda", dtype=torch.bfloat16)
        shared = nvfp4_base_fprop_many(x, bases)
        assert shared is not None
        individual = [nvfp4_base_fprop(x.reshape(-1, 256), base) for base in bases]
        for got, ref in zip(shared, individual, strict=False):
            assert got.shape == (2, 19, ref.shape[-1])
            assert torch.equal(got.reshape(-1, got.shape[-1]), ref)

    def test_shared_fast_base_fprop_many_matches_individual(self):
        """MSLK fast compute-base fprops can share one single-level activation pack."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4FastComputeBaseLinear,
            NVFP4Recipe,
            _mslk_available,
            nvfp4_base_fprop,
            nvfp4_base_fprop_many,
        )

        if not _mslk_available():
            pytest.skip("MSLK not available")

        torch.manual_seed(0)
        recipe = NVFP4Recipe(stochastic_rounding=False, hadamard=False)
        bases = [
            NVFP4FastComputeBaseLinear.from_linear(
                nn.Linear(256, out, bias=False).cuda().bfloat16(), recipe
            )
            for out in (512, 384, 640)
        ]
        x = torch.randn(2, 19, 256, device="cuda", dtype=torch.bfloat16)
        shared = nvfp4_base_fprop_many(x, bases)
        assert shared is not None
        individual = [nvfp4_base_fprop(x.reshape(-1, 256), base) for base in bases]
        for got, ref in zip(shared, individual, strict=False):
            assert got.shape == (2, 19, ref.shape[-1])
            assert torch.equal(got.reshape(-1, got.shape[-1]), ref)

    def test_sr_unbiased(self):
        """Averaging stochastic-rounded quant converges to the true value; RTN
        keeps a systematic bias."""
        from axolotl.integrations.nvfp4.nvfp4_training import QuantPolicy, _quantize

        torch.manual_seed(0)
        t = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16) * 0.01
        true = t.float()
        rtn = _quantize(t, QuantPolicy()).dequantize(torch.float32)
        acc = torch.zeros_like(true)
        n = 3000
        for _ in range(n):
            acc += _quantize(t, QuantPolicy(stochastic=True)).dequantize(torch.float32)
        sr_mean = acc / n
        rtn_bias = (rtn - true).abs().mean().item()
        sr_bias = (sr_mean - true).abs().mean().item()
        # SR is unbiased: its averaged error collapses far below RTN's fixed bias
        assert sr_bias < 0.25 * rtn_bias, (rtn_bias, sr_bias)

    def test_rht_cancels(self):
        """With FP4 quant bypassed, the Hadamard rotation applied to both wgrad
        operands cancels in the product (orthonormal H^T H = I)."""
        from axolotl.integrations.nvfp4.nvfp4_training import _apply_rht

        torch.manual_seed(1)
        g = torch.randn(32, 64, device="cuda", dtype=torch.float32)  # [N,M]
        x = torch.randn(64, 48, device="cuda", dtype=torch.float32)  # [M,K]
        ref = g @ x
        g_r = _apply_rht(g)  # rotate along contraction M (last dim)
        x_r = _apply_rht(x.t().contiguous()).t()  # rotate along M
        rel = ((g_r @ x_r) - ref).norm() / ref.norm()
        assert rel < 1e-4, rel

    def test_recipe_reduces_wgrad_noise(self):
        """RHT rotates contraction-dim outliers toward Gaussian, cutting the FP4
        wgrad quantization noise vs no-RHT."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            QuantPolicy,
            _fp4_mm,
            _pad_to_block,
        )

        torch.manual_seed(7)
        n, m, k = 128, 512, 256
        g = torch.randn(n, m, device="cuda", dtype=torch.bfloat16) * 0.01
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.01
        out_tok = torch.randperm(m)[:8]  # outliers concentrated on a few tokens (dim M)
        g[:, out_tok] *= 30.0
        x[out_tok, :] *= 30.0
        ref = g.float() @ x.float()
        gt, _ = _pad_to_block(g.contiguous(), 1)
        xp, _ = _pad_to_block(x.contiguous(), 0)

        def noise(hadamard):
            p = QuantPolicy(stochastic=True, hadamard=hadamard)
            vals = [
                ((_fp4_mm(gt, xp, p, p).float() - ref).norm() / ref.norm()).item()
                for _ in range(10)
            ]
            return sum(vals) / len(vals)

        assert noise(True) < noise(False)


@require_torch_2_8_0
@requires_sm_ge_100
class TestNVFP4Adapters:
    """NVFP4 base under LoRA adapters: FP4 compute, and FP4 storage (QLoRA)."""

    def _lora_model(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "axolotl-ai-co/tiny-qwen2-129m", device_map="auto", dtype=torch.bfloat16
        )
        lc = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, lc)

    def _train(self, model, steps=10):
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        ids = torch.randint(0, 1000, (2, 64), device=model.device)
        first = None
        for _ in range(steps):
            opt.zero_grad()
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            opt.step()
            first = first if first is not None else float(out.loss)
        return first, float(out.loss)

    def test_lora_fp4_compute(self):
        """base_layer -> NVFP4Linear (frozen HP weight), adapters train."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Linear,
            NVFP4Recipe,
            convert_lora_base_to_nvfp4,
        )

        model = self._lora_model()
        n = convert_lora_base_to_nvfp4(model, NVFP4Recipe(), quantized_storage=False)
        assert n > 0
        assert sum(1 for m in model.modules() if isinstance(m, NVFP4Linear)) == n
        l0, l1 = self._train(model)
        assert l1 < l0 and torch.isfinite(torch.tensor(l1))

    def test_lora_fp4_compute_base_prequant(self):
        """compute_base: frozen base pre-quantized into 2 FP4 layouts; fprop
        matches per-step requant bit-exactly, base quant prologue paid once."""
        from torch import nn

        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4ComputeBaseLinear,
            NVFP4Linear,
            NVFP4Recipe,
            convert_lora_base_to_nvfp4,
            is_nvfp4_base,
        )

        # bit-identical to the per-step path (same quantization, just cached)
        lin = nn.Linear(512, 768, bias=False).cuda().bfloat16()
        lin.weight.requires_grad_(False)
        x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
        oc = NVFP4ComputeBaseLinear.from_linear(lin, NVFP4Recipe())(x)
        op = NVFP4Linear.from_linear(lin, NVFP4Recipe())(x)
        assert torch.equal(oc, op)

        # two FP4 layouts are smaller than the bf16 weight
        cb = NVFP4ComputeBaseLinear.from_linear(lin, NVFP4Recipe())
        fp4 = (
            cb.w_fprop.qdata.numel()
            + cb.w_fprop.scale.numel()
            + cb.w_dgrad.qdata.numel()
            + cb.w_dgrad.scale.numel()
        )
        assert (512 * 768 * 2) / fp4 > 1.5

        model = self._lora_model()
        n = convert_lora_base_to_nvfp4(model, NVFP4Recipe(), compute_base=True)
        assert n > 0
        assert sum(1 for m in model.modules() if is_nvfp4_base(m)) == n
        l0, l1 = self._train(model)
        assert l1 < l0 and torch.isfinite(torch.tensor(l1))

    def test_qlora_fp4_storage(self):
        """base_layer -> NVFP4FrozenBaseLinear: weight stored in FP4 (memory win)."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Recipe,
            convert_lora_base_to_nvfp4,
            is_nvfp4_base,
        )

        model = self._lora_model()
        n = convert_lora_base_to_nvfp4(model, NVFP4Recipe(), quantized_storage=True)
        assert n > 0
        bases = [m for m in model.modules() if is_nvfp4_base(m)]
        assert len(bases) == n
        # FP4 storage is materially smaller than the bf16 weight it replaced
        b = bases[0]
        if hasattr(b, "w_q"):  # torchao base
            fp4_bytes = b.w_q.qdata.numel() + b.w_q.scale.numel()
        else:  # mslk-fast base: packed qdata + e4m3 block scales
            fp4_bytes = b.wq.numel() + b.wsc.numel()
        bf16_bytes = b.in_features * b.out_features * 2
        assert bf16_bytes / fp4_bytes > 3.0
        l0, l1 = self._train(model)
        assert l1 < l0 and torch.isfinite(torch.tensor(l1))

    def test_fsdp_hooks_present_and_reconstruct(self):
        """The FSDP-wrapped FP4 base carries all-gather hooks and reconstructs
        bit-exactly from row-sharded qdata/scale (the shard the hooks gather)."""
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            NVFP4Tensor,
            per_tensor_amax_to_scale,
        )

        from axolotl.integrations.nvfp4.nvfp4_training import _to_fsdp_nvfp4

        w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) * 0.05
        wq = NVFP4Tensor.to_nvfp4(
            w.contiguous(),
            block_size=16,
            per_tensor_scale=per_tensor_amax_to_scale(torch.max(torch.abs(w))),
        )
        fq = _to_fsdp_nvfp4(wq)
        assert hasattr(fq, "fsdp_pre_all_gather")
        assert hasattr(fq, "fsdp_post_all_gather")
        assert torch.equal(fq.dequantize(torch.bfloat16), wq.dequantize(torch.bfloat16))

        # simulate the gather: split qdata/scale by row, concat, reconstruct
        (qd, sc), (ctx, pts) = fq.fsdp_pre_all_gather(mesh=None)
        half = qd.shape[0] // 2
        gathered = (
            torch.cat([qd[:half], qd[half:]], dim=0),
            torch.cat([sc[:half], sc[half:]], dim=0),
        )
        rebuilt, _ = fq.fsdp_post_all_gather(gathered, (ctx, pts), torch.bfloat16)
        rel = (
            rebuilt.dequantize(torch.bfloat16) - wq.dequantize(torch.bfloat16)
        ).norm() / wq.dequantize(torch.bfloat16).norm()
        assert rel.item() < 1e-6

    def test_qlora_checkpoint_roundtrip(self):
        """The FP4-packed base must survive save/load (buffer in state_dict),
        else QLoRA resume silently reinitializes the base."""
        import io

        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Recipe,
            convert_lora_base_to_nvfp4,
        )

        model = self._lora_model()
        convert_lora_base_to_nvfp4(model, NVFP4Recipe(), quantized_storage=True)
        sd = model.state_dict()
        # torchao base stores "w_q"; the mslk-fast base stores "wq"/"wsc"/"w_inv".
        assert any(k.endswith(("w_q", "wq")) for k in sd)

        ids = torch.randint(0, 1000, (2, 32), device=model.device)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        opt.zero_grad()
        model(input_ids=ids, labels=ids).loss.backward()
        opt.step()

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        reloaded = self._lora_model()
        convert_lora_base_to_nvfp4(reloaded, NVFP4Recipe(), quantized_storage=True)
        missing, _ = reloaded.load_state_dict(
            torch.load(buf, weights_only=False), strict=False
        )
        assert not [
            k for k in missing if k.rsplit(".", 1)[-1] in ("w_q", "wq", "wsc", "w_inv")
        ]

        model.eval()
        reloaded.eval()
        with torch.no_grad():
            o1 = model(input_ids=ids).logits
            o2 = reloaded(input_ids=ids).logits
        assert torch.equal(o1, o2)

    def test_compute_base_checkpoint_roundtrip(self):
        """NVFP4ComputeBaseLinear: the two FP4 layouts (float4_e2m1fn_x2 qdata +
        float8_e4m3fn scales) must survive state_dict save/load bit-exactly,
        else compute-base resume reinitializes the base."""
        import io

        from torch import nn

        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4ComputeBaseLinear,
            NVFP4Recipe,
        )

        torch.manual_seed(0)
        lin = nn.Linear(512, 256, bias=True).cuda().bfloat16()
        mod = NVFP4ComputeBaseLinear.from_linear(lin, NVFP4Recipe())
        x = torch.randn(48, 512, device="cuda", dtype=torch.bfloat16)
        ref = mod(x)

        sd = mod.state_dict()
        assert any(k.endswith("w_fprop") for k in sd)
        assert any(k.endswith("w_dgrad") for k in sd)

        buf = io.BytesIO()
        torch.save(sd, buf)
        buf.seek(0)
        fresh = NVFP4ComputeBaseLinear.from_linear(
            nn.Linear(512, 256, bias=True).cuda().bfloat16(), NVFP4Recipe()
        )
        fresh.load_state_dict(torch.load(buf, weights_only=False))
        assert torch.equal(fresh.w_fprop.qdata, mod.w_fprop.qdata)
        assert torch.equal(fresh.w_fprop.scale, mod.w_fprop.scale)
        assert torch.equal(fresh(x), ref)

    def test_fast_compute_base_checkpoint_roundtrip(self):
        """NVFP4FastComputeBaseLinear (MSLK): the packed FP4 qdata + swizzled e4m3
        scale + inv-global buffers must survive state_dict save/load bit-exactly."""
        import io

        from torch import nn

        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4FastComputeBaseLinear,
            NVFP4Recipe,
            _mslk_available,
        )

        if not _mslk_available():
            pytest.skip("MSLK fused FP4 quant kernel not available")

        torch.manual_seed(0)
        lin = nn.Linear(512, 256, bias=True).cuda().bfloat16()
        mod = NVFP4FastComputeBaseLinear.from_linear(lin, NVFP4Recipe())
        x = torch.randn(48, 512, device="cuda", dtype=torch.bfloat16)
        ref = mod(x)

        sd = mod.state_dict()
        assert any(k.endswith("wq_f") for k in sd)
        assert any(k.endswith("wq_d") for k in sd)

        buf = io.BytesIO()
        torch.save(sd, buf)
        buf.seek(0)
        fresh = NVFP4FastComputeBaseLinear.from_linear(
            nn.Linear(512, 256, bias=True).cuda().bfloat16(), NVFP4Recipe()
        )
        fresh.load_state_dict(torch.load(buf, weights_only=False))
        assert fresh.wq_f.dtype == torch.float4_e2m1fn_x2
        assert fresh.wsc_f.dtype == torch.float8_e4m3fn
        assert torch.equal(fresh.wq_f, mod.wq_f)
        assert torch.equal(fresh.wsc_f, mod.wsc_f)
        assert torch.equal(fresh(x), ref)

    def test_fast_storage_base_checkpoint_roundtrip(self):
        """NVFP4FastFrozenBaseLinear (MSLK storage): single FP4 weight layout
        survives save/load and reproduces forward bit-exactly."""
        import io

        from torch import nn

        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4FastFrozenBaseLinear,
            NVFP4Recipe,
            _mslk_available,
        )

        if not _mslk_available():
            pytest.skip("MSLK fused FP4 quant kernel not available")

        torch.manual_seed(0)
        lin = nn.Linear(512, 256, bias=True).cuda().bfloat16()
        mod = NVFP4FastFrozenBaseLinear.from_linear(lin, NVFP4Recipe())
        x = torch.randn(48, 512, device="cuda", dtype=torch.bfloat16)
        ref = mod(x)

        sd = mod.state_dict()
        assert any(k.endswith("wq") for k in sd)

        buf = io.BytesIO()
        torch.save(sd, buf)
        buf.seek(0)
        fresh = NVFP4FastFrozenBaseLinear.from_linear(
            nn.Linear(512, 256, bias=True).cuda().bfloat16(), NVFP4Recipe()
        )
        fresh.load_state_dict(torch.load(buf, weights_only=False))
        assert fresh.wq.dtype == torch.float4_e2m1fn_x2
        assert torch.equal(fresh.wq, mod.wq)
        assert torch.equal(fresh(x), ref)

    def test_embedding_checkpoint_roundtrip(self):
        """NVFP4Embedding: the packed FP4 weight buffer survives save/load and
        reproduces the lookup bit-exactly (else quantize_embeddings resume
        reinitializes the embedding)."""
        import io

        from axolotl.integrations.nvfp4.nvfp4_training import NVFP4Embedding

        torch.manual_seed(0)
        emb = nn.Embedding(2048, 512).cuda().bfloat16()
        mod = NVFP4Embedding.from_embedding(emb)
        idx = torch.randint(0, 2048, (4, 32), device="cuda")
        ref = mod(idx)

        sd = mod.state_dict()
        assert any(k.endswith("w_q") for k in sd)
        buf = io.BytesIO()
        torch.save(sd, buf)
        buf.seek(0)
        fresh = NVFP4Embedding.from_embedding(nn.Embedding(2048, 512).cuda().bfloat16())
        fresh.load_state_dict(torch.load(buf, weights_only=False))
        assert torch.equal(fresh.w_q.qdata, mod.w_q.qdata)
        assert torch.equal(fresh(idx), ref)

    def test_tied_lmhead_shares_embedding_store(self):
        """NVFP4TiedLMHead must read the SAME FP4 store as its NVFP4Embedding
        (quantize-once), so the lookup and the GEMM use a bit-identical weight and
        only the embedding's buffer needs checkpointing."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Embedding,
            NVFP4Recipe,
            NVFP4TiedLMHead,
        )

        torch.manual_seed(0)
        nvemb = NVFP4Embedding.from_embedding(nn.Embedding(2048, 512).cuda().bfloat16())
        head = NVFP4TiedLMHead(nvemb, None, NVFP4Recipe())
        assert head.w_q is nvemb.w_q  # same object => quantized once
        assert torch.equal(head.weight, nvemb.weight)

    def test_save_nvfp4_frozen_base_size_and_bit_exact(self, tmp_path):
        """save_nvfp4 sidecar for a FROZEN storage base: materially smaller than
        the bf16 weight AND bit-exact on load (same FP4 buffers)."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4FrozenBaseLinear,
            NVFP4Recipe,
            load_nvfp4_packed,
            save_nvfp4_packed,
        )

        class Wrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.layer = m

        torch.manual_seed(0)
        lin = nn.Linear(512, 256, bias=False).cuda().bfloat16()
        model = Wrap(NVFP4FrozenBaseLinear.from_linear(lin, NVFP4Recipe()))
        x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
        ref = model.layer(x)

        n = save_nvfp4_packed(model, tmp_path)
        assert n == 1
        sidecar = tmp_path / "nvfp4_packed.pt"
        bf16_bytes = lin.weight.numel() * 2
        assert sidecar.stat().st_size < bf16_bytes / 3  # ~3.4x smaller

        fresh = Wrap(
            NVFP4FrozenBaseLinear.from_linear(
                nn.Linear(512, 256, bias=False).cuda().bfloat16(), NVFP4Recipe()
            )
        )
        load_nvfp4_packed(fresh, tmp_path)
        assert torch.equal(fresh.layer.w_q.qdata, model.layer.w_q.qdata)
        assert torch.equal(fresh.layer(x), ref)  # frozen FP4 => bit-exact

    def test_save_nvfp4_fft_lossy_roundtrip(self, tmp_path):
        """save_nvfp4 for an FFT NVFP4Linear (bf16 master): packs the weight to
        FP4 (lossy), ~3x smaller, load-back forward within FP4 tolerance."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Linear,
            NVFP4Recipe,
            load_nvfp4_packed,
            save_nvfp4_packed,
        )

        class Wrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.layer = m

        torch.manual_seed(0)
        lin = nn.Linear(512, 256, bias=False).cuda().bfloat16()
        model = Wrap(NVFP4Linear.from_linear(lin, NVFP4Recipe()))
        x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
        ref = model.layer(x)

        save_nvfp4_packed(model, tmp_path)
        bf16_bytes = lin.weight.numel() * 2
        assert (tmp_path / "nvfp4_packed.pt").stat().st_size < bf16_bytes / 3

        fresh = Wrap(
            NVFP4Linear.from_linear(
                nn.Linear(512, 256, bias=False).cuda().bfloat16(), NVFP4Recipe()
            )
        )
        load_nvfp4_packed(fresh, tmp_path)
        out = fresh.layer(x)
        cos = torch.nn.functional.cosine_similarity(
            out.flatten().float(), ref.flatten().float(), dim=0
        )
        assert cos.item() > 0.99

    def test_save_nvfp4_drops_bf16_keys(self, tmp_path):
        """collect_nvfp4_packed_state reports the bf16 state_dict keys that the FP4
        sidecar supersedes, so the main shard drops them (the disk win)."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            NVFP4Recipe,
            collect_nvfp4_packed_state,
            convert_lora_base_to_nvfp4,
        )

        model = self._lora_model()
        convert_lora_base_to_nvfp4(model, NVFP4Recipe(), quantized_storage=True)
        packed, drop = collect_nvfp4_packed_state(model)
        assert packed and drop
        sd = model.state_dict()
        assert drop <= set(sd)  # every dropped key really exists in the state_dict


def _is_sm_120():
    return (
        torch.cuda.is_available()
        and bool(torch.version.cuda)
        and torch.cuda.get_device_capability() == (12, 0)
    )


class TestNVFP4Sm120Gate:
    """C2: the recipe-fusion gate opens on sm_120 only if the codegen probe verifies.

    The probe / monkeypatched-capability parts run anywhere (no Blackwell
    needed); the real bit-exact assertion is guarded behind an sm_120 skip.
    """

    def _reset_probe_cache(self):
        import axolotl.integrations.nvfp4.nvfp4_training as nvmod

        nvmod._SM120F_RECIPE_VERIFIED = {}
        return nvmod

    def test_probe_returns_bool_and_is_cached(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        nvmod = self._reset_probe_cache()

        dev = torch.device("cuda")
        cap = torch.cuda.get_device_capability(dev)
        first = nvmod._sm120f_recipe_codegen_verified(dev)
        assert isinstance(first, bool)
        # Second call must return the per-device cached value without recomputing.
        assert nvmod._SM120F_RECIPE_VERIFIED[cap] is first
        assert nvmod._sm120f_recipe_codegen_verified(dev) is first

    def test_probe_env_kill_switch(self, monkeypatch):
        nvmod = self._reset_probe_cache()
        monkeypatch.setenv("AXOLOTL_NVFP4_SKIP_SM120F_PROBE", "1")
        assert nvmod._sm120f_recipe_codegen_verified(torch.device("cuda")) is False
        # Kill-switch must not poison the cache for a later (unset) run.
        assert nvmod._SM120F_RECIPE_VERIFIED == {}

    def test_gate_default_safe_on_non_blackwell(self, monkeypatch):
        """A spoofed sm_90 device must keep the gate closed regardless of probe."""
        nvmod = self._reset_probe_cache()
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
        monkeypatch.setattr(nvmod, "_mslk_available", lambda: True)

        t = torch.empty(16, 16)
        t_cuda = type(
            "FakeT",
            (),
            {
                "is_cuda": True,
                "dim": lambda self: 2,
                "shape": (16, 16),
                "device": "cuda:0",
            },
        )()
        # _recipe_fusion_available reads .is_cuda/.dim()/.shape/.device only.
        assert nvmod._recipe_fusion_available(t_cuda) is False
        del t

    def test_gate_sm120_requires_probe(self, monkeypatch):
        """With a spoofed sm_120 cap, the gate tracks the probe result exactly."""
        nvmod = self._reset_probe_cache()
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda *a, **k: (12, 0)
        )
        monkeypatch.setattr(nvmod, "_mslk_available", lambda: True)
        t_cuda = type(
            "FakeT",
            (),
            {
                "is_cuda": True,
                "dim": lambda self: 2,
                "shape": (16, 16),
                "device": "cuda:0",
            },
        )()

        monkeypatch.setattr(
            nvmod, "_sm120f_recipe_codegen_verified", lambda device: False
        )
        assert nvmod._recipe_fusion_available(t_cuda) is False

        monkeypatch.setattr(
            nvmod, "_sm120f_recipe_codegen_verified", lambda device: True
        )
        assert nvmod._recipe_fusion_available(t_cuda) is True

    @pytest.mark.skipif(not _is_sm_120(), reason="requires sm_120 (consumer Blackwell)")
    @require_torch_2_8_0
    def test_probe_verifies_and_gate_opens_on_real_sm120(self):
        """On real sm_120 the probe must verify and the gate must open."""
        from axolotl.integrations.nvfp4.nvfp4_training import (
            _BLOCK_SIZE,
            _abs_amax,
            _mslk_available,
            _mslk_quantize_recipe_op,
            _recipe_fusion_available,
            _sm120f_recipe_codegen_verified,
        )

        if not _mslk_available():
            pytest.skip("MSLK not importable; gate intentionally stays closed")

        self._reset_probe_cache()
        assert _sm120f_recipe_codegen_verified(torch.device("cuda")) is True

        assert (
            _recipe_fusion_available(
                torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            )
            is True
        )

        # Independent bit-exact re-check of the RTN cvt against torchao on the
        # SAME seeded tile the probe uses. Bit-exactness is the gross-miscompile
        # bar (a #3096-style mis-assembled cvt corrupts even this tile); it is
        # deliberately NOT a universal-input claim — adjacent-level scale-rounding
        # ties between the hw cvt and torchao's reference make a tiny fraction of
        # bytes differ on other random inputs, which is expected and benign.
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            NVFP4Tensor,
            per_tensor_amax_to_scale,
        )

        gen = torch.Generator(device="cuda").manual_seed(0)
        t = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, generator=gen)
        q, _s, _g = _mslk_quantize_recipe_op(t, False, False)
        pts = per_tensor_amax_to_scale(_abs_amax(t).clamp(min=1e-12))
        ref = NVFP4Tensor.to_nvfp4(
            t.contiguous(), block_size=_BLOCK_SIZE, per_tensor_scale=pts
        )
        assert torch.equal(q.view(torch.uint8).cpu(), ref.qdata.view(torch.uint8).cpu())
