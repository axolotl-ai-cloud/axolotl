"""Correctness tests for the fused MemFT-OT linear cross-entropy kernel."""

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton", reason="triton required for fused kernels")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for fused kernel tests", allow_module_level=True)

from axolotl.kernels.memft_xentropy import memft_linear_cross_entropy

LN2 = math.log(2.0)


def _reference(hidden, weight, labels, critical_loss=LN2, eps=1e-8, ignore_index=-100):
    logits = (hidden @ weight.t()).float()
    ce = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")
    mask = labels != ignore_index
    w = (mask & (ce > critical_loss)).float()
    return (w * ce).sum() / (w.sum() + eps)


@pytest.mark.parametrize(
    "n_tokens,hidden_size,vocab_size",
    [(64, 128, 1000), (130, 256, 2048), (512, 384, 32000)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_matches_reference(n_tokens, hidden_size, vocab_size, dtype):
    torch.manual_seed(0)
    dev = "cuda"
    hidden = torch.randn(n_tokens, hidden_size, device=dev, dtype=dtype) * 0.5
    weight = torch.randn(vocab_size, hidden_size, device=dev, dtype=dtype) * 0.5
    labels = torch.randint(0, vocab_size, (n_tokens,), device=dev)
    labels[:5] = -100

    hf = hidden.clone().requires_grad_(True)
    wf = weight.clone().requires_grad_(True)
    loss_f = memft_linear_cross_entropy(hf, wf, labels)
    loss_f.backward()

    hr = hidden.clone().float().requires_grad_(True)
    wr = weight.clone().float().requires_grad_(True)
    loss_r = _reference(hr, wr, labels)
    loss_r.backward()

    if dtype == torch.float32:
        ltol, gtol = 1e-4, 1e-4
    else:
        ltol, gtol = 5e-2, 5e-3

    assert abs(float(loss_f) - float(loss_r)) < ltol
    assert (hf.grad.float() - hr.grad).abs().max().item() < gtol
    assert (wf.grad.float() - wr.grad).abs().max().item() < gtol


def test_all_ignored_batch_is_zero_no_nan():
    dev = "cuda"
    n, h, v = 32, 64, 256
    hidden = torch.randn(n, h, device=dev, requires_grad=True)
    weight = torch.randn(v, h, device=dev, requires_grad=True)
    labels = torch.full((n,), -100, device=dev)

    loss = memft_linear_cross_entropy(hidden, weight, labels)
    loss.backward()
    assert torch.isfinite(loss) and float(loss) == 0.0
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().max() == 0.0
    assert torch.isfinite(weight.grad).all() and weight.grad.abs().max() == 0.0


def test_grad_weight_precision_under_many_chunks():
    # bf16 head, many tokens -> many chunks; fp32 accumulation must keep the
    # weight gradient close to the fp32 reference.
    torch.manual_seed(0)
    dev = "cuda"
    n, h, v = 8192, 512, 8000
    hidden = torch.randn(n, h, device=dev, dtype=torch.bfloat16) * 0.3
    weight = torch.randn(v, h, device=dev, dtype=torch.bfloat16) * 0.3
    labels = torch.randint(0, v, (n,), device=dev)

    hf = hidden.clone().requires_grad_(True)
    wf = weight.clone().requires_grad_(True)
    memft_linear_cross_entropy(hf, wf, labels).backward()

    hr = hidden.float().requires_grad_(True)
    wr = weight.float().requires_grad_(True)
    _reference(hr, wr, labels).backward()

    rel = (wf.grad.float() - wr.grad).abs().max() / (wr.grad.abs().max() + 1e-6)
    assert rel.item() < 0.05


def test_out_of_range_label_treated_as_ignore():
    # labels outside [0, vocab) (below 0 or >= vocab) and not the ignore
    # sentinel must not OOB-load; the kernel masks them like ignore_index.
    torch.manual_seed(0)
    dev = "cuda"
    n, h, v = 16, 32, 128
    hidden = torch.randn(n, h, device=dev)
    weight = torch.randn(v, h, device=dev)
    labels = torch.randint(0, v, (n,), device=dev)
    labels[3] = -5  # below range, not ignore_index
    labels[7] = v  # at the upper bound (== vocab_size)
    labels[9] = v + 4  # above range

    hf = hidden.clone().requires_grad_(True)
    wf = weight.clone().requires_grad_(True)
    loss_f = memft_linear_cross_entropy(hf, wf, labels)
    loss_f.backward()

    ref_labels = labels.clone()
    ref_labels[[3, 7, 9]] = -100  # reference: the bad positions are ignored
    hr = hidden.clone().requires_grad_(True)
    wr = weight.clone().requires_grad_(True)
    loss_r = _reference(hr, wr, ref_labels)
    loss_r.backward()

    assert torch.isfinite(loss_f)
    assert abs(float(loss_f) - float(loss_r)) < 1e-4
    assert (hf.grad - hr.grad).abs().max().item() < 1e-4


def _params(fused, variant="ot", **kw):
    p = {
        "fused": fused,
        "variant": variant,
        "critical_loss": LN2,
        "epsilon": 1e-8,
        "kappa": 1.0,
        "tau": 64.0,
        "window": 64,
        "floor": 0.0,
        "chunk_tokens": None,
        "ignore_index": -100,
    }
    p.update(kw)
    return p


def _tiny_llama(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(cfg).cuda().to(torch.float32)


def test_fused_forward_patch_matches_eager():
    from axolotl.monkeypatch.loss.memft import memft_loss, patch_memft

    model = _tiny_llama()
    bsz, seq = 2, 16
    input_ids = torch.randint(0, 512, (bsz, seq), device="cuda")
    labels = input_ids.clone()
    labels[:, :4] = -100

    out_ref = model(input_ids=input_ids)
    loss_ref = memft_loss(out_ref, labels, variant="ot")
    (g_ref,) = torch.autograd.grad(loss_ref, model.lm_head.weight)

    patch_memft("llama", _params(fused=True))
    out_f = model(input_ids=input_ids, labels=labels)
    (g_f,) = torch.autograd.grad(out_f.loss, model.lm_head.weight)

    assert out_f.logits is None  # logits not materialized during training
    assert abs(float(loss_ref) - float(out_f.loss)) < 1e-4
    assert (g_ref - g_f).abs().max().item() < 1e-5

    # generation path (no labels) still returns full logits
    out_gen = model(input_ids=input_ids)
    assert out_gen.logits.shape == (bsz, seq, 512)
    assert out_gen.loss is None


def test_nonfused_forward_patch_returns_logits_and_matches():
    from axolotl.monkeypatch.loss.memft import memft_loss, patch_memft

    model = _tiny_llama(seed=1)
    bsz, seq = 2, 16
    input_ids = torch.randint(0, 512, (bsz, seq), device="cuda")
    labels = input_ids.clone()
    labels[:, :3] = -100

    out_ref = model(input_ids=input_ids)
    loss_ref = memft_loss(out_ref, labels, variant="sw")

    patch_memft("llama", _params(fused=False, variant="sw"))
    out = model(input_ids=input_ids, labels=labels)

    assert out.logits is not None  # non-fused keeps logits for eval
    assert out.logits.shape == (bsz, seq, 512)
    assert abs(float(loss_ref) - float(out.loss)) < 1e-4


def test_packing_confines_loss_to_each_sample():
    # two samples packed into one row must give the same loss as the two
    # samples run unpacked (no cross-sample shift leakage).
    from axolotl.monkeypatch.loss.memft import memft_loss

    torch.manual_seed(2)
    bsz, total, vocab = 1, 12, 64
    logits = torch.randn(bsz, total, vocab, device="cuda")
    labels = torch.randint(0, vocab, (bsz, total), device="cuda")
    # sample A = positions 0..6, sample B = positions 7..11
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4]], device="cuda")

    packed = memft_loss((logits,), labels, variant="ot", position_ids=position_ids)

    # unpacked reference: weighted sums over each sample separately, no shift
    # across the A|B boundary
    a_logits, a_labels = logits[:, :7], labels[:, :7]
    b_logits, b_labels = logits[:, 7:], labels[:, 7:]

    def parts(lg, lb):
        sl = lg[:, :-1, :].reshape(-1, vocab)
        tl = lb[:, 1:].reshape(-1)
        ce = F.cross_entropy(sl.float(), tl, reduction="none")
        w = (ce > LN2).float()
        return (w * ce).sum(), w.sum()

    na, da = parts(a_logits, a_labels)
    nb, db = parts(b_logits, b_labels)
    expected = (na + nb) / (da + db + 1e-8)

    assert abs(float(packed) - float(expected)) < 1e-4
