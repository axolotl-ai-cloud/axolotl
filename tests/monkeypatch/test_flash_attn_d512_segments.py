"""Segment-masked head_dim-512 attention (packed encoder/canvas global layers).

CPU: the row-segment metadata reproduces the dense encoder/decoder reference masks exactly, and
layouts it cannot express are refused. GPU: the Triton kernel matches an fp32 reference in
forward and dq/dk/dv for Q != KV, GQA, packed documents and fully masked rows, and the tiny
DiffusionGemma model gives the same logits and gradients with the kernel as with dense attention.
"""

from __future__ import annotations

import copy

import pytest
import torch

from axolotl.core.trainers.diffusion_lm.attention import (
    decoder_prefix_canvas_mask,
    encoder_causal_mask,
)
from axolotl.monkeypatch.attention import large_head as lh
from axolotl.monkeypatch.attention.segment_mask import (
    SEGMENTS_ATTR,
    SegmentAttentionMeta,
    decoder_segment_meta,
    encoder_segment_meta,
)

FLEX_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "BLOCK_M1": 16,
    "BLOCK_N1": 16,
    "BLOCK_M2": 16,
    "BLOCK_N2": 16,
}

_cuda_skip = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton segment kernel needs CUDA"
)


def requires_gpu(fn):
    return pytest.mark.gpu(_cuda_skip(fn))


def _packed_layout(
    prompt_lengths,
    canvas_lengths,
    prefixes,
    canvas_valid,
    logical_ids,
    prompt_pad=0,
    canvas_pad=0,
):
    """Packed tensors shaped like EncoderCanvasBackend.pack output (B=1, bucket padding)."""
    encoder_docs, encoder_pos, canvas_docs, canvas_validity = [], [], [], []
    for length, canvas, valid, logical in zip(
        prompt_lengths, canvas_lengths, canvas_valid, logical_ids, strict=True
    ):
        encoder_docs.append(torch.full((length,), logical))
        encoder_pos.append(torch.arange(length))
        canvas_docs.append(torch.full((canvas,), logical))
        canvas_validity.append(torch.arange(canvas) < valid)
    encoder_docs.append(torch.full((prompt_pad,), -1))
    encoder_pos.append(torch.zeros(prompt_pad, dtype=torch.long))
    canvas_docs.append(torch.full((canvas_pad,), -1))
    canvas_validity.append(torch.zeros(canvas_pad, dtype=torch.bool))
    encoder_docs = torch.cat(encoder_docs)[None]
    return dict(
        encoder_document_ids=encoder_docs,
        canvas_document_ids=torch.cat(canvas_docs)[None],
        encoder_validity=encoder_docs >= 0,
        canvas_validity=torch.cat(canvas_validity)[None],
        decoder_prefix_lengths=torch.tensor(prefixes),
        encoder_position_ids=torch.cat(encoder_pos)[None],
        logical_ids=torch.tensor(logical_ids),
    )


LAYOUTS = [
    # prompt lengths, canvas lengths, decoder prefixes, valid canvas tokens, logical ids, pads
    ((7, 5), (3, 3), (4, 5), (3, 2), (17, 91), 4, 2),
    (
        (300, 517, 1000),
        (128, 128, 64),
        (250, 517, 700),
        (120, 128, 40),
        (0, 1, 2),
        33,
        0,
    ),
    (
        (40, 1, 9, 64),
        (16, 16, 16, 16),
        (40, 0, 9, 1),
        (16, 0, 5, 16),
        (5, 3, 9, 1),
        0,
        7,
    ),
]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_decoder_meta_reproduces_the_dense_decoder_mask(layout):
    *shape, prompt_pad, canvas_pad = layout
    inputs = _packed_layout(*shape, prompt_pad=prompt_pad, canvas_pad=canvas_pad)
    meta = decoder_segment_meta(**inputs)
    assert meta is not None
    reference = decoder_prefix_canvas_mask(**inputs)[:, 0]
    assert torch.equal(meta.dense(), reference)
    # dK/dV visit only kv_q_ranges: it must cover every query that sees each key.
    rows = torch.arange(meta.q_len)[None, :, None]
    covered = (rows >= meta.kv_q_ranges[:, None, :, 0]) & (
        rows < meta.kv_q_ranges[:, None, :, 1]
    )
    assert not (reference & ~covered).any()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_encoder_meta_reproduces_the_dense_causal_mask(layout):
    *shape, prompt_pad, canvas_pad = layout
    inputs = _packed_layout(*shape, prompt_pad=prompt_pad, canvas_pad=canvas_pad)
    docs, validity = inputs["encoder_document_ids"], inputs["encoder_validity"]
    positions = inputs["encoder_position_ids"]
    meta = encoder_segment_meta(docs, validity, positions)
    assert meta is not None
    reference = encoder_causal_mask(docs, validity, positions)[:, 0]
    assert torch.equal(meta.dense(), reference)
    rows = torch.arange(meta.q_len)[None, :, None]
    covered = (rows >= meta.kv_q_ranges[:, None, :, 0]) & (
        rows < meta.kv_q_ranges[:, None, :, 1]
    )
    assert not (reference & ~covered).any()


def test_meta_refuses_layouts_it_cannot_express():
    inputs = _packed_layout((4, 4), (2, 2), (4, 4), (2, 2), (0, 1))
    # Interleave the two documents' prompts: neither is a contiguous run any more.
    inputs["encoder_document_ids"] = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]])
    assert decoder_segment_meta(**inputs) is None
    assert (
        encoder_segment_meta(
            inputs["encoder_document_ids"],
            inputs["encoder_validity"],
            inputs["encoder_position_ids"],
        )
        is None
    )
    # Positions that go backwards inside a document break index-order causality.
    assert (
        encoder_segment_meta(
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.bool),
            torch.tensor([[0, 1, 3, 2]]),
        )
        is None
    )


def test_route_declines_without_meta_or_policy():
    q = torch.zeros(1, 2, 8, 512)
    lh.set_large_head_policy("triton_flash")
    try:
        assert lh.flash_d512_segment_route(None, q, q, q, None) is None
        mask = type("Mask", (), {})()
        setattr(
            mask,
            SEGMENTS_ATTR,
            SegmentAttentionMeta(
                torch.zeros(1, 8, 4, dtype=torch.int32),
                torch.ones(1, 8, dtype=torch.int8),
                torch.zeros(1, 8, 2, dtype=torch.int32),
            ),
        )
        # CPU tensors never reach the Triton kernel.
        assert lh.flash_d512_segment_route(None, q, q, q, mask) is None
        lh.set_large_head_policy("sdpa")
        assert not lh.segment_kernel_enabled()
    finally:
        lh.set_large_head_policy("sdpa")


def _reference(q, k, v, meta, scale):
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, 1)
    v = v.repeat_interleave(groups, 1)
    scores = (q @ k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~meta.dense()[:, None], float("-inf"))
    return torch.softmax(scores, -1).nan_to_num(0.0) @ v


def _check_kernel(meta, heads, kv_heads, dtype, rtol, seed=0):
    from axolotl.monkeypatch.attention.flash_attn_d512_segments import (
        flash_d512_segments,
    )

    torch.manual_seed(seed)
    device = torch.device("cuda")
    meta = SegmentAttentionMeta(
        meta.q_ranges.to(device), meta.kv_valid.to(device), meta.kv_q_ranges.to(device)
    )
    m, n, d = meta.q_len, meta.kv_len, 512
    # q arrives as a transposed [B, M, H, D] view, like the HF attention modules pass it.
    q = torch.randn(1, m, heads, d, device=device).transpose(1, 2)
    k = torch.randn(1, kv_heads, n, d, device=device)
    v = torch.randn(1, kv_heads, n, d, device=device)
    grad = torch.randn(1, heads, m, d, device=device)
    scale = d**-0.5
    leaves = [t.detach().to(dtype).requires_grad_() for t in (q, k, v)]
    out = flash_d512_segments(*leaves, meta, scale)
    out.backward(grad.to(dtype))
    gold = [t.detach().to(dtype).double().requires_grad_() for t in (q, k, v)]
    expected = _reference(*gold, meta, scale)
    expected.backward(grad.double())

    def relative(value, ref):
        return ((value.double() - ref).abs().max() / ref.abs().max()).item()

    # Max abs error relative to the reference's max magnitude: dK/dV of keys seen by few rows
    # are large, so a plain abs bound would test bf16 rounding of their size, not the kernel.
    errors = {"out": relative(out, expected)}
    for name, leaf, ref in zip(("dq", "dk", "dv"), leaves, gold, strict=True):
        errors[name] = relative(leaf.grad, ref.grad)
    print(errors)
    assert all(value <= rtol for value in errors.values()), errors
    # Rows that see no key are zero, like FlexAttention's fully masked rows.
    empty = ~meta.dense().any(-1)
    assert torch.all(out.transpose(1, 2)[empty] == 0)
    return errors


@requires_gpu
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("heads,kv_heads", [(16, 2), (4, 4), (8, 1)])
def test_kernel_matches_reference_on_decoder_masks(layout, heads, kv_heads):
    *shape, prompt_pad, canvas_pad = layout
    meta = decoder_segment_meta(
        **_packed_layout(*shape, prompt_pad=prompt_pad, canvas_pad=canvas_pad)
    )
    _check_kernel(meta, heads, kv_heads, torch.bfloat16, rtol=1e-2)


@requires_gpu
@pytest.mark.parametrize("layout", LAYOUTS)
def test_kernel_matches_reference_on_encoder_masks(layout):
    *shape, prompt_pad, canvas_pad = layout
    inputs = _packed_layout(*shape, prompt_pad=prompt_pad, canvas_pad=canvas_pad)
    meta = encoder_segment_meta(
        inputs["encoder_document_ids"],
        inputs["encoder_validity"],
        inputs["encoder_position_ids"],
    )
    _check_kernel(meta, 16, 2, torch.bfloat16, rtol=1e-2)


@requires_gpu
def test_kernel_fp16_and_fp32_routing():
    meta = decoder_segment_meta(
        **_packed_layout(*LAYOUTS[1][:5], prompt_pad=33, canvas_pad=0)
    )
    _check_kernel(meta, 16, 2, torch.float16, rtol=1e-2)
    # fp32 tiles do not fit sm120 shared memory at head_dim 512: fp32 stays on flex.
    device = torch.device("cuda")
    q = torch.zeros(1, 16, meta.q_len, 512, device=device)
    k = torch.zeros(1, 2, meta.kv_len, 512, device=device)
    mask = type("Mask", (), {})()
    setattr(
        mask,
        SEGMENTS_ATTR,
        SegmentAttentionMeta(
            meta.q_ranges.to(device),
            meta.kv_valid.to(device),
            meta.kv_q_ranges.to(device),
        ),
    )
    lh.set_large_head_policy("triton_flash")
    try:
        assert lh.flash_d512_segment_route(None, q, k, k, mask) is None
        routed = lh.flash_d512_segment_route(
            None, q.bfloat16(), k.bfloat16(), k.bfloat16(), mask
        )
        assert routed is not None and routed[0].shape == (1, meta.q_len, 16, 512)
    finally:
        lh.set_large_head_policy("sdpa")


@requires_gpu
def test_tiny_diffusion_gemma_kernel_matches_dense_logits_and_gradients():
    from dataclasses import replace

    from transformers import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion

    from axolotl.core.trainers.diffusion_lm.backends.encoder_canvas import (
        EncoderCanvasBackend,
    )
    from axolotl.core.trainers.diffusion_lm.collator import DiffusionCollator
    from axolotl.model_support.diffusion_gemma.modeling import (
        forward_packed_encoder_canvas,
    )

    text = {
        "vocab_size": 32,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "head_dim": 16,
        "global_head_dim": 512,
        "max_position_embeddings": 64,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 2,
        "use_bidirectional_attention": "vision",
        "num_experts": 2,
        "top_k_experts": 1,
        "moe_intermediate_size": 64,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "bos_token_id": 2,
    }
    vision = {
        "model_type": "gemma4_vision",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "patch_size": 16,
        "position_embedding_size": 16,
    }
    config = DiffusionGemmaConfig(
        text_config=text,
        vision_config=vision,
        canvas_length=8,
        boi_token_id=31,
        eoi_token_id=30,
        image_token_id=29,
    )
    device = torch.device("cuda:0")
    batch = DiffusionCollator(0, 3).build_batch(
        [
            {"input_ids": [2, 3, 4, 5, 8, 9, 10], "labels": [-100] * 4 + [8, 9, 10]},
            {"input_ids": [2, 6, 7, 11, 12, 13], "labels": [-100] * 3 + [11, 12, 13]},
            {"input_ids": [2, 6, 14, 15], "labels": [-100, -100, 14, 15]},
        ]
    )
    logical_ids = torch.tensor([17, 91, 4])
    batch = replace(
        batch,
        logical_ids=logical_ids,
        encoder_document_ids=torch.where(
            batch.encoder_validity,
            logical_ids[:, None].expand_as(batch.encoder_document_ids),
            batch.encoder_document_ids,
        ),
    ).to(device)
    dense = EncoderCanvasBackend(32, 2).pack(batch)
    lh.set_large_head_policy("triton_flash")
    route = lh.flash_d512_segment_route
    calls = {"kernel": 0, "reference": 0, "flex": 0}
    mode = {"name": "kernel"}

    # The kernel is 16-bit only, so the fp32 model hands the global layers bf16 q/k/v and
    # takes a bf16-rounded output back. The "reference" run does exactly that with dense
    # fp32 attention over meta.dense(), so kernel vs reference isolates the kernel's own
    # numerics; gold (fp32 eager on dense masks) bounds the bf16 rounding on top.
    def bf16_route(module, query, key, value, attention_mask, **kwargs):
        q, k, v = (t.bfloat16() for t in (query, key, value))
        if mode["name"] == "kernel":
            out = route(module, q, k, v, attention_mask, **kwargs)
            if out is None:
                return None
            calls["kernel"] += 1
            return out[0].float(), None
        meta = getattr(attention_mask, SEGMENTS_ATTR, None)
        if meta is None or query.shape[-1] != 512:
            return None
        calls[mode["name"]] += 1
        if mode["name"] == "flex":
            from transformers.integrations.flex_attention import flex_attention_forward

            out, _ = flex_attention_forward(
                module, q, k, v, attention_mask, scaling=kwargs.get("scaling"),
                kernel_options=FLEX_KERNEL_OPTIONS,
            )  # fmt: skip
            return out.float(), None
        out = _reference(q.float(), k.float(), v.float(), meta, kwargs.get("scaling"))
        return out.bfloat16().float().transpose(1, 2).contiguous(), None

    lh.flash_d512_segment_route = bf16_route
    try:
        kernel = EncoderCanvasBackend(32, 2, attention_backend="flex_attention").pack(
            batch
        )
        for masks in (kernel.encoder_attention_mask, kernel.decoder_attention_mask):
            assert getattr(masks["full_attention"], SEGMENTS_ATTR, None) is not None
            assert getattr(masks["sliding_attention"], SEGMENTS_ATTR, None) is None
        moved = kernel.to(device)
        assert (
            getattr(moved.decoder_attention_mask["full_attention"], SEGMENTS_ATTR)
            is not None
        )
        torch.manual_seed(7)
        reference = DiffusionGemmaForBlockDiffusion(config).eval()
        assert reference.model.decoder.layers[1].self_attn.head_dim == 512
        runs = {}
        for name, backend, packed in (
            ("gold", "eager", dense),
            ("reference", "flex_attention", kernel),
            ("flex", "flex_attention", kernel),
            ("kernel", "flex_attention", kernel),
        ):
            mode["name"] = name
            model = copy.deepcopy(reference).to(device).train()
            for cfg in (
                model.config,
                model.model.encoder.config,
                model.model.encoder.language_model.config,
                model.model.decoder.config,
                model.model.decoder.text_config,
            ):
                cfg._attn_implementation = backend
            logits = forward_packed_encoder_canvas(
                model,
                encoder_input_ids=packed.encoder_input_ids,
                encoder_attention_mask=packed.encoder_attention_mask,
                encoder_position_ids=packed.encoder_position_ids,
                decoder_input_ids=packed.canvas_clean_ids,
                decoder_attention_mask=packed.decoder_attention_mask,
                decoder_position_ids=packed.canvas_position_ids,
                # sm120's 99 KB shared memory needs the configs' small flex blocks
                # (the head_dim 16 sliding layers stay on flex).
                kernel_options=FLEX_KERNEL_OPTIONS if backend != "eager" else None,
            ).logits
            width = dense.canvas_clean_ids.shape[1]
            valid = dense.canvas_semantic_validity[0]
            logits = logits[:, :width][:, valid]
            logits.square().mean().backward()
            grads = {
                key: param.grad
                for key, param in model.named_parameters()
                if param.grad is not None
            }
            runs[name] = (logits.detach(), grads)
    finally:
        lh.flash_d512_segment_route = route
        lh.set_large_head_policy("sdpa")
        lh.unpatch_flex_large_head()
    # One encoder and one decoder global layer went through each path.
    assert calls == {"kernel": 2, "reference": 2, "flex": 2}

    def rel(value, ref):
        return ((value - ref).norm() / ref.norm().clamp_min(1e-12)).item()

    def compare(name, against):
        (ref_logits, ref_grads), (logits, grads) = runs[against], runs[name]
        assert grads.keys() == ref_grads.keys()
        return {"logits": rel(logits, ref_logits)} | {
            key: rel(grads[key], ref_grads[key])
            for key in ref_grads
            # top-1 of 2 experts: the renormalised routing weight is constant 1, so the
            # router's true gradient is ~0 and its relative error is pure noise.
            if ".router." not in key
        }

    kernel_vs_reference = compare("kernel", "reference")
    flex_vs_reference = compare("flex", "reference")
    kernel_vs_gold = compare("kernel", "gold")
    worst = max(kernel_vs_reference, key=kernel_vs_reference.get)
    print(
        "vs bf16-input fp32 reference, worst:", worst,
        "kernel", kernel_vs_reference[worst], "flex", flex_vs_reference[worst],
        "| logits kernel", kernel_vs_reference["logits"],
        "flex", flex_vs_reference["logits"],
        "| kernel vs fp32 gold logits", kernel_vs_gold["logits"],
    )  # fmt: skip
    # The kernel's own rounding (bf16 P and dS operands, as in flash attention) is no
    # larger than flex's on the same bf16 inputs.
    for key, flex_error in flex_vs_reference.items():
        assert kernel_vs_reference[key] <= 2 * flex_error + 2e-3, (
            key,
            kernel_vs_reference[key],
            flex_error,
        )
    assert kernel_vs_gold["logits"] <= 1e-2
