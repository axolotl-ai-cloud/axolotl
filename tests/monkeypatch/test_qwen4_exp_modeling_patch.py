"""Tests for the Qwen4-Exp (Qwen3.8-Flash-Next) QSA and sample-packing monkeypatches."""

import pytest
import torch
from transformers.masking_utils import create_causal_mask

qwen4_exp = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

from transformers.models.qwen4_exp.configuration_qwen4_exp import (  # noqa: E402
    Qwen4ExpTextConfig,
)

HIDDEN = 32
INDEX_HEAD_DIM = 16
DOC_LENS = [7, 11, 5]
# the shipped checkpoint's config.eos_token_id (<|endoftext|>)
EOS_TOKEN_ID = 248044


def _config(**overrides):
    kwargs = {
        "vocab_size": 512,
        "hidden_size": HIDDEN,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "hc_count": 2,
        "hc_lowrank": 8,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_conv_kernel_dim": 4,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": INDEX_HEAD_DIM,
        "indexer_budget": 8,
        "indexer_compress_ratio": 4,
        "ple_layer_ids": [2],
        "ple_embed_dim": HIDDEN,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 2003,
        "eos_token_id": 7,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 1.0,
        },
        "max_position_embeddings": 512,
    }
    kwargs.update(overrides)
    config = Qwen4ExpTextConfig(**kwargs)
    config._attn_implementation = "sdpa"
    return config


def _text_model(config, seed=0):
    torch.manual_seed(seed)
    model = qwen4_exp.Qwen4ExpTextModel(config).eval()
    for param in model.parameters():
        if param.dim() >= 2:
            torch.nn.init.normal_(param, std=0.05)
    # upstream zero-inits the PLE conv; make it live so cross-document leakage shows
    torch.nn.init.normal_(model.layers[1].ple.conv1d.weight, std=0.5)
    return model


def _varlen_stubs():
    """Reference varlen kernels standing in for FLA: upstream torch ops, per segment."""

    def causal_conv1d(x, weight, bias=None, activation=None, cu_seqlens=None):
        outputs = []
        for start, end in zip(
            cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True
        ):
            segment = x[:, start:end].transpose(1, 2)
            out = qwen4_exp.causal_conv1d_fn(
                segment, weight, bias, activation=activation
            )
            outputs.append(out.transpose(1, 2))
        return torch.cat(outputs, dim=1), None

    def chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    ):
        outputs = []
        last_state = None
        for start, end in zip(
            cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True
        ):
            out, last_state = qwen4_exp.torch_chunk_gated_delta_rule(
                query[:, start:end],
                key[:, start:end],
                value[:, start:end],
                g=g[:, start:end],
                beta=beta[:, start:end],
                initial_state=None,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            outputs.append(out)
        return torch.cat(outputs, dim=1), last_state

    return causal_conv1d, chunk_gated_delta_rule


def _packed_inputs(doc_lens=DOC_LENS, seed=1):
    torch.manual_seed(seed)
    docs = [torch.randint(10, 500, (1, n)) for n in doc_lens]
    input_ids = torch.cat(docs, dim=1)
    position_ids = torch.cat([torch.arange(n) for n in doc_lens]).view(1, -1)
    return docs, input_ids, position_ids


class _RecordingEmbedding(torch.nn.Module):
    """Captures the hashed n-gram ids the embedding is indexed with."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.weight = inner.weight
        self.last_ids = None

    def forward(self, ngram_ids):
        self.last_ids = ngram_ids.clone()
        return self.inner(ngram_ids)


def _recording_ngram(ngram_size=3):
    """An n-gram module using the shipped checkpoint's eos_token_id and vocab_size.

    `unigram_vocab_size` sizes the hash multipliers, so both have to be real for the
    eos oracle to mean anything.
    """
    config = _config(
        vocab_size=248320, eos_token_id=EOS_TOKEN_ID, ngram_size=ngram_size
    )
    torch.manual_seed(0)
    ngram = qwen4_exp.Qwen4ExpTextNGramEmbedding(config, HIDDEN, 1, 0)
    ngram.ngram_embedding = _RecordingEmbedding(ngram.ngram_embedding)
    return ngram, ngram.ngram_embedding


def _oracle_ngram_ids(docs, ngram_size=3):
    """Stock forward on documents terminated by a literal eos, separator rows dropped.

    `_shift_right_ignore_eos` puts the segment start at `previous_eos + 1`, so the
    separator is the last token of the document it follows.
    """
    ngram, recorder = _recording_ngram(ngram_size)
    separator = torch.full((docs[0].shape[0], 1), EOS_TOKEN_ID)
    pieces, keep = [], []
    for index, doc in enumerate(docs):
        if index:
            pieces.append(separator)
            keep.append(False)
        pieces.append(doc)
        keep.extend([True] * doc.shape[1])

    with torch.no_grad():
        qwen4_exp.Qwen4ExpTextNGramEmbedding.forward(
            ngram, torch.cat(pieces, dim=1), None
        )
    return recorder.last_ids[:, torch.tensor(keep)]


def _patched_ngram_ids(docs, ngram_size=3):
    """Patched forward on the same documents packed by position_ids, no separators."""
    from axolotl.monkeypatch.models.qwen4_exp import modeling as patch_module

    ngram, recorder = _recording_ngram(ngram_size)
    input_ids = torch.cat(docs, dim=1)
    position_ids = (
        torch.cat([torch.arange(doc.shape[1]) for doc in docs])
        .view(1, -1)
        .expand(docs[0].shape[0], -1)
        .contiguous()
    )

    unpatch = patch_module.patch_qwen4_exp_modeling_packing()
    try:
        with torch.no_grad():
            ngram(input_ids, None, position_ids)
    finally:
        unpatch()
    return recorder.last_ids


@pytest.fixture(name="packing_patch")
def fixture_packing_patch(monkeypatch):
    """Apply the packing patch with reference varlen kernels, then restore."""
    from axolotl.monkeypatch.models.qwen4_exp import modeling as patch_module

    monkeypatch.setattr(patch_module, "_load_fla", _varlen_stubs)
    unpatch = patch_module.patch_qwen4_exp_modeling_packing()
    yield patch_module
    unpatch()


class TestQwen4ExpPackingPatch:
    """Packing patches for linear attention and the PLE n-gram layer."""

    def test_patch_and_unpatch(self):
        from axolotl.monkeypatch.models.qwen4_exp.modeling import (
            patch_qwen4_exp_modeling_packing,
        )

        originals = (
            qwen4_exp.Qwen4ExpTextModel.forward,
            qwen4_exp.Qwen4ExpTextDecoderLayer.forward,
            qwen4_exp.Qwen4ExpTextGatedDeltaNet.forward,
            qwen4_exp.Qwen4ExpTextNGramEmbedding.forward,
            qwen4_exp.Qwen4ExpTextPLELayer.forward,
        )

        unpatch = patch_qwen4_exp_modeling_packing()
        assert qwen4_exp.Qwen4ExpTextDecoderLayer.forward != originals[1]
        assert patch_qwen4_exp_modeling_packing() is None, "patch should be idempotent"

        unpatch()
        assert (
            qwen4_exp.Qwen4ExpTextModel.forward,
            qwen4_exp.Qwen4ExpTextDecoderLayer.forward,
            qwen4_exp.Qwen4ExpTextGatedDeltaNet.forward,
            qwen4_exp.Qwen4ExpTextNGramEmbedding.forward,
            qwen4_exp.Qwen4ExpTextPLELayer.forward,
        ) == originals

    def test_unpacked_forward_is_unchanged(self, monkeypatch):
        """Without packed position_ids every patched path defers to the stock one."""
        from axolotl.monkeypatch.models.qwen4_exp import modeling as patch_module

        model = _text_model(_config())
        input_ids = torch.randint(10, 500, (1, 12))
        position_ids = torch.arange(12).view(1, -1)

        with torch.no_grad():
            stock = model(
                input_ids=input_ids, position_ids=position_ids, use_cache=False
            ).last_hidden_state

        monkeypatch.setattr(patch_module, "_load_fla", _varlen_stubs)
        unpatch = patch_module.patch_qwen4_exp_modeling_packing()
        try:
            with torch.no_grad():
                patched = model(
                    input_ids=input_ids, position_ids=position_ids, use_cache=False
                ).last_hidden_state
        finally:
            unpatch()

        assert torch.equal(patched, stock)

    def test_packed_documents_are_isolated(self, packing_patch):
        """Every packed document reproduces its standalone forward."""
        model = _text_model(_config())
        docs, input_ids, position_ids = _packed_inputs()

        with torch.no_grad():
            packed = model(
                input_ids=input_ids, position_ids=position_ids, use_cache=False
            ).last_hidden_state
            solo = torch.cat(
                [
                    model(
                        input_ids=doc,
                        position_ids=torch.arange(doc.shape[1]).view(1, -1),
                        use_cache=False,
                    ).last_hidden_state
                    for doc in docs
                ],
                dim=1,
            )

        assert torch.allclose(packed, solo, atol=1e-5)

    def test_packing_without_fla_raises(self, monkeypatch):
        from axolotl.monkeypatch.models.qwen4_exp import modeling as patch_module

        monkeypatch.setattr(patch_module, "_load_fla", lambda: (None, None))
        unpatch = patch_module.patch_qwen4_exp_modeling_packing()
        try:
            model = _text_model(_config())
            _, input_ids, position_ids = _packed_inputs()
            with pytest.raises(RuntimeError, match="flash-linear-attention"):
                model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        finally:
            unpatch()

    def test_ngram_embedding_segments_on_position_ids(self, packing_patch):
        """The n-gram context resets at packed boundaries, which eos matching cannot see."""
        config = _config()
        torch.manual_seed(0)
        ngram = qwen4_exp.Qwen4ExpTextNGramEmbedding(config, HIDDEN, 1, 0)
        docs, input_ids, position_ids = _packed_inputs()

        with torch.no_grad():
            packed = ngram(input_ids, None, position_ids)
            solo = torch.cat(
                [
                    ngram(doc, None, torch.arange(doc.shape[1]).view(1, -1))
                    for doc in docs
                ],
                dim=1,
            )
            without_positions = ngram(input_ids, None, None)

        assert torch.equal(packed, solo)
        assert not torch.equal(without_positions, solo)

    @pytest.mark.parametrize(
        "doc_lens, batch, ngram_size",
        [
            ([7, 11, 5], 1, 3),
            ([1, 1, 1, 1], 1, 3),
            ([2, 3, 1], 1, 3),
            ([9], 1, 3),
            ([1, 9, 4], 1, 3),
            ([4, 6, 5], 3, 3),
            ([6, 8], 1, 5),
        ],
    )
    def test_ngram_matches_eos_separated_oracle(self, doc_lens, batch, ngram_size):
        """Upstream's eos segmentation is the reference: reproduce it from position_ids.

        Feeding upstream a batch whose documents are terminated by a literal
        `eos_token_id` gives the segmentation the patch has to reproduce from
        position_ids on the same batch without those separators.
        """
        docs = [torch.randint(10, 200000, (batch, n)) for n in doc_lens]
        oracle = _oracle_ngram_ids(docs, ngram_size)
        assert torch.equal(oracle, _patched_ngram_ids(docs, ngram_size))

    def test_ngram_keeps_eos_boundaries_inside_a_document(self):
        """A literal eos inside a document still segments, so the two signals combine."""
        torch.manual_seed(3)
        docs = [
            torch.randint(10, 200000, (1, 5)),
            torch.randint(10, 200000, (1, 9)),
            torch.randint(10, 200000, (1, 6)),
        ]
        docs[1][0, 4] = EOS_TOKEN_ID

        assert torch.equal(_oracle_ngram_ids(docs), _patched_ngram_ids(docs))

    def test_ple_short_conv_does_not_cross_documents(self, packing_patch):
        """The dilated depthwise conv reaches back 9 tokens; taps must stop at the boundary."""
        config = _config()
        torch.manual_seed(0)
        ple = qwen4_exp.Qwen4ExpTextPLELayer(
            config, layer_idx=1, ple_layer_index=0
        ).eval()
        torch.nn.init.normal_(ple.conv1d.weight, std=0.5)

        _, _, position_ids = _packed_inputs()
        hidden = torch.randn(1, sum(DOC_LENS), config.hidden_size * config.hc_count)

        with torch.no_grad():
            packed = ple._short_conv(hidden, None, position_ids)
            offset = 0
            parts = []
            for length in DOC_LENS:
                parts.append(
                    ple._short_conv(hidden[:, offset : offset + length], None, None)
                )
                offset += length
            solo = torch.cat(parts, dim=1)
            without_positions = ple._short_conv(hidden, None, None)

        assert torch.allclose(packed, solo, atol=1e-5)
        assert not torch.allclose(without_positions, solo, atol=1e-5)

    def test_sdpa_varlen_is_skipped(self):
        """varlen drops the 4D mask, which the QSA indexer requires."""
        from types import SimpleNamespace
        from unittest.mock import patch as mock_patch

        from axolotl.loaders.patch_manager import PatchManager

        def _cfg(model_config_type):
            return SimpleNamespace(
                sdpa_varlen=None,
                attn_implementation="sdpa",
                sample_packing=True,
                model_config_type=model_config_type,
            )

        # `varlen_available` is only reached past the capability guard
        with mock_patch(
            "axolotl.monkeypatch.attention.sdpa_varlen.varlen_available",
            return_value=False,
        ) as available:
            PatchManager(
                cfg=_cfg("qwen4_exp"), model_config=object()
            )._apply_sdpa_varlen_patch()
            assert not available.called

            PatchManager(
                cfg=_cfg("llama"), model_config=object()
            )._apply_sdpa_varlen_patch()
            assert available.called


def _build_indexer(budget=16, ratio=4, seed=0):
    torch.manual_seed(seed)
    config = _config(indexer_budget=budget, indexer_compress_ratio=ratio)
    indexer = qwen4_exp.Qwen4ExpTextQSAIndexer(config, layer_idx=0)
    for param in indexer.parameters():
        torch.nn.init.normal_(param, std=0.5)
    return config, indexer, qwen4_exp.Qwen4ExpTextRotaryEmbedding(config)


def _indexer_inputs(config, rotary, batch, seq, attention_mask=None, position_ids=None):
    hidden = torch.randn(batch, seq, HIDDEN)
    if position_ids is None:
        position_ids = torch.arange(seq).view(1, -1).expand(batch, -1).contiguous()
    mask = create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
        allow_is_causal_skip=False,
    )
    return hidden, rotary(hidden, position_ids[None].expand(3, -1, -1)), mask


@pytest.fixture(name="original_qsa_forward")
def fixture_original_qsa_forward():
    original = qwen4_exp.Qwen4ExpTextQSAIndexer.forward
    yield original
    qwen4_exp.Qwen4ExpTextQSAIndexer.forward = original


class TestQwen4ExpQSAIndexerPatch:
    """QSA indexer vectorization patch."""

    def test_patch_and_unpatch(self, original_qsa_forward):
        from axolotl.monkeypatch.models.qwen4_exp.modeling import (
            patch_qwen4_exp_qsa_indexer,
        )

        unpatch = patch_qwen4_exp_qsa_indexer()
        assert qwen4_exp.Qwen4ExpTextQSAIndexer.forward != original_qsa_forward
        assert patch_qwen4_exp_qsa_indexer() is None, "patch should be idempotent"

        unpatch()
        assert qwen4_exp.Qwen4ExpTextQSAIndexer.forward == original_qsa_forward

    @pytest.mark.parametrize("seq", [1, 3, 5, 8, 17, 33])
    @pytest.mark.parametrize("budget", [8, 16])
    def test_matches_original_causal(self, original_qsa_forward, seq, budget):
        from axolotl.monkeypatch.models.qwen4_exp.modeling import qsa_indexer_forward

        config, indexer, rotary = _build_indexer(budget=budget)
        hidden, position_embeddings, mask = _indexer_inputs(config, rotary, 2, seq)

        with torch.no_grad():
            expected = original_qsa_forward(
                indexer, hidden, position_embeddings, mask, None
            )
            actual = qsa_indexer_forward(
                indexer, hidden, position_embeddings, mask, None
            )
        assert torch.equal(expected, actual)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_matches_original_padded(self, original_qsa_forward, side):
        from axolotl.monkeypatch.models.qwen4_exp.modeling import qsa_indexer_forward

        config, indexer, rotary = _build_indexer(budget=8)
        attention_mask = torch.ones(3, 33, dtype=torch.long)
        for batch_idx in range(3):
            if side == "right":
                attention_mask[batch_idx, 33 - (batch_idx + 1) :] = 0
            else:
                attention_mask[batch_idx, : batch_idx + 1] = 0
        hidden, position_embeddings, mask = _indexer_inputs(
            config, rotary, 3, 33, attention_mask
        )

        with torch.no_grad():
            expected = original_qsa_forward(
                indexer, hidden, position_embeddings, mask, None
            )
            actual = qsa_indexer_forward(
                indexer, hidden, position_embeddings, mask, None
            )
        assert torch.equal(expected, actual)

    @pytest.mark.parametrize(
        "doc_lens", [[4, 4, 4, 4], [5, 5, 5, 5], [3, 7, 1, 13, 8], [17, 19, 23]]
    )
    def test_matches_original_sample_packed(self, original_qsa_forward, doc_lens):
        """Block-diagonal document mask: doc starts land on every phase mod compress_ratio."""
        from axolotl.monkeypatch.models.qwen4_exp.modeling import qsa_indexer_forward

        config, indexer, rotary = _build_indexer(budget=8)
        position_ids = torch.cat([torch.arange(n) for n in doc_lens]).view(1, -1)
        hidden, position_embeddings, mask = _indexer_inputs(
            config, rotary, 1, sum(doc_lens), position_ids=position_ids
        )

        with torch.no_grad():
            expected = original_qsa_forward(
                indexer, hidden, position_embeddings, mask, None
            )
            actual = qsa_indexer_forward(
                indexer, hidden, position_embeddings, mask, None
            )
        assert torch.equal(expected, actual)

    def test_falls_back_on_non_contiguous_mask(self, original_qsa_forward):
        """A mask with holes breaks the contiguous-run assumption; the patch defers to the loop."""
        from axolotl.monkeypatch.models.qwen4_exp.modeling import qsa_indexer_forward

        config, indexer, rotary = _build_indexer(budget=8)
        hidden, position_embeddings, mask = _indexer_inputs(config, rotary, 2, 32)
        mask = mask.clone()  # create_causal_mask returns a broadcast view
        mask[:, :, :, 5] = False

        with torch.no_grad():
            expected = original_qsa_forward(
                indexer, hidden, position_embeddings, mask, None
            )
            actual = qsa_indexer_forward(
                indexer, hidden, position_embeddings, mask, None
            )
        assert torch.equal(expected, actual)

    def test_patch_manager_applies_qsa_without_packing(self, original_qsa_forward):
        """The QSA rewrite is a pure speedup, so it must not be gated on sample_packing."""
        from types import SimpleNamespace

        from axolotl.loaders.patch_manager import PatchManager
        from axolotl.monkeypatch.models.qwen4_exp import modeling as patch_module

        cfg = SimpleNamespace(
            fused_attn_kernel=False,
            model_config_type="qwen4_exp",
            llama4_linearized_experts=False,
            sample_packing=False,
            context_parallel_size=1,
            attn_uses_flash_lib=False,
            is_multimodal=False,
            use_kernels=False,
            inference=False,
        )
        try:
            PatchManager(
                cfg=cfg, model_config=object()
            )._apply_model_support_pre_load_hook()
            assert qwen4_exp.Qwen4ExpTextQSAIndexer.forward != original_qsa_forward
            assert not patch_module._ORIGINALS, "packing patch must stay off"
        finally:
            patch_module._QSA_PATCHED = False

    def test_no_op_below_token_budget(self, original_qsa_forward):
        """kv_length <= token_budget selects every visible token, so the mask is unchanged."""
        from axolotl.monkeypatch.models.qwen4_exp.modeling import qsa_indexer_forward

        config, indexer, rotary = _build_indexer(budget=64)
        hidden, position_embeddings, mask = _indexer_inputs(config, rotary, 2, 32)

        with torch.no_grad():
            assert torch.equal(
                qsa_indexer_forward(indexer, hidden, position_embeddings, mask, None),
                mask,
            )

    @pytest.mark.parametrize("prefill", [6, 8, 10])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_cached_decode_matches_original(
        self, original_qsa_forward, prefill, batch_size
    ):
        """A prefill at or below token_budget must still populate the indexer cache."""
        from axolotl.monkeypatch.models.qwen4_exp.modeling import (
            patch_qwen4_exp_qsa_indexer,
        )

        config = _config(indexer_budget=8, indexer_compress_ratio=4)
        model = _text_model(config)
        input_ids = torch.randint(10, 500, (batch_size, prefill))

        def decode(steps=6):
            out = model(input_ids=input_ids, use_cache=True)
            collected, cache = [out.last_hidden_state], out.past_key_values
            for step in range(steps):
                out = model(
                    input_ids=torch.full((batch_size, 1), 123 + step),
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.tensor([prefill + step]),
                )
                collected.append(out.last_hidden_state)
                cache = out.past_key_values
            return torch.cat(collected, dim=1)

        with torch.no_grad():
            expected = decode()

        unpatch = patch_qwen4_exp_qsa_indexer()
        try:
            with torch.no_grad():
                actual = decode()
        finally:
            unpatch()

        assert torch.allclose(actual, expected, atol=1e-5)
