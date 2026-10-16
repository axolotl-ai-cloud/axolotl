"""Tests that Inkling isolates packed documents from position_ids."""

import pytest
import torch

modeling = pytest.importorskip("transformers.models.inkling.modeling_inkling")

from axolotl.monkeypatch.models.inkling.modeling import (  # noqa: E402
    patch_inkling_modeling_packing,
)


def test_packed_documents_match_running_alone():
    patch_inkling_modeling_packing()
    config = modeling.InklingTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        swa_head_dim=16,
        sliding_window_size=8,
        d_rel=4,
        rel_extent=16,
        log_scaling_n_floor=2,
        layer_types=["hybrid_sliding", "hybrid"],
        mlp_layer_types=["dense", "dense"],
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = modeling.InklingTextModel(config).eval()
    docs = [torch.randint(0, 64, (1, n)) for n in (7, 5)]
    position_ids = torch.cat([torch.arange(doc.shape[1]) for doc in docs])[None]

    with torch.no_grad():
        packed = model(
            torch.cat(docs, dim=1), position_ids=position_ids
        ).last_hidden_state
        alone = torch.cat([model(doc).last_hidden_state for doc in docs], dim=1)

    torch.testing.assert_close(packed, alone)
