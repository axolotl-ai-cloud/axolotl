"""
Guards that sample packing does not leak attention across document boundaries.

Axolotl encodes each packed document as a distinct integer id in the 2D attention
mask, but modern transformers casts that mask to bool before flash attention and
instead derives per-document ``cu_seqlens`` from the (per-document resetting)
``position_ids`` (`_is_packed_sequence` / `_prepare_from_posids`). This is why the
`_get_unpad_data` override was dropped for in-tree models. If that native path ever
regresses, a packed forward would attend across documents and this test would fail.
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.testing_utils import require_flash_attn, require_torch_gpu


@require_torch_gpu
@require_flash_attn
def test_packed_forward_matches_per_document_forward():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=128,
        attn_implementation="flash_attention_2",
    )
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()

    # Two documents packed into a single row (batch_size == 1), exactly the shape
    # the multipack collator emits: segment-id attention mask + resetting position ids.
    doc1 = [11, 12, 13, 14, 15]
    doc2 = [21, 22, 23, 24]
    input_ids = torch.tensor([doc1 + doc2], device="cuda")
    attention_mask = torch.tensor([[1] * len(doc1) + [2] * len(doc2)], device="cuda")
    position_ids = torch.tensor(
        [list(range(len(doc1))) + list(range(len(doc2)))], device="cuda"
    )

    with torch.no_grad():
        packed = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits
        ref1 = model(
            input_ids=torch.tensor([doc1], device="cuda"),
            position_ids=torch.tensor([list(range(len(doc1)))], device="cuda"),
        ).logits
        ref2 = model(
            input_ids=torch.tensor([doc2], device="cuda"),
            position_ids=torch.tensor([list(range(len(doc2)))], device="cuda"),
        ).logits

    reference = torch.cat([ref1, ref2], dim=1)
    # Each document must see only itself; flash-attn isn't bit-exact across tiling shapes.
    assert torch.allclose(packed, reference, atol=1e-2)
