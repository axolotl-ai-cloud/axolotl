"""GPU check: TRL's activation offloader leaves gradients of layout-sensitive kernels unchanged."""

import contextlib

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

BATCH, SEQ, VOCAB = 4, 64, 4096


@pytest.fixture(scope="module", autouse=True)
def patched():
    from axolotl.monkeypatch.trainer.trl_offload import (
        patch_trl_logprob_entropy_saved_layout,
        patch_trl_offload_dense_layout,
    )

    patch_trl_offload_dense_layout()
    patch_trl_logprob_entropy_saved_layout()


def _offload(enabled, use_streams=True):
    from trl.models.activation_offloading import OffloadActivations

    if not enabled:
        return contextlib.nullcontext()
    return OffloadActivations(use_streams=use_streams, min_offload_size=0)


def _logits_grad(offload, use_streams=True):
    from trl.trainer.utils import selective_log_softmax

    torch.manual_seed(0)
    # peaked like a trained model's, so a misaligned read shows in the gradient
    logits = (torch.randn(BATCH, SEQ, VOCAB, device="cuda") * 8).bfloat16()
    logits.requires_grad_()
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda")
    with _offload(offload, use_streams):
        # shifted views: the offloader saves contiguous copies of both
        logps = selective_log_softmax(logits[:, :-1], labels[:, 1:])
        logps.float().sum().backward()
    return logits.grad


@pytest.mark.parametrize("use_streams", [True, False])
def test_offloaded_shifted_logits_keep_their_gradients(use_streams):
    assert torch.equal(_logits_grad(True, use_streams), _logits_grad(False))


def _sdpa_padded_grads(offload, use_streams=True):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=VOCAB,
            attn_implementation="sdpa",
        )
    ).to("cuda", torch.bfloat16)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda")
    mask = torch.ones_like(ids)
    mask[1:, SEQ // 2 :] = 0
    # a padding mask routes SDPA to memory-efficient attention, which saves transposed q / out
    with _offload(offload, use_streams), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        model(input_ids=ids, attention_mask=mask, labels=ids).loss.backward()
    return torch.cat([p.grad.float().flatten() for p in model.parameters()])


@pytest.mark.parametrize("use_streams", [True, False])
def test_offloaded_sdpa_efficient_attention_keeps_its_gradients(use_streams):
    assert torch.equal(_sdpa_padded_grads(True, use_streams), _sdpa_padded_grads(False))
