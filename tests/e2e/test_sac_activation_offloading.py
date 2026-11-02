"""GPU check: selective checkpointing still saves attention under TRL's activation offloader."""

import contextlib

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

BATCH, SEQ, VOCAB = 4, 128, 4096


@pytest.fixture(scope="module", autouse=True)
def patched():
    from axolotl.monkeypatch.trainer.trl_offload import patch_trl_offload_dense_layout

    patch_trl_offload_dense_layout()


class _AttentionForwards(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.name()
        if "scaled_dot_product" in name and "backward" not in name:
            self.count += 1
        return func(*args, **(kwargs or {}))


def _step(offload, selective, sac_offload=False):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import LlamaConfig, LlamaForCausalLM
    from trl.models.activation_offloading import get_act_offloading_ctx_manager

    from axolotl.monkeypatch.selective_checkpointing import (
        apply_selective_checkpointing,
    )

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
    if selective:
        apply_selective_checkpointing(model, save=["attention"], offload=sac_offload)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda")
    mask = torch.ones_like(ids)
    # padding routes SDPA to memory-efficient attention, whose backward is deterministic
    mask[1:, SEQ // 2 :] = 0
    context = (
        get_act_offloading_ctx_manager(model) if offload else contextlib.nullcontext()
    )
    recomputed = _AttentionForwards()
    with context, sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        loss = model(input_ids=ids, attention_mask=mask, labels=ids).loss
        with recomputed:
            loss.backward()
    grads = torch.cat([p.grad.float().flatten() for p in model.parameters()])
    return grads, recomputed.count


def test_plain_checkpointing_recomputes_attention():
    _, recomputed = _step(offload=True, selective=False)
    assert recomputed > 0


@pytest.mark.parametrize("sac_offload", [False, True])
def test_saved_attention_is_not_recomputed_under_trl_offload(sac_offload):
    expected, recomputed = _step(offload=False, selective=True, sac_offload=sac_offload)
    assert recomputed == 0

    actual, recomputed = _step(offload=True, selective=True, sac_offload=sac_offload)
    assert recomputed == 0
    assert torch.equal(actual, expected)
