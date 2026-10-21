import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    DeepseekV3Config,
    GraniteMoeConfig,
    Qwen3MoeConfig,
)
from trl.generation.vllm_client import VLLMClient

from axolotl.core.trainers.grpo.async_trainer import AsyncDataProducer, AsyncGRPOTrainer
from axolotl.core.trainers.grpo.routing_replay import (
    RoutingReplay,
    capturing_client,
    pad_routed_experts,
)
from axolotl.utils.schemas.trl import TRLConfig


def _encode(arr):
    """vLLM's ``numpy2base64`` wire format."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode()


COMMON = dict(
    vocab_size=64,
    hidden_size=32,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=4,
    num_experts_per_tok=2,
    attn_implementation="eager",
)
CONFIGS = {
    "softmax": lambda: Qwen3MoeConfig(
        **COMMON,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_experts=8,
        head_dim=8,
        norm_topk_prob=True,
    ),
    "sigmoid_grouped": lambda: DeepseekV3Config(
        **COMMON,
        intermediate_size=64,
        moe_intermediate_size=16,
        n_routed_experts=8,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        routed_scaling_factor=2.5,
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=8,
    ),
}


def _native_routes(model, ids):
    """[B, S, moe_layers, k] routes the model picks on its own, like vLLM returns them."""
    routes = {}
    handles = []
    for name, mod in model.named_modules():
        if hasattr(mod, "experts") and hasattr(mod, "gate"):
            layer = int(name.split("layers.")[1].split(".")[0])
            handles.append(
                mod.experts.register_forward_pre_hook(
                    lambda _m, a, layer=layer: routes.__setitem__(layer, a[1].clone())
                )
            )
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()
    out = torch.stack([routes[layer] for layer in sorted(routes)], 1)
    return out.view(*ids.shape, *out.shape[1:]).to(torch.int16)


def _forward(model, rr, ids, routes):
    with rr.replay(routes):
        rr.select(0, ids.size(0))
        return model(ids).logits


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_replay(kind):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(CONFIGS[kind]()).float()
    for name, p in model.named_parameters():
        if "e_score_correction_bias" in name:
            p.data.normal_()
    rr = RoutingReplay(model)
    ids = torch.randint(0, 64, (2, 6))
    native = _native_routes(model, ids)
    ref = model(ids).logits

    # Replaying the model's own routes is a no-op.
    torch.testing.assert_close(_forward(model, rr, ids, native), ref)
    assert rr.pop_agreement() == 1.0

    # Forced routes change the output, still train the router, and survive
    # gradient-checkpoint recompute.
    forced = torch.empty_like(native)
    forced[...] = torch.tensor([3, 5], dtype=torch.int16)
    grads = []
    for gc in (False, True):
        model.zero_grad()
        if gc:
            model.gradient_checkpointing_enable({"use_reentrant": False})
        model.train()
        out = _forward(model, rr, ids, forced)
        assert not torch.allclose(out, ref)
        out.sum().backward()
        rr.tokens = None
        grads.append(
            torch.cat(
                [p.grad.flatten() for n, p in model.named_parameters() if "gate" in n]
            )
        )
    assert grads[0].abs().sum() > 0
    torch.testing.assert_close(grads[0], grads[1])
    assert rr.pop_agreement() < 1.0


def test_rejects_unsupported_architecture():
    cfg = GraniteMoeConfig(**COMMON, intermediate_size=16, num_local_experts=4)
    with pytest.raises(ValueError, match="granitemoe"):
        RoutingReplay(AutoModelForCausalLM.from_config(cfg))


def test_rejects_routes_with_wrong_layer_count():
    rr = RoutingReplay(AutoModelForCausalLM.from_config(CONFIGS["sigmoid_grouped"]()))
    with pytest.raises(ValueError, match="3 MoE layers, model has 2"):
        with rr.replay(torch.zeros(1, 4, 3, 2, dtype=torch.int16)):
            pass


def test_pad_routed_experts():
    seqs = [
        np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2),
        np.arange(6 * 2 * 2, dtype=np.uint8).reshape(6, 2, 2),
    ]
    # prompt lens 2 and 3, completion lens 3 and 4 → vLLM returns p + c - 1 rows
    out = pad_routed_experts([_encode(s) for s in seqs], [2, 3], 3, 4)
    assert out.shape == (2, 7, 2, 2)
    assert (out[0, :1] == -1).all() and (out[0, 5:] == -1).all()
    torch.testing.assert_close(out[0, 1:5], torch.from_numpy(seqs[0]).to(torch.int16))
    torch.testing.assert_close(out[1, 0:6], torch.from_numpy(seqs[1]).to(torch.int16))
    assert (out[1, 6:] == -1).all()


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _VllmSession:
    """vLLM's native ``/v1/completions``: per-choice ``token_ids`` and ``routed_experts``."""

    def __init__(self, with_routes=True):
        self.with_routes = with_routes

    def post(self, url, json):
        assert url.endswith("/v1/completions")
        choices = [
            {
                "prompt_token_ids": prompt,
                "token_ids": [p * 10 + g],
                "logprobs": None,
                "routed_experts": f"r{p}{g}" if self.with_routes else None,
            }
            for p, prompt in enumerate(json["prompt"])
            for g in range(json["n"])
        ]
        return _Resp({"choices": choices})


def _trl_client(session):
    client = VLLMClient.__new__(VLLMClient)
    client.base_url, client.model, client.session = "http://vllm", "m", session
    return client


def test_capturing_client_reads_trl_completions():
    client = capturing_client(_trl_client(_VllmSession()))
    out = client.generate(prompts=[[1], [2]], n=2)
    assert client.session.routes_for(out["completion_ids"]) == [
        "r00",
        "r01",
        "r10",
        "r11",
    ]


def test_capturing_client_raises_without_routes():
    client = capturing_client(_trl_client(_VllmSession(with_routes=False)))
    out = client.generate(prompts=[[1]], n=2)
    with pytest.raises(RuntimeError, match="enable-return-routed-experts"):
        client.session.routes_for(out["completion_ids"])


def _r3_trainer(**kw):
    return SimpleNamespace(
        _routing_replay=object(),
        rollout_func=None,
        tools=None,
        model=SimpleNamespace(training=True),
        accelerator=SimpleNamespace(num_processes=1),
        **kw,
    )


@pytest.mark.parametrize("kw", [{"rollout_func": lambda *a: {}}, {"tools": [print]}])
def test_generate_rejects_rollout_func_and_tools(kw):
    trainer = _r3_trainer()
    vars(trainer).update(kw)
    with pytest.raises(ValueError, match="rollout_func or tools"):
        AsyncGRPOTrainer._generate_only(trainer, [{"prompt": "hi"}])


def test_training_forward_without_routes_raises():
    with pytest.raises(RuntimeError, match="no routed_experts"):
        AsyncGRPOTrainer._replaying(_r3_trainer(), None)


def test_sync_rollouts_use_routed_generation():
    inner = MagicMock()
    inner._trainer = _r3_trainer()
    inner.config.sync_warmup_rollouts = 1
    producer = AsyncDataProducer(inner, {"skip_policy_logps": True})
    producer.produce(None, 0)
    assert inner.produce.call_args.kwargs["skip_policy_logps"] is True


@pytest.mark.parametrize(
    "kw",
    [
        {"use_vllm": False},
        {"vllm_mode": "colocate"},
        {"async_prefetch": False, "use_data_producer": True},
        {"rollout_func": "my_module.rollout"},
    ],
)
def test_config_rejects_unsupported_rollout_paths(kw):
    base = {"routing_replay": True, "use_vllm": True, "async_prefetch": True}
    TRLConfig(**base)
    with pytest.raises(ValueError, match="routing_replay"):
        TRLConfig(**{**base, **kw})
