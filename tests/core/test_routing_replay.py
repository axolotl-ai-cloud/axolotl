import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, DeepseekV3Config, Qwen3MoeConfig

from axolotl.core.trainers.grpo.routing_replay import (
    RoutingReplay,
    capturing_client,
    pad_routed_experts,
)
from axolotl.utils.routed_experts import encode_routed_experts

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
    """[B, S, layers, k] routes the model picks on its own, -1 for dense layers."""
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
    k = next(iter(routes.values())).shape[-1]
    out = torch.full((ids.numel(), model.config.num_hidden_layers, k), -1)
    for layer, idx in routes.items():
        out[:, layer] = idx
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
    forced = native.clone()
    moe = native[0, 0, :, 0] >= 0
    forced[..., moe, :] = torch.tensor([3, 5], dtype=torch.int16)
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


def test_pad_routed_experts():
    seqs = [
        np.arange(4 * 2 * 2).reshape(4, 2, 2),
        np.arange(6 * 2 * 2).reshape(6, 2, 2),
    ]
    # prompt lens 2 and 3, completion lens 3 and 4 → vLLM returns p + c - 1 rows
    out = pad_routed_experts([encode_routed_experts(s) for s in seqs], [2, 3], 3, 4)
    assert out.shape == (2, 7, 2, 2)
    assert (out[0, :1] == -1).all() and (out[0, 5:] == -1).all()
    torch.testing.assert_close(out[0, 1:5], torch.from_numpy(seqs[0]).to(torch.int16))
    torch.testing.assert_close(out[1, 0:6], torch.from_numpy(seqs[1]).to(torch.int16))
    assert (out[1, 6:] == -1).all()


def test_capturing_client_matches_routes_by_completion():
    class Resp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class Session:
        def post(self, url, json):
            # vLLM OpenAI-style choice carrying its own routes
            return Resp(
                {
                    "choices": [
                        {"token_ids": [json["i"]], "routed_experts": f"r{json['i']}"}
                    ]
                }
            )

    class Client:
        session = Session()

        def chat(self, order):
            # Like trl's concurrent chat: requests finish out of order, results are reordered.
            responses = {
                i: self.session.post("/v1/chat/completions", json={"i": i}).json()
                for i in order
            }
            return {
                "completion_ids": [
                    responses[i]["choices"][0]["token_ids"] for i in sorted(order)
                ]
            }

    client = capturing_client(Client())
    out = client.chat([2, 0, 1])
    assert client.session.routes_for(out["completion_ids"]) == ["r0", "r1", "r2"]
