"""Rollout Routing Replay (R3): force vLLM's MoE expert choices in the training forward.

The replayed top-k *selection* comes from vLLM; the gate *weights* are recomputed from
the training router's logits so the router still receives gradients.
"""

import copy
import re
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from axolotl.utils.routed_experts import decode_routed_experts

_LAYER_IDX = re.compile(r"layers\.(\d+)\.")


class _CapturingSession:
    """Remembers each completion's routes, keyed by the identity of its token-id list.

    TRL passes the parsed ``token_ids`` lists through to ``completion_ids`` unchanged, so
    identity matching is exact even when requests run concurrently and finish out of order.
    """

    def __init__(self, session):
        self._session = session
        self._routes: dict[int, tuple[list, str]] = {}

    def post(self, *args, **kwargs):
        resp = self._session.post(*args, **kwargs)
        if resp.status_code == 200:
            payload = resp.json()
            # vLLM OpenAI server: per choice; axolotl /generate/ server: parallel list.
            pairs = [
                (c.get("token_ids"), c.get("routed_experts"))
                for c in payload.get("choices", [])
            ]
            if payload.get("routed_experts") is not None:
                pairs += zip(
                    payload["completion_ids"], payload["routed_experts"], strict=True
                )
            for ids, routes in pairs:
                if ids is not None and routes is not None:
                    self._routes[id(ids)] = (ids, routes)
            resp.json = lambda: payload
        return resp

    def routes_for(self, completion_ids) -> list[str] | None:
        routes = []
        for ids in completion_ids:
            hit = self._routes.get(id(ids))
            if hit is None:
                return None
            routes.append(hit[1])
        return routes


def capturing_client(client):
    """Shallow copy of a TRL ``VLLMClient`` whose responses keep ``routed_experts``.

    TRL's client returns a fixed set of keys; a private copy avoids racing the main
    thread's weight-sync requests on the shared session.
    """
    proxy = copy.copy(client)
    proxy.session = _CapturingSession(client.session)
    return proxy


def pad_routed_experts(encoded, prompt_lens, P, C) -> torch.Tensor:
    """Align vLLM's per-sequence ``[p + c - 1, layers, k]`` to ``[B, P + C, layers, k]``.

    Prompts are left-padded to ``P`` and completions right-padded to ``C``, so each
    sequence occupies a contiguous span starting at ``P - p``. ``-1`` means no replay.
    """
    arrs = [decode_routed_experts(e) for e in encoded]
    out = torch.full((len(arrs), P + C, *arrs[0].shape[1:]), -1, dtype=torch.int16)
    for i, (arr, p) in enumerate(zip(arrs, prompt_lens, strict=True)):
        arr = torch.from_numpy(arr.astype(np.int16))[: p + C]
        out[i, P - p : P - p + arr.shape[0]] = arr
    return out


class RoutingReplay:
    """Hooks every MoE block so ``experts(h, top_k_index, top_k_weights)`` uses replayed ids."""

    def __init__(self, model: nn.Module):
        self.batch: torch.Tensor | None = None
        # Kept until after backward so gradient-checkpoint recompute replays the same routes.
        self.tokens: torch.Tensor | None = None
        self.matched = self.total = 0
        self.num_blocks = 0
        for name, block in model.named_modules():
            gate = getattr(block, "gate", None) or getattr(block, "router", None)
            experts = getattr(block, "experts", None)
            m = _LAYER_IDX.search(name + ".")
            if m and isinstance(gate, nn.Module) and isinstance(experts, nn.Module):
                self._hook(block, gate, experts, int(m.group(1)))
                self.num_blocks += 1
        if self.num_blocks == 0:
            raise ValueError("routing_replay: no MoE blocks found in model")

    @contextmanager
    def replay(self, routed_experts: torch.Tensor | None):
        """Set the padded ``[B, S, layers, k]`` routes aligned with the forward's input_ids."""
        self.batch = routed_experts
        try:
            yield
        finally:
            self.batch = None
            if not torch.is_grad_enabled():
                self.tokens = None

    def select(self, start: int, end: int, valid: torch.Tensor | None = None):
        """Pick routes for rows ``start:end``; ``valid`` mirrors padding-free flattening."""
        if self.batch is None:
            self.tokens = None
            return
        if torch.is_grad_enabled() and start != 0:
            raise RuntimeError("routing_replay needs one grad forward per backward")
        r = self.batch[start:end]
        r = r[valid.to(r.device)] if valid is not None else r.flatten(0, 1)
        self.tokens = r

    def pop_agreement(self) -> float | None:
        if not self.total:
            return None
        rate = float(self.matched) / float(self.total)
        self.matched = self.total = 0
        return rate

    def _hook(self, block, gate, experts, layer_idx):
        logits = {}

        def gate_hook(_mod, _args, out):
            logits["v"] = out[0] if isinstance(out, tuple) else out

        def experts_hook(_mod, args, kwargs):
            if self.tokens is None:
                return None
            hidden, idx, weights = args[:3]
            replay = self.tokens[:, layer_idx].to(idx.device, torch.long)
            if replay.shape != idx.shape:
                raise ValueError(
                    f"routing_replay: layer {layer_idx} expects {tuple(idx.shape)} "
                    f"routes, got {tuple(replay.shape)}"
                )
            valid = (replay >= 0).all(-1, keepdim=True)
            same = (replay.sort(-1).values == idx.sort(-1).values).all(-1, keepdim=True)
            self.matched += (same & valid).sum()
            self.total += valid.sum()
            replay = torch.where(valid, replay, idx)
            new_w = replay_weights(block, gate, logits.pop("v"), replay)
            new_w = torch.where(valid, new_w.to(weights.dtype), weights)
            return (hidden, replay.to(idx.dtype), new_w, *args[3:]), kwargs

        gate.register_forward_hook(gate_hook)
        experts.register_forward_pre_hook(experts_hook, with_kwargs=True)


def replay_weights(block, gate, router_logits, idx) -> torch.Tensor:
    """The model's own top-k weight formula, evaluated at forced indices ``idx``."""
    gate = gate.get_base_layer() if hasattr(gate, "get_base_layer") else gate

    # Routing attributes live on the router or the block depending on model/version.
    def attr(name, default):
        return getattr(gate, name, getattr(block, name, default))

    sigmoid = attr("e_score_correction_bias", None) is not None
    logits = router_logits.float().reshape(idx.shape[0], -1)
    probs = logits.sigmoid() if sigmoid else logits.softmax(-1)
    w = probs.gather(1, idx)
    if attr("norm_topk_prob", True):
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    if sigmoid:
        w = w * attr("routed_scaling_factor", 1.0)
    return w
