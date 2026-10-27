"""CPU stand-ins for async GRPO tests: TRL's FA3 tree attention, a rollout worker
fed from a list, a weight transfer that records what it was sent, and a parity
check of packed rows against samples scored alone."""

import queue

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from axolotl.core.trainers.grpo.trl_compat import unsampled_fields


def reference_tree_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    *,
    tree_kv_index,
    tree_cu_seqlens_q,
    tree_cu_seqlens_k,
    tree_max_seqlen_q,
    tree_max_seqlen_k,
    scaling=None,
    **kwargs,
):
    """Pure-torch stand-in for TRL's FA3 tree attention."""
    q, k, v = (x.transpose(1, 2)[0] for x in (query, key, value))
    k, v = k.index_select(0, tree_kv_index), v.index_select(0, tree_kv_index)
    groups = q.shape[1] // k.shape[1]
    out = torch.empty_like(q)
    for s in range(len(tree_cu_seqlens_q) - 1):
        qs, qe = int(tree_cu_seqlens_q[s]), int(tree_cu_seqlens_q[s + 1])
        ks, ke = int(tree_cu_seqlens_k[s]), int(tree_cu_seqlens_k[s + 1])
        lq, lk = qe - qs, ke - ks
        mask = torch.arange(lk)[None, :] <= (lk - lq + torch.arange(lq))[:, None]
        # Batched (4D): torch 2.13 runs 3D CPU SDPA as its math decomposition, which
        # neither selective checkpointing nor AttentionForwards can see.
        out[qs:qe] = torch.nn.functional.scaled_dot_product_attention(
            q[qs:qe].transpose(0, 1)[None],
            k[ks:ke].repeat_interleave(groups, 1).transpose(0, 1)[None],
            v[ks:ke].repeat_interleave(groups, 1).transpose(0, 1)[None],
            attn_mask=mask,
            scale=scaling,
        )[0].transpose(0, 1)
    return out.unsqueeze(0), None


class StubWorker:
    def __init__(self, samples):
        self.rollout_buffer = queue.Queue()
        self.metrics_queue = queue.Queue()
        self._samples = samples
        self.versions = []

    def start(self):
        for sample in self._samples:
            self.rollout_buffer.put(sample)

    def stop(self):
        pass

    def update_model_version(self, version):
        self.versions.append(version)

    def check_health(self, stale_after_s):
        pass


class StubTransfer:
    syncs = 0

    def init_weight_transfer(self):
        pass

    def pause(self):
        pass

    def send_weights(self, iterator):
        self.names = [name for name, _ in iterator]
        self.syncs += 1

    def resume(self):
        pass

    def destroy(self):
        pass


def _sample(ids, prompt_len, group_id):
    ids = np.asarray(ids, dtype=np.int64)
    return {
        "input_ids": ids,
        "completion_mask": np.asarray([0] * prompt_len + [1] * (len(ids) - prompt_len)),
        "old_log_probs": np.zeros(len(ids), np.float32),
        **unsampled_fields(len(ids)),
        "advantage": 1.0,
        "group_id": group_id,
    }


def branching_samples(vocab_size=64):
    """Two prompts with three rollouts each, plus one branching off a rollout."""
    rng = np.random.default_rng(0)
    samples = []
    for group_id in range(2):
        prompt = rng.integers(1, vocab_size, size=6 + group_id).tolist()
        for g in range(3):
            completion = rng.integers(1, vocab_size, size=3 + g).tolist()
            samples.append(_sample(prompt + completion, len(prompt), group_id))
        branch = samples[-1]["input_ids"][: len(prompt) + 1].tolist() + [5, 6]
        samples.append(_sample(branch, len(prompt), group_id))
    return samples


def _set_attention(model, implementation):
    if any("Attention" in type(m).__name__ for m in model.modules()):
        model.set_attn_implementation(implementation)


def assert_tree_rows_match_alone(model, samples, on_row=None):
    """Trie and chain rows score ``samples`` (log-probs and grads) as they score alone.

    ``on_row(row_tokens)`` runs after each packed forward. Returns the trie's and
    the chains' row lengths.
    """
    from tree_recurrence.integrations.trl import ChainRow, sample_boundaries
    from trl.experimental.async_grpo import packing
    from trl.experimental.async_grpo.packing import TreeRow

    packing.register_tree_attention()

    def grads():
        found = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        model.zero_grad()
        return found

    def score_row(row):
        _set_attention(model, packing.TREE_ATTENTION)
        for sample in samples:
            row.add(sample)
        packed = row.pack()
        tree = {
            "tree_kv_index": packed["tree_kv_index"],
            "tree_cu_seqlens_q": packed["tree_cu_seqlens_q"],
            "tree_cu_seqlens_k": packed["tree_cu_seqlens_k"],
            "tree_max_seqlen_q": int(packed["tree_max_seqlens"][0]),
            "tree_max_seqlen_k": int(packed["tree_max_seqlens"][1]),
        }
        if isinstance(row, ChainRow):
            tree.update(sample_boundaries(tree))
        logits = model(
            input_ids=packed["input_ids"][None],
            position_ids=packed["position_ids"][None],
            use_cache=False,
            **tree,
        ).logits[0]
        if on_row is not None:
            on_row(len(packed["input_ids"]))
        index = packed["token_index"]
        log_probs = torch.log_softmax(logits[index], -1).gather(
            1, packed["shift_labels"][index][:, None]
        )[:, 0]
        log_probs.sum().backward()
        return len(packed["input_ids"]), log_probs.detach(), grads()

    def score_alone():
        _set_attention(model, "sdpa")
        log_probs = []
        for sample in samples:
            ids = torch.as_tensor(sample["input_ids"])[None]
            logits = model(input_ids=ids, use_cache=False).logits[0, :-1]
            trained = torch.as_tensor(sample["completion_mask"][1:]).bool()
            log_probs.append(
                torch.log_softmax(logits[trained], -1).gather(
                    1, ids[0, 1:][trained][:, None]
                )[:, 0]
            )
        torch.cat(log_probs).sum().backward()
        return torch.cat(log_probs).detach(), grads()

    model.train()
    shared_len, shared, shared_grads = score_row(TreeRow())
    chain_len, chains, chain_grads = score_row(ChainRow())
    alone, alone_grads = score_alone()
    assert shared_len < chain_len
    for packed_log_probs, packed_grads in (
        (shared, shared_grads),
        (chains, chain_grads),
    ):
        torch.testing.assert_close(packed_log_probs, alone, rtol=1e-5, atol=1e-5)
        assert packed_grads.keys() == alone_grads.keys()
        for name, grad in packed_grads.items():
            torch.testing.assert_close(grad, alone_grads[name], rtol=1e-4, atol=1e-5)
    return shared_len, chain_len


def offload_activations_on_cpu(monkeypatch):
    """Run axolotl's activation offload contexts on a CPU-only host: no accelerator,
    streams or pinned memory. Tensors already on the CPU stay where they are, so this
    covers the wiring and the gradients, not the device-to-host copies."""
    from axolotl.core.trainers.mixins import activation_checkpointing

    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda *a, **k: torch.device("cpu")
    )
    monkeypatch.setattr(torch.cpu, "empty_cache", lambda: None, raising=False)
    for name in (
        "get_lora_act_offloading_ctx_manager",
        "get_act_offloading_ctx_manager",
    ):
        offloader = getattr(activation_checkpointing, name)
        monkeypatch.setattr(
            activation_checkpointing,
            name,
            lambda model, offloader=offloader: offloader(
                model, use_pin_memory=False, use_streams=False
            ),
        )


class AttentionForwards(TorchDispatchMode):
    """Counts attention forwards, including those a backward pass recomputes."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.name()
        if "scaled_dot_product" in name and "backward" not in name:
            self.count += 1
        return func(*args, **(kwargs or {}))
