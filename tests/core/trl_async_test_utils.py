"""CPU stand-ins for async GRPO tests: TRL's FA3 tree attention, a rollout worker
fed from a list, and a weight transfer that records what it was sent."""

import queue

import torch


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
        out[qs:qe] = torch.nn.functional.scaled_dot_product_attention(
            q[qs:qe].transpose(0, 1),
            k[ks:ke].repeat_interleave(groups, 1).transpose(0, 1),
            v[ks:ke].repeat_interleave(groups, 1).transpose(0, 1),
            attn_mask=mask,
            scale=scaling,
        ).transpose(0, 1)
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
