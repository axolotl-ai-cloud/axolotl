# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Tree packing for tests: a packer laying rows out as TRL's ``TreeRow.pack`` does,
a PyTorch stand-in for TRL's tree attention, and a check that a model scores
tree-packed samples as it scores them alone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

TREE_ATTENTION = "tree_reference"


@dataclass
class Packed:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    kv_index: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    parents: torch.Tensor
    depths: torch.Tensor
    # each sample's tokens' positions in the row
    positions: list[torch.Tensor]

    def kwargs(self, fmt: str = "trl-1.16") -> dict:
        """The forward kwargs of TRL 1.16, of huggingface/trl#7686 (after its
        ``unpad_tree``), or the segment tree alone."""
        lengths = self.cu_seqlens_q.diff()
        paths = self.cu_seqlens_k.diff()
        if fmt == "parents":
            return {
                "tree_cu_seqlens_q": self.cu_seqlens_q,
                "tree_segment_parents": self.parents,
            }
        tree = {
            "tree_kv_index": self.kv_index,
            "tree_cu_seqlens_q": self.cu_seqlens_q,
            "tree_cu_seqlens_k": self.cu_seqlens_k,
            "tree_max_seqlen_q": int(lengths.max()),
            "tree_max_seqlen_k": int(paths.max()),
        }
        if fmt == "trl-7686":
            depths = self.depths.tolist()
            tree.update(
                tree_segment_parents=self.parents,
                tree_segment_depths=self.depths,
                tree_level_sizes=[depths.count(d) for d in range(max(depths) + 1)],
                tree_n_chunks=int(((lengths + 63) // 64).sum()),
            )
        else:
            assert fmt == "trl-1.16", fmt
        return tree


def pack(sequences, share: bool = True) -> Packed:
    """Pack token sequences into one row: a prefix trie when ``share``, else one
    chain per sequence. A node is keyed on its token and the next one, as in TRL."""
    tokens: list[int] = []
    children: list[dict] = []
    roots: dict = {}
    sample_nodes: list[list[int]] = []
    for ids in sequences:
        ids = [int(t) for t in ids]
        if share:
            level, nodes = roots, []
            for key in zip(ids, [*ids[1:], None], strict=True):
                if key not in level:
                    level[key] = len(tokens)
                    tokens.append(key[0])
                    children.append({})
                nodes.append(level[key])
                level = children[nodes[-1]]
        else:
            start = len(tokens)
            nodes = list(range(start, start + len(ids)))
            roots[("sample", len(sample_nodes))] = start
            for depth, token in enumerate(ids):
                tokens.append(token)
                children.append({})
                if depth:
                    children[-2][(token, None)] = nodes[depth]
        sample_nodes.append(nodes)

    input_ids: list[int] = []
    position_ids: list[int] = []
    offsets = [0]
    paths: list[list[int]] = []
    kv_index: list[int] = []
    cu_seqlens_k = [0]
    parents: list[int] = []
    depths: list[int] = []
    packed = [0] * len(tokens)
    stack: list[tuple[int, int | None, int]] = [
        (node, None, 0) for node in reversed(roots.values())
    ]
    while stack:
        node, parent, depth = stack.pop()
        segment, start = len(paths), len(input_ids)
        parents.append(-1 if parent is None else parent)
        depths.append(0 if parent is None else depths[parent] + 1)
        while True:
            packed[node] = len(input_ids)
            input_ids.append(tokens[node])
            position_ids.append(depth)
            depth += 1
            if len(children[node]) != 1:
                break
            node = next(iter(children[node].values()))
        offsets.append(len(input_ids))
        paths.append(
            (paths[parent] if parent is not None else [])
            + list(range(start, len(input_ids)))
        )
        kv_index.extend(paths[-1])
        cu_seqlens_k.append(len(kv_index))
        stack.extend(
            (child, segment, depth) for child in reversed(children[node].values())
        )
    return Packed(
        torch.tensor(input_ids),
        torch.tensor(position_ids),
        torch.tensor(kv_index, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.int32),
        torch.tensor(cu_seqlens_k, dtype=torch.int32),
        torch.tensor(parents, dtype=torch.int32),
        torch.tensor(depths, dtype=torch.int32),
        [torch.tensor([packed[n] for n in nodes]) for nodes in sample_nodes],
    )


def branching_sequences(vocab_size: int = 64, seed: int = 0) -> list[np.ndarray]:
    """Rollouts sharing a system prefix and per-group prompts, some branching late."""
    rng = np.random.default_rng(seed)
    system = rng.integers(1, vocab_size, size=5).tolist()
    sequences = []
    for _ in range(3):
        prompt = (
            system + rng.integers(1, vocab_size, size=int(rng.integers(1, 4))).tolist()
        )
        for _ in range(3):
            completion = rng.integers(
                1, vocab_size, size=int(rng.integers(1, 6))
            ).tolist()
            sequences.append(prompt + completion)
        # a branch of the last completion after its first tokens
        sequences.append(sequences[-1][: len(prompt) + 1] + [7, 8])
    return [np.asarray(s) for s in sequences]


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
    scaling=None,
    **kwargs,
):
    """PyTorch stand-in for TRL's tree attention: each segment attends to its path."""
    q, k, v = (x.transpose(1, 2)[0] for x in (query, key, value))
    k, v = k.index_select(0, tree_kv_index), v.index_select(0, tree_kv_index)
    groups = q.shape[1] // k.shape[1]
    out = torch.zeros_like(q)
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


def register_tree_attention() -> None:
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    AttentionInterface.register(TREE_ATTENTION, reference_tree_attention)
    AttentionMaskInterface.register(TREE_ATTENTION, lambda *args, **kwargs: None)


def _set_attention(model, implementation: str) -> None:
    if any("Attention" in type(m).__name__ for m in model.modules()):
        model.set_attn_implementation(implementation)


def assert_tree_matches_alone(
    model,
    sequences,
    *,
    fmt: str = "trl-1.16",
    share: bool = True,
    collation_padding: int = 0,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Packed:
    """A packed row scores ``sequences`` (log-probs and grads) as they score alone."""
    register_tree_attention()

    def grads():
        found = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        model.zero_grad()
        return found

    def log_probs(logits, ids):
        return torch.log_softmax(logits.float(), -1).gather(1, ids[:, None])[:, 0]

    packed = pack(sequences, share=share)
    input_ids, position_ids = packed.input_ids, packed.position_ids
    if collation_padding:
        input_ids = torch.cat([input_ids, input_ids.new_zeros(collation_padding)])
        position_ids = torch.cat(
            [position_ids, position_ids.new_zeros(collation_padding)]
        )
    _set_attention(model, TREE_ATTENTION)
    # called as TRL calls it, leaving use_cache to the config
    logits = model(
        input_ids=input_ids[None], position_ids=position_ids[None], **packed.kwargs(fmt)
    ).logits[0]
    tree = torch.cat(
        [
            log_probs(logits[positions[:-1]], torch.as_tensor(ids[1:]))
            for ids, positions in zip(sequences, packed.positions, strict=True)
        ]
    )
    tree.sum().backward()
    tree_grads = grads()

    _set_attention(model, "eager")
    alone = []
    for ids in sequences:
        ids = torch.as_tensor(ids)
        logits = model(input_ids=ids[None], use_cache=False).logits[0, :-1]
        alone.append(log_probs(logits, ids[1:]))
    alone_log_probs = torch.cat(alone)
    alone_log_probs.sum().backward()
    alone_grads = grads()

    torch.testing.assert_close(
        tree.detach(), alone_log_probs.detach(), rtol=rtol, atol=atol
    )
    assert tree_grads.keys() == alone_grads.keys()
    for name, grad in tree_grads.items():
        # summation order differs, so small entries of a large gradient are compared
        # at the gradient's scale
        expected = alone_grads[name]
        torch.testing.assert_close(
            grad,
            expected,
            rtol=rtol,
            atol=max(atol, rtol * float(expected.abs().max())),
            msg=lambda message, name=name: f"{name}: {message}",
        )
    return packed
