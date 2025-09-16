"""
Patches to support multipack for mixtral
"""

import torch


def patch_mixtral_moe_forward_zero3(cfg=None) -> None:
    import warnings

    import torch.nn.functional as F

    from axolotl.kernels.moe import backends as _moe_backends
    from axolotl.kernels.moe.backends import MOEBackend, get_moe_backend_name

    def mlp_forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(
            hidden_states
        )
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

    # Ref. https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py
    def moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        preferred = getattr(cfg, "moe_backend", None) if cfg is not None else None
        backend = get_moe_backend_name(preferred)
        if (
            backend == MOEBackend.TORCH_GROUPED
            and not _moe_backends._probe_torch_grouped()
        ):
            warnings.warn(
                "torch_grouped selected but not available; falling back to naive"
            )

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_weight, topk_idx = torch.topk(
            routing_weights, self.top_k, dim=-1, sorted=False
        )
        topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight.to(hidden_states.dtype)

        hidden_states_rep = hidden_states.repeat_interleave(self.top_k, dim=0)
        y = torch.empty_like(hidden_states_rep)
        flat_topk_idx = topk_idx.view(-1)
        for i in range(self.num_experts):
            expert = self.experts[i]
            sel = flat_topk_idx == i
            if sel.any():
                y[sel] = expert(hidden_states_rep[sel])
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        final_hidden_states = y.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    from transformers.models.mixtral.modeling_mixtral import (
        MixtralBlockSparseTop2MLP,
        MixtralSparseMoeBlock,
    )

    MixtralBlockSparseTop2MLP.forward = mlp_forward
    # Wrap forward to support optional torch_grouped backend via config
    from axolotl.kernels.moe import torch_grouped as _tg

    preferred = getattr(cfg, "moe_backend", None) if cfg is not None else None
    backend = get_moe_backend_name(preferred)

    if backend == MOEBackend.TORCH_GROUPED and _tg.available():

        def moe_forward_grouped(self, hidden_states: torch.Tensor) -> torch.Tensor:
            bsz, seqlen, hdim = hidden_states.shape
            y, router_logits = _tg.moe_ffn_forward_grouped(
                hidden_states, self.gate, self.experts, self.top_k
            )
            if y is None:
                return moe_forward(self, hidden_states)
            return y, router_logits

        MixtralSparseMoeBlock.forward = moe_forward_grouped
    else:
        MixtralSparseMoeBlock.forward = moe_forward
