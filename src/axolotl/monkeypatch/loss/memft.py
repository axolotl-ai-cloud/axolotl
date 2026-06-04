"""
MemFT (Memorization-oriented Fine-Tuning) loss.

Replaces the token-averaged cross-entropy with a token-weighted objective that
redirects gradient budget toward "sub-threshold" tokens — those whose target
probability is still below the deterministic recall threshold p = 0.5
(equivalently per-token loss above L_crit = ln(2)).

    L_MemFT = sum_t w_t * L_t / (sum_t w_t + eps)

Two weighting variants:
  - "ot" (only-threshold): hard mask  w_t = 1[L_t > L_crit]. No extra hyper-params.
  - "sw" (sliding-window):  soft base weight sigma(kappa * (L_t - L_crit)) modulated
    by an exponential spatial decay anchored at the first greedy-prediction error.

The loss is computed inside a patched CausalLM forward so it can see
``position_ids``: under sample packing each row holds several samples, and both
the label shift and the MemFT-SW anchor/window are kept strictly within a single
sample (no cross-sample leakage).

Reference: "How LoRA Remembers? A Parametric Memory Law for LLM Finetuning"
(arXiv:2605.30260).
"""

import math

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)


def _per_token_loss(shift_logits, shift_labels, ignore_index=-100):
    """Per-token cross-entropy, shape [B, S], zero at ignored positions."""
    bsz, seq_len, vocab_size = shift_logits.shape
    loss = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size).float(),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    return loss.view(bsz, seq_len)


def _ot_weights(per_token_loss, mask, critical_loss):
    """MemFT-OT hard mask: keep tokens still above the recall threshold."""
    return (mask & (per_token_loss > critical_loss)).to(per_token_loss.dtype)


def _segment_anchor(error, position, mask):
    """First-error intra-sample position for each token's sample segment.

    ``position`` is the intra-sample index of every shifted token; new samples
    start where ``position == 0``. Returns, per token, the smallest ``position``
    in its segment at which the greedy prediction is wrong (or a large sentinel
    when the segment has no error, so the window then covers the whole sample).
    """
    bsz, seq_len = position.shape
    device = position.device
    big = seq_len + 1

    new_sample = position == 0
    seg = new_sample.cumsum(dim=1)  # [B, S] segment index within each row
    num_seg = int(seg.max().item()) + 1

    flat_key = (torch.arange(bsz, device=device).unsqueeze(1) * num_seg + seg).reshape(
        -1
    )
    cand = torch.where(error, position, position.new_full((), big)).reshape(-1)
    anchor_flat = torch.full((bsz * num_seg,), big, device=device, dtype=position.dtype)
    anchor_flat.scatter_reduce_(0, flat_key, cand, reduce="amin", include_self=True)
    return anchor_flat[flat_key].view(bsz, seq_len)


def _sw_weights(
    per_token_loss,
    shift_logits,
    shift_labels,
    mask,
    critical_loss,
    kappa,
    tau,
    window,
    floor,
    position,
):
    """MemFT-SW: soft threshold modulated by anchor-based spatial decay.

    Tokens upstream of their sample's first-error anchor keep the base weight
    (phi = 1); downstream tokens within the window decay as exp(-(t - a) / tau);
    tokens beyond the window are floored. All offsets are intra-sample.
    """
    preds = shift_logits.argmax(dim=-1)
    error = mask & (preds != shift_labels)
    anchor = _segment_anchor(error, position, mask)

    offset = (position - anchor).to(per_token_loss.dtype)
    phi = torch.exp(-offset.clamp(min=0) / tau)
    within_window = offset < window

    base = torch.sigmoid(kappa * (per_token_loss - critical_loss))
    spatial = torch.where(within_window, phi, phi.new_full((), floor))
    weights = base * spatial
    return weights * mask.to(per_token_loss.dtype)


def memft_loss(
    outputs,
    labels,
    num_items_in_batch=None,  # noqa: ARG001 — MemFT self-normalizes by sum of weights
    variant="ot",
    critical_loss=LN2,
    epsilon=1e-8,
    kappa=1.0,
    tau=64.0,
    window=64,
    floor=0.0,
    ignore_index=-100,
    position_ids=None,
):
    """Compute the MemFT token-weighted loss.

    ``num_items_in_batch`` is accepted for API compatibility but unused: MemFT
    normalizes by the sum of token weights (Eq. 8). ``position_ids`` (when sample
    packing is active) confines the shift and the MemFT-SW anchor/window to each
    individual sample.
    """
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    mask = shift_labels != ignore_index

    if position_ids is not None:
        position = position_ids[..., 1:].contiguous().to(shift_logits.device)
        # a shifted pair crossing a sample boundary (target starts a new sample)
        # must not contribute
        mask = mask & (position != 0)
    else:
        seq_len = shift_logits.shape[1]
        position = torch.arange(seq_len, device=shift_logits.device).expand(
            shift_logits.shape[0], seq_len
        )

    per_token_loss = _per_token_loss(shift_logits, shift_labels, ignore_index)

    with torch.no_grad():
        if variant == "ot":
            weights = _ot_weights(per_token_loss, mask, critical_loss)
        elif variant == "sw":
            weights = _sw_weights(
                per_token_loss,
                shift_logits,
                shift_labels,
                mask,
                critical_loss,
                kappa,
                tau,
                window,
                floor,
                position,
            )
        else:
            raise ValueError(
                f"unknown memft variant {variant!r}; expected 'ot' or 'sw'"
            )

    numerator = (weights * per_token_loss).sum()
    denominator = weights.sum() + epsilon
    return numerator / denominator


def _build_memft_forward(params):
    """Build a llama-like CausalLM ``forward`` that computes the MemFT loss.

    ``params`` carries the resolved MemFT settings. The fused path computes the
    loss straight from hidden states (no logit materialization, MemFT-OT only);
    the non-fused path materializes logits and still returns them so evaluation
    metrics keep working.
    """
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from axolotl.kernels.memft_xentropy import memft_linear_cross_entropy

    fused = params["fused"]
    variant = params["variant"]
    critical_loss = params["critical_loss"]
    epsilon = params["epsilon"]
    kappa = params["kappa"]
    tau = params["tau"]
    window = params["window"]
    floor = params["floor"]
    chunk_tokens = params["chunk_tokens"]
    ignore_index = params["ignore_index"]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        loss = None
        logits = None
        if labels is not None:
            if fused:
                shift_hidden = hidden_states[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().clone()
                if position_ids is not None:
                    # drop cross-sample boundary pairs before the fused kernel
                    boundary = position_ids[..., 1:].to(shift_labels.device) == 0
                    shift_labels[boundary] = ignore_index
                loss = memft_linear_cross_entropy(
                    shift_hidden,
                    self.lm_head.weight,
                    shift_labels,
                    critical_loss=critical_loss,
                    epsilon=epsilon,
                    ignore_index=ignore_index,
                    chunk_tokens=chunk_tokens,
                )
            else:
                logits = self.lm_head(hidden_states)
                loss = memft_loss(
                    (logits,),
                    labels,
                    variant=variant,
                    critical_loss=critical_loss,
                    epsilon=epsilon,
                    kappa=kappa,
                    tau=tau,
                    window=window,
                    floor=floor,
                    ignore_index=ignore_index,
                    position_ids=position_ids,
                )
        else:
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    return forward


def patch_memft(model_type, params):
    """Patch the llama-like CausalLM ``forward`` to compute the MemFT loss."""
    from axolotl.utils.callbacks.models import get_causal_lm_model_cls_prefix

    model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
    module_path = f"transformers.models.{model_type}.modeling_{model_type}"
    try:
        module = __import__(module_path, fromlist=[f"{model_cls_prefix}ForCausalLM"])
        model_cls = getattr(module, f"{model_cls_prefix}ForCausalLM")
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"memft does not support model_type {model_type!r}: could not resolve "
            f"{model_cls_prefix}ForCausalLM. Use a llama-like architecture "
            "(e.g. llama, qwen3, mistral)."
        ) from e

    model_cls.forward = _build_memft_forward(params)
