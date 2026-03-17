"""
Strided block-parallel EBFT trainer for unstructured text data.

This trainer implements the full EBFT algorithm from the paper, including
strided block-parallel generation where multiple short rollouts are generated
at different anchor points within a single document. This is essential for
training on raw text data (code, prose, etc.) without prompt/completion splits.

Uses torch flex_attention with a compiled block mask for efficient strided
attention patterns. Falls back to eager attention with dense 4D masks when
flex_attention is not available.

Paper: "Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models"
       (Jelassi et al., 2026) https://arxiv.org/abs/2603.12248
"""

import copy
import functools
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from axolotl.core.trainers.ebft.rewards import (
    get_alignment_rewards,
    get_diversity_rewards,
    whiten_embeddings_batched,
)
from axolotl.core.trainers.mixins import (
    DistributedParallelMixin,
    RngLoaderMixin,
    SchedulerMixin,
)
from axolotl.core.trainers.mixins.optimizer import OptimizerInitMixin, OptimizerMixin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Check flex_attention availability
_FLEX_ATTENTION_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
    )
    _FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    pass


def _patch_flex_attention_dtype():
    """
    Patch HF's flex_attention_forward to cast q/k/v to a uniform dtype.

    This fixes the incompatibility between flex_attention and gradient
    checkpointing, where recomputation can produce q/k in float32 while
    v stays in bfloat16. flex_attention requires all three to match.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original_fn = ALL_ATTENTION_FUNCTIONS.get("flex_attention")
    if original_fn is None:
        return

    def patched_flex_attention_forward(module, query, key, value, attention_mask, **kwargs):
        # Cast q/k/v to the same dtype (use value's dtype as reference,
        # since model weights stay in the original dtype)
        target_dtype = value.dtype
        if query.dtype != target_dtype:
            query = query.to(target_dtype)
        if key.dtype != target_dtype:
            key = key.to(target_dtype)
        return original_fn(module, query, key, value, attention_mask, **kwargs)

    ALL_ATTENTION_FUNCTIONS["flex_attention"] = patched_flex_attention_forward


# ---------------------------------------------------------------------------
# Strided attention mask builders
# ---------------------------------------------------------------------------

def _strided_mask_mod(
    b, h, q_idx, kv_idx,
    prompt_length, context_length, max_generation_length, stride, num_blocks,
):
    """
    Mask mod function for flex_attention's create_block_mask.

    Defines the strided block-parallel attention pattern:
    - Prompt region: standard causal
    - Generated region: each block attends to its context window + same-block predecessors
    """
    # --- Prompt region: standard causal ---
    is_prompt_q = q_idx < prompt_length
    is_prompt_kv = kv_idx < prompt_length
    prompt_causal = is_prompt_q & is_prompt_kv & (q_idx >= kv_idx)

    # --- Generated region ---
    is_gen_q = ~is_prompt_q
    # Which generation step and block does this query token belong to?
    gen_offset = q_idx - prompt_length
    gen_step = gen_offset // num_blocks
    block_idx = gen_offset % num_blocks

    # Context window end for this block
    context_end = torch.clamp(
        block_idx * stride + context_length,
        max=prompt_length - max_generation_length,
    )

    # Rule 1: Generated token attends to its context window in the prompt
    in_context = is_gen_q & is_prompt_kv & (kv_idx < context_end)

    # Rule 2: Self-attention
    is_self = q_idx == kv_idx

    # Rule 3: Attend to earlier tokens in the SAME block (same block_idx, earlier gen_step)
    is_gen_kv = ~is_prompt_kv
    kv_gen_offset = kv_idx - prompt_length
    kv_gen_step = kv_gen_offset // num_blocks
    kv_block_idx = kv_gen_offset % num_blocks
    same_block_prev = (
        is_gen_q & is_gen_kv
        & (kv_block_idx == block_idx)
        & (kv_gen_step < gen_step)
    )

    return prompt_causal | in_context | is_self | same_block_prev


def create_strided_block_mask(
    prompt_length: int,
    context_length: int,
    max_generation_length: int,
    stride: int,
    num_blocks: int,
    full_sequence_length: int,
    batch_size: int,
    num_heads: int,
    device: torch.device,
):
    """
    Create a BlockMask for flex_attention using the strided EBFT pattern.

    Returns a BlockMask that can be passed directly to model.forward()
    when using attn_implementation="flex_attention".
    """
    mask_mod = functools.partial(
        _strided_mask_mod,
        prompt_length=prompt_length,
        context_length=context_length,
        max_generation_length=max_generation_length,
        stride=stride,
        num_blocks=num_blocks,
    )

    block_mask = create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,  # broadcast across heads
        Q_LEN=full_sequence_length,
        KV_LEN=full_sequence_length,
        device=device,
    )
    return block_mask


def build_strided_position_ids(
    full_sequence_length: int,
    prompt_length: int,
    context_length: int,
    generation_step: int,
    stride: int,
    num_blocks: int,
    device: torch.device,
    batch_size: int = 1,
):
    """Build position IDs for strided generation (shared between flex and eager modes)."""
    position_ids = torch.empty(
        (batch_size, full_sequence_length), dtype=torch.long, device=device
    )
    position_ids[:, :prompt_length] = torch.arange(prompt_length, device=device)

    block_starting_positions = (
        torch.arange(num_blocks, device=device) * stride + context_length
    )
    for gen_step in range(generation_step):
        start = prompt_length + gen_step * num_blocks
        end = start + num_blocks
        position_ids[:, start:end] = block_starting_positions + gen_step

    return position_ids


def build_strided_dense_mask_and_positions(
    full_sequence_length: int,
    prompt_length: int,
    context_length: int,
    generation_step: int,
    max_generation_length: int,
    stride: int,
    num_blocks: int,
    device: torch.device,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
):
    """Build dense 4D attention mask (eager fallback) + position IDs."""
    min_value = torch.finfo(dtype).min
    attention_mask = torch.full(
        (batch_size, 1, full_sequence_length, full_sequence_length),
        min_value, dtype=dtype, device=device,
    )

    causal_mask = torch.tril(
        torch.ones((prompt_length, prompt_length), dtype=torch.bool, device=device)
    )
    attention_mask[:, :, :prompt_length, :prompt_length].masked_fill_(
        causal_mask.view(1, 1, prompt_length, prompt_length), 0.0
    )

    for gen_step in range(generation_step):
        for block_idx in range(num_blocks):
            gen_pos = prompt_length + gen_step * num_blocks + block_idx
            context_end = min(
                block_idx * stride + context_length,
                prompt_length - max_generation_length,
            )
            attention_mask[:, 0, gen_pos, :context_end] = 0.0
            attention_mask[:, 0, gen_pos, gen_pos] = 0.0
            if gen_step > 0:
                for prev_s in range(gen_step):
                    prev_pos = prompt_length + prev_s * num_blocks + block_idx
                    attention_mask[:, 0, gen_pos, prev_pos] = 0.0

    position_ids = build_strided_position_ids(
        full_sequence_length, prompt_length, context_length,
        generation_step, stride, num_blocks, device, batch_size,
    )
    return attention_mask, position_ids


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AxolotlStridedEBFTTrainer(
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DistributedParallelMixin,
    Trainer,
):
    """
    Strided block-parallel EBFT trainer for unstructured text data.

    Takes full text documents (no prompt/completion split needed), generates
    short rollouts at multiple anchor points via strided attention, and trains
    with feature-matching rewards.

    When flex_attention is available (torch >= 2.5), uses compiled block masks
    for efficient fused attention kernels. Otherwise falls back to eager
    attention with dense 4D masks.
    """

    _tag_names = ["ebft", "strided", "axolotl"]

    def __init__(self, model, args, train_dataset, **kwargs):
        super().__init__(model=model, args=args, train_dataset=train_dataset, **kwargs)

        # EBFT config
        self.ebft_stride = getattr(args, "ebft_stride", 8)
        self.ebft_context_length = getattr(args, "ebft_context_length", 8)
        self.ebft_generate_max_len = getattr(args, "ebft_generate_max_len", 8)
        self.ebft_n_samples = getattr(args, "ebft_n_samples_per_prompt", 4)
        self.ebft_temperature = getattr(args, "ebft_temperature", 0.6)
        self.ebft_top_p = getattr(args, "ebft_top_p", 1.0)
        self.ebft_alignment_coef = getattr(args, "ebft_alignment_coef", 1.0)
        self.ebft_diversity_coef = getattr(args, "ebft_diversity_coef", 1.0)
        self.ebft_rl_coef = getattr(args, "ebft_rl_coef", 1.0)
        self.ebft_ce_coef = getattr(args, "ebft_ce_coef", 0.0)
        self.ebft_use_whitening = getattr(args, "ebft_use_whitening", False)
        self.ebft_advantage_estimator = getattr(args, "ebft_advantage_estimator", "rloo")

        # Validate config combinations
        if self.ebft_use_whitening and self.ebft_diversity_coef > 0:
            LOG.info(
                "ebft: whitening + diversity enabled. Per paper Variant (i) (eq 49): "
                "alignment uses cosine similarity (normalized), diversity uses raw dot product. "
                "Both are bounded after whitening."
            )
        if self.ebft_n_samples == 1 and self.ebft_diversity_coef > 0:
            LOG.warning(
                "ebft.n_samples_per_prompt=1 with diversity_coef > 0: diversity penalty requires "
                "multiple samples. Setting diversity_coef to 0."
            )
            self.ebft_diversity_coef = 0.0
        if self.ebft_n_samples == 1 and self.ebft_advantage_estimator == "rloo":
            LOG.warning(
                "ebft.n_samples_per_prompt=1 with advantage_estimator='rloo': RLOO requires "
                "multiple samples for baseline. Falling back to 'reinforce'."
            )
            self.ebft_advantage_estimator = "reinforce"

        # Feature network config
        feature_layers_frac = getattr(args, "ebft_feature_layers", [0.25, 0.5, 0.75])
        embed_method = getattr(args, "ebft_embed_method", "last_token")
        self.ebft_embed_method = embed_method

        # Attention implementation selection
        unwrapped = self.accelerator.unwrap_model(self.model)
        # Attention implementation selection
        self.use_flex_attention = _FLEX_ATTENTION_AVAILABLE and torch.cuda.is_available()

        if self.use_flex_attention:
            # Patch flex_attention to handle dtype mismatches from gradient
            # checkpointing recomputation (q/k may be float32, v bfloat16)
            _patch_flex_attention_dtype()
            LOG.info("Using flex_attention for strided EBFT (compiled block masks)")
            if hasattr(unwrapped.config, "_attn_implementation"):
                unwrapped.config._attn_implementation = "flex_attention"
            self._num_heads = unwrapped.config.num_attention_heads
        else:
            LOG.info("Using eager attention for strided EBFT (dense 4D masks)")
            if hasattr(unwrapped.config, "_attn_implementation"):
                unwrapped.config._attn_implementation = "eager"
            self._num_heads = None

        # Create frozen feature network
        # With FSDP2, parameters may be sharded DTensors — gather them first.
        # With standard training, this is a no-op.
        first_param = next(unwrapped.parameters())
        original_device = first_param.device
        if original_device.type == "meta":
            # FSDP2 with cpu_ram_efficient_loading — can't deepcopy from meta
            LOG.warning(
                "Model parameters on meta device (FSDP2 cpu_ram_efficient_loading). "
                "Feature network will be loaded separately from pretrained weights."
            )
            from transformers import AutoModelForCausalLM
            feature_model_name = getattr(args, "model_name_or_path", None) or unwrapped.config._name_or_path
            self.feature_network = AutoModelForCausalLM.from_pretrained(
                feature_model_name, torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
        else:
            self.feature_network = copy.deepcopy(unwrapped)
            self.feature_network.to(dtype=torch.bfloat16)

        actor_gpu = original_device.index if (original_device.type == "cuda" and original_device.index is not None) else 0
        visible_gpus = torch.cuda.device_count()
        if visible_gpus > 1:
            self._feature_device = torch.device(f"cuda:{(actor_gpu + 1) % visible_gpus}")
        else:
            self._feature_device = torch.device(f"cuda:{actor_gpu}")

        LOG.info(f"Creating frozen feature network on {self._feature_device}...")
        self.feature_network.to(device=self._feature_device)
        # Feature network can use flex_attention (no grad checkpointing → no dtype issue)
        if _FLEX_ATTENTION_AVAILABLE and self._feature_device.type == "cuda":
            if hasattr(self.feature_network.config, "_attn_implementation"):
                self.feature_network.config._attn_implementation = "flex_attention"
            self._feature_use_flex = True
            LOG.info("Feature network using flex_attention")
        else:
            if hasattr(self.feature_network.config, "_attn_implementation"):
                self.feature_network.config._attn_implementation = "eager"
            self._feature_use_flex = False
        for param in self.feature_network.parameters():
            param.requires_grad = False
        self.feature_network.eval()

        num_layers = unwrapped.config.num_hidden_layers
        self.feature_layer_indices = [
            int(frac * num_layers) for frac in feature_layers_frac
        ]
        LOG.info(
            f"Strided EBFT: layers={self.feature_layer_indices}, "
            f"stride={self.ebft_stride}, ctx={self.ebft_context_length}, "
            f"gen_len={self.ebft_generate_max_len}, n_samples={self.ebft_n_samples}, "
            f"embed={embed_method}, flex_attn={self.use_flex_attention}"
        )

    def _build_strided_mask(self, full_seq_len, seq_len, generation_step, num_blocks, batch_size, device, dtype):
        """Build strided attention mask + position IDs using flex or eager."""
        pos_ids = build_strided_position_ids(
            full_seq_len, seq_len, self.ebft_context_length,
            generation_step, self.ebft_stride, num_blocks, device, batch_size,
        )

        if self.use_flex_attention:
            block_mask = create_strided_block_mask(
                prompt_length=seq_len,
                context_length=self.ebft_context_length,
                max_generation_length=self.ebft_generate_max_len,
                stride=self.ebft_stride,
                num_blocks=num_blocks,
                full_sequence_length=full_seq_len,
                batch_size=batch_size,
                num_heads=self._num_heads,
                device=device,
            )
            return block_mask, pos_ids

        dense_mask, pos_ids = build_strided_dense_mask_and_positions(
            full_sequence_length=full_seq_len,
            prompt_length=seq_len,
            context_length=self.ebft_context_length,
            generation_step=generation_step,
            max_generation_length=self.ebft_generate_max_len,
            stride=self.ebft_stride,
            num_blocks=num_blocks,
            device=device,
            batch_size=batch_size,
            dtype=dtype,
        )
        return dense_mask, pos_ids

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Full strided EBFT training step.

        1. Take tokenized documents from inputs
        2. Generate n_samples short rollouts at strided anchor points
        3. Extract features from frozen network for both generated and GT blocks
        4. Compute alignment/diversity rewards per block
        5. Compute RLOO advantages
        6. Policy gradient loss on the strided forward pass
        """
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)  # (B, seq_len)
        B, seq_len = input_ids.shape

        stride = self.ebft_stride
        ctx_len = self.ebft_context_length
        gen_len = self.ebft_generate_max_len
        n_samples = self.ebft_n_samples

        num_blocks = (seq_len - gen_len - ctx_len) // stride + 1
        if num_blocks <= 0:
            LOG.warning(
                f"Sequence too short for strided EBFT: seq_len={seq_len}, "
                f"need >= {gen_len + ctx_len + stride}. Returning zero loss."
            )
            dummy_loss = input_ids.float().mean() * 0.0
            return (dummy_loss, None) if return_outputs else dummy_loss

        # --- Step 1: Generate strided blocks for n_samples ---
        repeated_ids = input_ids.repeat_interleave(n_samples, dim=0)

        with torch.no_grad():
            full_sequences = self._generate_strided_blocks(
                model, repeated_ids, num_blocks
            )

        # --- Step 2: Build strided mask for full generation ---
        full_seq_len = full_sequences.shape[1]
        model_dtype = next(model.parameters()).dtype

        # Free generation-phase memory before training forward pass
        torch.cuda.empty_cache()

        attn_mask, pos_ids = self._build_strided_mask(
            full_seq_len, seq_len, gen_len, num_blocks,
            B * n_samples, device, model_dtype,
        )

        # --- Step 3: Forward pass through actor for log probs ---
        # Only compute log probs for generated tokens to save memory
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                full_sequences,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                return_dict=True,
            )

        # Compute per-token log probs in chunks to avoid OOM on log_softmax
        logits = outputs.logits  # (B*N, full_seq_len, V)
        shift_labels = full_sequences[:, 1:]  # (B*N, full_seq_len - 1)
        gen_start = seq_len - 1  # shifted index where generated tokens start

        per_token_logps = torch.zeros(
            logits.shape[0], logits.shape[1] - 1,
            device=device, dtype=torch.float32,
        )

        # Compute log probs for prompt region (for CE loss) and generated region (for RL loss)
        # Process in chunks to avoid OOM on log_softmax over full vocab
        compute_start = 0 if self.ebft_ce_coef > 0 else gen_start
        region_logits = logits[:, compute_start:-1, :]
        region_labels = shift_labels[:, compute_start:]
        chunk_size = 256
        for i in range(0, region_logits.shape[1], chunk_size):
            chunk_logits = region_logits[:, i:i + chunk_size, :]
            chunk_labels = region_labels[:, i:i + chunk_size]
            chunk_lp = F.log_softmax(chunk_logits.float(), dim=-1)
            per_token_logps[:, compute_start + i:compute_start + i + chunk_logits.shape[1]] = (
                chunk_lp.gather(dim=-1, index=chunk_labels.unsqueeze(-1)).squeeze(-1)
            )
        del logits, region_logits

        action_mask = torch.zeros(per_token_logps.shape, dtype=torch.bool, device=device)
        action_mask[:, gen_start:] = True

        # --- Step 4: Extract features and compute rewards ---
        with torch.no_grad():
            block_rewards = self._compute_block_rewards(
                full_sequences, attn_mask, pos_ids,
                input_ids, num_blocks, B, n_samples,
            )

        del attn_mask, pos_ids
        torch.cuda.empty_cache()

        # --- Step 5: Compute advantages ---
        advantages_per_block = self._compute_advantages(
            block_rewards, B, n_samples, num_blocks
        )

        token_advantages = advantages_per_block.repeat_interleave(gen_len, dim=1)
        full_advantages = torch.zeros_like(per_token_logps)
        full_advantages[:, seq_len - 1:] = token_advantages[:, : full_advantages.shape[1] - seq_len + 1]

        # --- Step 6: Compute loss ---
        # RL loss: REINFORCE on generated tokens
        rl_loss_per_token = -per_token_logps * full_advantages.detach()
        rl_loss = (rl_loss_per_token * action_mask.float()).sum() / action_mask.float().sum().clamp(min=1)

        # CE loss: standard next-token prediction on prompt (ground-truth) tokens.
        # This provides dense supervision that stabilizes training alongside the
        # RL feature-matching signal. See paper Section 2.1, mixed objective.
        ce_loss = torch.tensor(0.0, device=device)
        if self.ebft_ce_coef > 0:
            ce_mask = ~action_mask  # prompt tokens only
            ce_loss = (-per_token_logps * ce_mask.float()).sum() / ce_mask.float().sum().clamp(min=1)

        loss = self.ebft_rl_coef * rl_loss + self.ebft_ce_coef * ce_loss

        # --- Log metrics ---
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "ebft/rl_loss": rl_loss.item(),
                "ebft/ce_loss": ce_loss.item(),
                "ebft/cfm_loss": getattr(self, "_last_cfm", 0.0),
                "ebft/mean_reward": block_rewards.mean().item(),
                "ebft/alignment": getattr(self, "_last_alignment", 0.0),
                "ebft/diversity": getattr(self, "_last_diversity", 0.0),
                "ebft/num_blocks": num_blocks,
                "ebft/advantages_std": advantages_per_block.std().item(),
            })

        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def _generate_strided_blocks(self, model, prompt_ids, num_blocks):
        """Generate tokens using strided block-parallel attention."""
        B, seq_len = prompt_ids.shape
        gen_len = self.ebft_generate_max_len
        stride = self.ebft_stride
        ctx_len = self.ebft_context_length
        temperature = self.ebft_temperature
        top_p = self.ebft_top_p
        device = prompt_ids.device
        model_dtype = next(model.parameters()).dtype

        full_sequence = prompt_ids.clone()

        for generation_step in range(gen_len):
            cur_len = full_sequence.shape[1]
            attn_mask, pos_ids = self._build_strided_mask(
                cur_len, seq_len, generation_step, num_blocks,
                B, device, model_dtype,
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    full_sequence,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    return_dict=True,
                )
            all_logits = output.logits

            logit_positions = []
            for block_idx in range(num_blocks):
                if generation_step == 0:
                    pos = block_idx * stride + ctx_len - 1
                else:
                    pos = seq_len + (generation_step - 1) * num_blocks + block_idx
                logit_positions.append(pos)

            position_indices = torch.tensor(logit_positions, device=device)
            block_logits = all_logits.index_select(1, position_indices)

            if temperature > 0:
                block_logits = block_logits / temperature
                probs = torch.softmax(block_logits, dim=-1)

                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumulative > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    mask = torch.zeros_like(probs, dtype=torch.bool)
                    mask.scatter_(-1, sorted_idx, remove)
                    probs[mask] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                flat_probs = probs.view(-1, probs.shape[-1])
                sampled = torch.multinomial(flat_probs, 1).squeeze(-1)
                sampled = sampled.view(B, num_blocks)
            else:
                sampled = torch.argmax(block_logits, dim=-1)

            full_sequence = torch.cat([full_sequence, sampled], dim=1)

        return full_sequence

    @torch.no_grad()
    def _compute_block_rewards(
        self, full_sequences, attn_mask, pos_ids,
        original_ids, num_blocks, batch_size, n_samples,
    ):
        """Extract features and compute per-block rewards. Returns (B, N, NB)."""
        device = full_sequences.device
        seq_len = original_ids.shape[1]
        gen_len = self.ebft_generate_max_len
        stride = self.ebft_stride
        ctx_len = self.ebft_context_length

        # Run feature network on its device WITH the strided attention mask.
        # Without the strided mask, generated tokens see tokens from other blocks
        # via default causal attention, corrupting the feature representations.
        fd = self._feature_device
        fn_seqs = full_sequences.to(fd)
        fn_pos = pos_ids.to(fd)

        # Build strided mask — flex block mask if available, else dense 4D
        if self._feature_use_flex:
            fn_attn_mask = create_strided_block_mask(
                prompt_length=seq_len,
                context_length=ctx_len,
                max_generation_length=gen_len,
                stride=stride,
                num_blocks=num_blocks,
                full_sequence_length=full_sequences.shape[1],
                batch_size=full_sequences.shape[0],
                num_heads=self.feature_network.config.num_attention_heads,
                device=fd,
            )
        else:
            fn_attn_mask, _ = build_strided_dense_mask_and_positions(
                full_sequence_length=full_sequences.shape[1],
                prompt_length=seq_len,
                context_length=ctx_len,
                generation_step=gen_len,
                max_generation_length=gen_len,
                stride=stride,
                num_blocks=num_blocks,
                device=fd,
                batch_size=full_sequences.shape[0],
                dtype=torch.bfloat16,
            )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fn_outputs = self.feature_network(
                fn_seqs, attention_mask=fn_attn_mask, position_ids=fn_pos,
                output_hidden_states=True, return_dict=True,
            )
        hidden_states_cpu = [fn_outputs.hidden_states[idx].to(device) for idx in self.feature_layer_indices]
        del fn_outputs, fn_seqs, fn_pos, fn_attn_mask

        # Normalize each layer's hidden states separately (like the reference critic),
        # then concatenate. This prevents one dominant layer from suppressing others.
        normalized_layers = [F.normalize(h, p=2, dim=-1) for h in hidden_states_cpu]
        features = torch.cat(normalized_layers, dim=-1).to(device)
        del hidden_states_cpu, normalized_layers

        gt_features = features[:, ctx_len:seq_len, :]
        gen_features = features[:, seq_len:, :]

        gt_block_features = gt_features.unfold(1, gen_len, stride).permute(0, 1, 3, 2)
        gen_block_features = gen_features.reshape(
            batch_size * n_samples, gen_len, num_blocks, -1
        ).transpose(1, 2)

        if self.ebft_embed_method == "mean_pooling":
            gt_emb = gt_block_features.mean(dim=2)
            gen_emb = gen_block_features.mean(dim=2)
        else:  # last_token
            gt_emb = gt_block_features[:, :, -1, :]
            gen_emb = gen_block_features[:, :, -1, :]

        gt_emb = gt_emb.view(batch_size, n_samples, num_blocks, -1)
        gen_emb = gen_emb.view(batch_size, n_samples, num_blocks, -1)

        if self.ebft_use_whitening:
            whitened_gen, whitened_gt = [], []
            for b in range(batch_size):
                for nb in range(num_blocks):
                    w_gen, w_gt = whiten_embeddings_batched(
                        gen_emb[b, :, nb, :], gt_emb[b, :, nb, :],
                    )
                    whitened_gen.append(w_gen)
                    whitened_gt.append(w_gt)
            gen_emb = torch.stack(whitened_gen).view(batch_size, num_blocks, n_samples, -1).transpose(1, 2)
            gt_emb = torch.stack(whitened_gt).view(batch_size, num_blocks, n_samples, -1).transpose(1, 2)

        alignment = F.cosine_similarity(gen_emb, gt_emb, dim=-1)

        diversity = torch.zeros_like(alignment)
        if n_samples > 1:
            for nb in range(num_blocks):
                block_gen = gen_emb[:, :, nb, :]
                sims = torch.bmm(block_gen, block_gen.transpose(1, 2))
                eye = torch.eye(n_samples, device=device, dtype=torch.bool)
                sims = sims.masked_fill(eye.unsqueeze(0), 0.0)
                diversity[:, :, nb] = sims.sum(dim=-1) / (n_samples - 1)

        # Scale by 2 per paper equation (7):
        #   r_j = 2*φ(ŷ_j)^T*φ(y) - 2/(n-1) * Σ_{j'≠j} φ(ŷ_j)^T*φ(ŷ_{j'})
        alignment = alignment * 2
        diversity = diversity * 2

        # Compute CFM loss: ||E[φ(ŷ)] - φ(y)||^2 (paper eq 2)
        # Mean generated embedding per prompt, squared distance to GT
        mean_gen_emb = gen_emb.mean(dim=1, keepdim=True)  # (B, 1, NB, D)
        gt_for_cfm = gt_emb[:, 0:1, :, :]  # (B, 1, NB, D) — one GT per prompt
        cfm_loss = ((mean_gen_emb - gt_for_cfm) ** 2).sum(dim=-1).mean()

        # Store for logging
        self._last_alignment = alignment.mean().item()
        self._last_diversity = diversity.mean().item()
        self._last_cfm = cfm_loss.item()

        return self.ebft_alignment_coef * alignment - self.ebft_diversity_coef * diversity

    def _compute_advantages(self, rewards, batch_size, n_samples, num_blocks):
        """Compute RLOO advantages. rewards: (B, N, NB) → (B*N, NB)."""
        if self.ebft_advantage_estimator == "rloo" and n_samples > 1:
            total = rewards.sum(dim=1, keepdim=True)
            baseline = (total - rewards) / (n_samples - 1)
            advantages = rewards - baseline
        elif self.ebft_advantage_estimator == "group_norm" and n_samples > 1:
            mean = rewards.mean(dim=1, keepdim=True)
            std = rewards.std(dim=1, keepdim=True) + 1e-8
            advantages = (rewards - mean) / std
        else:
            advantages = rewards
        return advantages.view(batch_size * n_samples, num_blocks)
