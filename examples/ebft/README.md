# Energy-Based Fine-Tuning (EBFT)

EBFT is an integration of ["Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models"](https://arxiv.org/abs/2603.12248) (Jelassi et al., 2026) into axolotl.

## Overview

EBFT fine-tunes language models by optimizing a **feature-matching loss** rather than relying on external reward functions or verifiers. A frozen copy of the model (the "feature network") extracts embeddings from both generated and ground-truth completions, and the generator is updated via REINFORCE to match the ground-truth feature moments.

**Key advantages over SFT:**
- Operates on model rollouts (not teacher forcing), reducing distribution shift
- Provides dense sequence-level supervision without a task-specific verifier
- Improves both downstream accuracy and validation cross-entropy simultaneously

**Key advantages over RLVR:**
- No reward model or verifier required — works on any (prompt, completion) data
- Applicable to non-verifiable tasks (e.g., raw code, translation, creative writing)
- Maintains distributional calibration (low feature-matching loss)

## Two Modes

EBFT supports two modes depending on your data format:

### Structured Mode (`mode: structured`, default)
For **QA/instruction data** with prompt + completion pairs (e.g., OpenCodeInstruct, ALMA translation).
- Extends GRPOTrainer — uses vLLM for fast rollout generation
- RLOO advantages and clipped policy gradient from GRPO
- Feature-matching rewards replace external reward functions

### Strided Mode (`mode: strided`)
For **unstructured text** without prompt/completion splits (e.g., raw code, prose, SwallowCode).
- Uses **strided block-parallel generation** — multiple short rollouts at different anchor points within a document
- No vLLM needed — generation uses custom strided attention masks
- Uses **torch flex_attention** with compiled block masks for efficient fused attention kernels (~2x faster than eager attention)
- Compatible with gradient checkpointing via automatic dtype normalization
- This is the core EBFT algorithm from the paper (Section F)

### Common to both modes:
- **Frozen feature network** — deep copy of the model at initialization (frozen, eval mode)
- **Feature extraction** — hidden states at configurable layer depths (default: 25%, 50%, 75%), L2-normalized per layer before concatenation
- **Feature-matching rewards** — cosine similarity (alignment) minus pairwise dot-product (diversity), scaled by 2 per paper equation (7)
- **SVD whitening** — decorrelates feature dimensions; the paper shows removing it causes the largest degradation
- **CFM loss tracking** — conditional feature-matching loss (paper eq 2) logged as `ebft/cfm_loss`
- **FSDP2 compatible** — feature network stays outside FSDP wrapping (frozen, inference-only)

## Quick Start

### Structured Mode (QA data + vLLM)

```bash
# 1. Start vLLM server (LoRA serve module auto-selected when vllm_lora_sync: true)
CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve examples/ebft/qwen3-4b-ebft-structured-async.yaml

# 2. Train on a separate GPU
CUDA_VISIBLE_DEVICES=1 axolotl train examples/ebft/qwen3-4b-ebft-structured-async.yaml
```

### Strided Mode (unstructured text)

```bash
# No vLLM needed — strided generation is built-in
axolotl train examples/ebft/llama-3b-ebft-strided-fft.yaml
```

## Configuration

### Common EBFT Settings

```yaml
rl: ebft

ebft:
  # Feature network: which layers to extract hidden states from
  # Values are fractions of total depth (0.0 = embedding, 1.0 = final layer)
  feature_layers: [0.25, 0.5, 0.75]

  # How to pool per-token hidden states into sequence embeddings
  # Options: "last_token" (recommended), "mean_pooling", "concat"
  embed_method: last_token

  # SVD whitening — strongly recommended (paper shows largest degradation without it)
  use_whitening: true

  # Reward = alignment_coef * alignment - diversity_coef * diversity
  # Per paper Variant (i) (eq 49): alignment uses cosine similarity (normalized),
  # diversity uses raw dot product — both are bounded after whitening.
  alignment_coef: 1.0
  diversity_coef: 1.0

  # Cross-entropy loss on ground-truth tokens (mixed objective, paper Section 2.1)
  # 0.0 = pure feature matching; 0.03 = recommended balance; 0.1 = CE-dominated
  ce_coef: 0.0
```

### Strided Mode Settings

```yaml
ebft:
  mode: strided
  stride: 8              # tokens between anchor points (paper default: 8)
  context_length: 8      # context window per block (paper default: 8)
  generate_max_len: 8    # tokens generated per block (paper default: 8)
  n_samples_per_prompt: 4  # independent rollouts per document (>= 2 for RLOO)
  temperature: 0.6
  rl_coef: 1.0           # RL loss weight
  advantage_estimator: rloo  # rloo (recommended), group_norm, or reinforce
```

### Structured Mode Settings (via TRL)

```yaml
trl:
  num_generations: 4           # samples per prompt
  max_completion_length: 256   # max tokens to generate
  temperature: 1.0
  use_vllm: true
  scale_rewards: true
  loss_type: grpo
  epsilon: 0.2
```

### Dataset Format

**Structured mode** — QA data with prompt + ground-truth completion:
```yaml
datasets:
  - path: nvidia/OpenCodeInstruct
    type: ebft_opencode.transform
```
Transform returns: `{"prompt": ..., "ground_truth": ...}`

**Strided mode** — raw text tokenized to fixed length:
```yaml
datasets:
  - path: sjelassi/swallow_code_20m
    type: ebft_pretrain.transform
```
Transform returns: `{"input_ids": ..., "attention_mask": ..., "labels": ...}`

## How It Works

### Structured Mode
1. **Generate**: For each prompt, generate `num_generations` completions via vLLM
2. **Extract features**: Forward both generated and ground-truth sequences through the frozen feature network
3. **Compute rewards**: `2 * alignment - 2 * diversity` (paper eq 7)
4. **RLOO advantages**: subtract leave-one-out group mean
5. **Policy gradient**: clipped PPO-style loss

### Strided Mode
1. **Anchor selection**: Pick `num_blocks = (seq_len - gen_len - ctx_len) / stride + 1` anchor points across the document
2. **Block-parallel generation**: At each anchor, generate `gen_len` tokens using a custom strided attention mask via `flex_attention` compiled block masks
3. **Feature extraction**: Forward the full sequence (prompt + generated) through the frozen feature network **with the strided attention mask** — this is critical for correct feature representations
4. **Per-block rewards**:
   - **Alignment** = `2 * cosine_similarity(gen_block_emb, gt_block_emb)` — normalized, bounded in [-2, 2]
   - **Diversity** = `2 * mean_pairwise_dot_product(gen_block_embs)` — raw dot product on whitened vectors
   - **Reward** = `alignment_coef * alignment - diversity_coef * diversity`
5. **RLOO advantages**: leave-one-out baseline across `n_samples_per_prompt` rollouts per block
6. **Policy gradient**: REINFORCE loss on generated tokens, weighted by per-block advantages

### Tracked Metrics

| Metric | Description |
|--------|-------------|
| `ebft/alignment` | Mean cosine similarity between generated and GT features (higher = better) |
| `ebft/diversity` | Mean pairwise similarity between samples (lower = more diverse) |
| `ebft/mean_reward` | alignment - diversity (should trend upward) |
| `ebft/cfm_loss` | Conditional feature-matching loss ‖E[φ(ŷ)] - φ(y)‖² (paper eq 2, lower = better) |
| `ebft/rl_loss` | REINFORCE policy gradient loss |
| `ebft/ce_loss` | Cross-entropy loss on ground-truth tokens (when `ce_coef > 0`) |
| `ebft/advantages_std` | RLOO advantage standard deviation (should be non-zero) |

## Tips and Recommendations

### Reward coefficients
- **`use_whitening: true`**: Strongly recommended. The paper's ablation (Figure 7) shows removing whitening causes the largest performance degradation. Safe to use with `diversity_coef > 0`.
- **`diversity_coef`**: Default 1.0. Per the paper's Variant (i) (eq 49), alignment uses cosine similarity while diversity uses raw dot product. After whitening, both are bounded and on compatible scales.
- **`n_samples_per_prompt`**: Must be >= 2 for diversity and RLOO. 4 is the paper's default.
- **`ce_coef`**: The paper ablates `γ ∈ {0, 0.03, 0.1}`. `0.03` balances CE and RL signals; `0.1` causes CE to dominate the gradient. `0.0` gives pure feature matching.

### Feature extraction
- **`feature_layers: [0.25, 0.5, 0.75]`**: Extracts and concatenates hidden states from 25%, 50%, 75% depth. Each layer is L2-normalized independently before concatenation. The paper shows this works better than mean pooling or single-layer extraction.
- **`embed_method: last_token`**: Uses the last token's hidden state per block. The paper shows this outperforms mean pooling (Figure 7).

### Performance
- **`torch_compile: true`**: Recommended for strided mode. Provides additional speedup via graph compilation.
- **flex_attention**: Strided mode automatically uses `flex_attention` with compiled block masks when available (~2x faster than eager attention). Works with gradient checkpointing via automatic dtype normalization. Falls back to eager attention with dense 4D masks if flex_attention is unavailable.

### Memory
- EBFT requires a frozen copy of the model (the feature network), roughly doubling model memory.
- **LoRA** is recommended to reduce trainable parameter memory. The feature network is always a frozen copy of the base model (without LoRA adapters).
- With 2 GPUs visible, the trainer automatically places the feature network on the second GPU.
- **FSDP2** is supported — the feature network stays outside FSDP wrapping since it's frozen and inference-only. With `cpu_ram_efficient_loading`, the feature network is loaded separately from pretrained weights.

## Example Configs

| Config | Mode | Model | Description |
|--------|------|-------|-------------|
| `llama-1b-ebft-opencode.yaml` | Structured | Llama-3.2-1B | QA coding with vLLM |
| `llama-1b-ebft-opencode-novllm.yaml` | Structured | Llama-3.2-1B | QA coding without vLLM |
| `llama-3b-ebft-strided-fft.yaml` | Strided | Llama-3.2-3B | Unstructured code with LoRA |
| `llama-1b-ebft-strided.yaml` | Strided | Llama-3.2-1B | Quick validation |

## Citation

```bibtex
@article{jelassi2026matching,
  title={Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models},
  author={Jelassi, Samy and Kwun, Mujin and Zhao, Rosie and Li, Yuanzhi and Fusi, Nicolo and Du, Yilun and Kakade, Sham M. and Domingo-Enrich, Carles},
  journal={arXiv preprint arXiv:2603.12248},
  year={2026}
}
```
