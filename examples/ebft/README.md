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
- No vLLM needed — generation happens via custom strided attention masks
- Uses **torch flex_attention** with compiled block masks for efficient fused attention kernels (falls back to eager attention if unavailable)
- This is the core EBFT algorithm from the paper (Section F)

### Common to both modes:
- **Frozen feature network** — a deep copy of the model at initialization (frozen, eval mode)
- **Feature extraction** — hidden states at configurable layer depths (default: 25%, 50%, 75%)
- **Feature-matching rewards** — cosine similarity (alignment) minus pairwise dot-product (diversity)
- **Optional SVD whitening** — decorrelates feature dimensions for balanced optimization

## Quick Start

### 1. Start a vLLM server

```bash
# On a separate GPU (or same GPU with colocate mode)
python -m trl.scripts.vllm_serve \
    --model meta-llama/Llama-3.2-1B \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.3
```

### 2. Run EBFT training

```bash
axolotl train examples/ebft/llama-1b-ebft-opencode.yaml
```

## Configuration

### EBFT-Specific Settings

```yaml
rl: ebft

ebft:
  # Feature network: which layers to extract hidden states from
  # Values are fractions of total depth (0.0 = embedding, 1.0 = final layer)
  feature_layers: [0.25, 0.5, 0.75]

  # How to pool per-token hidden states into sequence embeddings
  # Options: "last_token", "mean_pooling", "concat"
  embed_method: last_token

  # SVD whitening decorrelates feature dimensions (recommended)
  use_whitening: true

  # Reward = alignment_coef * alignment - diversity_coef * diversity
  alignment_coef: 1.0   # cosine similarity with ground-truth features
  diversity_coef: 1.0   # pairwise similarity penalty between samples

  # Optional cross-entropy loss on ground-truth tokens (mixed objective)
  ce_coef: 0.0
```

### Strided Mode Settings

For unstructured text (raw code, prose), use strided mode:

```yaml
ebft:
  mode: strided
  stride: 8              # tokens between anchor points
  context_length: 8      # context window per block
  generate_max_len: 8    # tokens generated per block
  n_samples_per_prompt: 4  # independent rollouts per document
  temperature: 0.6
  rl_coef: 1.0           # RL loss weight
  advantage_estimator: rloo  # rloo, group_norm, or reinforce
```

### Structured Mode Generation Settings (via TRL)

For QA data, EBFT reuses GRPOTrainer's generation infrastructure:

```yaml
trl:
  num_generations: 4           # samples per prompt (more = better RLOO baseline)
  max_completion_length: 256   # max tokens to generate
  temperature: 1.0             # sampling temperature
  use_vllm: true               # use vLLM server for fast generation
  scale_rewards: true          # normalize advantages by group std
  loss_type: grpo              # clipped policy gradient
  epsilon: 0.2                 # PPO clip range
```

### Dataset Format

**Structured mode** uses QA data with prompt + ground-truth completion:
```yaml
datasets:
  - path: nvidia/OpenCodeInstruct
    type: ebft_opencode.transform
```
Transform returns: `{"prompt": ..., "ground_truth": ...}`

**Strided mode** uses raw text (tokenized directly):
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
3. **Compute rewards**: alignment (cosine similarity) minus diversity (pairwise similarity)
4. **RLOO advantages**: subtract leave-one-out group mean
5. **Policy gradient**: clipped PPO-style loss

### Strided Mode
1. **Anchor selection**: Pick `num_blocks = (seq_len - gen_len - ctx_len) / stride + 1` anchor points
2. **Block-parallel generation**: At each anchor, generate `gen_len` tokens using strided attention mask (via flex_attention)
3. **Feature extraction**: Extract hidden states for both generated blocks and ground-truth blocks at same positions
4. **Compute rewards**:
   - **Alignment** = cosine similarity between generated and ground-truth block embeddings
   - **Diversity** = mean pairwise dot-product between samples (penalizes mode collapse)
   - **Reward** = `alignment_coef * alignment - diversity_coef * diversity`
4. **RLOO advantages**: Subtract leave-one-out group mean from each sample's reward
5. **Policy gradient update**: Clipped PPO-style loss on the advantage-weighted log probabilities

## Tips and Recommendations

### Reward coefficients

- **`alignment_coef`**: Controls the weight of cosine similarity between generated and ground-truth features. Default 1.0.
- **`diversity_coef`**: Controls the diversity penalty (pairwise dot-product similarity between samples). Default 1.0. Per the paper's Variant (i) (eq 49), alignment uses cosine similarity (normalized) while diversity uses raw dot product — both are bounded after whitening.
- **`use_whitening`**: Recommended. Whitening decorrelates feature dimensions and the paper shows removing it causes the largest degradation. Safe to use with `diversity_coef > 0`.
- **`n_samples_per_prompt`**: Must be >= 2 if `diversity_coef > 0` (diversity requires pairwise comparisons). Must be >= 2 for RLOO advantage estimation (falls back to REINFORCE with n=1).

### Feature extraction

- **`feature_layers`**: Default `[0.25, 0.5, 0.75]` extracts at 25%, 50%, 75% depth. Earlier layers capture syntax, middle layers capture semantics, later layers are biased toward next-token prediction.
- **`embed_method`**: `"last_token"` uses the last token's hidden state per block. `"mean_pooling"` averages across all tokens in the block. The paper uses `"last_token"`.
- **`use_whitening`**: SVD whitening decorrelates feature dimensions. Recommended when NOT using diversity penalty. See warning above about scale mismatch with diversity.

### Strided mode specifics

- **`stride`, `context_length`, `generate_max_len`**: The paper uses `stride=8, context_length=8, generate_max_len=8` for all experiments. These control the granularity and density of anchor points.
- **`flash_attention`**: Strided mode uses custom 4D attention masks. Set `flash_attention: false` — the trainer will automatically use `flex_attention` (compiled block masks) when gradient checkpointing is off, or `eager` attention as fallback. flex_attention is currently incompatible with gradient checkpointing due to dtype mismatches during recomputation.

### Memory

- EBFT requires a frozen copy of the model (the feature network), doubling model memory. For full-parameter training, ensure you have enough GPU memory for both models + optimizer states.
- **LoRA** is recommended to reduce trainable parameter memory. The feature network is always a frozen copy of the base model (without LoRA adapters).
- With 2 GPUs available, the trainer automatically places the feature network on the second GPU.

## Citation

```bibtex
@article{jelassi2026matching,
  title={Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models},
  author={Jelassi, Samy and Kwun, Mujin and Zhao, Rosie and Li, Yuanzhi and Fusi, Nicolo and Du, Yilun and Kakade, Sham M. and Domingo-Enrich, Carles},
  journal={arXiv preprint arXiv:2603.12248},
  year={2026}
}
```
