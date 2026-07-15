# SAR (Subspace-Aligned Rewiring)

Training-free post-processing that projects the weight delta of a post-trained model onto the base model's spectral (SVD) subspace, following "Spectral Rewiring for Exploration, Purification, and Model Merging" ([arXiv:2607.03065](https://arxiv.org/abs/2607.03065)). RL updates concentrate their reasoning-effective component in that subspace: keeping only the projected delta preserves nearly all Pass@1 gains with ~1% of the parameters, improves Pass@k at large k, purifies interference from mix-domain RL, and outperforms TIES/DARE for expert merging. One code path covers all three use cases: extraction, purification, and merging.

## Usage

### 1. Automatic post-training projection (extraction / purification)

With the plugin enabled, SAR runs on the main process after training completes, reading the trained model from `output_dir`:

```yaml
plugins:
  - axolotl.integrations.sar.SARPlugin

base_model: Qwen/Qwen2.5-7B
output_dir: ./outputs/grpo-run

sar:
  rank_ratio: 0.01
  # base_model, trained_model, output_dir default to
  # cfg.base_model, cfg.output_dir, {cfg.output_dir}/sar
```

```bash
axolotl train config.yaml
```

Only the main process runs the projection, which can take hours at 32B scale; list `SARPlugin` last under `plugins:` so other plugins' post-train hooks are not blocked behind it on the remaining ranks.

### 2. Standalone CLI

Project any `(base, trained)` pair without training:

```yaml
plugins:
  - axolotl.integrations.sar.SARPlugin

base_model: Qwen/Qwen2.5-7B
output_dir: ./outputs

sar:
  trained_model: ./outputs/grpo-run
  output_dir: ./outputs/grpo-run-sar
  rank_ratio: [0.005, 0.01, 0.05]   # rank sweep: one output per ratio
```

```bash
axolotl sar config.yaml
axolotl sar config.yaml --trained-model ./outputs/other-run --rank-ratio 0.05
```

Multi-ratio runs write to `{output_dir}/rank_{ratio}` subdirectories; a single ratio writes directly to `output_dir`.

### 3. Expert merging

Merge the projected delta of one expert into another model instead of the base:

```yaml
plugins:
  - axolotl.integrations.sar.SARPlugin

base_model: Qwen/Qwen2.5-7B
output_dir: ./outputs

sar:
  trained_model: ./outputs/math-expert
  merge_target: ./outputs/code-expert
  output_dir: ./outputs/merged
  rank_ratio: 0.01
```

Non-projected tensors (embeddings, norms, biases, `lm_head`) are copied verbatim from `merge_target` when set, otherwise from `base_model`.

## Config Reference

| Option | Default | Description |
|---|---|---|
| `sar.base_model` | `cfg.base_model` | Spectral reference model (local dir or HF hub id) |
| `sar.base_model_revision` | `cfg.revision_of_model` when `base_model` is inherited | HF hub revision for `base_model` |
| `sar.trained_model` | `cfg.output_dir` | Post-trained model supplying the weight delta |
| `sar.trained_model_revision` | `null` | HF hub revision for `trained_model` |
| `sar.merge_target` | `null` | Optional expert to merge the projected delta into |
| `sar.merge_target_revision` | `null` | HF hub revision for `merge_target` |
| `sar.output_dir` | `{cfg.output_dir}/sar` | Output directory for the projected model |
| `sar.rank_ratio` | `0.01` | Float or list; fraction of `min(dout, din)` used as per-layer rank, each in (0, 1] |
| `sar.delta_rank_ratio` | `rank_ratio` | Rank ratio for truncating the delta before projection |
| `sar.projection` | `spectral` | `spectral` \| `none` (`none` = low-rank-only ablation) |
| `sar.rewiring` | `full` | `full` \| `diagonal` \| `off_diagonal` masking of the rewiring matrix |
| `sar.scale` | `1.0` | Coefficient on the projected delta (must be > 0) |
| `sar.target_modules` | attention + MLP linears | Substrings matched against 2D `.weight` parameter names |
| `sar.exclude_modules` | `[]` | Substrings excluded even when matched |
| `sar.svd_device` | `auto` | `auto` \| `cuda` \| `cpu` (`auto` = CUDA when available) |
| `sar.save_dtype` | `float16` | `float16` \| `bfloat16` \| `float32` |
| `sar.save_rewiring_matrix` | `false` | Persist per-layer rewiring matrices under `{output_dir}/rewiring/` |
| `sar.run_after_training` | `true` | Run automatically in the post-training hook |

Default `target_modules`: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. Matching is by substring, so fused or unusually named layers (e.g. `Wqkv`, MoE experts) need explicit patterns; use `exclude_modules` to filter false positives.

## Precision Notes

All SVDs and projections run in fp32 regardless of input dtype; `save_dtype` only affects storage. The paper found fp16 preserves the fine-grained projected update better than bf16 for models up to ~7B, hence the `float16` default. When copying bf16 source tensors whose values exceed the fp16 range (|x| > 65504), set `save_dtype: bfloat16` to avoid overflow. On large models prefer `svd_device: auto`/`cuda`; CPU SVD on 32B-scale layers can take minutes per matrix. Inputs must be safetensors checkpoints in fp16/bf16/fp32 — merge LoRA adapters first (`axolotl merge-lora`) and dequantize compressed checkpoints before running SAR.

## Citation

```bib
@misc{sar2026,
  title  = {Spectral Rewiring for Exploration, Purification, and Model Merging},
  year   = {2026},
  eprint = {2607.03065},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2607.03065}
}
```
