# Nemotron typed decision training

Use both plugins: `DiffusionPlugin` supplies native diffusion training;
`DecisionPlugin` supplies typed datasets, answer supervision and readout.
The normalized format supports Choice, finite Score and Noul, with hard and soft
labels mixed within a record or batch.

## Data

Prepare separate train and dev JSONL files using
[DECISION_DATASET_FORMAT.md](DECISION_DATASET_FORMAT.md), then set their
`data_files` paths in `decision-lora-8b.yaml`. Preserve source row provenance and
keep protected test data out of train and dev. For public example data use
[PUBLIC_PROCEDURAL_DECISION_MIX.md](PUBLIC_PROCEDURAL_DECISION_MIX.md).

```bash
axolotl preprocess examples/nemotron-diffusion/decision-lora-8b.yaml
axolotl train examples/nemotron-diffusion/decision-lora-8b.yaml
```

## Recipe

The 8B recipe uses rank64/alpha128 LoRA on q/k/v/o and gate/up/down projections,
LoRA+ ratio8, AdamW LR6.5e-6, constant scheduling, and no warmup. Microbatch16
with eight accumulation steps gives EBS128. `sequence_len` is2048. Adjust batch
size to fit your hardware.

The objective combines candidate-restricted CE, full-vocabulary CE and Brier
(weight0.1). Full-vocabulary CE discourages probability outside valid answers.
For hard targets, optional `labels.hard_label_smoothing` spreads its fraction
uniformly over that question's valid answers; non-one-hot soft targets remain
unchanged. Brier and development targets remain unchanged.

The decision path uses two held-noise denoising reads and no latent slots.
`batch_flattening: true` preserves logical examples per microbatch. Physical
`sample_packing` is opt-in, with token budget implied by `sequence_len` and
`micro_batch_size`.

## Evaluation and export

The recipe evaluates and saves every50 steps. `load_best_model_at_end` selects
by dev loss; choose the explicit terminal checkpoint when benchmarking terminal
weights. Heldout results must not select checkpoints.

Use `src/axolotl/integrations/decision/scripts/decision_evaluate.py --help` for batched local reads.
The checkpoint's `diffusion_decision_manifest.json` records codebook, canvas,
read settings and model identity. Keep it with the adapter and use matching
settings for inference.

AutoJev routing examples can be normalized using
[AUTOJEV_ROUTE_DATASET_FORMAT.md](AUTOJEV_ROUTE_DATASET_FORMAT.md). Choice
candidate IDs and their order must be preserved between training and deployment.
