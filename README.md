<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

<p align="center">
    <img src="https://img.shields.io/github/license/axolotl-ai-cloud/axolotl.svg?color=blue" alt="GitHub License">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests.yml/badge.svg" alt="tests">
    <a href="https://codecov.io/gh/axolotl-ai-cloud/axolotl"><img src="https://codecov.io/gh/axolotl-ai-cloud/axolotl/branch/main/graph/badge.svg" alt="codecov"></a>
    <a href="https://github.com/axolotl-ai-cloud/axolotl/releases"><img src="https://img.shields.io/github/release/axolotl-ai-cloud/axolotl.svg" alt="Releases"></a>
    <br/>
    <a href="https://github.com/axolotl-ai-cloud/axolotl/graphs/contributors"><img src="https://img.shields.io/github/contributors-anon/axolotl-ai-cloud/axolotl?color=yellow&style=flat-square" alt="contributors" style="height: 20px;"></a>
    <img src="https://img.shields.io/github/stars/axolotl-ai-cloud/axolotl" alt="GitHub Repo stars">
    <br/>
    <a href="https://discord.com/invite/HhrNrHJPRb"><img src="https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord" alt="discord" style="height: 20px;"></a>
    <a href="https://twitter.com/axolotl_ai"><img src="https://img.shields.io/twitter/follow/axolotl_ai?style=social" alt="twitter" style="height: 20px;"></a>
    <br/>
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml/badge.svg" alt="tests-nightly">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml/badge.svg" alt="multigpu-semi-weekly tests">
</p>


## 🎉 Latest Updates

- 2025/07:
  - ND Parallelism support has been added into Axolotl. Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes. Check out the [blog post](https://huggingface.co/blog/accelerate-nd-parallel) for more info.
  - Axolotl adds more models: [GPT-OSS](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/gpt-oss), [Gemma 3n](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/gemma3n), [Liquid Foundation Model 2 (LFM2)](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/lfm2), and [Arcee Foundation Models (AFM)](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/afm).
  - FP8 finetuning with fp8 gather op is now possible in Axolotl via `torchao`. Get started [here](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8)!
  - [Voxtral](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/voxtral), [Magistral 1.1](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral), and [Devstral](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/devstral) with mistral-common tokenizer support has been integrated in Axolotl!
  - TiledMLP support for single-GPU to multi-GPU training with DDP, DeepSpeed and FSDP support has been added to support Arctic Long Sequence Training. (ALST). See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst) for using ALST with Axolotl!
- 2025/05: Quantization Aware Training (QAT) support has been added to Axolotl. Explore the [docs](https://docs.axolotl.ai/docs/qat.html) to learn more!
- 2025/03: Axolotl has implemented Sequence Parallelism (SP) support. Read the [blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [docs](https://docs.axolotl.ai/docs/sequence_parallelism.html) to learn how to scale your context length when fine-tuning.

<details>

<summary>Expand older updates</summary>

- 2025/06: Magistral with mistral-common tokenizer support has been added to Axolotl. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral) to start training your own Magistral models with Axolotl!
- 2025/04: Llama 4 support has been added in Axolotl. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-4) to start training your own Llama 4 models with Axolotl's linearized version!
- 2025/03: (Beta) Fine-tuning Multimodal models is now supported in Axolotl. Check out the [docs](https://docs.axolotl.ai/docs/multimodal.html) to fine-tune your own!
- 2025/02: Axolotl has added LoRA optimizations to reduce memory usage and improve training speed for LoRA and QLoRA in single GPU and multi-GPU training (DDP and DeepSpeed). Jump into the [docs](https://docs.axolotl.ai/docs/lora_optims.html) to give it a try.
- 2025/02: Axolotl has added GRPO support. Dive into our [blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code) and have some fun!
- 2025/01: Axolotl has added Reward Modelling / Process Reward Modelling fine-tuning support. See [docs](https://docs.axolotl.ai/docs/reward_modelling.html).

</details>

## ✨ Overview

Axolotl is a tool designed to streamline post-training for various AI models.

### 📌 Project Objectives (Authoritative Reference)

This section defines the canonical objectives for the embedded Bethpage Black strategy + description prototype. Refer to this list in issues, PRs, and planning; do not redefine elsewhere.

1. Unified Model Interface: A single chat endpoint where the same fine‑tuned base + LoRA adapter produces BOTH tee‑shot strategy recommendations and hole descriptions.
2. Strategy Accuracy: Predict normalized discrete cutoff yardages aligned with on-hole options; gracefully coerce outputs to the nearest valid available strategy when the raw generation drifts.
3. Description Specificity: Generate grounded hole descriptions (par, yardage, hazards, strategic theme, preferred miss) without inventing unsupported features or yardages.
4. Data Evolution (v3 Goal): Incorporate structured course attributes and curated ("gold") descriptions directly into training prompts and targets so the model internalizes domain grounding instead of relying on deterministic post-processing.
5. Transparent Inference UX: Model path is the default for both tasks; deterministic description fallback remains optional for debugging / regression comparison.
6. Reproducibility & Maintainability: Clean LoRA training script, manifest-verified datasets, evaluation metrics (MAE / exact-match for strategies), lightweight Windows-friendly environment.
7. Extensibility: Architecture leaves room for adding approach-shot logic, expanded hazard taxonomy, and multi-course generalization without refactoring core trainer or chat surface.

If an enhancement conflicts with any of these objectives, document the trade-off before merging.

Features:

- **Multiple Model Support**: Train various models like LLaMA, Mistral, Mixtral, Pythia, and more. We are compatible with HuggingFace transformers causal language models.
- **Training Methods**: Full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), Multimodal, and Reward Modelling (RM) / Process Reward Modelling (PRM).
- **Easy Configuration**: Re-use a single YAML file between dataset preprocess, training, evaluation, quantization, and inference.
- **Performance Optimizations**: [Multipacking](https://docs.axolotl.ai/docs/multipack.html), [Flash Attention](https://github.com/Dao-AILab/flash-attention), [Xformers](https://github.com/facebookresearch/xformers), [Flex Attention](https://pytorch.org/blog/flexattention/), [Liger Kernel](https://github.com/linkedin/Liger-Kernel), [Cut Cross Entropy](https://github.com/apple/ml-cross-entropy/tree/main), [Sequence Parallelism (SP)](https://docs.axolotl.ai/docs/sequence_parallelism.html), [LoRA optimizations](https://docs.axolotl.ai/docs/lora_optims.html), [Multi-GPU training (FSDP1, FSDP2, DeepSpeed)](https://docs.axolotl.ai/docs/multi-gpu.html), [Multi-node training (Torchrun, Ray)](https://docs.axolotl.ai/docs/multi-node.html), and many more!
- **Flexible Dataset Handling**: Load from local, HuggingFace, and cloud (S3, Azure, GCP, OCI) datasets.
- **Cloud Ready**: We ship [Docker images](https://hub.docker.com/u/axolotlai) and also [PyPI packages](https://pypi.org/project/axolotl/) for use on cloud platforms and local hardware.



## 🚀 Quick Start

**Requirements**:

- NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
- Python 3.11
- PyTorch ≥2.6.0

### Installation

#### Using pip

```bash
pip3 install -U packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation axolotl[flash-attn,deepspeed]

# Download example axolotl configs, deepspeed configs
axolotl fetch examples
axolotl fetch deepspeed_configs  # OPTIONAL
```

#### Using Docker

Installing with Docker can be less error prone than installing in your own environment.
```bash
docker run --gpus '"all"' --rm -it axolotlai/axolotl:main-latest
```

Other installation approaches are described [here](https://docs.axolotl.ai/docs/installation.html).

#### Cloud Providers

<details>

- [RunPod](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)
- [Vast.ai](https://cloud.vast.ai?ref_id=62897&template_id=bdd4a49fa8bce926defc99471864cace&utm_source=github&utm_medium=developer_community&utm_campaign=template_launch_axolotl&utm_content=readme)
- [PRIME Intellect](https://app.primeintellect.ai/dashboard/create-cluster?image=axolotl&location=Cheapest&security=Cheapest&show_spot=true)
- [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl)
- [Novita](https://novita.ai/gpus-console?templateId=311)
- [JarvisLabs.ai](https://jarvislabs.ai/templates/axolotl)
- [Latitude.sh](https://latitude.sh/blueprint/989e0e79-3bf6-41ea-a46b-1f246e309d5c)

</details>

### Your First Fine-tune

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

That's it! Check out our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a more detailed walkthrough.


## 📚 Documentation

- [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Detailed setup instructions for different environments
- [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Full configuration options and examples
- [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Loading datasets from various sources
- [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported formats and how to use them
- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
- [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated code documentation
- [FAQ](https://docs.axolotl.ai/docs/faq.html) - Frequently asked questions

## 🤝 Getting Help

- Join our [Discord community](https://discord.gg/HhrNrHJPRb) for support
- Check out our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/) directory
- Read our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
- Need dedicated support? Please contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai) for options

## 🌟 Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

## ❤️ Sponsors

Interested in sponsoring? Contact us at [wing@axolotl.ai](mailto:wing@axolotl.ai)

## 📝 Citing Axolotl

If you use Axolotl in your research or projects, please cite it as follows:

```bibtex
@software{axolotl,
  title = {Axolotl: Post-Training for AI Models},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 📓 Example: Bethpage Multitask Prompted Dataset

An internal example (golf course strategy) demonstrates a full pipeline:

1. Enrichment & schema validation of structured hole data.
2. Multi-task sample generation (strategy_selection, strategy_selection_negative, description_generation, style_rewrite).
3. Style compression + QA (word cap & rationale grounding).
4. Weighting / oversampling of narrative tasks.
5. Chat-style prompt construction into `messages` format.

Artifacts in `data/bethpage_black/`:
- `training_bethpage_multitask.weighted.train.jsonl` – weighted raw tasks with `sample_weight`.
- `build_prompted_dataset.py` – builds chat-formatted dataset.
- `training_bethpage_multitask.weighted.prompted.train.jsonl` – output with `messages` list.
- `prompt_template_spec.md` – detailed template spec.

Regenerate prompted dataset:
```bash
python data/bethpage_black/build_prompted_dataset.py
```

JSONL line schema (abridged):
```json
{
  "task": "description_generation",
  "hole": 5,
  "sample_weight": 2.0,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Training integration: reference the prompted file with a `chat_template` dataset entry. Future improvement path: leverage `sample_weight` for probabilistic sampling instead of physical replication.

### Dedupe, Normalization & Weighted Sampling (Bethpage Example)

After generating the physically replicated weighted prompted dataset, we:

1. Run `scripts/dedupe_normalize_prompted.py` to:
   - Merge duplicate assistant outputs, summing `sample_weight`.
   - Normalize mojibake artifacts (e.g., `â€™` -> `’`).
2. Evaluate quality with `scripts/evaluate_prompted_dataset.py` to confirm:
   - Zero encoding artifacts
   - High 4-gram uniqueness (no replication collapse)
3. Sample per-epoch using `scripts/sample_prompted_dataset.py` with probabilities ∝ `sample_weight`.

python scripts/dedupe_normalize_prompted.py \
  data/bethpage_black/training_bethpage_multitask.weighted.prompted.train.jsonl \
  -o data/bethpage_black/training_bethpage_multitask.weighted.prompted.dedup.train.jsonl

# Re-evaluate metrics
python scripts/evaluate_prompted_dataset.py \
  data/bethpage_black/training_bethpage_multitask.weighted.prompted.dedup.train.jsonl

# Sample an epoch (probabilistic weighting)
python scripts/sample_prompted_dataset.py \
  --data data/bethpage_black/training_bethpage_multitask.weighted.prompted.dedup.train.jsonl \
  --epoch-size 1000 --seed 7 --summary \
  --out data/bethpage_black/epoch_sample_1000.jsonl
```

Benefits vs physical replication:
- Preserves intended task balance while maximizing lexical diversity.
- Eliminates duplicate gradient contributions for identical targets.
- Normalizes text early, avoiding downstream tokenization inconsistencies.

Result snapshot (before → after dedupe):
- 4-gram uniqueness (description_generation): 0.000 → 0.947
- Encoding artifact rate: 0.333 → 0.000
- Duplicate assistant outputs: many → none

You can adjust sampling sharpness via `--temperature` (e.g., >1 to flatten, <1 to emphasize high-weight tasks).

```
python scripts/freeze_prompted_dataset.py \
  --data data/bethpage_black/training_bethpage_multitask.weighted.prompted.dedup.train.jsonl \
  ### Quick Launch (Bethpage Example)

  Reproduce the small multi-task fine-tune (strategy selection + description + style rewrite) with deterministic weighted sampling and manifest verification:

  1. Freeze / verify dataset (already frozen if you have the manifest):
  ```bash
  python scripts/verify_prompted_manifest.py --data data/bethpage_black/prompted_dedup.train.jsonl \
    --manifest data/bethpage_black/prompted_dedup.train.manifest.json
  ```
  2. (Optional) Validate environment & hardware:
  ```bash
  python scripts/validate_env.py --out-json env_report.json --output-dir outputs --min-free-gb 1 \
    --manifest data/bethpage_black/prompted_dedup.train.manifest.json
  ```
  3. Launch training (locked config):
  ```bash
  python -m axolotl.cli.train examples/bethpage/weighted_prompted_locked.yml \
    dataset=data/bethpage_black/prompted_dedup.train.jsonl
  ```
  4. For stratified per-window task diversity add:
  ```bash
  python -m axolotl.cli.train examples/bethpage/weighted_prompted_stratified.yml \
    dataset=data/bethpage_black/prompted_dedup.train.jsonl
  ```

  Early stopping (patience=3) is enabled in the locked config. A manifest mismatch causes startup failure, ensuring reproducibility.
  --out data/bethpage_black/training_bethpage_multitask.weighted.prompted.dedup.train.manifest.json
```

Manifest includes sha256, line counts, per-task weights, and a format tag for reproducibility audits.

## Python 3.11 Environment (Windows) 🔧

If you're on Windows and ran into build/import issues on newer Python (e.g. 3.13), use a supported 3.11 environment for full Axolotl features:

1. Install Python 3.11 (verify with `py -3.11 -V`).
2. From repo root run:
  ```powershell
  scripts/setup_py311.ps1
  ```
  Add `-ForceRecreate` to rebuild or `-SkipTests` to omit test dependencies.
3. Activate later:
  ```powershell
  . .venv311\Scripts\Activate.ps1
  ```
4. Train (example):
  ```powershell
  python -m axolotl.cli.main train examples/bethpage/weighted_prompted_locked.yml `
     dataset=data/bethpage_black/prompted_dedup.train.jsonl `
     max_steps=50 eval_steps=50 save_steps=50 logging_steps=10
  ```

The setup script retries with a conservative NumPy pin if the first install fails. Heavy optional GPU extras (xformers, bitsandbytes) are skipped on Windows unless you manually enable them.

### Using Weighted Prompted Iterable in Training

Example config: `examples/bethpage/weighted_prompted.yml`

Config block:
```yaml
weighted_prompted_jsonl:
  path: data/bethpage_black/training_bethpage_multitask.weighted.prompted.dedup.train.jsonl
  epoch_size: 2000
  seed: 42
  temperature: 1.0
  enforce_quota: false
```

When present, the loader constructs a `WeightedPromptedIterableDataset` providing on-the-fly weighted sampling. Set `enforce_quota: true` to eliminate multinomial variance in per-task counts.

### Stratified Batch / Window Mixing

For very small multi-task datasets, per-batch stochastic sampling can yield batches dominated by a single task early in training. Enable stratified windows to guarantee local task diversity:

Config additions:
```yaml
weighted_prompted_jsonl:
  path: ...
  epoch_size: 4000
  seed: 42
  stratify_window_size: 8   # emit windows containing (roughly) one sample per task
  stratify_shuffle: true    # shuffle within each window before yielding
```

Behavior:
- Builds windows by round-robin over tasks, sampling each example within its task by relative intra-task weight.
- Optional shuffle randomizes order inside each window to avoid position bias.
- Falls back to standard multinomial sampling when `stratify_window_size` unset or 0.
- Can combine with `enforce_quota: true` to honor exact per-epoch task counts, though later windows may concentrate a single task once other quotas are exhausted (expected when quotas not divisible by window size).

Demo:
```
python scripts/demo_stratified_windows.py \
  --data data/bethpage_black/training_bethpage_multitask.weighted.prompted.dedup.train.jsonl \
  --epoch-size 32 --window-size 8
```
Sample output:
```
Window 0: description_generation=2 strategy_selection=2 strategy_selection_negative=2 style_rewrite=2
...
```
This ensures gradient signals per optimizer step (with packing/accumulation) reflect all tasks early, stabilizing initial loss curves. When combining with quotas, expect the final windows to skew toward tasks with remaining quota remainders.
