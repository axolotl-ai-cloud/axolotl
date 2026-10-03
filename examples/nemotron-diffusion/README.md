# Nemotron diffusion LoRA smoke

This two-step synthetic chat run checks absorbing-mask diffusion training through `DiffusionPlugin`,
physical sample packing, gradient accumulation and adapter saving. It is not a
quality benchmark or a production fine-tuning recipe.

Use the isolated Torch 2.14 environment described in
[`docs/diffusion_lm.qmd`](../../docs/diffusion_lm.qmd). From the repository
root, install this checkout with
`python -m pip install -r requirements/diffusion-lm-torch214.txt -e .` in that
environment, then put its `bin` directory first on `PATH` so `axolotl train`
launches the matching Accelerate installation:

```bash
axolotl preprocess examples/nemotron-diffusion/lora-smoke.yaml
axolotl train examples/nemotron-diffusion/lora-smoke.yaml
```

This example pins the 3B model and remote implementation to
`0d51902da1f8869f83413ce642fab402fa5641e0` for reproducibility. The loader uses
normal Hugging Face revision handling: `revision_of_model` is optional, and
other compatible revisions and Nemotron-Labs-Diffusion-8B are supported with
`trust_remote_code: true`. When changing models, remove or replace the example
revision with one belonging to the new model. The recipe uses the published
bidirectional diffusion mode, aligned logits and the existing mask token 100.
It reuses the tokenizer EOS token for padding without resizing the vocabulary.
The explicit SDPA backend is the dense reference path.

A run with the pinned checkpoint cached locally completed on an RTX PRO 6000
Blackwell with Torch 2.14.0+cu130 and BF16. It used 8.41 GiB peak allocated GPU
memory for this small 256-token packing budget. Source-objective and native inference checks are recorded in the native-core
evidence ledger; native-core integration acceptance passed.

See [the native diffusion guide](../../docs/diffusion_lm.qmd) for FlexAttention,
packing budgets, objective options and adapter constraints.

See [the 8B decision training recipe](DECISION_TRAINING.md) for the plugin YAML,
portable dataset paths, batch flattening, and opt-in typed-decision packing.
For the reproducibly materialized public procedural decision mix, see
[PUBLIC_PROCEDURAL_DECISION_MIX.md](PUBLIC_PROCEDURAL_DECISION_MIX.md).
