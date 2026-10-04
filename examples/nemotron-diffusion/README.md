# Nemotron diffusion LoRA smoke

This two-step synthetic chat run checks absorbing-mask diffusion training through `DiffusionPlugin`,
physical sample packing, gradient accumulation and adapter saving. It is not a
quality benchmark or a production fine-tuning recipe.

Use the installation instructions in
[`docs/diffusion_lm.qmd`](../../docs/diffusion_lm.qmd). From the repository
root, install this checkout with
`python -m pip install -e .` in an isolated
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

For image-conditioned choice, score, and noul training, see
[the VLM starter recipe](decision-vlm-lora-8b.yaml) and
[the image dataset contract](DECISION_DATASET_FORMAT.md#image-conditioned-decisions).
It shares the decision objectives and core media collation with text-only
training, with adapters scoped to the language decoder.

The VLM recipe uses BF16 LoRA with a frozen vision tower and projector. It is
a starting configuration, not a tuned accuracy or VRAM guarantee. CUDA
FlexAttention compiles both the attention kernel and block-mask construction;
whole-model compilation is optional. Selected-position logits avoid projecting
image/context positions through the vocabulary head. QLoRA is not enabled for
native diffusion yet.

### VLM memory measurements

A direct pretrained-model training measurement on Torch 2.14.0 with an RTX PRO
6000 Blackwell used BF16, rank-16 decoder LoRA, frozen BF16 vision/projector,
microbatch 1, nonreentrant checkpointing, compiled FlexAttention and block masks,
and selected-token cross entropy. Peaks include backward and an AdamW step.

| Actual input tokens | Image longest edge | Peak allocated | Peak reserved |
| ---: | ---: | ---: | ---: |
| 512 | 280 pixels | 17.56 GiB | 17.87 GiB |
| 2,048 | 560 pixels | 18.59 GiB | 19.02 GiB |

These are synthetic-input memory measurements, not task-quality benchmarks or
tests on a physical 24GB card. Both sizes also passed with the PyTorch allocator
capped at 24 GiB, with 4.98 GiB of reserved-memory headroom at 2,048 tokens.
This cap excludes allocations outside PyTorch. They measure a direct model training loop;
full trainer overhead and different numbers of images/decisions can change the
peak. The example bounds context to 2,048 tokens and image size to 560 pixels.
The model and adapters alone occupied 16.78 GiB, so this BF16 configuration
cannot fit a 16 GiB device. Quantized training requires separate validation.

A separate two-step `axolotl train` smoke passed on the published model with
LoRA+ ratio 8 and mixed hard/soft choice, score, and noul labels on one visual
example. It saved the terminal adapter and processor; a fresh process reloaded both
and completed an image-conditioned forward pass. Training reported 17.61 GiB
peak allocated / 17.83 GiB reserved. This checks trainer integration, not
accuracy or generalization; the larger-context table above uses the direct
measurement loop.
