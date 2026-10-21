# Next-Concept Modeling (NCM)

Two-stage, coarse-to-fine parallel text generation from ["Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation"](https://arxiv.org/abs/2610.14162) (Kumbong et al., 2026). The model code is ported from [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (MIT, see `LICENSE`).

1. **Tokenizer** (`ncm.stage: tokenizer`): a multiscale residual VQ-VAE encodes a fixed block of `L` text tokens into `K` levels of discrete codes, from 1 code (coarse) to `L` codes (fine).
2. **Generator** (`ncm.stage: generator`): a transformer over the frozen tokenizer predicts every code of level `k` in parallel from the reconstruction of levels `< k`. Generation takes `K` forward passes instead of `L`.

Both stages train from scratch.

## Quickstart

```bash
axolotl train examples/ncm/tinystories-tokenizer.yaml   # writes ./outputs/ncm-tinystories-tokenizer
axolotl train examples/ncm/tinystories-generator.yaml   # base_model: the stage-1 output
```

Stage 1 takes its text tokenizer from `tokenizer_config` and builds the VQ-VAE from `model_config`; pointing `base_model` at a stage-1 output continues it instead. Stage 2 points `base_model` at a stage-1 output: the plugin starts a fresh generator (configured by `model_config`) around the trained tokenizer and freezes it. Pointing stage 2 at an existing generator continues training it. Continued runs load in fp32 (training uses autocast), and `model_config` cannot add parameters to an existing checkpoint.

Data are raw text (`type: completion`) cut into blocks of `levels[-1]` tokens, padding the last block of each document. Set `sequence_len` to that length with `pad_to_sequence_len: true`.

## Configuration

```yaml
plugins:
  - axolotl.integrations.ncm.NCMPlugin
ncm:
  stage: tokenizer        # or generator
  sample_every: 0         # generator: log unconditional samples every N steps
  num_samples: 4
```

Architecture and objective go in `model_config`.

| Stage | Key | Default | Meaning |
|---|---|---|---|
| tokenizer | `levels` | 12 levels, 1 to 128 | codes per level; the last is the block length |
| tokenizer | `codebook_size` | 4096 | per-level codebook size (int or list) |
| tokenizer | `dim`, `num_heads`, `decoder_layers` | 384, 6, 1 | latent width and decoder transformer |
| tokenizer | `ema_decay` | 0.99 | EMA codebook decay |
| tokenizer | `fine_drop_prob` | 0.5 | nested dropout: train on a truncated hierarchy this often |
| tokenizer | `dead_code_threshold` | 0.05 | re-seed codes used less than this fraction of their uniform share |
| generator | `n_layer`, `n_head`, `n_embd` | 12, 12, 768 | transformer size |
| generator | `head_type` | `tied` | `tied` (distance to the codebook), `linear`, or `shared` |
| generator | `corruption_rate` | 0.0 | mean fraction of conditioning codes replaced at random (paper: 0.5) |
| generator | `corruption_schedule` | `linear` | `linear` decreases the rate from coarse to fine; `constant` |
| generator | `level_loss_alpha` | 0.0 | level `k` weighs `ell_k^-alpha`; 0 weighs every code equally, 1 every level |
| generator | `query_level0` | false | predict level 0 from learned queries instead of sampling its unigram |

## Sampling

```python
from transformers import AutoModel, AutoTokenizer
import axolotl.integrations.ncm  # registers the model types

from axolotl.integrations.ncm.trainer import truncate_at_eos

model = AutoModel.from_pretrained("./outputs/ncm-tinystories-generator").eval()
tok = AutoTokenizer.from_pretrained("./outputs/ncm-tinystories-generator")
for row in model.generate(batch_size=4, temperature=1.0).tolist():
    print(tok.decode(truncate_at_eos(row, tok.eos_token_id), skip_special_tokens=True))
```

Generation fills the whole block; positions after the first EOS are untrained filler. Unconditional generation samples level 0 from its empirical distribution, counted during generator training.

## Released checkpoints

`convert.load_reference_checkpoint` loads a `hazyresearch/ncm-checkpoints` task folder (`config.yaml`, `tokenizer.pt`, `generator.pt`). The port reproduces the reference on the TinyStories checkpoint: identical codes, identical greedy and seeded samples, and teacher-forced logits within 4e-7 relative. Released checkpoints use their own text tokenizers (custom BPE for TinyStories), which are not converted.

```python
from axolotl.integrations.ncm.convert import load_reference_checkpoint

model = load_reference_checkpoint("ncm-checkpoints/tinystories", null_prefix_token_id=0)
model.save_pretrained("./ncm-tinystories-ref")
```

## Differences from the reference

The reference release ships the model code (including the quantizer's EMA updates and nested dropout) but no training loop, generator loss, code corruption, or level-0 statistics; those follow the paper.

- **Codebook health.** EMA cluster sizes start at 1 instead of 0, and codes unused for a while are re-seeded from current inputs (the reference's absolute `restart_threshold` is replaced by `dead_code_threshold`, relative to the uniform share). Starting at 0, the first EMA update scales every rarely-hit code out of range, which collapsed a from-scratch tokenizer to one code per level in testing.
- **Corruption schedule.** The paper uses a linearly decreasing per-level rate with a given mean. Here the rate falls from about `2 * corruption_rate` at the coarsest corrupted level towards 0 at the finest. Level 0 is spared unless `corruption_skip_level0: false`; the finest level is never an input.
- **Not ported yet.** Prompt-conditioned training data (translation, summarization, Sudoku), classifier-free guidance, MBR decoding, the frequency penalties used to decode LM1B and XSum, and WMT14's `predict_level0`.

## Limitations

- Fixed-length blocks only. `sample_packing`, FSDP, and DeepSpeed are rejected.
- EMA codebook statistics are all-reduced across ranks, but multi-GPU runs are untested.
- The EMA updates once per micro-batch, so gradient accumulation steps change codebook dynamics: 8 micro-batches of 512 make 8 EMA updates, 8 GPUs with 512 each make 1.
- PyTorch's MPS backend crashes in the tokenizer's convolution backward; train on CUDA or CPU.
