# Knowledge Distillation

Top-k logprob distillation. The student is trained against a teacher's top-k next-token
distribution (soft loss, forward/reverse KL or JSD) mixed with the usual cross-entropy on
the hard labels.

## Offline teacher (pre-computed logprobs)

```yaml
plugins:
  - "axolotl.integrations.kd.KDPlugin"

kd_trainer: True
kd_ce_alpha: 0.1
kd_alpha: 0.9
kd_temperature: 1.0

torch_compile: True  # recommended to reduce vram

datasets:
  - path: ...
    type: "axolotl.integrations.kd.chat_template"
    field_messages: "messages_combined"
    logprobs_field: "llm_text_generation_vllm_logprobs"  # for kd only, field of logprobs
    temperature: 1.0  # temperature the logprobs were generated at
```

An example dataset can be found at [`axolotl-ai-co/evolkit-logprobs-pipeline-75k-v2-sample`](https://huggingface.co/datasets/axolotl-ai-co/evolkit-logprobs-pipeline-75k-v2-sample)

## Pre-prepared datasets and `kd_prepared_targets_alignment`

Datasets whose `target_logprobs` / `target_token_ids` / `target_mask` columns were baked
ahead of time (typically used with `skip_prepare_dataset: true`) carry whichever row
alignment the axolotl version that prepared them used:

| value | meaning | who produces it |
| --- | --- | --- |
| `current` (default) | row `j` holds the teacher distribution over token `j + 1` | axolotl > 0.18.0, and every online teacher |
| `legacy` | row `j` holds the distribution over token `j` | axolotl <= 0.18.0 |

`legacy` datasets are one position out of step with what the loss consumes. Rather than
rebuilding them — teacher logprobs are expensive to produce — declare the convention and
the collator shifts the rows into place per sequence at collation time:

```yaml
kd_trainer: True
skip_prepare_dataset: true
kd_prepared_targets_alignment: legacy
```

This applies to any dataset prepared before the alignment fix, including the published
`winglian/OpenThoughts-*-prepared-*` sets. Re-preparing the dataset with a current axolotl
is the alternative; freshly prepared data is `current` and needs no knob. The shift is a
per-sequence slice done before packing (so rows never cross a sequence boundary), costs
nothing measurable, and never mutates the dataset on disk.

### Alignment detector

On the first batch that has enough unmasked positions, the collator checks the declared
alignment against the labels and logs a warning — never a hard failure — if the data looks
mis-declared. It compares two hypotheses on the same batch:

- **probability mass**: how much of the stored top-k distribution (renormalized over the
  stored slice, padded slots excluded) sits on the token a row should describe;
- **top-k containment**: how often that token appears anywhere in the stored top-k.

Top-1 equality is deliberately *not* used: a dataset sampled with temperature or top-p, or
one where the teacher scored text it never generated, routinely has the actual token
outside the teacher's argmax, which would make correctly-aligned data look broken. Mass and
containment degrade gracefully in those cases while still separating the hypotheses
sharply.

The warning fires only when the alternative hypothesis beats the declared one by both a
ratio (1.5x) and an absolute margin (0.10 mass), over at least 64 valid positions, and it
reports both hypotheses' numbers so the call is yours:

```
KD teacher targets look mis-aligned: with kd_prepared_targets_alignment='current' the
teacher puts 10.6% of its mass on the token each row should describe (top-k containment
33.3%), but 32.9% on the token -1 positions away (containment 100.0%), over 84 positions.
If this dataset was prepared by an older axolotl, set kd_prepared_targets_alignment: legacy
```

## Reasoning-mode mix (hybrid students)

A student that has to work with reasoning both on and off needs training data in both
renderings. That is a *dataset-side* choice, not a KD one: per dataset entry, declare the
mix of chat-template kwargs to render under, and axolotl assigns each example one mode.

```yaml
datasets:
  - path: ...
    type: chat_template          # or axolotl.integrations.kd.chat_template
    chat_template_kwargs_mix:
      - kwargs: {enable_thinking: true}
        weight: 0.3
      - kwargs: {enable_thinking: false}
        weight: 0.7
```

Weights are relative and normalized (`0.3/0.7` and `3/7` are the same mix). Any template
kwargs work — `enable_thinking` is simply what the Qwen3 family and its peers call the
switch. A single fixed rendering needs no mix, just `chat_template_kwargs` on the dataset
entry (which overrides the top-level `chat_template_kwargs`):

```yaml
datasets:
  - path: ...
    type: chat_template
    chat_template_kwargs: {enable_thinking: false}
```

**The teacher inherits the mix for free.** Both the HTTP online teacher and any in-process
teacher score the *rendered token ids* of each example, so whatever mode an example was
rendered in is the mode the teacher is asked about. There is no server-side flag for this
and none is needed — do not set `enable_thinking` on the teacher server.

**Determinism.** An example's mode is a pure function of `(seed, dataset index)` — a
blake2b draw, no RNG state. The same config produces the same assignment across runs,
across resumes, and regardless of the `num_proc` used to tokenize (`dataset.map` hands out
absolute indices, so sharding does not shift anything). The seed is the run's `seed` unless
the dataset entry sets `chat_template_kwargs_mix_seed`.

**Eval datasets do not inherit it.** The mix belongs to the dataset entry it is written on,
so a `test_datasets` entry renders exactly how *it* is configured (or with the template
default). This is deliberate: an eval split whose reasoning mix drifts with the training
config is not comparable across runs. A `val_set_size` split is carved out of the already
rendered training data and therefore carries the same mix — configure a separate
`test_datasets` entry to evaluate in one fixed mode.

**Offline / pre-prepared datasets bake the mix at preparation time.** The rendering happens
during tokenization, so a dataset prepared once and reused with `skip_prepare_dataset: true`
keeps the mix it was prepared with; changing `chat_template_kwargs_mix` (or its seed)
changes the dataset hash and triggers re-preparation. This is the same reasoning as the
`kd_prepared_targets_alignment` section above: what is on disk is what trains.

**Label masking of think spans is unchanged.** Which tokens of an assistant turn are
trained follows the template's existing assistant-span rules (`roles_to_train`,
`train_on_eos`/`train_on_eot`, `split_thinking`); rendering a `<think>` span does not by
itself mask or unmask it. Masking think spans independently of the rest of the turn would
be a separate knob — out of scope here.

## Online teacher

The teacher is queried for prompt logprobs at collation time, so no logprobs need to be
pre-computed.

```yaml
plugins:
  - "axolotl.integrations.kd.KDPlugin"

kd_trainer: True
kd_ce_alpha: 0.1
kd_alpha: 0.9
kd_temperature: 1.0
kd_temperature_min: 1.0   # optional cosine decay of kd_temperature over training

kd_online_server_base_url: "http://teacher-host:8000"
kd_online_server: vllm    # or sglang
kd_online_topk: 64
kd_online_timeout: 120

sample_packing: true
```

### Serving the teacher

The teacher must be a **stock OpenAI-compatible vLLM server with prefix caching off**:

```bash
vllm serve <teacher-model> --port 8000 \
  --max-logprobs <kd_online_topk> \
  --no-enable-prefix-caching
```

Both flags are required:

- **`--max-logprobs >= kd_online_topk`**. vLLM's default cap is 20; asking for more makes
  the server reject *every* request with
  `Requested prompt logprobs of N, which is greater than max allowed: 20`.
- **`--no-enable-prefix-caching`**, because prompt logprobs are not produced for cached
  prefix blocks.

`axolotl vllm-serve` **cannot** be used here: it starts the weight-syncing server used by
GRPO/EBFT, which exposes neither `/v1/completions` nor `prompt_logprobs`.

The collator posts one request per (sub-)batch:

```json
{"prompt": [[...token ids...]], "max_tokens": 0, "echo": true, "prompt_logprobs": <kd_online_topk>}
```

`max_tokens: 0` means the teacher scores the prompt and generates nothing. The teacher and
the student must share a tokenizer/vocabulary, since teacher token ids are used directly to
gather student logits.

For sglang (`kd_online_server: sglang`) the equivalent `/generate` request is sent with
`return_logprob`, `top_logprobs_num` and `max_new_tokens: 0`; the server must allow at least
`kd_online_topk` top logprobs per position and run with `--disable-radix-cache`.

### Startup preflight

Before training starts — in the main process, before any dataloader worker exists — the
collator sends a single 3-token probe to the teacher and checks connectivity, the response
contract, and that the server actually returns `kd_online_topk` logprobs per position. A
teacher that cannot serve the config fails the run immediately, quoting the server's own
error and the remedy, instead of failing once per batch inside every worker:

```
TeacherRequestError: the online teacher rejected the request with HTTP 400
(http://teacher:8000/v1/completions): {"object": "error", "message": "Requested prompt
logprobs of 64, which is greater than max allowed: 20", ...}. The teacher must be a stock
OpenAI-compatible `vllm serve` started with --max-logprobs >= kd_online_topk (64) and
--no-enable-prefix-caching.
```

Rejected requests (any 4xx) are never retried at any point — a configuration error cannot be
fixed by trying again, and retrying it turns a fast failure into a silent stall. Set
`kd_online_preflight: false` to skip the probe.

## Temperature

`kd_temperature` is applied on both sides of the KL: student logits are divided by it, and
the teacher's top-k logprobs are rescaled to the same temperature
(`p_T2(k) = p_T1(k) ** (T1 / T2) / Z`) before the loss sees them. The soft loss is scaled by
`temperature ** 2`, in both the reported loss and the gradient.

For the offline path, `T1` is the per-dataset `temperature` (the temperature the stored
logprobs were generated at). For the online path, `T1` is 1.0: the temperature is *not* sent
to the teacher, because sampling parameters do not affect prompt logprobs.

With `kd_temperature_min`, the temperature is decayed with a cosine schedule during training
and applied to the collator and the loss on every step.

## Config reference

| key | default | notes |
| --- | --- | --- |
| `kd_trainer` | `false` | enables the KD trainer and collators |
| `kd_alpha` | `1.0` | weight of the soft (KD) loss |
| `kd_ce_alpha` | `0.0` | weight of the hard-label cross-entropy loss |
| `kd_temperature` | `1.0` | KD temperature, applied to student and teacher |
| `kd_temperature_min` | `None` | online only; cosine-decays `kd_temperature` to this value |
| `kd_beta` | `0.0` | `0.0` forward KL, `1.0` reverse KL, in-between JSD |
| `kd_normalize_topk` | `true` | renormalize the top-k slice into a distribution |
| `kd_compiled_kernel` | `true` | `torch.compile` the chunked loss kernel |
| `kd_prepared_targets_alignment` | `current` | convention of baked `target_*` columns; `legacy` shifts them at collation time |
| `kd_online_server_base_url` | `None` | enables the online teacher |
| `kd_online_server` | `vllm` | `vllm` or `sglang` |
| `kd_online_topk` | `None` | required with an online teacher |
| `kd_online_timeout` | `120` | per-request timeout, in seconds |
| `kd_online_preflight` | `true` | probe the teacher once at startup and fail fast if it cannot serve the config |

Per-dataset (not `kd_*`, they work for any chat-template dataset):

| key | default | notes |
| --- | --- | --- |
| `chat_template_kwargs` | `None` | fixed template kwargs for this dataset; overrides the top-level setting |
| `chat_template_kwargs_mix` | `None` | list of `{kwargs, weight}` renderings to mix per example |
| `chat_template_kwargs_mix_seed` | run `seed` | seed for the mix assignment |

Online and offline KD are mutually exclusive: `kd_online_server_base_url` cannot be combined
with the `axolotl.integrations.kd.chat_template` dataset type.

## Target alignment

All teacher producers emit one target row per input token, aligned to what the student
predicts at that position:

```
target_token_ids[j] / target_logprobs[j]  ->  distribution over input_ids[j + 1]
target_mask[j]                            ->  0 where labels[j + 1] == -100
```

The last row of every sequence is padding (it would describe a token outside the sequence),
as are the positions before the teacher's coverage starts. Padded slots use a large negative
logprob with `target_mask == 0`; the loss drops them, so they never contribute a `NaN`.

Because teacher rows are right-padded to the batch length, KD requires a right-padding
tokenizer; the collator raises if `padding_side` is anything else.
