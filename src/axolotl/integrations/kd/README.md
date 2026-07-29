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
vllm serve <teacher-model> --port 8000 --no-enable-prefix-caching
```

Prefix caching has to be disabled because prompt logprobs are not produced for cached
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
`return_logprob`, `top_logprobs_num` and `max_new_tokens: 0`.

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
| `kd_online_server_base_url` | `None` | enables the online teacher |
| `kd_online_server` | `vllm` | `vllm` or `sglang` |
| `kd_online_topk` | `None` | required with an online teacher |
| `kd_online_timeout` | `120` | per-request timeout, in seconds |

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
