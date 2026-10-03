# Projection Sampling

Implements offline sampling from [Finetuning with Sampling: SFT Learns Better Than You Think](https://arxiv.org/abs/2610.02140), followed by Axolotl's ordinary SFT pipeline. Enable `axolotl.integrations.projection_sampling.ProjectionSamplingPlugin` in `plugins:`. See [the example config](../../../../examples/projection-sampling/qwen2.5-3b.yaml).

```bash
axolotl preprocess examples/projection-sampling/qwen2.5-3b.yaml
axolotl train examples/projection-sampling/qwen2.5-3b.yaml
```

Run preprocessing in a single process. It loads the unadapted `base_model` through the selected inference backend, rewrites trainable assistant replies or flat question/expert response pairs, saves an atomic JSONL cache, unloads the sampling model, and runs normal dataset preparation. Training requires that cache and never loads a sampling model. Evaluation datasets use normal Axolotl processing. `val_set_size` splits the transformed training data; use `test_datasets` for held-out original data.

Preprocessing also exports a separate inspection dataset at `<output_dir>/projection-sampling/rewritten.jsonl`. It contains rewritten `messages` (and tools where available) for chat sources, or `prompt`, `response`, and `expert_response` for flat pairs, plus readable sampling metadata and `sampling_seed`. Token IDs, labels, masks, and sampled-token arrays stay in the internal caches and are omitted from this export. Files are written atomically, so inspection sees only a complete dataset.

The inspection export is refreshed from the sampling cache on cache hits too, including when training uses a different `output_dir`. Changing only `output_dir` does not trigger sampling. Distributed training exports on rank zero. The file contains all sampled source rows before the normal SFT preparation filters and validation split; a later run in the same output directory replaces it with that run's dataset. `projection_sampling.cache_dir` and `dataset_prepared_path` still control the reusable sampling and prepared caches.

The input `datasets:` use Axolotl's dataset loader, including local JSON/Parquet, Hub datasets, splits, shards, and weights. Use the existing `type: chat_template` for role-based message datasets:

```yaml
chat_template: tokenizer_default
datasets:
  - path: ./expert-traces.jsonl
    ds_type: json
    type: chat_template
    roles_to_train: [assistant]
```

```json
{"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Expert reasoning and answer"}]}
```

The plugin loads Axolotl's standard `chat_template` strategy with the dataset configuration. `field_messages`, `message_property_mappings`, `roles`, `drop_system_message`, `roles_to_train`, per-message training flags, EOS/EOT policies, configured templates, and `chat_template_kwargs` keep their usual meaning. For example, ShareGPT-style conversations can use `field_messages: conversations`, `message_property_mappings: {role: from, content: value}`, and `roles: {user: [human], assistant: [gpt]}`. JSON-encoded message lists are also supported. See the [chat template dataset documentation](../../../../docs/dataset-formats/conversation.qmd).

Each trainable assistant reply is sampled with the entire preceding conversation as the target context, including system instructions and earlier replies. Rewrites proceed in conversation order, so subsequent turns use rewritten history. The current expert reply appears only in the proposal. The proposal uses the same selected template, template kwargs, and tools. Its question field contains the preceding messages serialized as JSON; native chat control tokens would otherwise interrupt the surrounding proposal message. Training flags and loss-mask offsets are omitted from this text. After rewriting, the existing parser produces the full conversation's token IDs and loss labels; preprocessing caches those labels for training. If rendering a rewritten turn would change the sampled token IDs or generation prefix, that turn falls back to the expert reply.

Non-trainable turns remain unchanged. Turns with character-based training spans or per-part training masks remain unchanged because rewriting would invalidate their offsets. Assistant tool calls and separately stored reasoning fields also remain unchanged; their structured fields and masks are retained by the standard parser. Cache metadata records skipped turns. This initial chat implementation rewrites fully trainable text assistant replies.

For compatibility, a dataset without `type:` can contain flat string columns selected by `question_field` and `response_field` (defaults: `prompt`, `response`). This path preserves sampled token IDs and masks the question unless `train_on_inputs: true`. Its `prompt_format: chat` uses the tokenizer's chat template; `raw` supports base models. These flat-pair options do not control `type: chat_template` datasets. Other dataset strategies are rejected. Model context limits include the conversation, expert solution, proposal template, and continuation; shorten inputs or reduce `max_new_tokens` if needed.

`block_size` defaults to 32, `max_new_tokens` to 1856, and `mcmc_steps` to 10. The reply budget includes reasoning and answer tokens. With `type: chat_template`, use `chat_template_kwargs: {enable_thinking: true}` to enable reasoning for templates that support it. See the [Qwen3.5 base-model math example](../../../../examples/projection-sampling/qwen3.5-4b-base-math.yaml) for a reasoning-enabled configuration with the chat turn terminator. The final block can be shorter. Set `mcmc_steps: 0` for the rewrite-only baseline. `temperature` defaults to 0.6 and `repetition_penalty` to 1.0. `proposal_template` must contain `{question}`, `{expert_response}`, and `{prefix}`; escaped braces are allowed. `device` defaults to `cuda`; use `cpu` and `dtype: float32` for Transformers CPU sampling.

`acceptance: metropolis_hastings` (default) uses summed base-model log probabilities and explicitly recomputes both proposal densities under the same expert-conditioned prefix. Proposal scoring uses the same temperature and repetition penalty as generation. The cut-index probability ratio accounts for EOS changing trajectory length. Finite blockwise sampling approximates the paper's target; prompting alone does not enforce semantic equivalence.

`acceptance: greedy` reproduces the public [reference sampler's](https://github.com/aakaran/finetuning-with-sampling/tree/6d3e9f0bfaa98dcca534247dd35dc1b33dd8c428) acceptance criterion: accept only an improvement in mean base-model token likelihood. To match its sampling settings, also set `repetition_penalty: 1.1`. The reference uses vLLM and task-specific templates; this implementation defaults to Transformers and a general template, so it does not promise identical traces across backends or prompts. Greedy acceptance is a heuristic rather than the paper's MH kernel.

An optional `verifier: my_package.check_response` resolves a callable receiving keyword arguments `question`, `expert_response`, and `response` and returning a boolean. Verification occurs on the final trace, not partial block states. A failed verification, empty response, or trace that exhausts the generation budget without EOS falls back to the original expert response. Cache metadata records acceptance counts, the sampled trace's likelihood, completion, verification, and fallback status. The verifier must be deterministic for reproducibility. Without a verifier, completed responses rely on the proposal instruction to preserve information; validate their correctness for your task.

Caches depend on backend selection and backend options, sampling settings, model/tokenizer configuration, dataset configuration, chat tokenization and mask settings, and local source file contents. Local model/tokenizer file sizes and modification times also enter the fingerprint. Pin Hub dataset and model revisions for reproducible remote inputs. Delete the projection cache and prepared dataset cache to intentionally resample unchanged configurations or changed unpinned remote inputs or verifier implementations. The sampling cache preserves EOS, exact token boundaries, and the chat parser's loss labels. Sampling uses Axolotl's top-level `seed` (default 42) for its private Python RNG, backend generation, and source shuffling. Changing the top-level seed creates a new cache fingerprint and resamples during preprocessing; there is no `projection_sampling.seed` override. Transformers scopes its PyTorch RNG; vLLM uses a new deterministic seed for each generation request. Scoring requests do not advance the generation seed sequence. Existing flat-pair caches from the default Transformers backend remain reusable.

This integration supports text SFT and adapters, including LoRA/QLoRA training after sampling. Transformers sampling loads full base-model weights on one device; vLLM supports its own tensor parallelism. Standard training quantization, distributed sharding, custom model-loading patches, and adapters do not apply to the sampler. Adding vocabulary tokens to the base model is unsupported. Streaming, RL, pretraining, multimodal processors, `skip_prepare_dataset`, dataset `input_transform`, and `preprocess_shards` are unsupported. Teacher-forced scoring can be expensive for long traces, particularly MH proposal scoring. Neither backend promises the paper's reported throughput or benchmark results.


## Inference backends

`projection_sampling.backend` defaults to `transformers`. Select `vllm` to use a local vLLM engine during preprocessing; install Axolotl's optional `vllm` dependencies first. The sampling engine shuts down before standard dataset preparation, and training consumes cached tokens without requiring the selected inference runtime to be installed. [Example vLLM config](../../../../examples/projection-sampling/qwen2.5-3b-vllm.yaml):

```yaml
projection_sampling:
  backend: vllm
  device: cuda
  dtype: bfloat16
  backend_kwargs:
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.8
    max_model_len: 8192
    enforce_eager: false
    enable_prefix_caching: true
    score_batch_size: 4
```

`backend_kwargs` are validated engine options. All options above except `max_model_len` show their defaults; omitted `max_model_len` uses the model's context limit. Optional `attention_backend` selects a vLLM attention kernel, such as `TRITON_ATTN`; omitted, vLLM selects one for the hardware. `enforce_eager: true` disables vLLM compilation and CUDA graph capture. Select visible GPUs with `CUDA_VISIBLE_DEVICES`; vLLM requires `device: cuda` and supports `tensor_parallel_size` across those visible devices. Its accepted temperature is at least 0.01 to avoid vLLM's low-temperature clamping. The backend passes token IDs directly and uses the same Axolotl tokenizer as SFT, including configured chat templates. Model generation defaults cannot silently introduce extra penalties or sampling filters. EOS IDs from the model's generation configuration are retained. The backend factory also adds configured `eot_tokens` as stop IDs so chat turns can finish even when the model's EOS differs from its chat terminator.

vLLM prompt log probabilities are unprocessed, so they score the base-model target. Proposal scores come from the processed next-token distributions with the same temperature and repetition penalty as generation. The adapter scores known suffix tokens in batches of at most `score_batch_size`. Dataset rows are sampled serially; this setting batches scoring requests, and `dataset_processes` controls normal dataset preparation rather than sampling concurrency. vLLM versions with `SamplingParams.logprob_token_ids` return only the requested token's score; earlier supported versions return the full next-token distribution. The full-distribution path can use substantial host memory with large vocabularies; reduce `score_batch_size` if needed. This design avoids treating unscaled prompt probabilities as the proposal distribution. No Transformers model is loaded alongside vLLM for scoring.

## Batched proposals

Set `projection_sampling.proposal_batch_size` to generate multiple candidate replies at the same cut. The default is 1, preserving ordinary MH and existing cache identity. For example:

```yaml
projection_sampling:
  backend: vllm
  proposal_batch_size: 4
  acceptance: metropolis_hastings
  mcmc_steps: 3
  backend_kwargs:
    enable_prefix_caching: true
    score_batch_size: 32
```

Each MH step sends all candidates in one generation batch with the same expert-conditioned prefix. vLLM handles their independent request seeds and can reuse cached prompt prefixes. Sharing depends on the model's prefix-cache support and block boundaries. Later MH steps use the selected chain state, so they remain sequential. The final dataset still has one rewritten row per source row.

Acceptance uses [multiple-try independent Metropolis](https://arxiv.org/pdf/2111.15084), applied conditionally at the fixed cut. Candidate log weights are `target_logprob - proposal_logprob - log(length)`. The length factor accounts for the uniformly sampled cut. A candidate is selected proportionally to those weights. The reverse balancing set reuses the unselected candidates and includes the current state; the acceptance probability compares the two sums of weights. Conditional independence avoids a second generation batch. All proposal scores still match temperature and repetition penalties.

With `acceptance: greedy`, batching selects the candidate with the highest mean target likelihood and accepts it only if it improves on the current state. This extends the reference heuristic; it does not use MH weights. `mcmc_steps: 0` skips candidate batches. Sampling metadata reports `proposal_batch_size`, `forward_proposals`, and `balancing_proposals_reused` when batching is enabled; `attempts` and `accepted` count chain steps.

The backend contract also has optional `sample_batch(contexts, max_tokens)`, `target_logprob_batch(contexts, tokens)`, and `proposal_logprob_batch(contexts, tokens)` methods. Results must follow input order, with one item per input. Budgets and continuations may have different lengths. Native vLLM implementations batch generation and target scoring and bound processed next-token scoring across candidates by `score_batch_size`. Default implementations call the single-request methods sequentially, so existing external backends and Transformers remain compatible. New runtimes can override these methods for native batching.

## Adding a backend

The algorithm and dataset plugin depend only on `SamplingBackend` in `backend.py`. Transformers and vLLM live in separate, lazily imported modules under `backends/`. An external backend, including a future SGLang implementation, can subclass `SamplingBackend` and be selected without changing the sampler or plugin:

```yaml
projection_sampling:
  backend: my_package.backends.SGLangBackend
  backend_kwargs:
    server_url: http://localhost:30000
```

The external class implements `from_config(cfg, config)`, `sample(context, max_tokens)`, `target_logprob(context, tokens)`, `proposal_logprob(context, tokens)`, and idempotent `close()`. It exposes `tokenizer` and `eos_token_ids`; tokenization must match Axolotl's SFT tokenizer. `sample()` returns continuation token IDs and stops only on EOS or the token budget. Both score methods sum log probabilities of the supplied continuation, including EOS, and exclude the prompt. Proposal scoring must reproduce the actual generation distribution, including temperature, penalties, and normalization; target scoring uses the unmodified base distribution. Never length-normalize the score at the backend layer.

The factory owns `close()` on success and failure. A backend's `from_config()` must release partially initialized resources if construction fails. Override the optional `rng_context(cfg, config)` class method when a runtime needs its RNG state scoped. External implementations validate their own `backend_kwargs`. An SGLang adapter is not bundled yet.
