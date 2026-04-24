# Multimodal assistant-only loss masking

## What this fixes

For multimodal fine-tuning, `cfg.train_on_inputs`, `cfg.roles_to_train`, and
`cfg.train_on_eos` were silently ignored. Every non-pad, non-media token in
the sequence — including system prompts, user turns, and role markers —
contributed to the loss. Only Gemma3n had a working per-role mask; every
other multimodal path (LLaVA, Qwen2-VL, Qwen3.5, Gemma3, Llama-3.2 Vision,
Llama 4, Pixtral, Mistral V7 Tekken, Voxtral, SmolVLM2, Mistral3, InternVL,
GLM4V) trained on the full sequence.

## Root cause

`MultiModalChatDataCollator` re-tokenizes raw `messages` via
`processor.apply_chat_template(...)` at collation time, discarding the
per-role labels already computed by `ChatTemplateStrategy.tokenize_prompt` in
the preprocessing path. It then calls
`processing_strategy.process_labels(input_ids)`, which was supposed to rebuild
role-aware labels — but the base `_mask_non_assistant` was a no-op `return
labels`, and only `Gemma3nProcessingStrategy` overrode it. So for every other
multimodal model, the retokenized labels are never masked by role.

## Design

We make role masking a first-class, declarative capability of the base
`ProcessingStrategy` and thread the masking knobs through from the trainer
builder.

### Why this over alternatives

- **Option (b): preserve the per-role labels from `tokenize_prompt`.**
  Rejected. The preprocessing labels were computed against a text-only
  tokenization; they don't align with the MM collator's re-tokenization after
  image/audio/video placeholders expand into hundreds of placeholder tokens.
  Preserving them would require either a second tokenization pass with image
  stand-ins, or rewriting the collator to never re-tokenize. Either is
  high-blast-radius for an incremental bugfix.
- **Option (c): `apply_chat_template(return_assistant_tokens_mask=True)`.**
  Rejected. This requires `{% generation %}` / `{% endgeneration %}` jinja
  markers. Only `llava.jinja` and `phi_4.jinja` have them in
  `src/axolotl/utils/chat_templates/templates/`. Adding these markers to
  upstream-mirrored templates (gemma3, qwen2_vl, llama3_2_vision, etc.)
  diverges from the reference templates and is fragile when HF updates them.
- **Option (a): parametrized token-boundary scanner in the base class.**
  Chosen. Each strategy declares its per-role boundary markers
  (`<|im_start|>assistant\n` ... `<|im_end|>` for Qwen2-VL,
  `<|turn>model` ... `<turn|>` for Gemma 4, etc.). The base scanner walks the
  re-tokenized sequence, locates role spans, and masks everything outside
  `cfg.roles_to_train`. Works with existing jinja templates, is testable
  offline with fake tokenizers, and fails visible (unverified strategies emit
  a one-shot warning rather than silently mis-masking).

### Components

1. **`RoleBoundary`** dataclass in `src/axolotl/processing_strategies.py`
   describing one role's `(start_tokens, end_tokens, include_start, include_end)`.
2. **`_apply_role_boundaries`** function: a longest-prefix-match scanner that
   implements `roles_to_train` / `train_on_inputs` / `train_on_eos` (`"turn"`
   keeps role-end markers on trainable turns, `"all"` keeps them on every
   turn, `"none"` excludes them).
3. **`ProcessingStrategy._build_role_boundaries`**: empty by default;
   overridden by each subclass. `_mask_non_assistant` delegates to the
   scanner; if no boundaries are declared it short-circuits and emits a
   one-shot warning (legacy behavior preserved).
4. **Plumbing**: `cfg.train_on_inputs` (top-level) and the first dataset's
   `roles_to_train` / `train_on_eos` (per-dataset, under `datasets[0]`) are
   threaded through `build_collator` → `get_processing_strategy` → each
   strategy's constructor. See *Where to put the masking knobs in YAML*
   below — top-level `roles_to_train` / `train_on_eos` are silently ignored.

## Audit table

| Strategy / chat template | Honors `roles_to_train`? (before) | (after) | Role-boundary markers | Media tokens masked |
|---|---|---|---|---|
| `ProcessingStrategy` (fallback for `llava`, `lfm2vl`, `mistral_v3_tekken`, unknown) | ✗ | fallback + warn | *unverified* | `image_token_id` if processor exposes it |
| `Qwen2VLProcessingStrategy` (`qwen2_vl`) | ✗ | ✓ | `<\|im_start\|>{role}\n` ... `<\|im_end\|>` | `<\|image_pad\|>` |
| `Qwen3_5ProcessingStrategy` (`qwen3_5`) | ✗ | ✓ | same as Qwen2VL | `<\|image_pad\|>`, `<\|video_pad\|>` |
| `Gemma3ProcessingStrategy` (`gemma3`) | ✗ | ✓ | `<start_of_turn>{model/user/system}\n` ... `<end_of_turn>` | `boi_token`, `<image_soft_token>` (262144) |
| `Gemma3nProcessingStrategy` (`gemma3n`) | ✓ (ad-hoc) | ✓ (shared scanner) | same as Gemma 3 | `image_token_id`, `audio_token_id`, `boi_token_id`, `eoi_token_id` |
| `Gemma4ProcessingStrategy` (`gemma4`) | n/a (new) | ✓ | `<\|turn>{model/user/system}` ... `<turn\|>` | `image_token_id`, `audio_token_id`, `boi/eoi/boa/eoa` (resolved via `convert_tokens_to_ids`), `video_token_id` (on processor) |
| `Llama3_2VisionProcessingStrategy` (`llama3_2_vision`) — **new** | ✗ | ✓ | `<\|start_header_id\|>{role}<\|end_header_id\|>\n\n` ... `<\|eot_id\|>` | `image_token_id` via base |
| `Llama4ProcessingStrategy` (`llama4`) — **new** | ✗ | ✓ | `<\|header_start\|>{role}<\|header_end\|>\n\n` ... `<\|eot\|>` | `image_token_id` via base |
| `PixtralProcessingStrategy` (`pixtral`) — **new** | ✗ | ✓ | user: `[INST]` ... `[/INST]` (`include_end=False`), assistant: `[/INST]` ... `eos_token` | `image_token_id` via base |
| `MistralV7TekkenProcessingStrategy` (`mistral_v7_tekken`) — **new** | ✗ | ✓ | `[SYSTEM_PROMPT]` ... `[/SYSTEM_PROMPT]`, `[INST]` ... `[/INST]` (`include_end=False`), assistant: `[/INST]` ... `eos_token` | `image_token_id` via base |
| `VoxtralProcessingStrategy` | ✗ | fallback + warn | *unverified* (mistral-common tokenizer) | `audio_token`, `begin_audio_token` |
| `SmolVLM2ProcessingStrategy` | ✗ | fallback + warn | *unverified* (checkpoint-dependent default) | `<image>` |
| `Mistral3ProcessingStrategy` | ✗ | fallback + warn | *unverified* (mistral-common tokenizer) | `img`, `img_break`, `img_end` |
| `InternVLProcessingStrategy` | ✗ | fallback + warn | *unverified* (InternLM-family) | `processor.image_ids` |
| `Glm4vProcessingStrategy` | ✗ | fallback + warn | *unverified* | image/video + begin/end markers |

Pixtral and Mistral V7 Tekken share a token (`[/INST]`) between the user-end
and assistant-start markers. The scanner supports this via `include_end=False`
on the user boundary: when the scanner hits an end marker that is also another
boundary's start, it rewinds past it so the next iteration can match the
shared token as the next role's start. See commit `acfe4fe4` and the full
per-position assertions in `tests/test_processing_strategies.py`.

*unverified*: the right boundary markers cannot be confirmed without a real
checkpoint; the fallback preserves the legacy "mask pad + media tokens only"
behavior and emits a one-shot warning naming the strategy class so the miss
is visible in training logs. To enable role masking for one of these models,
subclass the strategy and implement `_build_role_boundaries` — see the Gemma
and Qwen implementations for the pattern.

## Where to put the masking knobs in YAML

`roles_to_train` and `train_on_eos` are **per-dataset** fields — they live
under each entry of `datasets:` (and `test_datasets:`), not at the root of
the config. Only `train_on_inputs` is read from the top level. The
multimodal collator resolves the mask knobs in `build_collator` with:

```python
ds_cfg = (self.cfg.datasets or [None])[0]
roles_to_train = _ds_get(ds_cfg, "roles_to_train")
train_on_eos   = _ds_get(ds_cfg, "train_on_eos")
# ... then passed to get_processing_strategy(..., roles_to_train, train_on_eos)
```

There is no fallback to a top-level `cfg.roles_to_train` / `cfg.train_on_eos`,
and the schema (`ChatTemplateDatasetConfig` in `utils/schemas/datasets.py`)
only defines these fields at the dataset level. If you put them at the root,
they are silently ignored for the MM path.

**Why this is a trap:** when the resolver returns `None`,
`ProcessingStrategy.__init__` falls back to its defaults — `["assistant"]`
and `"turn"` — which happen to be what most users want. So the loss *looks*
correctly masked to assistant-only, but the declared intent in the YAML is
dead code. Any future change to those defaults, or to a non-default value
the user intended to set, will silently flip the behavior.

### Correct placement

```yaml
# Top-level: only train_on_inputs lives here.
train_on_inputs: false

datasets:
  - path: data/train.jsonl
    type: chat_template
    roles_to_train:          # per-dataset — this is what the MM scanner reads
      - assistant
    train_on_eos: turn       # per-dataset — same

test_datasets:
  - path: data/val.jsonl
    type: chat_template
    split: train
    roles_to_train:
      - assistant
    train_on_eos: turn
```

### How to verify at runtime

`build_collator` logs the resolved knobs at INFO:

```
MM collator: train_on_inputs=False roles_to_train=['assistant'] train_on_eos=turn role_boundaries_override=none
```

If `roles_to_train` logs as `None`, the YAML knobs are not reaching the
scanner — check that they are under `datasets[0]`, not at the root.

Each verified strategy additionally logs its resolved boundary token ids at
strategy init (e.g. `<|turn>model` → `[105, 4368]`, `<turn|>` → `[106]` for
Gemma 4). If a strategy emits the "has no built-in role boundaries ... only
pad and media tokens are masked" one-shot warning instead, it is on the
fallback path — declare per-role markers in YAML via `cfg.role_boundaries`
(below) to activate masking. The strategies currently on this path are
listed in the audit table above under `fallback + warn`.

## Config-based override: `cfg.role_boundaries`

For the "unverified" strategies above, or for custom chat templates that
don't match a built-in strategy's markers, users can declare role boundaries
directly in YAML without subclassing:

```yaml
role_boundaries:
  - role: assistant
    start: "<|turn>model"
    end: "<turn|>"
  - role: user
    start: "<|turn>user"
    end: "<turn|>"
  # Optional keys:
  # include_start: false   # default False
  # include_end: true      # default True, respects cfg.train_on_eos
  # end: eos_token         # sentinel: resolves to tokenizer.eos_token_id
  # end: null              # span runs to end of sequence
```

Semantics:

- `start` and `end` are literal strings; axolotl encodes them at strategy
  init via `tokenizer.encode(..., add_special_tokens=False)` and logs the
  resolved token-id sequences at INFO level.
- The special value `end: eos_token` is the portable way to express
  "Pixtral-style assistant turns end at EOS" without hard-coding an id.
- `role_boundaries` is an **opt-in override**. A non-empty list **replaces**
  the strategy's built-in declarations wholesale (partial overlays are
  intentionally unsupported — they're hard to reason about at review time).
  Leaving the field unset *or* setting it to an empty list (`[]`) both mean
  "use the strategy's built-ins." Writing `role_boundaries: []` is almost
  always a typo or leftover — honoring it literally would produce all-masked
  labels and zero gradient, so it is treated the same as unset.
- `cfg.roles_to_train` still governs which declared roles contribute to
  loss. You can declare `user` and `assistant` boundaries and set
  `roles_to_train: ["assistant"]` to have the scanner correctly identify
  user spans as masking boundaries without training on their content.
- Invalid specs fail loudly at strategy init (missing `role`/`start`,
  unencodable markers), not silently at loss-compute time.

## Commits on this branch

Run `git log main..HEAD --oneline` for the authoritative sequence. As of
this revision the logical units are:

1. **`feat: systemic multimodal assistant-only loss masking`** — core
   refactor of `processing_strategies.py` (`RoleBoundary`,
   `_apply_role_boundaries`, `_build_role_boundaries`), per-strategy boundary
   declarations, dispatcher routing for new subclasses.
2. **`feat: thread cfg.train_on_inputs / roles_to_train / train_on_eos into
   MM collator`** — `build_collator` reads the knobs from `cfg` and the
   first dataset entry and passes them to `get_processing_strategy`.
3. **`docs: multimodal assistant-mask design doc`** — this file.
4. **`feat: cfg.role_boundaries YAML override for MM role-mask scanner`** —
   schema field (`MultiModalConfig.role_boundaries`), resolver that converts
   string markers to token ids at strategy init, ``eos_token`` sentinel, and
   wiring through ``build_collator`` / ``get_processing_strategy`` /
   every strategy constructor.
5. **`test: additional coverage for MM role-mask scanner edge cases`** —
   expands the unit test suite covering scanner semantics, per-strategy
   masking, media-token masking within assistant spans, dispatcher
   routing, and override semantics (replace built-in, enable on unverified
   strategy, eos_token sentinel, null end, validation errors, pydantic
   model input).
6. **`chore: tighten docstrings and comments in multimodal mask refactor`**
   — no-behavior-change polish.
7. **`fix: resolve MM per-dataset masking knobs for pydantic SFTDataset`**
   — `build_collator` resolver now uses `.get` → `getattr` fallback so
   `roles_to_train` / `train_on_eos` are honored when datasets are supplied
   as pydantic models (not just `DictDefault`). Adds an INFO log of the
   resolved collator knobs.

## Verification

- All 64 unit tests pass offline (`pytest tests/test_processing_strategies.py`).
- End-to-end check against real tokenizers:
  - `google/gemma-4-E2B-it`: 13/40 tokens kept for a 2-turn chat; decoded
    preview shows only assistant responses + `<turn|>` markers remain.
  - `axolotl-ai-co/Llama-3.3-70B-Instruct-tokenizer` (with bundled
    `llama3_2_vision.jinja`): 11/64 tokens kept; content correctly resolves
    to `"The capital of France is Paris.<|eot_id|>"` and `"Berlin.<|eot_id|>"`.
- Verified boundary token ids against the real Gemma 4 tokenizer:
  `<|turn>model` → `[105, 4368]`, `<turn|>` → `[106]`, `<|image|>` → `258880`,
  `<|audio|>` → `258881`, `<|video|>` → `258884`.

## Draft upstream PR description

> Fix silently-ignored `train_on_inputs` / `roles_to_train` / `train_on_eos`
> in the multimodal training path.
>
> **Why this matters**: for every multimodal model except Gemma 3n, loss was
> computed on the entire sequence (minus pad and media tokens) regardless of
> what `roles_to_train` / `train_on_inputs` the config specified. This
> silently turned assistant-only SFT into full-sequence SFT for thousands of
> users, degrading sample efficiency and introducing spurious gradient signal
> on system and user content.
>
> **What changed**:
> - `ProcessingStrategy._build_role_boundaries` declares per-role start/end
>   token sequences. The base `_mask_non_assistant` now consumes those
>   declarations via a shared scanner that honors `train_on_inputs`,
>   `roles_to_train`, and `train_on_eos`.
> - Per-strategy boundary declarations added for Qwen2-VL, Qwen3.5, Gemma 3,
>   Gemma 3n (refactored from ad-hoc scanner), Gemma 4 (new), Llama 3.2
>   Vision (new), Llama 4 (new), Pixtral (new), Mistral V7 Tekken (new).
> - Strategies whose boundary tokens we couldn't verify against a real
>   tokenizer (Voxtral, SmolVLM2, Mistral3, InternVL, GLM4V, and the
>   llava/lfm2vl/unknown-template fallback) retain legacy behavior but emit a
>   one-shot warning so the miss is visible in training logs.
> - `cfg.train_on_inputs` / `cfg.datasets[0].roles_to_train` /
>   `cfg.datasets[0].train_on_eos` are threaded through
>   `HFCausalTrainerBuilder.build_collator` → `get_processing_strategy` →
>   strategy constructor.
>
> **Testing**: 64 offline unit tests; end-to-end verified with the real
> Gemma 4 and Llama 3.x tokenizers.
