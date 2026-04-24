# Multimodal assistant-only loss masking

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
