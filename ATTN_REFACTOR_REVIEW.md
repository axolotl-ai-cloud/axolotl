# `attn-implementation-refactor` branch review

Review target: `attn-implementation-refactor` (5 commits ahead of main, merge base `69904781`).
Scope: 16 files, +682 / −96.

## 1. What the branch is trying to do

Collapse seven boolean attention flags (`flash_attention`, `sdp_attention`, `xformers_attention`, `flex_attention`, `sage_attention`, `s2_attention`, `eager_attention`) into a single `attn_implementation` field, with derived capability flags (`attn_supports_packing`, `attn_uses_flash_lib`, `attn_needs_dtype_cast`) for the gates that used to be ad-hoc OR-chains.

Mechanism: `normalize_attn_implementation` (a `@model_validator(mode="before")` on `AxolotlInputConfig`) maps bidirectionally between the new field and the legacy flags, with a priority list for legacy combos (`s2 + flash → s2`), and then computes the three capability flags from frozen sets in `enums.py`.

Adjacent changes: `xformers` and `sage` now register as their own entries in `ALL_ATTENTION_FUNCTIONS` (with FA2 mask behavior) instead of stomping the `flash_attention_2` slot. New `fp8` backend wires `torchao.prototype.attention.apply_low_precision_attention` in `apply_post_model_load_patches`.

## 2. Target design

**`cfg.attn_implementation` is the single source of truth on the validated config.**

- Its type is `str | None`. Accepted values are **canonical names only** — no short-form aliases:
  - HF-native: `eager`, `sdpa`, `flash_attention_2`, `flash_attention_3`, `flex_attention`. (`flash_attention_3` is net-new to axolotl — the current branch only encodes `flash_attention_2` under the short name `flash`.)
  - Axolotl-owned (registered into `ALL_ATTENTION_FUNCTIONS` under exactly these names): `xformers`, `sage`, `s2`, `fp8`.
  - Hub-kernel paths: `kernels-community/sage-attention`, `kernels-community/flash-attn3`, etc. — passthrough. Known-kernel allowlist in `enums.py` classifies the common ones into the capability tables.
  Short forms like `flash`, `fa2`, `fa3`, `sdp`, `flex` are rejected (Pydantic validation error with a pointer to the canonical name).
- `model.py:_set_attention_config` passes `cfg.attn_implementation` to HF verbatim — no `_ATTN_IMPL_TO_HF` translation dict needed.
- Legacy booleans (`flash_attention: true`, `sdp_attention: true`, …) are the **only** input aliases, kept for backwards compatibility. The normalizer maps them to the canonical `attn_implementation` value, emits a one-time `DeprecationWarning` per flag, and removes them from `data` so they're never readable on the validated `cfg`. `deprecated=True` on each Field surfaces this in JSON schema. Mapping is 1:1 with the current legacy-flag semantics (`flash_attention → flash_attention_2`, `sdp_attention → sdpa`, `flex_attention → flex_attention`, `xformers_attention → xformers`, `sage_attention → sage`, `s2_attention → s2`, `eager_attention → eager`).
- Capability flags (`attn_supports_packing`, `attn_uses_flash_lib`, `attn_needs_dtype_cast`) are **`@computed_field` on the model**, not settable inputs. Lookup is keyed by the canonical `attn_implementation` string.
- Unknown / user-supplied strings (custom hub kernels) pass through to HF but get **conservative capability defaults** (packing=False, flash-lib=False, dtype-cast=True). Known hub kernels axolotl can classify live in a small allowlist.
- Downstream consumers read *only* `cfg.attn_implementation` and the capability flags. No `cfg.flash_attention`, `cfg.xformers_attention`, etc. anywhere in `src/`.

This is strictly what the branch is already *trying* to do — the gaps below are places it hasn't landed that goal yet.

## 3. Gaps and holes

### A. Legacy flags are still parallel state, not input-only

1. The normalizer *sets* the legacy flags on `data` (`impl_to_flag[attn_impl]` branch). It does not delete them. So `cfg.flash_attention` is still truthy after validation, and downstream code still reads it (see G).
2. Short-form enum values (`flash`, `sdpa`, `fp8`) are persisted as-is on `cfg.attn_implementation`, which is why `model.py` needs `_ATTN_IMPL_TO_HF` to translate before passing to HF. Source-of-truth implies canonicalize at normalize-time, not translate at consume-time.
3. Legacy flag + `attn_implementation` (consistent combo, e.g. `attn_implementation: flash + flash_attention: true`) emits no deprecation warning — only legacy-only path warns.
4. Legacy Field descriptions (`xformers_attention`, `sdp_attention`, etc.) don't have `deprecated=True`, so JSON schema still advertises them as first-class.

### B. Validators that still only check the legacy flag

5. **`check_ebft_activation_offloading`** (`validation.py:1607-1619`) reads only `data.get("flex_attention")`. Users on `attn_implementation: flex_attention` bypass the incompatibility check.
6. **`check_sample_packing_without_attention`** (`validation.py:188-203`) early-returns when `attn_implementation` is set but never validates the chosen backend actually supports packing. `attn_implementation: eager + sample_packing: true` silently passes; the old legacy-flag check warned.

### C. Non-enum strings fall through the capability tables

7. **HF-native `"flash_attention_2"`** is neither in `impl_to_flag` nor `FLASH_ATTN_LIB_IMPLS`. A user copy-pasting from HF docs gets `attn_uses_flash_lib=False`, silently disabling FA4 auto-apply, LLaMA flash hijack, `_patch_attention` (btlm, stablelm_epoch, mistral3, llava), and `_apply_flash_attention_peft_patches`.
8. **Hub kernel strings** (`kernels-community/flash-attn3`, `kernels-community/sage-attention`) default to `attn_supports_packing=True` (silently enters multipack with varlen `position_ids` — correctness depends on the kernel honoring them) and `attn_uses_flash_lib=False` (so `context_parallel_size > 1` raises "requires flash attention" even for FA3 hub kernels).
9. **Conflict trap for hub-kernel + legacy flag** (`config.py:1414-1419`): `attn_implementation: kernels-community/flash-attn3 + flash_attention: true` always raises, because `impl_to_flag.get(custom) is None` and the loop treats `flag != None` as conflict. Common combo in existing YAMLs breaks hard on upgrade.

### D. Silent behaviour change for xformers

10. Old `_apply_flash_attention_patches` did `self.cfg.flash_attention = True` for `xformers + sample_packing`. The new version doesn't, and xformers is not in `FLASH_ATTN_LIB_IMPLS`. Consumers that keyed off `cfg.flash_attention` now see falsy for xformers, silently dropping `_patch_attention` (btlm / stablelm_epoch+packing / mistral3 / llava model-type FA patches). Some of this is arguably correct cleanup (xformers has its own HF registry entry now), but the btlm/stablelm/mistral3 regression is not called out and not tested. Decide consciously, not by omission.

### E. Capability flags are writable Pydantic fields, not computed

11. `attn_supports_packing`, `attn_uses_flash_lib`, `attn_needs_dtype_cast` are declared `bool | None = Field(default=None)` on `AxolotlInputConfig`. YAML is not rejected — a user can set `attn_uses_flash_lib: true` and override the normalizer.

### F. Validator ordering (not covered by tests)

12. `AttentionValidationMixin.check_attention_fields` (inherited, `mode="before"`) and `normalize_attn_implementation` (subclass, `mode="before"`) both run during `model_validator` phase. Pydantic MRO may run the inherited one first. For legacy-only `s2_attention: true + flash_attention: true` (the test `test_s2_plus_flash_maps_to_s2` asserts this maps to `s2`), the inherited check may raise "multiple attention implementations set" before the normalizer runs. The test calls the classmethod directly and does not build the model, so this is unverified either way.

### G. Remaining legacy reads in `src/`

13. `src/axolotl/integrations/lm_eval/cli.py:120` reads `cfg.flash_attention`. Works for `attn_implementation=flash` only.
14. `tests/e2e/multigpu/test_llama.py:524-526` writes `cfg.flash_attention = True` / `cfg.flex_attention = True`. Stale pattern.
15. Dual-check idioms in `config.py` (lines 1464, 1478, 1570, 1586, 1774) and `validation.py` (198, 209, 221, 850, 1586, 1611) — `data.get("x_attention") or data.get("attn_implementation") == "x"` — are redundant once legacy flags are input-only; remove them.

### H. fp8 operational risk

16. The `fp8` docstring documents hard requirements (PyTorch ≥ 2.11, SM90+, flash-attn with FA3, torchao ≥ 0.17.0) and a runtime constraint (`config.use_cache = False`). None are validated — misconfig surfaces as a torchao runtime error. `xformers` and `sage` availability/compute-capability guards exist; `fp8` should match.

### I. Test coverage gaps

17. `test_attn_implementation.py` exercises the classmethod in isolation plus the constant sets. It does **not**:
   - Build a full `AxolotlInputConfig(**data)` (so validator ordering — item 12 — is untested).
   - Verify capability flags can't be overridden from YAML (item 11).
   - Cover `check_sample_packing_without_attention` with `attn_implementation: eager` (item 6).
   - Cover `check_ebft_activation_offloading` with `attn_implementation: flex_attention` (item 5).
   - Cover hub-kernel + legacy flag combo (item 9).
   - Cover `flash_attention_2` canonicalization (item 7).

## 4. Fix plan

Four phases, each commit-sized. Phases 1–2 are local and low-risk; phase 3 is the behaviour-changing cleanup; phase 4 is tests + docs.

### Phase 1 — Lock down the data model

1. Drop the `AttnImplementation` enum. `attn_implementation` becomes `str | None`, validated against a canonical allowlist (`eager`, `sdpa`, `flash_attention_2`, `flash_attention_3`, `flex_attention`, `xformers`, `sage`, `s2`, `fp8`) **or** a hub-kernel path (`startswith("kernels-")` / contains `/`). Reject short-form strings like `flash` / `fa2` / `sdp` / `flex` with an explicit error pointing at the canonical name.
2. Rewrite `normalize_attn_implementation` so its only job is mapping **legacy booleans → canonical `attn_implementation`** (for BC). Mapping is fixed:
   - `flash_attention → flash_attention_2`
   - `sdp_attention → sdpa`
   - `flex_attention → flex_attention`
   - `xformers_attention → xformers`
   - `sage_attention → sage`
   - `s2_attention → s2`
   - `eager_attention → eager`
   Priority for legacy combos stays as in the current branch (`s2 > sage > xformers > flex > flash > sdp > eager`). Emit a one-time `DeprecationWarning` per unique legacy flag seen. After mapping, delete the legacy flag keys from `data` so they never appear on validated `cfg`. If both a canonical `attn_implementation` and any legacy flag are set, raise (no silent precedence).

   **Merge `AttentionValidationMixin.check_attention_fields` into this normalizer and delete the mixin method.** Pydantic v2 runs inherited `mode="before"` validators before subclass ones per MRO, so leaving them as siblings causes the inherited check to reject legacy combos like `s2 + flash` before the normalizer can map them. One validator, one source of conflict detection.

   **Fix the gemma4-hybrid path**: change `data["attn_implementation"] = "flash"` to `data["attn_implementation"] = "flash_attention_2"` (the short name no longer validates after step 1).
3. Convert `attn_supports_packing`, `attn_uses_flash_lib`, `attn_needs_dtype_cast` to `@computed_field`. The three capability tables move to `enums.py` as module constants keyed by the canonical `attn_implementation` string (including `flash_attention_3` — missing from the current branch — and known hub kernels):
   - Packing-capable: `{flash_attention_2, flash_attention_3, flex_attention, xformers, sage, kernels-community/flash-attn3, kernels-community/sage-attention}`.
   - Flash-lib (axolotl's monkeypatch targets): `{flash_attention_2, flash_attention_3, s2, kernels-community/flash-attn3}`.
   - No-dtype-cast: `{eager, sdpa}`.
   Unknown strings: conservative defaults (`packing=False, flash_lib=False, dtype_cast=True`).
4. Delete `_ATTN_IMPL_TO_HF` from `model.py` and pass `cfg.attn_implementation` straight through. The gemma4-hybrid branch continues to override to `flash_attention_2` before passing to HF.
5. `deprecated=True` on each legacy boolean Field so JSON schema + Pydantic surface the deprecation.

### Phase 2 — Fix the validators

6. `check_sample_packing_without_attention`: drop the early-return and gate on `attn_supports_packing`. Warn (or raise — pick one and be consistent) if packing is enabled with a non-packing backend.
7. `check_ebft_activation_offloading`: replace `data.get("flex_attention")` with `attn_implementation == "flex_attention"`.
8. Sweep items (item 15): remove every `data.get("x_attention") or data.get("attn_implementation") == "x"` dual-check — after phase 1 the legacy side is always `None`. Reduces ~10 lines of noise and eliminates the "which side wins" class of bugs.
9. fp8 preflight (item 16): require `env_capabilities.compute_capability ≥ sm_90`, `torch_version ≥ 2.11`, and `torchao_version ≥ 0.17`. Warn if `use_cache` isn't explicitly `False`.

### Phase 3 — Migrate remaining consumers

10. `lm_eval/cli.py:120` → `flash_attention=cfg.attn_uses_flash_lib`.
11. `lm_eval/__init__.py:26` currently reads `(cfg.attn_implementation == "flash")` — after canonicalization `"flash"` is never stored, so this evaluates `False` for every backend. Change to `cfg.attn_uses_flash_lib`.
12. `validation.py:1137-1142` (NPU check) currently iterates `["flash_attention", "sdp_attention", "s2_attention"]` as string keys. Replace with `cfg.attn_implementation in {"flash_attention_2", "flash_attention_3", "sdpa", "s2"}` or the equivalent canonical-string set.
13. `tests/e2e/multigpu/test_llama.py:524-526` → `cfg.attn_implementation = "flash_attention_2"` / `"flex_attention"`.
14. **Xformers decision** (item 10): the old `cfg.flash_attention = True` side-effect activated `_patch_attention` for btlm/stablelm_epoch+packing/mistral3/llava. Two choices:
    - Add `xformers` to the set that gates `_patch_attention` (restore old behaviour, keeps patches live).
    - Document that those patches don't apply to xformers post-refactor and drop the paths if they're dead.
    Pick one explicitly and leave a commit note. Do not leave it as silent breakage.
15. Add a repo-level check (`tests/test_no_legacy_attn_reads.py` or a ruff/grep pre-commit) that fails if anything outside `config.py`'s normalizer reads `cfg.flash_attention` / `cfg.sdp_attention` / etc. Keeps the invariant from rotting.

### Phase 4 — Tests + docs

14. Rewrite `test_attn_implementation.py` to build full `AxolotlInputConfig(**data)`, not just the classmethod. Covers validator ordering and the Pydantic-field-override issue.
15. Add one test per gap closed above: `attn_implementation: eager + sample_packing`; `attn_implementation: flex_attention + activation_offloading`; short-form `flash` rejected; `flash_attention_2` passthrough; `kernels-community/flash-attn3` capability lookup; `attn_uses_flash_lib: true` in YAML rejected; legacy boolean emits `DeprecationWarning` and is absent from validated `cfg`; fp8 preflight failures.
16. Update `docs/attention.qmd` for the single `attn_implementation` field + the deprecation table for legacy flags. One-paragraph migration note in the changelog.
17. `examples/` contains ~170 YAML files using legacy flags (`flash_attention: true` etc.). They still validate post-refactor (normalizer maps them with deprecation), but a follow-up sweep to convert them to `attn_implementation: flash_attention_2` is worth scheduling — call this out in the migration note so users know examples will be migrated on a later pass.

## 5. Ordering & risk

- Phase 1 is the keystone: it's the largest diff but each step is mechanical once the alias map is in place. No behaviour change for any consumer that was using `attn_implementation` correctly; behaviour change only for consumers that were reading legacy flags (phase 3 step 13 is the explicit decision point).
- Phase 2 is independent of phase 1 and can land first as a quick safety net.
- Phase 3 step 13 is the only judgment call — flag for review before choosing.
- Total: ~10-13 commits beyond what's on the branch, each scoped and individually revertable.
