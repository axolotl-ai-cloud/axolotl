---
name: liger-upstream-sync
description: "Audits axolotl's Liger integration against liger-kernel to catch silent dispatch drift, and previews a liger version before bumping the pin. Use when asked to launch the liger skill or run the liger sync, when bumping the liger-kernel version, when a Liger-patched model trains but loss/throughput looks off, when reviewing src/axolotl/integrations/liger/plugin.py, or when adding/removing a model from the hand-patch elif chain or axolotl_override_liger_fn set. Finds hand-patches that upstream now shadows with native dispatch, stale override-set entries, and glu-activation drift in the generic-path parameter probes."
---

# Liger upstream-sync

Axolotl's Liger plugin (`src/axolotl/integrations/liger/plugin.py`) dispatches a
model type to upstream's **generic native path** when:

```python
model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN
and model_config_type not in axolotl_override_liger_fn
```

A hand-written `elif cfg.model_config_type == "X"` branch therefore runs **only**
when `X` is in `axolotl_override_liger_fn` (or absent from the upstream table).
The override set is the *only* lever that routes a natively-supported type to its
hand-patch.

So when liger ships native support for a type axolotl still hand-patches, the
generic path silently shadows the hand-patch: the `elif` becomes dead code, a
different kernel set is applied, and **CI stays green** — it is not a version
error or an import failure, just a correctness/perf regression. This was real in
the liger-0.8.0 bump (qwen3_5 / qwen3_5_moe / gemma4_text override shadowing).

This skill is a static + introspection audit that flags exactly these cases.

## When to run

- Bumping the `liger-kernel` pin
- Editing the `elif` dispatch chain or `axolotl_override_liger_fn` in `plugin.py`
- A Liger-patched model trains but loss or throughput looks wrong
- Reviewing a PR that touches `integrations/liger/`

## Stage 1: Audit

Run the bundled audit **in the training environment** (it introspects the
*installed* liger-kernel, so it reports against the exact version you train with):

```bash
python .agents/skills/liger-upstream-sync/scripts/audit_liger_sync.py
```

It parses `plugin.py` (override set, dispatch `elif` types, the
`"<param>" in liger_fn_sig.parameters` probes) and diffs them against the live
`MODEL_TYPE_TO_APPLY_LIGER_FN`. Exit code is non-zero when anything needs review,
so it can gate CI.

### Preview a version before bumping

To check what a *different* liger version would shadow without installing it, pass
its `monkey_patch.py` (or an extracted package dir) as `--liger-source`:

```bash
python .agents/skills/liger-upstream-sync/scripts/audit_liger_sync.py \
  --liger-source /path/to/liger_kernel  # a checkout, an unzipped wheel, or the file
```

This is keys-only (no installed fns), so `[4]` signature drift is skipped; `[1]`–`[3]`
work normally. Use it in a liger-bump PR to see the impact before changing the pin.

## Stage 2: Interpret

| Section | Meaning | Action |
|---------|---------|--------|
| **[1] Shadowed hand-patches** | Type is in the upstream table **and** has an `elif`, but is **not** in the override set. The generic native path runs; the `elif` is dead. | Decide per type: if the hand-patch adds kernels native lacks, **add the type to `axolotl_override_liger_fn`** to force the `elif`. If native now suffices, **delete the `elif`**. Either way, go to Stage 3 first. |
| **[2] Override-set health** | Entries should be native (else the override is pointless) and must have an `elif` to route to. | Remove stale entries (not in the upstream table); add a missing `elif` (else liger is never applied for that type). |
| **[3] Custom hand-patches** | `elif` types absent from the upstream table — genuinely custom, correct to hand-patch. | None (informational). Recheck after a bump in case upstream added native support (it would jump to [1]). |
| **[4] Signature drift** | A generic-path fn exposes a glu-activation toggle under a name `plugin.py` does not check (e.g. `swiglu`→`glu`). | Update the parameter probes in the generic path so `liger_glu_activation` is forwarded again. |

## Stage 3: Validation gate (before changing the override set)

For each `[1]` finding, confirm which path *actually* fires and which modules get
swapped, so you don't trade a working generic patch for a broken hand-patch (or
vice-versa). In the training env:

```bash
# What params does upstream's native fn accept for this type?
python -c "import inspect; from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN as M; print(inspect.signature(M['gemma4']))"
```

Then compare against the hand-patch body in `plugin.py`: list the modules it
swaps (e.g. `Gemma4RMSNorm`, `Gemma4TextMLP`, the `lce_forward`) and check whether
the generic native fn swaps the same set with the same semantics (Gemma4, for
example, needs `offset=0`, `in_place=False` RMSNorm — confirm native matches).
Only after this comparison decide override-vs-delete. Re-run Stage 1 to confirm
the finding clears.

## Notes

- The audit reflects the **installed** liger-kernel; results differ across
  versions, which is the point — run it wherever you train.
- It does not modify `plugin.py`; it only reports. Edits are a human decision.
