# Bethpage Multitask Prompt Template Specification

This document defines the standardized chat-style prompt formatting for the four multitask training types derived from `training_bethpage_multitask.weighted.train.jsonl`.

All prompted records will be emitted as JSONL with fields:
- `task`: original task id
- `hole`: hole number
- `sample_weight`: float carried through from weighted dataset (no further scaling)
- `messages`: list of chat messages, each having `{"role": "system"|"user"|"assistant", "content": "..."}`
- `meta` (optional): lightweight structure with original fields for auditing (NOT used in tokenization)

We do NOT inline rationale chains into the assistant output for selection tasks; instead we append a short justification only when originally present. For description tasks we keep the description as the assistant response and (optionally) append a concise rationale bullet list separated by a delimiter when rationale objects exist.

## Global System Message
A single consistent system message enforces domain style:

```
You are a concise, knowledgeable golf course strategy analyst. Always be precise, avoid hallucinations, and keep units in yards when distances are given. If data is absent, do not invent it.
```

## Task-Specific User Message Templates
Placeholders (curly braces) are substituted then redundant blank lines collapsed.

### 1. strategy_selection
```
Task: Select optimal tee shot strategy.
Hole: {hole} | Par {par} | Yardage {yardage}y
Attributes:
- Gradient: {gradient}
- Shape: {hole_shape}
- Length: {hole_length}
- Elevation Δ: {elevation_delta_yards}y
- Landing Zone Width: {tee_landing_zone_width_yards}y
- Primary Hazard: {primary_hazard_type}
- Secondary Hazard: {secondary_hazard_type}
- Green Size: {green_size}
Key Hazards: {key_hazards_joined}
Strategic Theme: {strategic_theme}
Candidate Strategies:
{strategies_block}
Instruction: Return ONLY the chosen strategy_id on a single line. If rationale known, append a short justification after a tab.
```
`strategies_block` example line: `- cutoff_308 (aggressive) remaining=70y`

Assistant target format (training):
```
{strategy_id}\t{optional_rationale}
```
If no rationale present, it's just the id.

### 2. strategy_selection_negative
Same as above but first line:
```
Task: Identify least optimal tee shot strategy.
```
Assistant target format identical.

### 3. description_generation
User template:
```
Task: Write hole description.
Hole: {hole} | Par {par} | Yardage {yardage}y
Theme: {strategic_theme}
Strategy: {strategy_id} ({classification}) cutoff={cutoff_distance}y
Key Hazards: {key_hazards_joined}
Derived Notes: {derived_notes_condensed}
Instruction: Write a vivid, factual description of the hole incorporating strategic context and hazards. Do not add scoring statistics unless provided. Avoid repetition.
```
Assistant target:
```
{description}
```
If rationale list exists we append:
```
---
Rationale: {semicolon_joined_key_value_pairs}
```

### 4. style_rewrite
User template:
```
Task: Rewrite hole description into concise TV commentary (<45 words).
Original:
{base_description}
Instruction: Keep strategic tone and core hazards; remove fluff.
```
Assistant target:
```
{description}
```

## Null / Missing Handling
- Any `null` numeric becomes `unknown` token.
- Empty rationale or notes omitted completely.
- Empty hazard list -> `None`.

## Safety / Guardrails
System instruction already forbids fabrication. We intentionally do not ask model to produce rationale unless present.

## Example (strategy_selection)
```
System: You are a concise, knowledgeable golf course strategy analyst...
User: Task: Select optimal tee shot strategy.\nHole: 2 | Par 4 | Yardage 389y\n...\nAssistant: cutoff_308\tright approach bunker, uphill angle advantage
```

This spec is consumed by `build_prompted_dataset.py`.
