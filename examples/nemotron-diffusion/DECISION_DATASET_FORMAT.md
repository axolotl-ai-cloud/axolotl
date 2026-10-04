# Decision-training JSONL format

Use `type: decision.jsonl` for already-normalized JSONL. Each line is
one record. The loader renders a fixed decision system prompt from the question
schema, then sends `state` as the user message. There is no per-row `system`
field today. Put shared context in `state`; put global decision instructions in
the optional top-level `instructions` string and per-question instructions in
each question.

`source`, `id`, and `group` identify the row. `family` is optional but useful
for split isolation and grouped development splits. `group` joins records with
the same source, family, state, and top-level instructions so their questions
share one canvas. A single record may already contain several questions. Keep
question IDs stable and nonempty.

```json
{"source":"support_demo","family":"returns","id":"case-0042","group":"case-0042","instructions":"Decide from the supplied case only.","state":{"customer":"A customer says their unopened order arrived yesterday and asks for a refund.","policy":"Unopened items delivered within 30 days may be refunded."},"questions":{"refund":{"type":"choice","instructions":"Choose the next action.","options":[{"name":"approve","description":"Approve a refund under the stated policy."},{"name":"deny","description":"Deny the request because the policy does not apply."},{"name":"escalate","description":"Send the case to a human reviewer."}]},"urgency":{"type":"score","instructions":"Rate how urgently this needs review.","levels":["routine","soon","urgent"]},"eligible":{"type":"noul","instructions":"Is the customer eligible for a refund?","criteria":{"true":"The stated policy permits a refund.","false":"The stated policy does not permit a refund."}}},"labels":{"refund":{"kind":"dist","probs":[0.78,0.02,0.20]},"urgency":{"kind":"hard","gold_idx":0},"eligible":{"kind":"set","allowed_set":[0]}} ,"source_metadata":{"provenance":{"dataset":"my-org/support-decisions","revision":"immutable-revision","split":"train","source_id":"case-0042"}}}
```

The current parser accepts these finite question types.

| Type | Required schema field | Canonical target order |
|---|---|---|
| `choice` | `options` | List order. An option may be a string or an object with a name and written description. Keep descriptions: they are part of the rendered decision schema. |
| `score` | `levels` | List order, from the lowest to highest rubric level as defined by the source. Do not sort levels lexically. |
| `noul` | no alternatives field required | The loader's canonical order is `yes`, then `no`; use that target order. `criteria` may document the true/false conditions and is rendered context. |

For a `choice` source with named options, normalize it into the desired list
order before writing JSONL. The index in a label always refers to this
canonical order, never to an option name, a source ID, or a token ID. Candidate
labels are encoded by the configured decision codebook during preparation; do
not provide token IDs in dataset JSONL.

## Labels

Every question must have exactly one label with one of these forms:

```json
{"kind":"hard","gold_idx":1}
{"kind":"dist","probs":[0.10,0.70,0.20]}
{"kind":"set","allowed_set":[0,2]}
```

`hard` uses one valid zero-based alternative index. `set` uses a nonempty set
of unique valid indices. `dist` must contain exactly one finite, nonnegative
probability per alternative, sum to one within `1e-4`, and is normalized only
after that validation. Zero probabilities are valid and contribute no
`p log p` term. Full soft labels keep all supplied alternatives; do not omit
zero-mass entries.

With `decision.labels.label_softmax: both`, a distribution target has
candidate-restricted KL, full-vocabulary soft cross-entropy over its allowed
label tokens, and the configured Brier term. The full-vocabulary term retains
the model's probability mass penalty outside the allowed candidate set. Hard
and set behavior remains available for legacy data.

## Splits and provenance

Keep train, dev, validation, and test in separate source files or dataset
entries. Training must use only `split: train`; never put test rows in a train
entry. Preserve a `source_metadata.provenance` mapping with immutable source
location/revision, split, and source row ID. The typed-decisions adapter records
dataset, workflow, declared split, row split when provided, and source ID; it
rejects typed training unless the declared split is exactly `train`, and rejects
a row split that disagrees with it.

The loader checks split isolation after normalization. Do not reuse identical
state/grouped context across protected splits. When grouped rows are emitted,
the loader retains per-record and per-question source metadata.

## Mixture and weighting

`decision.mixture.weights` controls sampling frequency by `source`.
`loss_weight` multiplies each source's per-example loss. Loss reduction first
averages questions inside a canvas, then averages canvases in the batch; a row
with many questions does not automatically outweigh a one-question row.
`premixed: true` requires `source_metadata.premix` draw provenance and is for a
prebuilt draw sequence, not ordinary source JSONL.

## Not currently accepted

The normalized loader has no `system`, `candidate_token_ids`, `other_mass`,
partial-distribution renormalization, ranking/Plackett--Luce target, or
hard/soft blending field. Those are proposed formats, not dataset fields. A
partial label must be modeled explicitly in a future contract; silently
renormalizing it would change the loss.

## Image-conditioned decisions

Nemotron-Labs-Diffusion-VLM-8B accepts an optional top-level `images` list of
local image paths or HTTP(S) URLs. Images appear in list order before `state`
in the user message. Use `state` and question instructions to identify images
by their order when comparing several images. Supply paths relative to the
training process's working directory, or use absolute paths.

```json
{"source":"visual-inspection","id":"parcel-42","group":"parcel-42","images":["/data/parcel-42.jpg"],"state":"Inspect the parcel shown in the image.","questions":{"action":{"type":"choice","instructions":"Choose the appropriate next action.","options":["accept","inspect manually","reject"]},"damage":{"type":"score","instructions":"Rate visible damage.","levels":["none","minor","major"]},"open":{"type":"noul","instructions":"Is the parcel visibly open?"}},"labels":{"action":{"kind":"hard","gold_idx":1},"damage":{"kind":"dist","probs":[0.1,0.8,0.1]},"open":{"kind":"hard","gold_idx":1}}}
```

All questions share the visual context and are scored in the same decision
canvas. Hard and soft targets use the same semantics as text-only records.
Use `processor_kwargs.max_image_size` to bound the longest image edge before
patch alignment (the starter recipe uses 560 pixels). Image expansion counts
toward `sequence_len`; images and their marker tokens
are context, never decision-loss targets. Oversized examples follow the
existing decision length policy rather than truncating through an image.

Install the image preprocessing dependencies with `pip install 'axolotl[vision]'`
(or `pip install -e '.[vision]'` from a checkout), then start from
[decision-vlm-lora-8b.yaml](decision-vlm-lora-8b.yaml). Its LoRA
module pattern targets only the language decoder; vision and the projector
remain frozen. This is a starter configuration, not a benchmark-tuned recipe.
The VLM reserves image marker IDs 18–21, so a reserved-token codebook must not
reuse these IDs. Use the example's `spreadsheet151` codebook initially.

Decoded Hugging Face `Image` columns are not a normalized decision source yet;
materialize their images to local files and supply those paths when converting
the source into this format. Local-image cache identity includes image contents;
remote image URLs bypass prepared-cache reuse.
