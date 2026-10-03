# AutoJev route-capture dataset format

Use `type: decision.autojev_route` for JSONL where each record stores
one raw AutoJev `/api/alpha/decisions` request in `request`, plus a supervised
target. The adapter accepts the request only when it has exactly `model`,
`state`, and one `questions.route` choice question. It never accepts a prompt,
messages, another question, or another question type.

```json
{"id":"route-capture-001","request":{"model":"~typesafe/jev-latest","state":{"requirements":{"requires_vision":false},"candidates":[{"id":"local/model-a"},{"id":"cloud/model-b"}]},"questions":{"route":{"type":"choice","instructions":"Choose exactly one eligible candidate ID.","criteria":{"local/model-a":"{\"id\":\"local/model-a\",\"estimated_request_cost\":0.1}","cloud/model-b":"{\"id\":\"cloud/model-b\",\"estimated_request_cost\":1.2}"}}}},"gold_unique_model_id":"local/model-a","source_metadata":{"provenance":{"dataset":"my-route-captures","revision":"immutable-revision","split":"train"}}}
```

For a full soft target, replace `gold_unique_model_id` with
`unique_model_id_probabilities`. The map must include every `criteria` key,
with finite nonnegative values that sum to one.

```json
{"id":"route-capture-002","request":{"model":"~typesafe/jev-latest","state":{"candidates":[{"id":"local/model-a"},{"id":"cloud/model-b"}]},"questions":{"route":{"type":"choice","instructions":"Choose exactly one eligible candidate ID.","criteria":{"local/model-a":"{\"id\":\"local/model-a\"}","cloud/model-b":"{\"id\":\"cloud/model-b\"}"}}}},"unique_model_id_probabilities":{"local/model-a":0.8,"cloud/model-b":0.2}}
```

Each criteria value must remain the original JSON string from AutoJev. The
adapter preserves `state`, candidate order, candidate IDs, and description
strings; it maps labels to the normalized option order before the plugin's
seeded deterministic permutation remaps those indices. No decision-codebook
tokens appear in the source data. The normalized record also retains the exact
raw request under `source_metadata.autojev_request` for provenance.

The adapter derives a canonical `request_id` from the full raw request. You may
include it to validate a capture, but it is optional, so existing capture rows
such as the examples above remain valid. `id` identifies an observation; an
optional top-level `group` can identify a task or session for group-disjoint
downstream splits.
