# Benchmark API Plugin

Forwards Axolotl checkpoint events to an external benchmark runner and logs the
returned scalar metrics back into the trainer, with optional single-metric early
stopping.

Axolotl stays generic: it does **not** know how the benchmark works, what
datasets are used, whether adapters are merged, how the model is loaded, or what
the metrics mean. All benchmark-specific logic lives in the external runner. The
plugin is a checkpoint-event webhook:

```
training event -> POST checkpoint info -> external runner -> scalar metrics -> log + optional stop
```

## Config

```yaml
plugins:
  - axolotl.integrations.benchmark_api.BenchmarkAPIPlugin

benchmark_api:
  endpoint: http://localhost:8765/eval
  run_on:
    - save            # save | eval | train_end (default: [save])
  timeout_sec: 3600
  fail_training_on_error: false

  early_stopping:
    enabled: true
    metric: eval/ocr/cer_mean   # a metric name returned by the runner
    mode: lower                 # lower/higher (aliases of min/max)
    patience: 3
    min_delta: 0.002
    threshold: 0.075
```

`save` is the recommended trigger because a real `checkpoint-{step}` directory
exists at that point.

## API contract

Request (Axolotl -> runner):

```json
{
  "event": "save",
  "step": 1200,
  "checkpoint_dir": "/abs/path/output/checkpoint-1200",
  "output_dir": "/abs/path/output"
}
```

Response (runner -> Axolotl):

```json
{
  "status": "completed",
  "metrics": {
    "eval/ocr/cer_mean": 0.081,
    "eval/ocr/wer_mean": 0.194
  }
}
```

- `status` must be `completed` for metrics to be logged.
- `metrics` is a flat dict; only scalar `int`/`float` values are logged, exactly
  as keyed by the runner. Non-scalar values are ignored.
- Metric keys are fully qualified — there is no `metric_prefix`. Early stopping
  watches one of the returned metric names via `early_stopping.metric`.
- Any `artifacts` field is ignored in this version.

## Early stopping

Stops when **either** condition is met:

- **threshold** — the metric reaches the target (`<=` for `lower`, `>=` for
  `higher`).
- **patience** — the metric fails to improve by at least `min_delta` for
  `patience` consecutive benchmark runs.

`mode` accepts `lower`/`min`/`smaller`/`decrease` and
`higher`/`max`/`larger`/`increase`.

## Scope

Version 1 is synchronous, main-process-only, single early-stopping metric. Not
supported: async/queued runs, multiple stopping rules, artifact handling, or any
model loading/merging inside Axolotl.
