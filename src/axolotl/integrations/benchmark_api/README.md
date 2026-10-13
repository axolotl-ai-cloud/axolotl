# Benchmark API Plugin

Forwards Axolotl checkpoint events to an external benchmark runner and logs the
returned scalar metrics back into the trainer, with optional single-metric early
stopping.

Axolotl stays generic: it does **not** know how the benchmark works, what
datasets are used, whether adapters are merged, how the model is loaded, or what
the metrics mean. All benchmark-specific logic lives in the external runner. The
plugin is a checkpoint-event webhook:

```text
training event -> POST checkpoint info -> external runner -> scalar metrics -> log + optional stop
```

## Config

```yaml
plugins:
  - axolotl.integrations.benchmark_api.BenchmarkAPIPlugin

benchmark_api:
  endpoint: http://localhost:8765/eval
  # auth_env: BENCHMARK_API_TOKEN   # env var holding a bearer token (optional)
  execution_mode: sync     # sync (default) | async
  poll_interval_steps: 10  # async only: how often to poll pending jobs
  run_on:
    - save            # save | eval | train_end (default: [save])
  timeout_sec: 3600   # sync: HTTP read timeout; async: per-job deadline (0 = no timeout)
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

## Sync vs async

**sync** (default) — the POST blocks until the runner returns `completed` with
metrics. Simplest, and metrics attribute cleanly to the current step. Best when
the benchmark is quick.

**async** — the runner replies immediately so training keeps moving while an
expensive benchmark runs in the background:

```json
{ "status": "queued", "job_id": "bench-123", "poll_url": "http://host/eval/bench-123" }
```

The plugin tracks the job and polls `poll_url` (GET) every `poll_interval_steps`
training steps; when it returns `completed`, the metrics are logged and early
stopping is evaluated. Any jobs still outstanding at the end are drained
(blocking) at `on_train_end`. Notes:

- `poll_url` is optional; if omitted it defaults to `<endpoint>/<job_id>`.
- Statuses `queued`/`running`/`accepted`/`pending` mean "not done yet"; any other
  status is treated as a failed job (dropped, or fails training if
  `fail_training_on_error: true`).
- `timeout_sec` becomes a per-job deadline; a job that never completes is
  dropped (or fails training if `fail_training_on_error: true`).
- Because results lag, early stopping reacts a few steps after the metric
  plateaus — the throughput/promptness trade-off inherent to async.
- The runner may also answer a submit with `completed` directly (running
  synchronously under an async client); that is handled too.

## Where metrics land

Each completed benchmark result is written to the **axolotl log** (via the
logger) with its values and the true source/checkpoint step, e.g.
`Benchmark API metrics for job-3 (step 8): {'eval/ocr/cer_mean': 0.03, ...}`.
It is also sent through `trainer.log` for live trackers (W&B/TensorBoard) — but
at the `on_train_end` drain the trainer's printer and reporting callbacks have
already closed, so `trainer.log` alone would silently drop those final results;
the explicit log line is what guarantees they are captured. Non-finite values
(`nan`/`inf`) are dropped before logging.

## Security & offline mode

The plugin only makes network calls when `benchmark_api.endpoint` is explicitly configured — it is fully opt-in, and the endpoint is treated as **user-trusted** (you point it at your own runner; there is no allow-list beyond the `http(s)://` scheme check).

- **Offline mode** — when `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE` is set, the plugin refuses to register for any non-local endpoint and logs a warning. Loopback endpoints (`localhost`, `127.0.0.1`, `::1`, `0.0.0.0`) are still allowed, since a local runner never leaves the machine.
- **SSRF hardening** — a runner-supplied `poll_url` is only trusted if it shares the configured endpoint's scheme/host/port; otherwise the plugin falls back to `<endpoint>/<job_id>`. HTTP redirects are never followed (a redirect response is treated as an error), so the origin pin cannot be bypassed via a 3xx hop.
- **Authentication** — set `auth_env` to the *name* of an environment variable holding a bearer token; the plugin sends it as `Authorization: Bearer <token>` on every submit and poll request. The token itself never appears in the YAML config or in log output. If `auth_env` is set but the variable is empty/unset, the run fails fast at startup.
- The checkpoint payload contains local filesystem paths (`checkpoint_dir`, `output_dir`) — only point `endpoint` at a runner you control.

## Distributed (multi-GPU)

The HTTP call, polling, and metric logging run on the main process only; the
early-stop/error decision is broadcast to all ranks so `should_training_stop`
stays consistent. Because a blocking benchmark holds rank 0 while the other
ranks wait on the next collective, a **sync** benchmark that runs longer than the
process-group/NCCL watchdog (default ~10 min) can abort the run — prefer
**async** for slow benchmarks, and keep the runner responsive to submit/poll
requests. The `on_train_end` drain runs on rank 0 without a collective, but a
very long drain can still delay shutdown under sharded saving (FSDP/DeepSpeed);
keep `timeout_sec` modest so outstanding jobs don't stall teardown.

Polling and the stop/error decision are driven on the main process and
broadcast to all ranks, so early stopping stays consistent under multi-GPU.

## Early stopping

Stops when **either** condition is met:

- **threshold** — the metric reaches the target (`<=` for `lower`, `>=` for
  `higher`).
- **patience** — the metric fails to improve by at least `min_delta` for
  `patience` consecutive benchmark runs.

`mode` tells the plugin which direction counts as an improvement for your
metric — i.e. whether "better" means the value goes down or up:

- **`lower`** — a smaller value is better (minimize). Use for error-style
  metrics like loss, CER, WER, perplexity. Threshold fires when `metric <= threshold`.
- **`higher`** — a larger value is better (maximize). Use for score-style
  metrics like accuracy, F1, BLEU. Threshold fires when `metric >= threshold`.

Each direction has interchangeable aliases: `lower`/`min`/`smaller`/`decrease`
all mean minimize, and `higher`/`max`/`larger`/`increase` all mean maximize
(they normalize to `min`/`max` internally). For example, with
`metric: eval/ocr/cer_mean`, `mode: lower` — since lower CER is better —
patience counts a run as progress only when CER drops by at least `min_delta`.

### Record-only (let loss drive early stopping)

The plugin's early stopping is **opt-in**. Omit the `early_stopping` block (or
set `enabled: false`) and the plugin only records benchmark metrics — it never
stops training. This is the common case when you want Axolotl's built-in
loss-based early stopping to govern the run while still logging benchmark
metrics on every checkpoint:

```yaml
# loss-based early stopping (Axolotl native — watches eval_loss)
val_set_size: 0.05
eval_steps: 50
save_steps: 50                 # eval_steps must evenly divide save_steps
early_stopping_patience: 3

benchmark_api:
  endpoint: http://localhost:8765/eval
  run_on: [save]
  # no early_stopping block -> metrics are logged, the plugin never halts training
```

The two are independent `TrainerCallback`s and do **not** interfere:
`early_stopping_patience` wires HF's `EarlyStoppingCallback`, which reads only
`metric_for_best_model` (default `eval_loss`) from the eval results; the
benchmark metrics are logged separately and never feed that decision. Use one
mechanism or the other — not both — to avoid two competing stop rules.

## Scope

Main-process-driven (with stop/error broadcast to all ranks), single
early-stopping metric, sync or async submission. Not supported: multiple
stopping rules, artifact handling, or any model loading/merging inside Axolotl.
