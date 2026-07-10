# Benchmark / eval during training

Example configs for the [Benchmark API plugin](../../src/axolotl/integrations/benchmark_api/README.md)
(`axolotl.integrations.benchmark_api.BenchmarkAPIPlugin`) — fire an external
benchmark/eval runner on each checkpoint and log the returned scalar metrics
back into training, with optional early stopping.

The YAMLs here are symlinks to the canonical copies under the plugin package:

| Example | Mode |
|---------|------|
| `lora-1b-benchmark.yaml` | sync (POST blocks until the runner returns metrics) |
| `lora-1b-benchmark-async.yaml` | async (runner replies immediately; Axolotl polls + drains) |

```bash
axolotl train examples/benchmark-eval/lora-1b-benchmark.yaml
```

Both point at a runner on `http://localhost:8765/eval`. A minimal reference
runner implementing the contract lives at
https://github.com/thad0ctor/axolotl-benchmark-server.
