# Arctic Platform integration for Axolotl

Remote SFT (and later RL) against an [Arctic Platform](https://github.com/Snowflake-AI-Research/arctic-platform)
training server. Axolotl keeps its full data pipeline (tokenize, chat templates,
packing, collation, logging); GPU work — weights, forward/backward, optimizer —
runs on the server. The Axolotl process can be **CPU-only**.

## Install

```bash
# Axolotl, then the soft dependency:
pip install -e .                        # or: pip install axolotl
pip install "axolotl[arctic-sft]"       # pulls arctic_platform
# or from a checkout:
pip install -e /path/to/arctic-platform
```

Without `arctic_platform` installed, enabling the plugin raises an
`ImportError` with the same install hint.

## Activate

There is no auto-discovery. Opt in from your Axolotl YAML with the plugin
dotted path (same mechanism as Hatchery / Liger / …):

```yaml
plugins:
  - axolotl.integrations.arctic_platform.sft.ArcticSFTPlugin

arctic_sft:
  host: localhost
  port: 8765
  training_gpus: 2
  launch_local_server: true
  server_cuda_visible_devices: "0,1"
  checkpoint_path: ./arctic_sft_ckpt   # server-side; required for new jobs
```

Full worked example:
[`sft/examples/arctic_sft.yaml`](sft/examples/arctic_sft.yaml).

### Colocated HTTP server (CPU-only client)

```bash
CUDA_VISIBLE_DEVICES= axolotl train path/to/your_config.yaml
```

- Empty `CUDA_VISIBLE_DEVICES` keeps Axolotl off the GPUs.
- `launch_local_server: true` starts `arctic_platform.common.http_server` as a child.
- `server_cuda_visible_devices` is the GPU list for that child.

### Already-running server

```yaml
arctic_sft:
  host: training-node.example.com
  port: 8765
  training_gpus: 4
  launch_local_server: false
  # training_job_id: 1   # optional: reconnect instead of creating a job
```

```bash
python -m arctic_platform.common.http_server \
  --host 0.0.0.0 --port 8765 \
  --training-gpus 4 --sampling-gpus 0 --log-prob-gpus 0
```

## Config

### Nested `arctic_sft:` (connection / server)

| Key | Default | Meaning |
|-----|---------|---------|
| `host` / `port` | `localhost` / `8000` | Server address |
| `training_gpus` | **required** (≥1) | GPUs on the server for training |
| `comm_protocol` | `http` | `http` or `ray` |
| `launch_local_server` | `false` | Spawn a local HTTP server from the client |
| `server_cuda_visible_devices` | `null` | GPU list for that subprocess (e.g. `"0,1"`) |
| `loss_fn` | `sft` | `sft` (HF CE) or `sft_ce` (explicit CE) |
| `model_name` | `null` | Override; else top-level `base_model` |
| `checkpoint_path` | `null` | Server checkpoint dir (defaults from `output_dir` if unset) |
| `attn_implementation` | `sdpa` | Passed into the server worker config |
| `gradient_checkpointing` | `false` | Server-side activation checkpointing |
| `sampling_gpus` | `0` | `>0` enables remote sample generation (vLLM) |
| `colocate` | `false` | Share GPUs between training and sampling |
| `training_job_id` / `sampling_job_id` | `null` | Reattach to existing jobs |
| `ds_config` / `training_config` / `ds_worker_config` | `null` | Escape hatches; else synthesized from top-level knobs |

CLI nested overrides work as usual, e.g. `arctic_sft__port=9000`.

### Top-level knobs (still required)

Keep these at YAML top level — the plugin forwards them to the server optimizer /
schedule:

- `base_model`, `sequence_len`, `seed`
- `learning_rate`, `adam_beta1`, `adam_beta2`, `weight_decay`, `warmup_ratio` / `warmup_steps`
- `micro_batch_size`, `gradient_accumulation_steps`
- `num_epochs` / `max_steps`, `logging_steps`, `output_dir`
- Datasets, chat templates, packing — unchanged from normal Axolotl SFT

`micro_batch_size` must be a positive multiple of `training_gpus` (server DP split).

### Sample generation (optional)

```yaml
generate_samples: true
num_generation_samples: 3

arctic_sft:
  training_gpus: 1
  sampling_gpus: 1
  colocate: true
  # … plus a vllm_config if you need non-defaults
```

Requires `arctic_platform[rl]` (vLLM) and ArcticInference on the server side.
The plugin replaces the stock local `SFTGenerationCallback` with a remote one.

### Ray transport

```yaml
arctic_sft:
  comm_protocol: ray
  training_gpus: 2
  launch_local_server: false   # Ray is in-process; no HTTP child
```

Do **not** blank `CUDA_VISIBLE_DEVICES` for Ray (actors need visible GPUs).

## What to expect

- Logs: plugin active → stub model (no local weight load) →
  `Arctic SFT training session created: … transport=http, job=…`.
- Step metrics look like normal Axolotl (`loss`, `learning_rate`, `grad_norm`).
  Client-side `memory/*` stays near zero — compute is remote.
- Checkpoints land under `arctic_sft.checkpoint_path` on the **server**.
  Local `output_dir` may still get a tiny stub write; ignore it for weight recovery.

## Pitfalls

| Symptom | Fix |
|---------|-----|
| `micro_batch_size … must be a multiple of training_gpus` | Raise `micro_batch_size` (e.g. 2 GPUs → `micro_batch_size: 2`) |
| `/initialize` 422 / seed errors | Set top-level `seed:` (plugin defaults to `42` if omitted) |
| Client OOM / CUDA init on Axolotl | `CUDA_VISIBLE_DEVICES=` + `server_cuda_visible_devices` |
| `generate_samples` without sampling | Set `arctic_sft.sampling_gpus > 0` |
| Port already in use | Change `arctic_sft.port` or stop the leftover server |

## Layout

```
arctic_platform/
  README.md          ← this file
  sft/
    plugin.py        ArcticSFTPlugin
    trainer.py       ArcticSFTTrainer (remote train loop)
    args.py          arctic_sft: pydantic schema
    generation.py    remote sample generation
    callbacks.py     ArcticSFTGenerationCallback
    examples/        worked YAML
  # rl/              (future)
```
