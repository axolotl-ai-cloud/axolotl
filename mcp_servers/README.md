# MCP Servers (Learning Path)

This directory contains a staged Model Context Protocol (MCP) implementation for the golf strategy project.

## Phase 0 status

Implemented now:
- hello_world server with one `ping` tool
- package scaffolding for future servers

Planned next:
- inference_server (Phase 1)
- training_control (Phase 2)
- debug_assistant (Phase 3)
- dataset_validator (Phase 4)
- checkpoint_browser (Phase 5)

## Phase 1 status

Implemented now:
- inference MCP server tools:
	- `strategy_for_hole`
	- `description_for_strategy`
	- `analyze_scenarios`

Model configuration for inference server:
- `AXOLOTL_MCP_MODEL_NAME` (default: `gpt2`)
- `AXOLOTL_MCP_ADAPTER_DIR` (optional path to LoRA adapter checkpoint)

If the model or adapter is unavailable, tools return deterministic fallback responses with a warning field.

## Run the hello server

From repo root, with the project virtual environment active:

```powershell
./axo-env/Scripts/python.exe -m mcp_servers.hello_world.server
```

Expected behavior:
- Server starts using StdIO transport.
- MCP host can discover tool `ping`.

## Run the inference server

From repo root:

```powershell
$env:AXOLOTL_MCP_MODEL_NAME = "gpt2"
$env:AXOLOTL_MCP_ADAPTER_DIR = "outputs/bethpage-lora/checkpoint-quick"
./axo-env/Scripts/python.exe -m mcp_servers.inference_server.server
```

Expected behavior:
- MCP host can discover tools `strategy_for_hole`, `description_for_strategy`, and `analyze_scenarios`.
- First request may be slower due to model load; subsequent requests use cached model state.

## Phase 2 status

Implemented now:
- training control MCP server tools:
	- `start_training_job`
	- `training_status`
	- `stop_training_job`
	- `resume_training_job`

Training server notes:
- Valid modes are `debug`, `debug_training`, `1_hour`, `8_hour`.
- Jobs run as subprocesses and are tracked by generated `job_id` values.
- Status returns recent log tail so clients can poll progress without attaching to process output.

## Run the training control server

From repo root:

```powershell
./axo-env/Scripts/python.exe -m mcp_servers.training_control.server
```

Optional interpreter override for spawned training jobs:

```powershell
$env:AXOLOTL_MCP_PYTHON = "./axo-env/Scripts/python.exe"
```

## Phase 3 status

Implemented now:
- debug assistant MCP server tools:
	- `sample_strategy_failures`
	- `diagnose_task_prefix`
	- `compare_adapters`

Debug server notes:
- Default failure files target the strategy-only validation flow outputs.
- Adapter comparison can use environment overrides:
	- `AXOLOTL_MCP_SINGLE_ADAPTER_DIR`
	- `AXOLOTL_MCP_MULTITASK_ADAPTER_DIR`
- A reproducible fixture file is available at `mcp_servers/debug_assistant/test_scenarios.jsonl`.

## Run the debug assistant server

From repo root:

```powershell
$env:AXOLOTL_MCP_SINGLE_ADAPTER_DIR = "outputs/bethpage-lora/checkpoint-quick"
$env:AXOLOTL_MCP_MULTITASK_ADAPTER_DIR = "outputs/bethpage-lora/checkpoint-multitask"
./axo-env/Scripts/python.exe -m mcp_servers.debug_assistant.server
```

## Phase 4 status

Implemented now:
- dataset validator MCP server tools:
	- `validate_dataset_file`
	- `dataset_stats`
	- `filter_dataset_by_task`
	- `repair_dataset`

Dataset validator notes:
- Validation checks JSON parseability, required fields, task labels, and strategy completion parseability.
- Repair tool writes output to a new file with suffix `.all.fixed.jsonl`, `.case.fixed.jsonl`, or `.format.fixed.jsonl`.
- Filter tool mirrors your existing task split workflow and returns filtered record count.

## Run the dataset validator server

From repo root:

```powershell
./axo-env/Scripts/python.exe -m mcp_servers.dataset_validator.server
```

## Phase 5 status

Implemented now:
- checkpoint browser MCP server tools:
	- `list_model_checkpoints`
	- `checkpoint_metadata`
	- `compare_model_checkpoints`
	- `export_checkpoint_artifacts`

Checkpoint browser notes:
- Discovers checkpoint folders under `outputs/bethpage-lora` by default.
- Metadata includes LoRA config fields when `adapter_config.json` is present.
- Exports are written under `outputs/bethpage-lora/exports`.
- ONNX export currently copies existing `.onnx` artifacts; it does not convert checkpoints automatically.

## Run the checkpoint browser server

From repo root:

```powershell
./axo-env/Scripts/python.exe -m mcp_servers.checkpoint_browser.server
```

## Phase 6 status

Implemented now:
- integration demo script at `mcp_servers/demo/integration_demo.py`
- end-to-end workflow checks for dataset, inference, diagnostics, and checkpoints

## Run end-to-end demo

From repo root:

```powershell
./axo-env/Scripts/python.exe -m mcp_servers.demo.integration_demo
```

The demo returns a single JSON payload with:
- dataset validation summary
- dataset stats
- strategy + description inference examples
- task prefix diagnostic output
- single vs multitask adapter comparison
- checkpoint listing summary

## Phase 7 status

Implemented now:
- GitHub Actions workflow at `.github/workflows/mcp-tests.yml`

CI coverage includes:
- syntax compilation for all MCP servers and the integration demo
- smoke imports for MCP server modules

## Manual learning checklist for Phase 0

1. Read MCP quickstart docs for Python server setup and tool registration.
2. Start the hello server and connect from an MCP-compatible host/client.
3. Call `ping` with different payloads and verify response structure.
4. Observe request/response shape to understand JSON-RPC flow.
