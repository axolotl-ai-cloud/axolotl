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

## Manual learning checklist for Phase 0

1. Read MCP quickstart docs for Python server setup and tool registration.
2. Start the hello server and connect from an MCP-compatible host/client.
3. Call `ping` with different payloads and verify response structure.
4. Observe request/response shape to understand JSON-RPC flow.
