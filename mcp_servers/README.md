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

## Run the hello server

From repo root, with the project virtual environment active:

```powershell
./axo-env/Scripts/python.exe -m mcp_servers.hello_world.server
```

Expected behavior:
- Server starts using StdIO transport.
- MCP host can discover tool `ping`.

## Manual learning checklist for Phase 0

1. Read MCP quickstart docs for Python server setup and tool registration.
2. Start the hello server and connect from an MCP-compatible host/client.
3. Call `ping` with different payloads and verify response structure.
4. Observe request/response shape to understand JSON-RPC flow.
