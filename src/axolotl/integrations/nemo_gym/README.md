# NeMo Gym Integration for Axolotl

Train LLMs with reinforcement learning using [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) environments as reward sources. NeMo Gym provides 50+ verified RL environments spanning math, coding, tool-use, reasoning, and safety — each with deterministic reward signals.

## Validated Training Paths

| Path | Speed | Multi-turn | Config |
|------|-------|------------|--------|
| **Async GRPO + Agent /run** | Fastest (3x) | Yes | `async_prefetch: true` + `nemo_gym_multi_turn: true` |
| Standard GRPO + Agent /run | Baseline | Yes | `nemo_gym_multi_turn: true` |
| Standard GRPO + /verify | Simplest | No | Single-turn reward function |
| FSDP2 + /verify (2 GPU) | Distributed | No | `fsdp_version: 2` |

All paths tested end-to-end with Qwen3-0.6B + LoRA, logged to wandb project `nemo-gym-rl`.

## Quick Start

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager (for NeMo Gym's venv)
- Git
- Two GPUs recommended (one for vLLM server, one for training)

### 1. Set Up NeMo Gym

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git ~/Gym
cd ~/Gym
uv venv --python 3.12 && source .venv/bin/activate && uv sync

# Fix pycosat build (GCC 13+)
CFLAGS="" uv pip install pycosat --python .venv/bin/python --no-build-isolation

# Pre-build resource server venvs
for dir in resources_servers/reasoning_gym resources_servers/example_single_tool_call responses_api_models/vllm_model responses_api_agents/simple_agent; do
    uv venv --seed --allow-existing --python 3.12 $dir/.venv
    CFLAGS="" uv pip install --python $dir/.venv/bin/python pycosat --no-build-isolation 2>/dev/null
    uv pip install --python $dir/.venv/bin/python -e . "ray[default]==2.52.1"
done

# Install extra deps for reasoning_gym
uv pip install --python resources_servers/reasoning_gym/.venv/bin/python \
    reasoning-gym matplotlib pillow cycler contourpy kiwisolver
```

### 2. Multi-Turn with Async GRPO (Recommended — Fastest Path)

This is the fully validated, highest-performance path. NeMo Gym's agent server handles
multi-turn tool execution while axolotl's async GRPO prefetches data in background threads.

**Step 1: Create the NeMo Gym agent config**

Create `~/Gym/configs/axolotl_tool_calling.yaml`:
```yaml
# Resource server (tools + verify)
example_single_tool_call:
  resources_servers:
    example_single_tool_call:
      entrypoint: app.py
      domain: agent
      verified: false

# Model server proxy (forwards to your vLLM)
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: http://localhost:8000/v1
      api_key: dummy_key
      model: Qwen/Qwen3-0.6B   # Must match your training model
      return_token_id_information: true
      uses_reasoning_parser: false

# Agent server (orchestrates multi-turn via /run)
example_single_tool_call_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: example_single_tool_call
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: weather
        type: example
        jsonl_fpath: resources_servers/example_single_tool_call/data/weather_tool_calling.jsonl
```

**Step 2: Start three services**

```bash
# Terminal 1: vLLM OpenAI server on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B --max-model-len 2048 --gpu-memory-utilization 0.85

# Terminal 2: NeMo Gym (resource server + model proxy + agent)
cd ~/Gym && .venv/bin/ng_run \
    "+config_paths=[configs/axolotl_tool_calling.yaml]" "+skip_venv_if_present=true"

# Terminal 3: Training on GPU 1
cd experiments && CUDA_VISIBLE_DEVICES=1 CUDA_HOME=$HOME/env-claude-cu130/cuda_shim \
    axolotl train nemo_gym_async_agent.yaml
```

**Step 3: Training config** (`nemo_gym_async_agent.yaml`):
```yaml
base_model: Qwen/Qwen3-0.6B
adapter: lora
lora_r: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
sequence_len: 2048

rl: grpo
chat_template: tokenizer_default

trl:
  use_vllm: true
  vllm_mode: server
  vllm_server_host: localhost
  vllm_server_port: 8000
  vllm_lora_sync: true
  vllm_sync_interval: 5
  # Async GRPO — 3x faster than standard
  use_data_producer: true
  async_prefetch: true
  num_generations: 4
  max_completion_length: 512
  temperature: 0.8
  reward_funcs:
    - nemo_gym_rewards.reward_env

plugins:
  - axolotl.integrations.nemo_gym.NemoGymPlugin

nemo_gym_enabled: true
nemo_gym_auto_start: false
nemo_gym_head_port: 11000
nemo_gym_multi_turn: true
nemo_gym_verify_timeout: 120
nemo_gym_datasets:
  - path: ~/Gym/resources_servers/example_single_tool_call/data/weather_tool_calling.jsonl
    server_name: example_single_tool_call

datasets:
  - path: ~/Gym/resources_servers/example_single_tool_call/data/weather_tool_calling.jsonl
    type: chat_template
    field_messages: responses_create_params.input
    message_field_content: content
    message_field_role: role

vllm:
  gpu_memory_utilization: 0.85
  max_model_len: 2048
  tensor_parallel_size: 1

learning_rate: 5e-6
micro_batch_size: 1
gradient_accumulation_steps: 4
max_steps: 30
gradient_checkpointing: true
bf16: true
output_dir: ./outputs/nemo_gym_async

use_wandb: true
wandb_project: nemo-gym-rl
```

### 3. Single-Turn Training (Simplest — No Agent Server Needed)

For environments that only need single-turn verify (math, coding challenges), you don't need
an agent server. The plugin's reward function calls `/verify` directly.

```yaml
base_model: Qwen/Qwen2.5-0.5B-Instruct
rl: grpo
chat_template: tokenizer_default

trl:
  use_vllm: true
  vllm_mode: colocate
  vllm_enable_sleep_mode: false
  num_generations: 8
  max_completion_length: 128
  temperature: 0.9
  reward_funcs:
    - nemo_gym_rewards.reward_nemo_gym_verify

plugins:
  - axolotl.integrations.nemo_gym.NemoGymPlugin

nemo_gym_enabled: true
nemo_gym_auto_start: false
nemo_gym_head_port: 11000
nemo_gym_datasets:
  - path: ~/Gym/resources_servers/reasoning_gym/data/train_basic_arithmetic.jsonl
    server_name: reasoning_gym

datasets:
  - path: ~/Gym/resources_servers/reasoning_gym/data/train_basic_arithmetic.jsonl
    type: chat_template
    field_messages: responses_create_params.input
    message_field_content: content
    message_field_role: role

vllm:
  gpu_memory_utilization: 0.3
  max_model_len: 512
  tensor_parallel_size: 1

learning_rate: 1e-5
micro_batch_size: 4
gradient_accumulation_steps: 2
max_steps: 50
output_dir: ./outputs/nemo_gym_arithmetic
```

Only needs `ng_run` with resource servers (no agent config):
```bash
cd ~/Gym && ng_run "+config_paths=[resources_servers/reasoning_gym/configs/resources_only.yaml]" "+skip_venv_if_present=true"
```

## How It Works

### Single-Turn
```
axolotl train → GRPO Trainer generates completions
  → NeMo Gym plugin reward_fn calls POST /verify on resource server
  → reward flows back to GRPO for advantage computation
```

### Multi-Turn (Agent /run)
```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  axolotl    │     │  NeMo Gym    │────▶│  vLLM OpenAI     │
│  train      │────▶│  Agent /run  │◀────│  Server (GPU 0)  │
│  (GPU 1)    │     │              │     │  /v1/completions  │
└─────────────┘     └──────┬───────┘     └──────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Resource    │
                    │  Server     │
                    │  (tools +   │
                    │   verify)   │
                    └─────────────┘
```

The agent server orchestrates the entire multi-turn loop:
1. Calls our vLLM server for model generation
2. Parses tool calls from model output
3. Executes tools against resource servers
4. Feeds tool results back to the model
5. Repeats until done, then calls /verify for reward
6. Returns token IDs + logprobs + reward to our rollout_func

### Async GRPO

With `async_prefetch: true`, axolotl's async data producer prefetches rollouts in a
background thread while the main thread trains on the previous batch. This gives ~3x
speedup over standard synchronous GRPO since generation and training overlap.

The async trainer's own `_sync_lora_adapter()` handles weight sync to the vLLM server
with proper interval tracking (`vllm_sync_interval`).

### Weight Sync (LoRA Mode)

With `vllm_lora_sync: true`, the plugin (or async trainer) replaces NCCL-based weight
sync with filesystem + HTTP:

1. `accelerator.get_state_dict()` gathers LoRA weights from all ranks
2. Rank 0 saves adapter to `/tmp/lora_sync_*/vN/`
3. Rank 0 POSTs to `/set_lora_adapter/` on vLLM server
4. vLLM loads adapter natively via Punica kernels
5. Only ~40MB transferred (vs ~1.4GB for full model weights)

### Multi-Environment Support

Datasets support per-row environment routing via `agent_ref`:
```jsonl
{"agent_ref": {"name": "reasoning_gym"}, "responses_create_params": {...}}
{"agent_ref": {"name": "instruction_following"}, "responses_create_params": {...}}
```

Or use the simpler per-dataset routing:
```yaml
nemo_gym_datasets:
  - path: reasoning_data.jsonl
    server_name: reasoning_gym
  - path: tool_data.jsonl
    server_name: example_single_tool_call
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nemo_gym_enabled` | bool | `null` | Enable the NeMo Gym integration |
| `nemo_gym_dir` | str | `~/Gym` | Path to NeMo Gym repo |
| `nemo_gym_auto_clone` | bool | `true` | Auto-clone NeMo Gym repo if missing |
| `nemo_gym_auto_start` | bool | `true` | Auto-start resource servers |
| `nemo_gym_config_paths` | list[str] | — | Server config YAMLs (relative to gym_dir) |
| `nemo_gym_datasets` | list[dict] | required | Dataset configs with `path` and optional `server_name` |
| `nemo_gym_head_port` | int | `11000` | Head server port |
| `nemo_gym_server_timeout` | int | `360` | Server startup timeout (seconds) |
| `nemo_gym_verify_timeout` | int | `30` | Per-request timeout (seconds) |
| `nemo_gym_multi_turn` | bool | `false` | Enable multi-turn via agent /run |

### Dataset JSONL Format

Each line must have `responses_create_params` with `input` messages:
```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "What's the weather in SF?"}],
    "tools": [{"name": "get_weather", "type": "function", "strict": true, "parameters": {...}}]
  }
}
```

For multi-turn agent routing, include `agent_ref`:
```json
{"agent_ref": {"name": "my_agent"}, "responses_create_params": {...}}
```

Note: Tool definitions MUST include `"strict": true` and `"additionalProperties": false` for NeMo Gym agent compatibility.

### Reward Functions

Create in a Python file importable from your working directory:

```python
# nemo_gym_rewards.py

def reward_env(completions, prompts=None, **kwargs):
    """Passthrough for multi-turn: agent /run already computed rewards."""
    env_rewards = kwargs.get("env_reward")
    if env_rewards is not None:
        return [float(r) for r in env_rewards]
    return [0.0 for _ in completions]

def reward_nemo_gym_verify(completions, prompts=None, **kwargs):
    """Single-turn: call /verify per completion."""
    # (see experiments/nemo_gym_rewards.py for full implementation)
    ...
```

## Known Issues / Troubleshooting

### NeMo Gym Server Setup
- **pycosat build failure**: `CFLAGS="" uv pip install pycosat --no-build-isolation`
- **Ray version mismatch**: Pin `ray[default]==2.52.1` in all server venvs
- **Pre-build venvs**: `ng_run` creates per-server venvs via Ray. Pre-build them and use `+skip_venv_if_present=true`
- **Tool `strict` field required**: Agent server validates tool definitions require `strict: true`

### vLLM / Weight Sync
- **NCCL weight sync broken with vLLM 0.17**: Use `vllm_lora_sync: true` (filesystem + HTTP)
- **`CUDA_HOME` required**: DeepSpeed import needs it for the nvcc shim
- **Use vLLM native OpenAI server**: For multi-turn agent mode, use `python -m vllm.entrypoints.openai.api_server` (not `trl vllm-serve` or `axolotl vllm-serve`)

### Multi-Turn
- **Agent server required**: Multi-turn delegates to NeMo Gym's agent server `/run` endpoint. Without an agent, the plugin falls back to single-turn `/verify`
- **Model server proxy**: NeMo Gym needs a `responses_api_models` server that proxies to your vLLM. See the agent config example above

### FSDP2
- Validated on 2 GPUs with single-turn + LoRA
- Async field filtering: The builder automatically filters async-only config fields when using the standard GRPO trainer

## Comparison with Other Integrations

| Feature | Axolotl + NeMo Gym | Unsloth + NeMo Gym | NeMo RL (native) |
|---------|-------------------|-------------------|-------------------|
| Server management | Automatic | Manual (notebook) | Built-in |
| Multi-environment | Per-row routing | Manual code | YAML config |
| Multi-turn / tool use | Agent /run delegation | No | Agent /run (Ray) |
| Async GRPO (3x speedup) | Yes | No | Yes |
| LoRA sync | Filesystem + HTTP | N/A | NCCL |
| Multi-GPU (FSDP2) | Yes | No | Yes (Ray) |
| Config-driven | Yes | No (code) | Yes |
