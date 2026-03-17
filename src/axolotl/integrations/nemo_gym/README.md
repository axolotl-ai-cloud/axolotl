# NeMo Gym Integration for Axolotl

Train LLMs with reinforcement learning using [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) environments as reward sources. NeMo Gym provides 50+ verified RL environments spanning math, coding, tool-use, reasoning, and safety — each with deterministic reward signals via `/verify` endpoints.

## Quick Start

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager (for NeMo Gym's venv)
- Git
- Two GPUs recommended (one for vLLM server, one for training)

### 1. Set Up NeMo Gym

```bash
# Clone
git clone https://github.com/NVIDIA-NeMo/Gym.git ~/Gym
cd ~/Gym
uv venv --python 3.12
source .venv/bin/activate
uv sync

# Fix pycosat build (GCC 13+)
CFLAGS="" uv pip install pycosat --python .venv/bin/python --no-build-isolation

# Install reasoning-gym deps
uv pip install --python .venv/bin/python reasoning-gym matplotlib pillow

# Pre-build the resource server venv (avoids build failures in ng_run)
uv venv --seed --allow-existing --python 3.12 resources_servers/reasoning_gym/.venv
CFLAGS="" uv pip install --python resources_servers/reasoning_gym/.venv/bin/python \
    pycosat --no-build-isolation
uv pip install --python resources_servers/reasoning_gym/.venv/bin/python \
    -e . reasoning-gym matplotlib pillow cycler contourpy kiwisolver "ray[default]==2.52.1"
```

### 2. Create Dataset and Start Servers

```bash
cd ~/Gym

# Create sudoku dataset
.venv/bin/python resources_servers/reasoning_gym/scripts/create_dataset.py \
  --task mini_sudoku --size 2000 --seed 42 \
  --output resources_servers/reasoning_gym/data/train_mini_sudoku.jsonl

# Start servers (use skip_venv_if_present for pre-built venvs)
ng_run "+config_paths=[resources_servers/reasoning_gym/configs/resources_only.yaml]" \
       "+skip_venv_if_present=true"
```

### 3. Single-Turn Training (Simplest)

```yaml
# single_turn.yaml
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

```bash
cd experiments && axolotl train single_turn.yaml
```

### 4. Multi-Turn with NeMo Gym Agent Servers (Recommended)

For multi-turn environments (tool use, multi-step reasoning), the plugin delegates to
NeMo Gym's agent servers via the `/run` endpoint. The agent handles generation (by
calling our vLLM server), tool execution, session management, and reward computation.

**Requirements:** NeMo Gym agent servers running via `ng_run` with agent configs that
reference your vLLM server as the policy model.

LoRA + vLLM server mode is the validated path:

```yaml
# multi_turn_lora.yaml
base_model: Qwen/Qwen3-0.6B

adapter: lora
lora_r: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

rl: grpo
chat_template: tokenizer_default

trl:
  use_vllm: true
  vllm_mode: server
  vllm_server_host: localhost
  vllm_server_port: 8000
  vllm_lora_sync: true          # Key: uses LoRA sync instead of NCCL
  vllm_sync_interval: 5
  num_generations: 4
  max_completion_length: 256
  temperature: 0.8
  reward_funcs:
    - nemo_gym_rewards.reward_tool_use

plugins:
  - axolotl.integrations.nemo_gym.NemoGymPlugin

nemo_gym_enabled: true
nemo_gym_auto_start: false
nemo_gym_head_port: 11000
nemo_gym_multi_turn: true
nemo_gym_max_turns: 3
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
  dtype: auto

learning_rate: 5e-6
micro_batch_size: 1
gradient_accumulation_steps: 4
max_steps: 30
gradient_checkpointing: true
bf16: true
output_dir: ./outputs/nemo_gym_multi_turn
```

Run with two terminals:

```bash
# Terminal 1: vLLM LoRA server on GPU 0
cd experiments
CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve multi_turn_lora.yaml

# Terminal 2: Training on GPU 1
cd experiments
CUDA_VISIBLE_DEVICES=1 CUDA_HOME=$HOME/env-claude-cu130/cuda_shim \
  axolotl train multi_turn_lora.yaml
```

## How It Works

### Architecture

**Single-Turn** (reward_fn calls /verify directly):
```
axolotl train → GRPO Trainer generates completions
  → NeMo Gym plugin reward_fn calls POST /verify on resource server
  → reward flows back to GRPO for advantage computation
```

**Multi-Turn** (rollout_func delegates to NeMo Gym agent /run):
```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  axolotl    │     │  NeMo Gym    │────▶│  vLLM LoRA       │
│  train      │────▶│  Agent /run  │◀────│  Server (GPU 0)  │
│  (GPU 1)    │     │              │     │  /v1/responses    │
└─────────────┘     └──────┬───────┘     └──────────────────┘
      │  LoRA sync         │                      ▲
      │  (filesystem)      ▼                      │
      └───────────▶ ┌──────────────┐              │
                    │  Resource    │   model weights
                    │  Server     │   synced via
                    │  (tools +   │   /set_lora_adapter
                    │   verify)   │
                    └─────────────┘
```

The agent server orchestrates the multi-turn loop:
1. Calls our vLLM server for model generation
2. Parses tool calls from model output
3. Executes tools against resource servers
4. Feeds tool results back to the model
5. Repeats until done, then calls /verify for reward
6. Returns token IDs + logprobs + reward to our rollout_func

### Weight Sync (LoRA Mode)

With `vllm_lora_sync: true`, the plugin replaces TRL's NCCL-based `sync_weights` with:

1. `accelerator.get_state_dict()` gathers LoRA weights (all ranks participate)
2. Rank 0 saves adapter via `model.save_pretrained()` to `/tmp/lora_sync_*/vN/`
3. Rank 0 POSTs to `/set_lora_adapter/` on vLLM server
4. vLLM loads adapter natively via Punica kernels
5. Only ~40MB transferred (vs ~1.4GB for full model weights)

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nemo_gym_enabled` | bool | `null` | Enable the NeMo Gym integration |
| `nemo_gym_dir` | str | `~/Gym` | Path to NeMo Gym repo |
| `nemo_gym_auto_clone` | bool | `true` | Auto-clone NeMo Gym repo if missing |
| `nemo_gym_auto_start` | bool | `true` | Auto-start resource servers |
| `nemo_gym_config_paths` | list[str] | — | Server config YAMLs (relative to gym_dir) |
| `nemo_gym_datasets` | list[dict] | required | Dataset configs (see below) |
| `nemo_gym_head_port` | int | `11000` | Head server port |
| `nemo_gym_server_timeout` | int | `360` | Server startup timeout (seconds) |
| `nemo_gym_verify_timeout` | int | `30` | Per-request verify timeout (seconds) |
| `nemo_gym_model_name` | str | `base_model` | Model name in verify requests |
| `nemo_gym_multi_turn` | bool | `false` | Enable multi-turn rollouts with tool execution |
| `nemo_gym_max_turns` | int | `10` | Max turns per multi-turn rollout |

### Dataset Config

Each entry in `nemo_gym_datasets`:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | str | yes | JSONL file path (absolute or relative to gym_dir) |
| `server_name` | str | yes | NeMo Gym resource server name |
| `max_samples` | int | no | Max samples to use from this file |

### Dataset JSONL Format

Each line must have:
```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "What's the weather in SF?"}],
    "tools": [{"name": "get_weather", "type": "function", "parameters": {...}}]
  }
}
```

Use standard `"user"` role (not `"developer"`) for prompts. Tools in the dataset are automatically extracted and passed to the model's chat template.

## Known Issues / Troubleshooting

### NeMo Gym Server Setup

- **pycosat build failure**: `CFLAGS="" uv pip install pycosat --no-build-isolation`
- **Ray version mismatch**: Pin `ray[default]==2.52.1` in resource server venvs to match head server
- **Resource server port is dynamic**: Discovered via head server, don't hardcode
- **Head server YAML double-encoding**: Plugin handles this automatically
- **Pre-build resource server venvs**: Use `+skip_venv_if_present=true` with `ng_run`

### vLLM / Weight Sync

- **vLLM V1 has no `init_communicator`**: NCCL weight sync is broken with vLLM 0.17. Use `vllm_lora_sync: true` which uses filesystem + HTTP instead
- **`GuidedDecodingParams` import error**: Renamed to `StructuredOutputsParams` in vLLM 0.17. The plugin's `vllm_serve_lora.py` handles this with a try/except import
- **`CUDA_HOME` required**: DeepSpeed import needs it. Set `CUDA_HOME=$HOME/env-claude-cu130/cuda_shim`

### Multi-Turn Tool Calling

- **Model must support tool calling**: Qwen3 family works well. The plugin adds `enable_thinking=False` to prevent the model wasting tokens on `<think>` blocks
- **Dataset tool format matters**: NeMo Gym datasets may have extra fields (`strict`, `type`, `description`) in tool definitions. The plugin strips these to clean OpenAI format
- **Logprobs format**: Server mode returns nested `list[list[float]]`, colocate returns flat `list[float]`. The plugin normalizes both to TRL's expected `list[list[list[float]]]` format
- **Colocate mode doesn't produce tool calls**: Known issue — works in standalone tests but not inside axolotl train. Use server mode instead
- **`datasets` field required**: Axolotl validation requires a `datasets` field even when the plugin loads data. Point it at the same JSONL file

### Reward Functions

Create reward functions in a Python file importable from your working directory:

```python
# nemo_gym_rewards.py
def reward_tool_use(completions, prompts=None, **kwargs):
    """1.0 if tool was used, 0.0 otherwise."""
    tool_used = kwargs.get("tool_used")
    if tool_used is not None:
        return [float(r) for r in tool_used]
    return [0.0 for _ in completions]
```

Reference in config: `trl.reward_funcs: ["nemo_gym_rewards.reward_tool_use"]`

## Comparison with Other Integrations

| Feature | Axolotl + NeMo Gym | Unsloth + NeMo Gym | NeMo RL (native) |
|---------|-------------------|-------------------|-------------------|
| Server management | Automatic | Manual (notebook) | Built-in |
| Multi-environment | YAML config | Manual code | YAML config |
| Multi-turn / tool use | Yes (rollout_func) | No | Yes (Ray actor) |
| LoRA support | Full (LoRA sync) | Full | Full |
| Async GRPO | Yes (axolotl) | No | Yes |
| Multi-GPU (FSDP/DeepSpeed) | Yes | No | Yes (Ray) |
| vLLM generation | Yes (server mode) | No | Yes |
| Config-driven | Yes | No (code) | Yes |
