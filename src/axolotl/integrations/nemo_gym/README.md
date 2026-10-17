# NeMo Gym Integration for Axolotl

Train LLMs with reinforcement learning using [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) environments as reward sources. NeMo Gym provides 50+ verified RL environments spanning math, coding, tool-use, reasoning, and safety — each with deterministic reward signals.

## Training Paths

| Path | Multi-turn | Architecture |
|------|------------|--------------|
| **Async GRPO + agent /run** | Yes | Rollout process sends each rollout to an agent's `/run` |
| Async GRPO + /verify | No | TRL's rollout process generates, reward function calls /verify |
| Standard GRPO + /verify | No | Reward function calls /verify directly |
| FSDP2 + /verify (2 GPU) | No | `fsdp_version: 2` |

Async GRPO (`trl.async_prefetch: true`) runs on TRL's `AsyncGRPOTrainer`: a CUDA-free
rollout process keeps generating and scoring while the trainer consumes scored samples,
so generation and training overlap. Multi-turn (`nemo_gym_multi_turn: true`) requires it.

## Quick Start

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager (for NeMo Gym's venv)
- Two GPUs recommended (one for vLLM server, one for training)

### 1. Set Up NeMo Gym

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git ~/Gym
cd ~/Gym
# Gym requires Python >= 3.13
uv venv --python 3.13 && source .venv/bin/activate && uv sync

# Fix pycosat build (GCC 13+)
CFLAGS="" uv pip install pycosat --python .venv/bin/python --no-build-isolation

# Pre-build resource server venvs
for dir in resources_servers/reasoning_gym resources_servers/example_single_tool_call responses_api_models/vllm_model responses_api_agents/simple_agent; do
    uv venv --seed --allow-existing --python 3.13 $dir/.venv
    uv pip install --python $dir/.venv/bin/python setuptools wheel  # pycosat build deps
    CFLAGS="" uv pip install --python $dir/.venv/bin/python pycosat --no-build-isolation 2>/dev/null
    uv pip install --python $dir/.venv/bin/python -e . "ray[default]==2.58.0"
done

# Install extra deps for reasoning_gym
uv pip install --python resources_servers/reasoning_gym/.venv/bin/python \
    reasoning-gym matplotlib pillow cycler contourpy kiwisolver
```

### 2. Multi-Turn with Async GRPO

NeMo Gym's agent server handles multi-turn tool execution while async GRPO keeps
rollouts in flight and trains on them as they finish.

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
      model: Qwen/Qwen3-0.6B   # Must match nemo_gym_model_name (default: base_model)
      return_token_id_information: true   # Required: training needs token IDs + logprobs
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
      - name: example
        type: example
        jsonl_fpath: resources_servers/example_single_tool_call/data/example.jsonl
```

**Step 2: Start three services**

```bash
# Terminal 1: vLLM on GPU 0, with the weight sync routes, runtime LoRA loading and tool calling
VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True CUDA_VISIBLE_DEVICES=0 \
    vllm serve Qwen/Qwen3-0.6B --max-model-len 2048 --gpu-memory-utilization 0.85 \
    --enable-lora --max-lora-rank 16 --max-loras 7 \
    --weight-transfer-config '{"backend":"nccl"}' --logprobs-mode processed_logprobs \
    --enable-auto-tool-choice --tool-call-parser hermes

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
  async_prefetch: true   # Async GRPO, required for multi-turn
  num_generations: 4
  max_completion_length: 512
  temperature: 0.8

plugins:
  - axolotl.integrations.nemo_gym.NemoGymPlugin

nemo_gym_enabled: true
nemo_gym_auto_start: false
nemo_gym_head_port: 11000
nemo_gym_multi_turn: true
nemo_gym_verify_timeout: 120
nemo_gym_datasets:
  - path: ~/Gym/resources_servers/example_single_tool_call/data/example.jsonl
    server_name: example_single_tool_call

datasets:
  - path: ~/Gym/resources_servers/example_single_tool_call/data/example.jsonl
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
an agent server. The plugin adds a reward function that calls `/verify` directly.

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
```text
axolotl train → GRPO Trainer generates completions
  → NeMo Gym plugin reward function calls POST /verify on resource server
  → reward flows back to GRPO for advantage computation
```

### Multi-Turn (Agent /run)
```text
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
6. Returns token IDs + logprobs + reward to the async GRPO rollout process

### Async GRPO Architecture (Multi-Turn)

Multi-turn runs on TRL's `AsyncGRPOTrainer`. Its CUDA-free rollout process runs the
plugin's `NemoGymRolloutLoop`, which:

1. Takes the next dataset row and sends `num_generations` rollouts of it to the agent
   named by its `agent_ref` (the first agent otherwise), with `max_completion_length`
   as each model call's `max_output_tokens`
2. Turns the prompt and generated token IDs and logprobs of every model call into
   training rows with TRL's multi-turn reconciler: generated tokens are trained, tool
   results and other context are masked, and a rewritten history starts a new row
3. Scores the agent's reward as the `nemo_gym` reward column, alongside any
   `trl.reward_funcs`, and normalizes advantages within the group
4. Queues the samples tagged with the policy version they were sampled under

The trainer drops samples more than `max_staleness` weight syncs old, packs the rest
by token budget and bounds each token's policy/sampler ratio with
`vllm_importance_sampling_cap`. A failed `/run` is left out of its group's baseline.

### Weight Sync (LoRA Mode)

Async GRPO picks the weight sync from what the vLLM server reports: with
`--enable-lora` it publishes only the adapter, otherwise it merges the adapter and
broadcasts full weights over NCCL.

1. Each sync saves the adapter and loads it into vLLM as `trl-policy-v<N>`
2. In multi-turn mode the adapter is also loaded under `nemo_gym_model_name` (default:
   `base_model`), because agent servers address vLLM by the fixed `model` in their
   config. Keep that `model` equal to `nemo_gym_model_name`, and to the name vLLM serves
   the base model under
3. That alias keeps its name across versions, and vLLM keys its prefix cache on the
   name, so a sync waits for in-flight requests to finish (which clears the cache)
   before moving the alias
4. vLLM needs `--max-loras` of at least `max_staleness + 3` (default 7); `axolotl
   vllm-serve` sizes it itself

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
| `nemo_gym_verify_timeout` | int | `30` | Per `/verify` request timeout (seconds) |
| `nemo_gym_run_timeout` | int | `300` | Seconds an agent `/run` may go without sending data |
| `nemo_gym_model_name` | str | `base_model` | Model name the agents' model proxy sends to vLLM |
| `nemo_gym_multi_turn` | bool | `false` | Enable multi-turn via agent /run (needs `trl.async_prefetch: true`) |

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

### Rewards

The plugin adds the NeMo Gym reward itself, as the last reward column (`nemo_gym`):

- Single-turn: a reward function that calls the resource servers' `/verify` for a
  group's completions concurrently
- Multi-turn: the reward the agent's `/run` returns

Rewards listed in `trl.reward_funcs` are scored alongside it, so `trl.reward_weights`
needs one more entry. Configs that still list
`axolotl.integrations.nemo_gym.rewards.reward_nemo_gym_verify` or `reward_env` have them
dropped with a warning. `reward_nemo_gym_verify` remains importable for use without the
plugin.

## Known Issues / Troubleshooting

### NeMo Gym Server Setup
- **pycosat build failure**: `CFLAGS="" uv pip install pycosat --no-build-isolation`
- **Ray version mismatch**: Pin all server venvs to the `ray[default]` version in Gym's `pyproject.toml` (2.58.0 as of Oct 2026)
- **Pre-build venvs**: `ng_run` creates per-server venvs via Ray. Pre-build them and use `+skip_venv_if_present=true`
- **Tool `strict` field required**: Agent server validates tool definitions require `strict: true`

### vLLM / Weight Sync
- **Start vLLM with weight sync routes, LoRA, tool calling and runtime loading** (async GRPO
  needs `VLLM_SERVER_DEV_MODE=1` for `/pause`, `/resume` and `/server_info`):
  ```bash
  VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7 \
    --enable-lora --max-lora-rank 64 --max-loras 7 \
    --weight-transfer-config '{"backend":"nccl"}' --logprobs-mode processed_logprobs \
    --enable-auto-tool-choice --tool-call-parser hermes
  ```
- **`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`** (set automatically by `axolotl vllm-serve` when LoRA is enabled): Required for `vllm_lora_sync: true`. Without it, vLLM won't expose the `/v1/load_lora_adapter` endpoint; the plugin warns at startup and the first sync raises.
- **`--enable-lora`**: Enables LoRA adapter support in vLLM
- **`--enable-auto-tool-choice --tool-call-parser hermes`**: Required for Qwen3 tool calling
- **`max_model_len` must be > `max_completion_length`**: Leave room for prompt tokens (~200). If equal, the NeMo Gym model proxy gets a 400 error and returns empty completions.
- **`CUDA_HOME` required**: DeepSpeed import needs it for the nvcc shim
- **Full-parameter weight sync needs the native NCCL routes**: start the server with `axolotl vllm-serve` (sets `VLLM_SERVER_DEV_MODE=1` and the weight-transfer config), or run `vllm serve` with `VLLM_SERVER_DEV_MODE=1 --weight-transfer-config '{"backend":"nccl"}'`. For LoRA, use `vllm_lora_sync: true` (filesystem + `/v1/load_lora_adapter`)

### Multi-Turn
- **Agent server required**: Multi-turn delegates to NeMo Gym's agent server `/run` endpoint. Without an agent, the plugin falls back to single-turn `/verify`
- **`return_token_id_information: true`**: The agents' `vllm_model` server must return token IDs and logprobs; the rollout process stops with an error otherwise
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
| Async GRPO | Yes | No | Yes |
| LoRA sync | Filesystem + HTTP | N/A | NCCL |
| Multi-GPU (FSDP2) | Yes | No | Yes (Ray) |
| Config-driven | Yes | No (code) | Yes |
