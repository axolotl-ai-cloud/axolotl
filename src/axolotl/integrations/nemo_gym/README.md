# NeMo Gym Integration for Axolotl

Train LLMs with reinforcement learning using [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) environments as reward sources. NeMo Gym provides 50+ verified RL environments spanning math, coding, tool-use, reasoning, and safety — each with deterministic reward signals via `/verify` endpoints.

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Axolotl    │────▶│  GRPO        │────▶│  NeMo Gym        │
│  Config     │     │  Trainer     │     │  /verify endpoint │
│  (YAML)     │     │  (TRL)       │◀────│  (reward signal)  │
└─────────────┘     └──────────────┘     └──────────────────┘
      │                    ▲                      ▲
      ▼                    │                      │
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  NeMo Gym   │     │  Model       │     │  Resource Server  │
│  Plugin     │────▶│  Completions │────▶│  (Sudoku, Math,   │
│  (lifecycle)│     │              │     │   Coding, etc.)   │
└─────────────┘     └──────────────┘     └──────────────────┘
```

1. The plugin starts NeMo Gym resource servers (or connects to existing ones)
2. Datasets are loaded from NeMo Gym JSONL files containing task prompts
3. The GRPO trainer generates completions from the model
4. Completions are sent to NeMo Gym `/verify` endpoints for reward scoring
5. Rewards drive GRPO policy optimization

## Quick Start

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager (for NeMo Gym's venv)
- Git (for auto-cloning NeMo Gym)

### 1. Minimal Sudoku Example

Create a config file `nemo_gym_sudoku.yaml`:

```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct
model_type: AutoModelForCausalLM
sequence_len: 4096

rl: grpo
chat_template: tokenizer_default

trl:
  use_vllm: false
  num_generations: 8
  max_completion_length: 2048

plugins:
  - axolotl.integrations.nemo_gym.NemoGymPlugin

nemo_gym_enabled: true
nemo_gym_config_paths:
  - resources_servers/reasoning_gym/configs/resources_only.yaml
nemo_gym_datasets:
  - path: resources_servers/reasoning_gym/data/train_mini_sudoku.jsonl
    server_name: reasoning_gym
    max_samples: 2000

learning_rate: 1.0e-5
optimizer: adamw_8bit
micro_batch_size: 1
gradient_accumulation_steps: 64
max_steps: 100
output_dir: ./outputs/nemo_gym_sudoku
```

Before first run, set up NeMo Gym and create the dataset:

```bash
# 1. Clone and set up NeMo Gym (plugin can auto-clone, but manual setup gives more control)
git clone https://github.com/NVIDIA-NeMo/Gym.git ~/Gym
cd ~/Gym
uv venv --python 3.12
source .venv/bin/activate
uv sync

# 2. Fix pycosat build on some systems (compiler flag issue)
CFLAGS="" uv pip install pycosat --python .venv/bin/python --no-build-isolation

# 3. Install reasoning-gym and its deps
uv pip install --python .venv/bin/python reasoning-gym matplotlib pillow

# 4. Pre-build the resource server venv (ng_run creates one per server via Ray)
uv venv --seed --allow-existing --python 3.12 resources_servers/reasoning_gym/.venv
CFLAGS="" uv pip install --python resources_servers/reasoning_gym/.venv/bin/python \
    pycosat --no-build-isolation
uv pip install --python resources_servers/reasoning_gym/.venv/bin/python \
    -e . reasoning-gym matplotlib pillow cycler contourpy kiwisolver "ray[default]==2.52.1"

# 5. Create the dataset
.venv/bin/python resources_servers/reasoning_gym/scripts/create_dataset.py \
  --task mini_sudoku --size 2000 --seed 42 \
  --output resources_servers/reasoning_gym/data/train_mini_sudoku.jsonl

# 6. Start NeMo Gym servers (use skip_venv_if_present to use pre-built venv)
ng_run "+config_paths=[resources_servers/reasoning_gym/configs/resources_only.yaml]" \
       "+skip_venv_if_present=true"
```

Then train:

```bash
axolotl train nemo_gym_sudoku.yaml
```

### 2. Multi-Environment Training

Train on multiple environments simultaneously (Sudoku + Instruction Following):

```yaml
nemo_gym_config_paths:
  - resources_servers/reasoning_gym/configs/resources_only.yaml
  - resources_servers/instruction_following/configs/instruction_following.yaml
nemo_gym_datasets:
  - path: resources_servers/reasoning_gym/data/train_mini_sudoku.jsonl
    server_name: reasoning_gym
    max_samples: 1000
  - path: resources_servers/instruction_following/data/instruction_following.jsonl
    server_name: instruction_following
    max_samples: 1000
```

For the instruction following dataset, download from HuggingFace:

```python
from huggingface_hub import hf_hub_download
import shutil
src = hf_hub_download(
    repo_id="nvidia/Nemotron-RL-instruction_following",
    filename="instruction_following.jsonl",
    repo_type="dataset",
)
shutil.copy(src, "~/Gym/resources_servers/instruction_following/data/instruction_following.jsonl")
```

### 3. Multi-Turn with Tool Execution

For agentic environments that require multi-step interactions (tool use, multi-hop QA, etc.),
enable multi-turn mode. This uses TRL's `rollout_func` to orchestrate:

1. Model generates a response (possibly with tool calls)
2. Tool calls are executed against NeMo Gym resource servers
3. Tool results are appended to the conversation
4. Model generates again
5. Repeat until done or max turns reached
6. Final reward from `/verify`

Only model-generated tokens are trained on (environment feedback is masked via `env_mask`).

```yaml
# Requires vLLM for generation
trl:
  use_vllm: true
  vllm_mode: server
  vllm_server_host: localhost
  vllm_server_port: 8000
  num_generations: 4
  max_completion_length: 2048

# Enable multi-turn
nemo_gym_multi_turn: true
nemo_gym_max_turns: 10
```

The multi-turn rollout flow:

```
Turn 1: [Prompt] → Model generates "I'll check the weather" + tool_call(get_weather)
         → Execute tool via POST /get_weather on resource server
         → Tool result: "Sunny, 72F"  (masked from loss via env_mask=0)
Turn 2: [Full conversation] → Model generates final answer
         → POST /verify → reward=1.0
```

Tool calls are parsed from model output in two formats:
- XML: `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
- JSON: `{"name": "func", "arguments": {...}}`

Resource servers define their tools via `/seed_session`. The plugin automatically discovers
and routes tool calls to the correct server endpoints.

### 4. Using with LoRA

Add LoRA configuration to train more efficiently:

```yaml
adapter: lora
lora_r: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

### 4. Pre-started Servers

If you prefer to manage NeMo Gym servers yourself:

```bash
# Start servers manually
cd ~/Gym && source .venv/bin/activate
ng_run "+config_paths=[resources_servers/reasoning_gym/configs/resources_only.yaml]"
```

```yaml
# In your axolotl config
nemo_gym_auto_start: false
nemo_gym_auto_clone: false
nemo_gym_head_port: 11000  # port where your servers are running
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nemo_gym_enabled` | bool | `null` | Enable the NeMo Gym integration |
| `nemo_gym_dir` | str | `~/Gym` | Path to NeMo Gym repo |
| `nemo_gym_auto_clone` | bool | `true` | Auto-clone NeMo Gym repo if missing |
| `nemo_gym_auto_start` | bool | `true` | Auto-start resource servers |
| `nemo_gym_config_paths` | list[str] | required | Server config YAMLs (relative to gym_dir) |
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

## Available NeMo Gym Environments

NeMo Gym ships with 50+ environments. Some popular ones:

- **Math**: `mini_sudoku`, `basic_arithmetic`, `polynomial_equations`, `calculus`
- **Coding**: `text_to_sql`, `code_generation`, `swe_bench`
- **Reasoning**: `logic_puzzles`, `word_sorting`, `number_sequences`
- **Agentic**: `tool_use`, `multi_hop_qa`, `calendar_scheduling`
- **Safety**: `jailbreak_detection`, `refusal_calibration`

See the [NeMo Gym environments docs](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers) for the full list.

## How Rewards Work

The reward function creates OpenAI Responses API-compatible verify requests:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "Solve this sudoku..."}]
  },
  "response": {
    "output": [{
      "role": "assistant",
      "content": [{"type": "output_text", "text": "4 2 1 3\n..."}]
    }]
  },
  "answer": "...",
  "metadata": {...}
}
```

The resource server returns `{"reward": 1.0}` for correct solutions and `{"reward": 0.0}` otherwise. These rewards drive GRPO advantage estimation.

## Comparison with Other Integrations

| Feature | Axolotl + NeMo Gym | Unsloth + NeMo Gym | NeMo RL (native) |
|---------|-------------------|-------------------|-------------------|
| Server management | Automatic | Manual (notebook) | Built-in |
| Multi-environment | YAML config | Manual code | YAML config |
| Multi-turn / tool use | Yes (rollout_func) | No | Yes (Ray actor) |
| LoRA support | Full | Full | Full |
| Async GRPO | Yes (axolotl) | No | Yes |
| Multi-GPU (FSDP/DeepSpeed) | Yes | No | Yes (Ray) |
| vLLM generation | Yes | No | Yes |
| Config-driven | Yes | No (code) | Yes |

Unsloth's NeMo Gym "support" is a pair of Colab notebooks that manually wire up verify endpoints as TRL reward functions. There is no library-level integration — it's effectively example code. Axolotl's plugin provides a production-ready, config-driven integration with automatic server lifecycle management.

## Known Issues / Gotchas

### NeMo Gym Server Setup

- **pycosat build failure**: On some systems (e.g., GCC 13+), `pycosat` fails to build with
  `error: unrecognized command-line option '-fdebug-default-version=4'`. Fix by installing with:
  ```bash
  CFLAGS="" uv pip install pycosat --no-build-isolation
  ```

- **Ray version mismatch**: `ng_run` starts a Ray cluster with a specific version. The resource
  server venvs must use the same Ray version. Check the head server's version with
  `uv pip list --python .venv/bin/python | grep ray` and pin accordingly in resource server venvs.

- **Resource server port is dynamic**: `ng_run` assigns ports dynamically. The plugin discovers
  them via `/global_config_dict_yaml` on the head server (port 11000). Don't hardcode resource
  server ports.

- **Head server double-encodes YAML**: The `/global_config_dict_yaml` endpoint returns a YAML
  string wrapped in another YAML string. The plugin handles this automatically with double-parse.

- **Pre-build resource server venvs**: `ng_run` creates a fresh venv per resource server (via Ray).
  This can be slow and may fail due to build issues. Use `+skip_venv_if_present=true` with a
  pre-built venv to avoid this.

### Multi-Turn

- **Requires vLLM**: Multi-turn mode uses `generate_rollout_completions()` from TRL's experimental
  OpenEnv module, which requires `use_vllm: true` in the TRL config.

- **Logprobs format**: TRL expects logprobs as `list[list[list[float]]]` (batch x seq x top-k).
  The plugin wraps scalar logprobs in single-element lists automatically.

- **Tool call parsing**: The plugin parses tool calls in XML (`<tool_call>...</tool_call>`) and
  JSON formats. Model must generate one of these formats for tool execution to work.

- **Single-turn environments work with multi_turn=true**: If `/seed_session` returns no tools,
  the rollout degrades to single-turn (one generation + verify). No harm in leaving multi-turn
  enabled for mixed environments.
