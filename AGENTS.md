# Axolotl

Fine-tuning framework for LLMs. Config-driven: every training run is defined by a single YAML file.

## Tech Stack

Python, PyTorch, HuggingFace Transformers, TRL, PEFT (LoRA/QLoRA), DeepSpeed, FSDP, vLLM (for GRPO generation).

## Commands

```bash
axolotl train config.yaml              # Train (single or multi-GPU, auto-detected)
axolotl preprocess config.yaml         # Tokenize dataset and validate config
axolotl preprocess config.yaml --debug # Inspect tokenized samples and label masking
axolotl inference config.yaml          # Interactive inference
axolotl merge-lora config.yaml         # Merge LoRA adapter into base model
axolotl vllm-serve config.yaml         # Start vLLM server for GRPO/EBFT training
axolotl fetch examples                 # Download example configs
axolotl agent-docs                     # Show agent-optimized docs (bundled with pip package)
axolotl agent-docs grpo                # Topic-specific agent reference
axolotl config-schema                  # Dump config JSON schema
```

## Training Methods

| Method | Config Key | When to Use |
|--------|-----------|-------------|
| SFT | *(default)* | Input-output pairs, instruction tuning |
| DPO/IPO | `rl: dpo` / `rl: dpo, dpo_loss_type: ["ipo"]` | Paired preference data (chosen vs rejected) |
| KTO | `rl: kto` | Unpaired binary preference labels |
| ORPO | `rl: orpo` | Single-stage alignment, no ref model |
| GRPO | `rl: grpo` | RL with verifiable reward functions (math, code) |
| EBFT | `rl: ebft` | Feature-matching rewards from internal representations |

Agent-specific references:
- [docs/agents/sft.md](docs/agents/sft.md) — supervised fine-tuning
- [docs/agents/preference_tuning.md](docs/agents/preference_tuning.md) — DPO, IPO, KTO, ORPO, SimPO
- [docs/agents/grpo.md](docs/agents/grpo.md) — GRPO online RL with reward functions
- [docs/agents/reward_modelling.md](docs/agents/reward_modelling.md) — outcome and process reward models
- [docs/agents/pretraining.md](docs/agents/pretraining.md) — continual pretraining
- [docs/agents/model_architectures.md](docs/agents/model_architectures.md) — model-specific quirks (Gemma4, Qwen3.5 MoE, etc.)
- [docs/agents/new_model_support.md](docs/agents/new_model_support.md) — debugging and adding support for new model architectures

## Config Pattern

All training is config-driven. A YAML file specifies model, adapter, dataset(s), and hyperparameters:

```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
adapter: lora                    # or qlora, or omit for full fine-tune
datasets:
  - path: my_dataset
    type: chat_template          # prompt strategy (see docs/dataset-formats/)
output_dir: ./outputs/lora-out
```

Config schema: `src/axolotl/utils/schemas/config.py` (AxolotlInputConfig).

## Project Structure

```
src/axolotl/
  cli/                           # CLI entry points (train, preprocess, inference, merge_lora, vllm_serve)
  core/
    builders/                    # TrainerBuilder classes (causal.py for SFT, rl.py for RLHF)
    trainers/                    # Trainer classes, mixins (optimizer, scheduler, packing)
      dpo/                       # DPO trainer and config
      grpo/                      # GRPO trainer and sampler
  loaders/                       # Model, tokenizer, adapter, processor loading
  prompt_strategies/             # Dataset format handlers (chat_template, alpaca, dpo/, kto/, orpo/)
  utils/schemas/                 # Pydantic config schemas (config, model, training, peft, trl, fsdp)
  integrations/                  # Plugins (liger, cut_cross_entropy, swanlab, nemo_gym)
  monkeypatch/                   # Runtime patches for HF transformers

examples/                        # Example YAML configs by model (llama-3/, qwen2/, mistral/, ebft/)
deepspeed_configs/               # DeepSpeed JSON configs (zero2, zero3)
docs/                            # Quarto documentation site
```

## Code Conventions

- Config-driven: features are toggled via YAML, not code changes
- Prompt strategies: `src/axolotl/prompt_strategies/` — each `type:` value maps to a function
- Plugin system: `plugins:` list in config loads integration modules
- Trainer mixins: `core/trainers/mixins/` for composable trainer behaviors
- Schemas: all config validation via Pydantic in `utils/schemas/`

## Key Documentation

- [Getting Started](docs/getting-started.qmd) — quickstart tutorial
- [Choosing a Method](docs/choosing_method.qmd) — SFT vs DPO vs GRPO decision guide
- [Config Reference](docs/config-reference.qmd) — all config options
- [Dataset Formats](docs/dataset-formats/) — chat_template, alpaca, input_output, completion
- [RLHF](docs/rlhf.qmd) — DPO, KTO, ORPO, GRPO, EBFT configs and dataset formats
- [GRPO Deep Dive](docs/grpo.qmd) — async training, custom rewards, scaling
- [vLLM Serving](docs/vllm_serving.qmd) — vLLM setup for GRPO/EBFT
- [Multi-GPU](docs/multi-gpu.qmd) — FSDP and DeepSpeed
- [Training Stability](docs/training_stability.qmd) — debugging loss, NaN, OOM
- [Debugging](docs/debugging.qmd) — VSCode setup, Docker debugging
