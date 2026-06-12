# TorchSpec — EAGLE-3 speculator training

Train EAGLE-3 speculative-decoding draft models ("speculators") from an axolotl
config, using [TorchSpec](https://github.com/lightseekorg/TorchSpec).

## How it works

TorchSpec runs **disaggregated**: SGLang/vLLM inference engines run the *target*
model and emit its hidden states (3 aux layers + last hidden state + input_ids),
stream them through a [Mooncake](https://github.com/kvcache-ai/Mooncake) store,
and FSDP **training workers** consume them to train the draft model — all
orchestrated by Ray.

Because TorchSpec owns the whole run, axolotl does **not** use its normal
in-process HF-`Trainer`. Instead it validates the YAML, translates it into the
`argparse.Namespace` TorchSpec expects, and launches
`torchspec.train_entry.train_async_no_generation` in-process (as the Ray driver).

## Install

```bash
pip install -e '.[torchspec]'
# plus a backend, e.g. SGLang, and Mooncake's system requirements
```

## Run

```bash
# dry-run: print the translated TorchSpec config and exit (no Ray/GPUs)
axolotl train-speculator examples/speculators/qwen3-8b-eagle3.yaml --dry-run

# train (NOT under accelerate/torchrun)
axolotl train-speculator examples/speculators/qwen3-8b-eagle3.yaml

# forward extra OmegaConf dotlist overrides to TorchSpec
axolotl train-speculator examples/speculators/qwen3-8b-eagle3.yaml training.num_train_steps=10
```

The Qwen3-8B example needs ≥4 GPUs (2 inference + 2 training).

## Config

Add the plugin and a `speculator:` block (see `args.py` → `TorchSpecArgs` for all
fields):

```yaml
plugins:
  - axolotl.integrations.torchspec.TorchSpecPlugin

base_model: Qwen/Qwen3-8B
datasets:
  - path: ./conversations.jsonl   # raw conversations; TorchSpec tokenizes it
    type: chat_template
chat_template: qwen3

speculator:
  ttt_length: 7
  inference_engine: sgl           # sgl | vllm | hf
  inference_num_gpus: 2
  training_num_gpus: 2            # per node
  mooncake_protocol: tcp          # rdma requires mooncake_device_name
```

Top-level axolotl keys (`base_model`, `sequence_len`, `learning_rate`,
`num_epochs`, `micro_batch_size`, `warmup_ratio`, `max_grad_norm`, `seed`,
`gradient_checkpointing`, `output_dir`) are mapped to the corresponding TorchSpec
`model.*`/`training.*` fields by `translate.py`.

### Dataset

TorchSpec does its own tokenization from a conversations-format dataset (rows with
a `conversations` key); axolotl's prompt strategies are not used here. The
`chat_template` is mapped to TorchSpec's template name (`qwen3`/`chatml` → `qwen`,
`llama3` → `llama3`); set `speculator.chat_template` to override.

## Export

TorchSpec writes draft-model checkpoints under `output_dir`; convert to an
HF-loadable EAGLE-3 draft with TorchSpec's `tools/convert_to_hf.py`.

## Notes / limitations

- `draft_model_config: null` → TorchSpec auto-generates a reduced-layer EAGLE-3
  config from the target model.
- Optional dependency surface is large (Ray, Mooncake, SGLang/vLLM, `numpy<2.4`)
  and stays isolated in the `torchspec` extra; nothing is imported unless you run
  `train-speculator`.
- Phase 2 will add a `plugins:` + `axolotl train` dispatch path; today, use the
  `train-speculator` command.
