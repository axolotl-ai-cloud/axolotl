# Speculators (EAGLE-3) — Agent Reference

Train EAGLE-3 speculative-decoding draft models via
[TorchSpec](https://github.com/lightseekorg/TorchSpec). Full notes:
[src/axolotl/integrations/torchspec/README.md](../../src/axolotl/integrations/torchspec/README.md).

## Architecture (disaggregated)

```
Inference engines (SGLang/vLLM)        Mooncake store         Training workers (FSDP)
┌───────────────────────────┐   hidden  ┌───────────┐  hidden  ┌────────────────────────┐
│ Run TARGET model          │  states   │ RDMA/TCP  │  states  │ Train DRAFT (EAGLE-3)  │
│ emit 3 aux layers +       │──────────►│ tensor    │─────────►│ forward-KL TTL loss    │
│ last hidden + input_ids   │           │ transfer  │          │ over ttt_length steps  │
└───────────────────────────┘           └───────────┘          └────────────────────────┘
        Ray orchestrates engines, controller, and training actors
```

TorchSpec owns the whole run. Axolotl validates the YAML, translates it, and
launches `torchspec.train_entry.train_async_no_generation` in-process.

## Components Required

1. A YAML config with `plugins: [axolotl.integrations.torchspec.TorchSpecPlugin]`
   and a `speculator:` block
2. `pip install -e '.[torchspec]'` + an SGLang/vLLM backend + Mooncake
3. A conversations-format dataset (rows with a `conversations` key)
4. ≥4 GPUs for the Qwen3-8B example (2 inference + 2 training)

## Commands

```bash
axolotl train-speculator config.yaml --dry-run          # print translated config, no GPUs
axolotl train-speculator config.yaml                    # launch (NOT under accelerate)
axolotl train-speculator config.yaml training.num_train_steps=10   # dotlist override
axolotl train --launcher python config.yaml             # standard verb, single-process only
axolotl preprocess config.yaml                          # just materialize the standardized JSONL
axolotl export-speculator config.yaml [--prune-vocab]   # FSDP draft checkpoint -> HF EAGLE-3 model
```

Both entry points must run single-process — TorchSpec is the sole Ray driver; a
multi-process `accelerate`/`torchrun` launch is rejected at train time.

## Dataset reuse

With `speculator.prepare_dataset: true` (default), axolotl's dataset **loading**
standardizes the `datasets:` list (ShareGPT/OpenAI/etc., multi-source, merged)
into a `conversations` JSONL at `<output_dir>/torchspec_data/train.jsonl`;
TorchSpec then tokenizes + masks it (EAGLE-3-correct). Set it to `false` to pass
`datasets[0].path` through untouched.

## Config keys

| Axolotl key | Maps to TorchSpec |
|---|---|
| `base_model` | `model.target_model_path` |
| `datasets[0].path` | `dataset.train_data_path` (raw conversations; TorchSpec tokenizes) |
| `chat_template` | `dataset.chat_template` (`qwen3`/`chatml`→`qwen`, `llama3`→`llama3`) |
| `sequence_len` | `training.max_seq_length` |
| `learning_rate`, `num_epochs`, `micro_batch_size`, `warmup_ratio`, `max_grad_norm`, `seed` | `training.*` |
| `speculator.ttt_length` | `training.ttt_length` (speculative depth) |
| `speculator.inference_engine` | `inference.inference_engine_type` (`sgl`/`vllm`/`hf`) |
| `speculator.inference_num_gpus` / `training_num_gpus` | GPU split |
| `speculator.mooncake_protocol` | `mooncake.protocol` (`tcp`/`rdma`) |

See `TorchSpecArgs` in `src/axolotl/integrations/torchspec/args.py` for the full
`speculator:` schema.

## Draft architecture

`draft_model_config: null` + no `draft_*` knobs → TorchSpec auto-generates a
1-layer EAGLE-3 head. Set `speculator.draft_num_hidden_layers` /
`draft_hidden_size` / `draft_intermediate_size` / `draft_vocab_size` /
`draft_config_overrides` to generate a tuned config from the target. An explicit
`draft_model_config` JSON path overrides the knobs.

## Export

`axolotl export-speculator config.yaml` converts the FSDP draft checkpoint under
`output_dir/checkpoints` into an HF-loadable EAGLE-3 model (wraps TorchSpec's
`tools/convert_to_hf.py`). `--prune-vocab` prunes to `draft_vocab_size` using the
training dataset.

## Gotchas

- Run as a plain process — it becomes the Ray driver; do **not** wrap in
  `accelerate`/`torchrun`.
- `mooncake_protocol: rdma` requires `speculator.mooncake_device_name` (the NIC).
- `inference_engine: vllm` needs vLLM ≥ 0.18.0 and is incompatible with
  `train_with_decode`.
