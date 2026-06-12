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
```

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

## Export

Draft checkpoints land in `output_dir`; convert with TorchSpec's
`tools/convert_to_hf.py` to get an HF-loadable EAGLE-3 draft.

## Gotchas

- Run as a plain process — it becomes the Ray driver; do **not** wrap in
  `accelerate`/`torchrun`.
- `draft_model_config: null` auto-generates a reduced-layer draft from the target.
- `mooncake_protocol: rdma` requires `speculator.mooncake_device_name` (the NIC).
- `inference_engine: vllm` needs vLLM ≥ 0.18.0 and is incompatible with
  `train_with_decode`.
