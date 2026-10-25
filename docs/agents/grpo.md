# GRPO — Agent Reference

Online RL with verifiable reward functions. For full config reference, async features, and scaling, see [grpo.qmd](../grpo.qmd). For vLLM setup, see [vllm_serving.qmd](../vllm_serving.qmd).

## Architecture

```
Terminal 1 (GPU 0)                    Terminal 2 (GPU 1)
┌──────────────────────┐              ┌──────────────────────────────────┐
│  vLLM Server         │   HTTP       │  Trainer                         │
│  Serves base model   │◄────────────►│  1. Send prompts to vLLM         │
│  + LoRA adapter      │  /v1/*       │  2. Score completions (rewards)  │
│                      │  LoRA load   │  3. Compute advantages           │
│  Punica kernels for  │              │  4. PPO-clip gradient update     │
│  LoRA inference      │              │  5. Sync LoRA weights to vLLM    │
└──────────────────────┘              └──────────────────────────────────┘
```

## Components Required

1. A YAML config with `rl: grpo`
2. A reward module (Python file with reward functions)
3. A vLLM backend: a running server (`axolotl vllm-serve config.yaml`, `vllm_mode: server`), or `vllm_mode: colocate` to host the engine on the training GPU (single-GPU; configure via the `vllm:` block and set `vllm.max_model_len`)

## Reward Function Signature

```python
def my_reward(completions, **kwargs) -> list[float]:
    # completions[i][0]["content"] = text of i-th completion
    # **kwargs contains dataset columns not removed by transform
    return [score_for_each_completion]
```

Multiple rewards: `reward_funcs: [r1, r2]` with `reward_weights: [1.0, 0.5]`.

Image datasets (`images`/`image` column, `processor_type: AutoProcessor`) also work; rewards receive the loaded images as the `images` kwarg (one list per sample, for either column). See [rlhf.qmd](../rlhf.qmd#multimodal).

## Key Async Features

`async_prefetch: true` with `rl: grpo` runs TRL's `AsyncGRPOTrainer` (vLLM server, FSDP2/DDP, `beta: 0`, conversational prompts, picklable reward functions).

| Feature | Config | Purpose |
|---------|--------|---------|
| Async GRPO | `async_prefetch: true` | Generate and score continuously while training |
| Staleness bound | `max_staleness: 4` | Drop rollouts more than N weight syncs old |
| LoRA sync | `vllm_lora_sync: true` | Fast adapter sync via filesystem |
| Zero-adv filter | `skip_zero_advantage_batches: true` | Drop samples with no learning signal before packing |
| Reward workers | `reward_num_workers: 4` | Score rewards in processes, several groups at once |
| Loss | `loss_type: dapo` (default) | TRL's surrogate, clipped against the vLLM sampler at `epsilon` / `epsilon_high` |
| Importance weights | `loss_type: cispo`, `vllm_importance_sampling_cap: 3.0` | Weight each token by its policy/sampler ratio, capped, instead of clipping; faster with LoRA / higher lr |

OPSM (`off_policy_mask_threshold`) applies to synchronous GRPO only. Axolotl's own data producer trainer was removed; `use_data_producer` is a deprecated alias for `async_prefetch` with `max_staleness: 0` (it discards in-flight rollouts at every sync and is slower than the default staleness), and its streaming, replay and re-roll options fail validation (see docs/grpo.qmd#sec-data-producer-migration).

## Health Checks

- `rewards/*/mean` > 0.15 within 20 steps (else: test reward function standalone)
- `reward_std` > 0 on most steps (else: no learning signal)
- `entropy` 0.05-0.5 (< 0.01 = mode collapse)
- `grad_norm` 0.001-1.0 (> 10 = unstable, 0.0 = zero-advantage skip)

See [training_stability.qmd](../training_stability.qmd) for detailed diagnostics.

## File Map

```
src/axolotl/
  cli/train.py                     # Entry point
  cli/vllm_serve.py                # Launches vLLM's native OpenAI server (dev mode, NCCL weight transfer, optional LoRA)
  core/trainers/grpo/
    trainer.py                     # AxolotlGRPOTrainer (sync) and its context-parallel variant
    trl_async_trainer.py           # Async GRPO: TRL's AsyncGRPOTrainer + axolotl extensions
    trl_async_rollout.py           # Async GRPO rollout process: rewards, advantages, reward pool
    trl_async_scoring.py           # Async GRPO: rewards scored on the trainer (EBFT)
    trl_async_traces.py            # Async GRPO: the on_rollouts_scored plugin hook
    sampler.py                     # Sampling utilities
  core/builders/rl.py              # HFRLTrainerBuilder — routes rl type → trainer
  utils/schemas/trl.py             # TRL config schema (all trl: options)

docs/grpo.qmd                     # Full user docs: async, rewards, scaling, config reference
docs/vllm_serving.qmd             # vLLM server modes, LoRA sync, weight sync
```
