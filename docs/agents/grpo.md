# GRPO — Agent Reference

Online RL with verifiable reward functions. For full config reference, async features, and scaling, see [grpo.qmd](../grpo.qmd). For vLLM setup, see [vllm_serving.qmd](../vllm_serving.qmd).

## Architecture

```
Terminal 1 (GPU 0)                    Terminal 2 (GPU 1)
┌──────────────────────┐              ┌──────────────────────────────────┐
│  vLLM Server         │   HTTP       │  Trainer                         │
│  Serves base model   │◄────────────►│  1. Send prompts to vLLM         │
│  + LoRA adapter      │  /generate   │  2. Score completions (rewards)  │
│                      │  /set_lora   │  3. Compute advantages           │
│  Punica kernels for  │              │  4. PPO-clip gradient update     │
│  LoRA inference      │              │  5. Sync LoRA weights to vLLM    │
└──────────────────────┘              └──────────────────────────────────┘
```

## Components Required

1. A YAML config with `rl: grpo`
2. A reward module (Python file with reward functions)
3. A running vLLM server (`axolotl vllm-serve config.yaml`)

## Reward Function Signature

```python
def my_reward(completions, **kwargs) -> list[float]:
    # completions[i][0]["content"] = text of i-th completion
    # **kwargs contains dataset columns not removed by transform
    return [score_for_each_completion]
```

Multiple rewards: `reward_funcs: [r1, r2]` with `reward_weights: [1.0, 0.5]`.

## Key Async Features

| Feature | Config | Purpose |
|---------|--------|---------|
| Async prefetch | `async_prefetch: true` | Overlap generation with training |
| LoRA sync | `vllm_lora_sync: true` | Fast adapter sync via filesystem |
| Streaming scoring | `streaming_partial_batch: true` | Score one group at a time |
| Zero-adv skip | `skip_zero_advantage_batches: true` | Skip batches with no learning signal |
| Replay buffer | `replay_buffer_size: 100` | Cache high-signal groups |
| IS correction | `vllm_importance_sampling_correction: true` | Fix off-policy distribution shift |
| SCOPE-RL | `scope_rl: true` | Hold entropy at a target instead of collapsing |

## SCOPE-RL

Entropy control from [arXiv:2510.08141](https://arxiv.org/abs/2510.08141). Each rollout
resamples a small fraction of prompt groups at `T = clip(1 + H0 - H(pi_old), 0.8, 1.2)`,
keeps only the correct completions, and trains on them with advantage 1 weighted by `alpha`.
Entropy below the target makes the extra samples hotter (pushing entropy back up); above it,
colder. Ignored for multimodal batches.

```yaml
trl:
  use_data_producer: true
  async_prefetch: true            # REQUIRED -- auxiliary rollout runs only on this path
  loss_type: grpo                 # REQUIRED -- TRL defaults to dapo, which is incompatible
  scope_rl: true
  scope_target_entropy: 0.5       # H0
  scope_alpha: 0.015625           # 1/64: share of groups resampled AND loss weight
  scope_positive_threshold: 1.0   # total reward counting as a positive sample
```

Both marked lines are required and rejected if missing. `scope_alpha` is honoured only when a
batch holds at least `1 / scope_alpha` groups; smaller batches resample one whole group and pay
more generation than configured.

Metrics: `scope/temperature` (drifts to 1.2 as entropy collapses), `scope/positive_frac`
(0.0 for many steps means the threshold is too high, and the branch contributes nothing).

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
  cli/vllm_serve.py                # Entry point for vLLM server
  core/trainers/grpo/
    trainer.py                     # AxolotlGRPOTrainer
    async_trainer.py               # AsyncGRPOTrainer — rollout, deferred scoring, loss
    scope.py                       # SCOPE-RL entropy control helpers
    sampler.py                     # Sampling utilities
  core/builders/rl.py              # HFRLTrainerBuilder — routes rl type → trainer
  scripts/vllm_serve_lora.py       # vLLM serve script with LoRA sync support
  utils/schemas/trl.py             # TRL config schema (all trl: options)

docs/grpo.qmd                     # Full user docs: async, rewards, scaling, config reference
docs/vllm_serving.qmd             # vLLM server modes, LoRA sync, weight sync
```
