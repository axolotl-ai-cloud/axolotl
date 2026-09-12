# SCOPE-RL Entropy Control

Entropy control for async GRPO from [SCOPE-RL](https://arxiv.org/abs/2510.08141).

GRPO entropy usually falls monotonically until the policy stops exploring. SCOPE-RL holds it at a target instead: each rollout resamples a small fraction of the prompt groups at an adjusted temperature, keeps only the completions that were rewarded, and adds them to the loss as a positive-advantage term weighted by `scope_alpha`.

## Usage

```yaml
plugins:
  - axolotl.integrations.scope_rl.ScopeRLPlugin

rl: grpo
trl:
  use_vllm: true
  use_data_producer: true
  async_prefetch: true            # required -- the auxiliary rollout runs only here
  loss_type: grpo                 # required -- TRL defaults to dapo, which is incompatible

scope_rl: true
scope_target_entropy: 0.5         # H0 -- the entropy level to hold
scope_alpha: 0.015625             # 1/64: share of groups resampled, and the weight of the term
scope_positive_threshold: 1.0     # total reward at which a sample counts as positive
```

The plugin routes training through its own async GRPO trainer, so `trainer_cls` must be left unset.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `scope_rl` | `false` | Enable SCOPE-RL |
| `scope_target_entropy` | `0.5` | Target policy entropy `H0` |
| `scope_alpha` | `1/64` | Fraction of groups resampled and the auxiliary loss weight |
| `scope_temperature_min` | `0.8` | Lower clip for the auxiliary sampling temperature |
| `scope_temperature_max` | `1.2` | Upper clip for the auxiliary sampling temperature |
| `scope_positive_threshold` | `1.0` | Total reward at or above which an auxiliary sample is positive |

## Behaviour

The auxiliary temperature is `clip(1 + H0 - H, scope_temperature_min, scope_temperature_max)` times the sampling temperature, where `H` is the entropy measured on the previous step: below target it samples hotter, above target it samples colder. Only positive samples are kept; the paper's ablations show that mixing in negatives re-collapses entropy.

Cost is roughly `scope_alpha` extra generation, but only once a rollout batch holds at least `1 / scope_alpha` groups. Whole groups are resampled and never fewer than one, so 4 groups per batch costs 25% whatever `scope_alpha` says. Only generation cost is affected; the loss weight stays at `scope_alpha`.

Requirements: `rl: grpo`, vLLM generation, `trl.async_prefetch: true`, and `trl.loss_type` of `grpo`, `sapo` or `dr_grpo`. Not supported with `streaming_partial_batch`. Multimodal batches skip the auxiliary branch.

Metrics: `scope/temperature` (drifts to `scope_temperature_max` as entropy collapses) and `scope/positive_frac` (pinned at 0 means `scope_positive_threshold` is above what the reward functions emit, and the branch contributes nothing).

## Citation

```bib
@article{scope-rl-2025,
    title={SCOPE-RL: Entropy Control via Temperature-Adaptive Positive Samples},
    journal={arXiv preprint arXiv:2510.08141},
    year={2025}
}
```
