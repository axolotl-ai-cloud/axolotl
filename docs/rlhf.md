# RLHF (Beta)

### Overview

Reinforcement Learning from Human Feedback is a method whereby a language model is optimized from data using human
feedback. Various methods include, but not limited to:

- Proximal Policy Optimization (PPO) (not yet supported in axolotl)
- Direct Preference Optimization (DPO)
- Identity Preference Optimization (IPO)


### RLHF using Axolotl

[!IMPORTANT]
This is a BETA feature and many features are not fully implemented. You are encouraged to open new PRs to improve the integration and functionality.

The various RL training methods are implemented in trl and wrapped via axolotl. Below are various examples with how you can use various preference datasets to train models that use ChatML

#### DPO
```yaml
rl: true
datasets:
  - path: Intel/orca_dpo_pairs
    split: train
    type: intel_apply_chatml
  - path: argilla/ultrafeedback-binarized-preferences
    split: train
    type: argilla_apply_chatml
```

#### IPO
```yaml
rl: ipo
```
