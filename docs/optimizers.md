# Optimizers

Optimizers are an important component when training LLMs. Optimizers are responsible for updating the model's weights (parameters) based on the gradients computed during backpropagation.
The goal of an optimizer is to minimize the loss function.

### Adam/AdamW Optimizers

```yaml
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1e-8
weight_decay: 0.0
```

### GaLore Optimizer

https://huggingface.co/papers/2403.03507

```yaml
optimizer: galore_adamw | galore_adamw_8bit | galore_adafactor
optim_args:
  rank: 128
  update_proj_gap: 200
  scale: 0.25
  proj_type: std
optim_target_modules:
  - mlp
  - attn
```
