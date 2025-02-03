# Low-rank Linear Conversion via Attention Transfer (LoLCATs)

https://github.com/HazyResearch/lolcats/

### Usage

TODO: Add instruction to install `causal_dot_product`.

Step 1:

```yaml
plugins:
  - axolotl.integrations.lolcats.LinearizePlugin

linearize: true
```

Step 2: Remove the config above and finetune with lora with below possible targets.

```yaml
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# with optional config below but this requires patching axolotl
# to allow this config to work with lora
# unfrozen_parameters: ['.*feature_map_q.mlp.layer.*', '.*feature_map_k.mlp.layer.*', '.*window_factors.*']
```
