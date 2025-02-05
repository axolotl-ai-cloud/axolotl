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

Run axolotl: `python -m axolotl.cli.convert_linear_attention config.yaml` TODO: change path CLI

Step 2: Remove the config `linearize: true` and finetune with lora with below possible targets.

```yaml
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# with optional config below but this requires patching axolotl
# to allow this config to work with lora
# unfrozen_parameters: ['.*feature_map_q.mlp.layer.*', '.*feature_map_k.mlp.layer.*', '.*window_factors.*']
```

`axolotl train config.yaml --base-model={output_dir}/distilled --trust-remote-code`
