# Low-rank Linear Conversion via Attention Transfer (LoLCATs)

https://github.com/HazyResearch/lolcats/

### Usage

Install `causal_dot_product` CUDA kernel (check the README in the `csrc` directory):

```bash
cd src/axolotl/integrations/lolcats/linear_llama/csrc

# Edit `setup.py` to point to the correct CUDA capabilities L40-44
# nano setup.py

# Build the CUDA kernel
python setup.py install
```

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

`axolotl train config.yaml --base-model={output_dir}/distilled --trust-remote-code --learning-rate=0.0001 # --wandb-project="..."`

Step 3: Run inference on the finetuned model

`axolotl inference config.yaml --lora-model-dir="{output_dir}" --trust-remote-code # --prompter="AlpacaPrompter"`
