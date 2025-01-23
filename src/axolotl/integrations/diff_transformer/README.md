# Differential Transformer

### Installation

```shell
pip install git+https://github.com/axolotl-ai-cloud/diff-transformer.git
```

Editable:

```shell
git clone git@github.com:axolotl-ai-cloud/diff-transformer.git
cd diff-transformer
pip install -e .
```

### Usage

**Note:** The following with be set in the model config output by the `axolotl convert-diff-transformer` command.

```yaml
plugins:
  - axolotl.integrations.diff_transformer.DifferentialTransformerPlugin

diff_attention: true
```

Additional, optional arguments include:

```yaml
# How often to log diffential attention-related metrics to wandb
diff_attn_log_every: 100

# How many differential attention layers to monitor (strided from 0..k..num_layers)
diff_attn_num_monitor_layers: 3

# How many steps to "warmup" the mixing parameter for the negative component of differential attention
# Follows a linear warmup schedule from 0 to 1; if not specified, the mixing component is set to 1
diff_attn_warmup_steps: 1000
```
