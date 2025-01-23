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
