# Arctic Long Sequence Training (ALST)

Artic Long Sequence Training (ALST) is a technique for training long context models using a variety of optimization
techniques. It is a combination of:
- TiledMLP: Leverage tiling over the sequence dimension on MLP layers to reduce memory usage
- Tiled Loss: Using optimized loss functions like Liger-Kernel or Cut Cross Entropy to reduce memory usage
- Activation Offloading: Offload activations to CPU RAM to reduce memory usage

For more information, you can check out the ALST paper [here](https://www.arxiv.org/abs/2506.13996).

## Usage

```yaml
tiled_mlp: true

# See Sequence Parallelism docs
# https://docs.axolotl.ai/docs/sequence_parallelism.html
context_parallel_size: int

plugins:
# See Cut Cross Entropy docs
# https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin

# or Liger Kernel docs
# https://docs.axolotl.ai/docs/custom_integrations.html#liger-kernels
  - axolotl.integrations.liger.LigerPlugin
# ...

```
