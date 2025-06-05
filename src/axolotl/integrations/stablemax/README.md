# StableMax Integration

> **Note:** StableMax is incompatible with the CutCrossEntropy integration. Do not enable both plugins at the same time.

> **Note:** StableMax is intended to be used in combination with the orthograd optimizer ([implementation here](https://github.com/cognitivecomputations/dolphinflow-optimizer)) to fully implement the solution described in Prieto et al.

StableMax is a numerically stable alternative to the softmax activation, designed to prevent Softmax Collapse (SC) and enable grokking without regularization, as described in [Grokking at the Edge of Numerical Stability (ICLR 2025)](https://arxiv.org/abs/2501.04697).

## How it works

StableMax replaces the exponential in softmax with a piecewise function:
- For x >= 0: s(x) = x + 1
- For x < 0:  s(x) = 1 / (1 - x)

The StableMax probability is:
```
StableMax(x_i) = s(x_i) / sum_j s(x_j)
```
This prevents floating point absorption errors that can halt learning in grokking tasks.

## Usage

1. **Install the integration** (if not already part of your Axolotl environment).

2. **Enable StableMax in your config:**
```yaml
plugins:
  - axolotl.integrations.stablemax.StableMaxPlugin

stablemax: true
```

3. **What it does:**  
When enabled, this plugin patches `torch.nn.functional.cross_entropy` to use StableMax cross-entropy loss for all classification tasks.

## References

- [Grokking at the Edge of Numerical Stability (Prieto et al., ICLR 2025)](https://arxiv.org/abs/2501.04697)
- [Axolotl Integrations Documentation](../../custom_integrations.qmd)
