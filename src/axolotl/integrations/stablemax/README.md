# StableMax Integration

> **⚠️ WARNING:** StableMax performs **global patching** of `torch.nn.functional.cross_entropy`, replacing it with `stablemax_cross_entropy` for ALL subsequent calls throughout the entire application. This affects not only your model training but also any other libraries, models, or code that use `torch.nn.functional.cross_entropy`.

> **⚠️ COMPATIBILITY:** Do not enable StableMax simultaneously with other cross-entropy patches such as **Liger** (`liger_cross_entropy`, `liger_fused_linear_cross_entropy`) or **CutCrossEntropy** (`cut_cross_entropy`). The system will detect and prevent such conflicts, but enabling multiple patches can lead to unpredictable runtime behavior.

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
