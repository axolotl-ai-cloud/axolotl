## Spectrum: Targeted Training on Signal to Noise Ratio

by Eric Hartford, Lucas Atkins, Fernando Fernandes, David Golchinfar

This plugin contains code to freeze the bottom fraction of modules in a model, based on the Signal-to-Noise Ratio (SNR).

### Overview

Spectrum is a tool for scanning and evaluating the Signal-to-Noise Ratio (SNR) of layers in large language models.
By identifying the top n% of layers with the highest SNR, you can optimize training efficiency.

### Usage

```yaml
plugins:
  - axolotl.integrations.spectrum.SpectrumPlugin

spectrum_top_fraction: 0.5
# Optional if using a pre-scanned model as your base_model. Useful if using a model mirror
spectrum_model_name: meta-llama/Meta-Llama-3.1-8B
```
