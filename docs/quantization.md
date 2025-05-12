---
title: "Quantization with torchao"
back-to-top-navigation: true
toc: true
toc-expand: 2
toc-depth: 4
---

Quantization is a technique to lower the memory footprint of your model, potentially at the cost of accuracy or model performance. We support quantizing your model using the [torchao](https://github.com/pytorch/ao) library. Quantization is supported for both post-training quantization (PTQ) and quantization-aware training (QAT).


## Configuring Quantization in Axolotl

Quantization is configured using the `quantization` key in your configuration file.

```yaml
quantization:
  weight_dtype: # Optional[str] = "int8". Fake quantization layout to use for weight quantization. Valid options are uintX or intX for X in [1, 2, 3, 4, 5, 6, 7, 8]
  activation_dtype: # Optional[str] = "int8". Fake quantization layout to use for activation quantization. Valid options are "int4" and "int8"
  group_size: # Optional[int] = 32. The number of elements in each group for per-group fake quantization
  
```
