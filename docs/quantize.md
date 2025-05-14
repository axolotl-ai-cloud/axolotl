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
base_model: # The path to the model to quantize.
quantization:
  weight_dtype: # Optional[str] = "int8". Fake quantization layout to use for weight quantization. Valid options are uintX for X in [1, 2, 3, 4, 5, 6, 7], or int4, or int8
  activation_dtype: # Optional[str] = "int8". Fake quantization layout to use for activation quantization. Valid options are "int4" and "int8"
  group_size: # Optional[int] = 32. The number of elements in each group for per-group fake quantization
  quantize_embedding: # Optional[bool] = False. Whether to quantize the embedding layer.

output_dir:  # The path to the output directory.
```

Once quantization is complete, your quantized model will be saved in the `output_dir/quantized` directory.

You may also use the `quantize` command to quantize a model which has been trained with [QAT](./qat.md) - you can do this by using the existing QAT configuration file which
you used to train the model:

```bash
# qat.yml
qat:
  activation_dtype: int8
  weight_dtype: int8
  group_size: 256
  quantize_embedding: true

output_dir: # The path to the output directory used during training where the final checkpoint has been saved.
```

```bash
axolotl quantize qat.yml
```

This ensures that an identical quantization configuration is used to quantize the model as was used to train it.
