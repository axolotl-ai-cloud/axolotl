# LLMCompressor Integration

Fine-tune sparsified models in Axolotl using Neural Magic's [LLMCompressor](https://github.com/vllm-project/llm-compressor).

This integration enables fine-tuning of models sparsified using LLMCompressor within the Axolotl training framework. By combining LLMCompressor's model compression capabilities with Axolotl's distributed training pipelines, users can efficiently fine-tune sparse models at scale.

It uses Axolotl’s plugin system to hook into the fine-tuning flows while maintaining sparsity throughout training.

---

## Requirements

- Axolotl with `llmcompressor` extras:

  ```bash
  pip install "axolotl[llmcompressor]"
  ```

- Requires `llmcompressor >= 0.5.1`

This will install all necessary dependencies to fine-tune sparsified models using the integration.

---

## Usage

To enable sparse fine-tuning with this integration, include the plugin in your Axolotl config:

```yaml
plugins:
  - axolotl.integrations.llm_compressor.LLMCompressorPlugin

llmcompressor:
  recipe:
    finetuning_stage:
      finetuning_modifiers:
        ConstantPruningModifier:
          targets: [
            're:.*q_proj.weight',
            're:.*k_proj.weight',
            're:.*v_proj.weight',
            're:.*o_proj.weight',
            're:.*gate_proj.weight',
            're:.*up_proj.weight',
            're:.*down_proj.weight',
          ]
          start: 0
# ... (other training arguments)
```

This plugin **does not apply pruning or sparsification itself** — it is intended for **fine-tuning models that have already been sparsified**.

Pre-sparsified checkpoints can be:
- Generated using [LLMCompressor](https://github.com/vllm-project/llm-compressor)
- Or downloaded from [Neural Magic's Hugging Face page](https://huggingface.co/neuralmagic)

To learn more about writing and customizing LLMCompressor recipes, refer to the official documentation:
[https://github.com/vllm-project/llm-compressor/blob/main/README.md](https://github.com/vllm-project/llm-compressor/blob/main/README.md)

### Example Config

See [`examples/llama-3/sparse-finetuning.yaml`](examples/llama-3/sparse-finetuning.yaml) for a complete example.

---

## Learn More

For details on available sparsity and quantization schemes, fine-tuning recipes, and usage examples, visit the official LLMCompressor repository:

👉 [https://github.com/vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)
