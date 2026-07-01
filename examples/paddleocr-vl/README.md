# Finetune PaddleOCR-VL with Axolotl

[PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) is a compact document parsing vision-language model from PaddlePaddle for OCR, table, formula, chart, seal, and spotting tasks.

This guide shows how to fine-tune PaddleOCR-VL with Axolotl's multimodal SFT path.

## Getting Started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Run one of the fine-tuning examples:

    ```bash
    axolotl train examples/paddleocr-vl/paddleocr-vl-1_6-qlora.yaml
    ```

    ```bash
    axolotl train examples/paddleocr-vl/paddleocr-vl-1_6-full-finetune.yaml
    ```

    To use a local checkout of the model, replace `base_model` with your local path:

    ```yaml
    base_model: /path/to/local/PaddleOCR-VL-1.6
    ```

## Tips

- The model uses its bundled chat template through `processor_type: AutoProcessor`; no explicit `chat_template` is needed.
- Do not set `trust_remote_code` for this example; Axolotl's pinned Transformers version includes the PaddleOCR-VL model and processor implementation.
- Keep `sample_packing: false`; PaddleOCR-VL uses 3D multimodal RoPE positions that are not compatible with Axolotl's packed 2D position IDs.
- `attn_implementation: flash_attention_2` works with PaddleOCR-VL; use `sdpa` as a fallback if your environment does not have Flash Attention 2 kernels available.
- Do not enable Liger or Cut Cross Entropy; neither path currently patches PaddleOCR-VL's multimodal `ForConditionalGeneration` class.
- PaddleOCR-VL task prompts include `OCR:`, `Table Recognition:`, `Formula Recognition:`, `Chart Recognition:`, `Seal Recognition:`, and `Spotting:`.
- Dataset rows should use Axolotl's multimodal `messages` format with image content in the user turn and the parsed text or markup in the assistant turn.
- The QLoRA example targets the language decoder, vision encoder, and multimodal projector with LoRA adapters.

## Related Resources

- [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- [PaddleOCR documentation](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [Axolotl multimodal docs](https://docs.axolotl.ai/docs/multimodal.html)
