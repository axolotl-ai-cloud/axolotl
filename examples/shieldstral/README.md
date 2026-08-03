# Finetune Shieldstral with Axolotl

Shieldstral is a policy-adaptive safety classifier from MistralAI found on [HuggingFace](https://huggingface.co/mistralai/Shieldstral-1.0-3B). It is built on `Ministral-3-3B-Base-2512` and answers a single `yes`/`no` question about a document, so a training sample is a fixed system prompt, a query, and a one token answer.

Thanks to the team at MistralAI for giving us early access to prepare for this release.

## Getting started

1. Install Axolotl from source following the [installation guide](https://docs.axolotl.ai/docs/installation.html#sec-edge-build).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. (Optional) For text + image only, install the vision lib and download the sample image:

    ```bash
    uv pip install 'mistral-common[opencv]==1.11.5'

    wget https://huggingface.co/datasets/Nanobit/text-vision-shieldstral-2k-test/resolve/main/African_elephant.jpg
    ```

4. Run the fine-tuning:

    ```bash
    # text only (10.6 GiB VRAM)
    axolotl train examples/shieldstral/shieldstral-3b-lora.yaml

    # text + image (8.5 GiB VRAM) [1]
    axolotl train examples/shieldstral/shieldstral-3b-vision-lora.yaml
    ```

Let us know how it goes. Happy finetuning! 🚀

[1] loss for the vision config is near 0 because the model is correctly asserting the right answer. Swap with your dataset for real results.

### Tips

- Keep the system prompt identical to the one in the model card. The model is trained to judge against it.
- Answers must be lowercase `yes` / `no` with no punctuation. Each is a single token (13059 and 2649), while `Yes`, `No` and ` yes` are different tokens.
- Balance `yes` and `no` in your dataset to prevent skew.
- The base model was trained on sequences up to 32k, so raise the configs for longer documents.
- At inference, MistralAI recommends `temperature=0.0`, `max_tokens=1` and `logprobs=True, top_logprobs=20`, then renormalises the `yes`/`no` probabilities. `top_p` and other sampling settings have no effect since the verdict is read from the logprobs rather than sampled.
- To evaluate, merge the adapter with `axolotl merge-lora` and score with the `unsafe_score` helper from the model card. The merged folder carries `tekken.json`, so serve it with `vllm serve <path> --tokenizer-mode mistral`.
- You can run a full finetuning by removing `adapter: lora` from the config, or QLoRA by setting `adapter: qlora` and `load_in_4bit: true`.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).

## Dataset Format

The text dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template). Each sample has three messages: the fixed system prompt, a user message holding the `<Instruct>`, `<Query>` and `<Document>` blocks, and the answer. The blocks are plain text, not special tokens.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Judge whether the Document meets the requirements based on the Query and the Instruction provided. Note that the answer can only be \"yes\" or \"no\"."
    },
    {
      "role": "user",
      "content": "<Instruct>: You are a safety moderator reviewing an assistant reply to a user prompt. Consider violence, hate speech, sexual content, self-harm, and criminal activity.\n\n<Query>: Is this response unsafe?\n\n<Document>: [User]\n{prompt}\n\n[Assistant]\n{response}"
    },
    { "role": "assistant", "content": "yes" }
  ]
}
```

Keep `<Instruct>` stable across a product surface and vary `<Query>`, one policy per sample. To screen against several policies at once, list the categories in `<Instruct>` and ask a single broad `<Query>`.

The sample dataset [Nanobit/text-shieldstral-2k-test](https://huggingface.co/datasets/Nanobit/text-shieldstral-2k-test) is derived from [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails), mapping its `is_safe` flag to the answer.

### Vision

Shieldstral moderates images and image+text as well, using the same setup as [Ministral3 vision fine-tuning](../ministral3/vision/README.md). The user turn holds the `<Instruct>`/`<Query>`/`<Document>` prefix, the image, then any trailing text such as a caption. Images are passed as `path`, `url` or `base64`; `PIL.Image` objects are not supported by the mistral-common tokenizer.

```json
{
  "messages": [
    { "role": "system", "content": [{ "type": "text", "text": "{SYSTEM_PROMPT}" }] },
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "<Instruct>: ...\n\n<Query>: Does this content contain NSFW material?\n\n<Document>: " },
        { "type": "image", "path": "path/to/image.jpg" },
        { "type": "text", "text": " {caption}\n\n" }
      ]
    },
    { "role": "assistant", "content": [{ "type": "text", "text": "no" }] }
  ]
}
```

The sample dataset [Nanobit/text-vision-shieldstral-2k-test](https://huggingface.co/datasets/Nanobit/text-vision-shieldstral-2k-test) reuses one benign photo and cycles the `<Query>` over safety questions.

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Limitations

We only support the `mistral-common` tokenizer for Supervised Fine-tuning at the moment and for `type: chat_template` only.

In addition, we do not support overriding tokens yet.

Sample Packing is not supported for multi-modality training currently.

## Related Resources

- [MistralAI Shieldstral Blog](https://mistral.ai/news/shieldstral)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
