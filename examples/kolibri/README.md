# Finetune Aleph Alpha's Kolibri-1 with Axolotl

[Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16) is Aleph Alpha's German and English mixture-of-experts reasoning model. This guide shows how to fine-tune it with Axolotl.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Run the finetuning example:

    ```bash
    axolotl train examples/kolibri/kolibri-1-qlora.yaml
    ```

Let us know how it goes. Happy finetuning! 🚀

### Routed and Shared Experts (MoE)

Routed experts are fused 3D parameters, so target them with `lora_target_parameters`:

```yaml
lora_target_parameters:
  - mlp.experts.gate_up_proj
  - mlp.experts.down_proj
```

The shared expert uses `nn.Linear`, so uncomment `gate_proj`, `up_proj` and `down_proj` in `lora_target_modules` to train it.

### TIPS

- Fine-tune `Aleph-Alpha/Kolibri-1-BF16`; `Aleph-Alpha/Kolibri-1` is an FP8 serving checkpoint.
- transformers has no Kolibri-1 implementation, so Axolotl ships its own and `trust_remote_code` is not needed. Axolotl detects the model from `model_type: kolibri1` in the checkpoint's `config.json`, on the Hub or locally.
- Cut Cross Entropy is not supported for this model.
- Kolibri-1 balances its experts with a routing bias, so the MoE load-balancing auxiliary loss (`output_router_logits`, `router_aux_loss_coef`) is not supported.
- For inference hyperparameters and reasoning modes, please see the model card.
- Read more on loading your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Optimizations Guide](https://docs.axolotl.ai/docs/optimizations.html)

## Related Resources

- [Kolibri-1 Blog](https://aleph-alpha.com/en/blog/kolibri-has-landed-a-sovereign-open-weight-model/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
