# Liquid Foundation Models 2

Liquid Foundation Models are a family of small models focused on quality, speed and memory efficiency.

For more information, see [Liquid Foundation Models 2](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models).

We include two examples in Axolotl:

- LFM2 Text Generation
  - [YAML](./lfm2-350m-fft.yaml)
-LFM2-VL Image-Text-to-Text
  - [YAML](./lfm2-vl.yaml)

> [!TIP]
> If you get the following error: `undefined symbol: _ZN3c104cuda9SetDeviceEab`, try uninstalling causal-conv1d.
> causal-conv1d is installed by default in most Axolotl Docker images.
>
> ```bash
> pip uninstall -y causal-conv1d
> ```
