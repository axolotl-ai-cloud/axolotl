# Cut Cross Entropy

Cut Cross Entropy (CCE) reduces VRAM usage through optimization on the cross-entropy operation during loss calculation.

See https://github.com/apple/ml-cross-entropy

## Requirements

- PyTorch 2.4.0 or higher

## Installation

Run the following command to install `cut_cross_entropy[transformers]` if you don't have it already.

- If you are in dev environment
```bash
python scripts/cutcrossentropy_install.py | sh
```

- If you are installing from pip
```bash
pip3 uninstall -y cut-cross-entropy && pip3 install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@58d6572"
```

## Usage

```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
```

## Supported Models

- afmoe
- apertus
- arcee
- cohere
- cohere2
- deepseek_v3
- exaone4
- gemma
- gemma2
- gemma3
- gemma3_text
- gemma3n
- gemma3n_text
- glm
- glm4
- glm_moe
- glm4_moe
- glm4_moe_lite
- glm46v
- glm4v
- glm4v_moe
- glm_image
- glm_moe_dsa
- gpt_oss
- granite
- granitemoe
- granitemoehybrid
- granitemoeshared
- hunyuan_v1_dense
- hunyuan_v1_moe
- internvl
- kimi_linear
- lfm2
- lfm2_moe
- lfm2_vl
- llama
- llama4
- llama4_text
- llava
- ministral
- ministral3
- mistral
- mistral3
- mixtral
- mllama
- olmo
- olmo2
- olmo3
- olmoe
- phi
- phi3
- phi4_multimodal
- qwen2
- qwen2_5_vl
- qwen2_moe
- qwen2_vl
- qwen3
- qwen3_5
- qwen3_5_moe
- qwen3_5_moe_vl
- qwen3_5_vl
- qwen3_moe
- qwen3_next
- qwen3_vl
- qwen3_vl_moe
- seed_oss
- smollm3
- step3p5
- voxtral

## Citation

```bib
@article{wijmans2024cut,
  author       = {Erik Wijmans and
                  Brody Huval and
                  Alexander Hertzberg and
                  Vladlen Koltun and
                  Philipp Kr\"ahenb\"uhl},
  title        = {Cut Your Losses in Large-Vocabulary Language Models},
  journal      = {arXiv},
  year         = {2024},
  url          = {https://arxiv.org/abs/2411.09009},
}
```
