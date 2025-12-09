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
pip3 uninstall -y cut-cross-entropy && pip3 install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@f643b88"
```

## Usage

```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
```

## Supported Models

- apertus
- arcee
- cohere
- cohere2
- deepseek_v3
- gemma
- gemma2
- gemma3
- gemma3_text
- gemma3n
- gemma3n_text
- glm
- glm4
- glm4_moe
- glm4v
- glm4v_moe
- gpt_oss
- granite
- granitemoe
- granitemoeshared
- granitemoehybrid
- hunyuan_v1_dense
- hunyuan_v1_moe
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
- phi
- phi3
- phi4_multimodal
- qwen2
- qwen2_vl
- qwen2_moe
- qwen2_5_vl
- qwen3
- qwen3_moe
- qwen3_vl
- qwen3_vl_moe
- qwen3_next
- smollm3
- seed_oss
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
