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
pip3 uninstall -y cut-cross-entropy && pip3 install "cut-cross-entropy[transformers] @ git+https://github.com/apple/ml-cross-entropy.git@bad6f7b49c75fdec69471abb71b4cddd0f0c6438"
```

## Usage

**NOTE**: If you are training a VLM model, please use older version of Axolotl as upstream has applied a major VLM refactor, and our patches have not been updated yet.

```bash
git checkout 787880215b3ab32ccaf81c1b2e9588c6f3e6e764

pip3 install --no-build-isolation -e .
```

```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
```

## Supported Models

- llama
- llama4
- llama4_text
- mllama
- phi3
- gemma
- gemma2
- gemma3
- gemma3_text
- mistral
- mistral3
- qwen2
- qwen2_moe
- qwen2_vl
- qwen2_5_vl
- qwen3
- qwen3_moe
- cohere
- cohere2
- glm
- glm4

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
