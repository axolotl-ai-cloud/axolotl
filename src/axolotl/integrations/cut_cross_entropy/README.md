# Cut Cross Entropy

Cut Cross Entropy (CCE) reduces VRAM usage through optimization on the cross-entropy operation during loss calculation.

See https://github.com/apple/ml-cross-entropy

## Requirements

- PyTorch 2.4.0 or higher

## Installation

Run the following command to install `cut_cross_entropy[transformers]` if you don't have it already.

```bash
# if you are in dev environment
python scripts/cutcrossentropy_install.py | sh

# if you are not in dev environment
pip3 uninstall -y cut-cross-entropy && pip3 install "cut-cross-entropy[transformers] @ git+https://github.com/apple/ml-cross-entropy.git@24fbe4b5dab9a6c250a014573613c1890190536c"
```

## Usage

```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin

cut_cross_entropy: true
```

## Supported Models

- llama
- llama4_text
- llama4
- mllama
- phi3
- gemma
- gemma2
- gemma3
- gemma3_text
- mistral
- mistral3
- qwen2
- cohere
- cohere2

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
