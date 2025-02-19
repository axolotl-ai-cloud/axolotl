# Cut Cross Entropy

Cut Cross Entropy reduces VRAM usage through optimization on the cross-entropy operation during loss calculation.

See https://github.com/apple/ml-cross-entropy

## Usage

```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin

cut_cross_entropy: true
```

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
