# Causal linear attention CUDA kernel

Usage:
```bash
cd src/axolotl/integrations/lolcats/linear_llama/csrc

# Edit `setup.py` to point to the correct CUDA capabilities L40-44
# nano setup.py

# Build the CUDA kernel
python setup.py install
```

Reference: https://github.com/idiap/fast-transformers/

```bib
@inproceedings{katharopoulos_et_al_2020,
    author = {Katharopoulos, A. and Vyas, A. and Pappas, N. and Fleuret, F.},
    title = {Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention},
    booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
    year = {2020}
}

@article{vyas_et_al_2020,
    author={Vyas, A. and Katharopoulos, A. and Fleuret, F.},
    title={Fast Transformers with Clustered Attention},
    booktitle = {Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS)},
    year={2020}
}
```
