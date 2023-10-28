# LLaVA

### Installing dependencies

```shell
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install --no-deps -e .
```

### Downloading assets

LLaVA doesn't support remote datasets, so both the JSON and image assets need to be downloaded locally

```shell
mkdir llava
mkdir data
cd llava
curl -L -O https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip
unzip images.zip

cd ../data
curl -L -O https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json
```

### Pretraining

Pretraining aligns the vision model with the language model.

```shell
accelerate launch -m axolotl.cli.train_mm examples/multimodal/pretrain-llava-llama.yml
```

### Finetuning

TBD
