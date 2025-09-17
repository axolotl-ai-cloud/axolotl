# Finetune Swiss-AI's Apertus with Axolotl

[Apertus](https://huggingface.co/collections/swiss-ai/apertus-llm-68b699e65415c231ace3b059) is a family of opensource models trained by Swiss-ai.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html). You need to install from main as Apertus is only on nightly or use our latest [Docker images](https://docs.axolotl.ai/docs/docker.html).

    Here is an example of how to install from main for pip:

```bash
# Ensure you have Pytorch installed (Pytorch 2.6.0 min)
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl

pip3 install packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation -e '.[flash-attn]'

# Install CCE https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy
python scripts/cutcrossentropy_install.py | sh
```

2. (Optional, highly recommended) Install XIELU CUDA

```bash
## Recommended for reduced VRAM and faster speeds

# Point to CUDA toolkit directory
# For those using our Docker image, use the below path.
export CUDA_HOME=/usr/local/cuda

pip3 install git+https://github.com/nickjbrowning/XIELU@59d6031 --no-build-isolation --no-deps
```

For any installation errors, see [XIELU Installation Issues](#xielu-installation-issues)

3. Run the finetuning example:

```bash
axolotl train examples/apertus/apertus-8b-qlora.yaml
```

This config uses about 8.7 GiB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### Tips

- For inference, the official Apertus team recommends `top_p=0.9` and `temperature=0.8`.
- You can instead use full paremter fine-tuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

### XIELU Installation Issues

#### `ModuleNotFoundError: No module named 'torch'`

Please check these one by one:
- Running in correct environment
- Env has PyTorch installed
- CUDA toolkit is at `CUDA_HOME`

If those didn't help, please try the below solutions:

1. Pass env for CMAKE and try install again:

    ```bash
    Python_EXECUTABLE=$(which python) pip3 install git+https://github.com/nickjbrowning/XIELU@59d6031 --no-build-isolation --no-deps
    ```

2. Git clone the repo and manually hardcode python path:

    ```bash
    git clone https://github.com/nickjbrowning/XIELU
    cd xielu
    git checkout 59d6031

    cd xielu
    nano CMakeLists.txt  # or vi depending on your preference
    ```

    ```diff
    execute_process(
    -    COMMAND ${Python_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    +    COMMAND /root/miniconda3/envs/py3.11/bin/python -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
        RESULT_VARIABLE TORCH_CMAKE_PATH_RESULT
        OUTPUT_VARIABLE TORCH_CMAKE_PATH_OUTPUT
        ERROR_VARIABLE TORCH_CMAKE_PATH_ERROR
    )
    ```

    ```bash
    pip3 install . --no-build-isolation --no-deps
    ```

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [LoRA Optimizations](https://docs.axolotl.ai/docs/lora_optims.html)

## Related Resources

- [Apertus Tech Report](https://github.com/swiss-ai/apertus-tech-report/blob/main/Apertus_Tech_Report.pdf)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
