FROM huggingface/transformers-pytorch-deepspeed-latest-gpu:latest

RUN export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
RUN apt-get update && \
    apt-get install -y build-essential ninja-build vim git-lfs && \
    git lfs install --skip-repo && \
    mkdir /tmp/wheels && \
    cd /tmp/wheels && \
    curl -L -O https://github.com/winglian/axolotl/raw/wheels/wheels/deepspeed-0.9.2%2B7ddc3b01-cp38-cp38-linux_x86_64.whl && \
    curl -L -O https://github.com/winglian/axolotl/raw/wheels/wheels/flash_attn-1.0.4-cp38-cp38-linux_x86_64.whl && \
    pip install deepspeed-0.9.2%2B7ddc3b01-cp38-cp38-linux_x86_64.whl && \
    pip install flash_attn-1.0.4-cp38-cp38-linux_x86_64.whl && \
    pip install wheel && \
    pip install "peft @ git+https://github.com/huggingface/peft.git@main" --force-reinstall --no-dependencies

WORKDIR /workspace
ARG REF=main
RUN git clone https://github.com/winglian/axolotl && cd axolotl && git checkout $REF && \
    pip install -e .[int4]

RUN pip3 install --force-reinstall https://download.pytorch.org/whl/nightly/cu117/torch-2.0.0.dev20230301%2Bcu117-cp38-cp38-linux_x86_64.whl --index-url https://download.pytorch.org/whl/nightly/cu117

