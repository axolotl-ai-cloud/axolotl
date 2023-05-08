ARG BASE_TAG=main-base
FROM winglian/axolotl-base:$BASE_TAG

ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"

RUN apt-get update && \
    apt-get install -y vim

WORKDIR /workspace

# The base image ships with `pydantic==1.8.2` which is not working
RUN python3 -m pip install -U --no-cache-dir pydantic

ARG REF=main
RUN git clone https://github.com/winglian/axolotl && cd axolotl && git checkout $REF && \
    pip install -e .[int4]
