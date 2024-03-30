ARG BASE_TAG=main-base
FROM winglian/axolotl-base:$BASE_TAG

ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ARG AXOLOTL_EXTRAS=""
ARG AXOLOTL_ARGS=""
ARG CUDA="118"
ENV BNB_CUDA_VERSION=$CUDA
ARG PYTORCH_VERSION="2.1.2"

ENV PYTORCH_VERSION=$PYTORCH_VERSION

RUN apt-get update && \
    apt-get install -y --allow-change-held-packages vim curl nano libnccl2 libnccl-dev apt-transport-https ca-certificates gnupg

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && apt-get install google-cloud-sdk -y

WORKDIR /workspace

RUN git clone --depth=1 https://github.com/Y-IAB/axolotl.git

WORKDIR /workspace/axolotl

# If AXOLOTL_EXTRAS is set, append it in brackets
RUN pip install causal_conv1d
RUN if [ "$AXOLOTL_EXTRAS" != "" ] ; then \
        pip install -e .[deepspeed,flash-attn,mamba-ssm,galore,$AXOLOTL_EXTRAS] $AXOLOTL_ARGS; \
    else \
        pip install -e .[deepspeed,flash-attn,mamba-ssm,galore] $AXOLOTL_ARGS; \
    fi

# So we can test the Docker image
RUN pip install pytest

# fix so that git fetch/pull from remote works
RUN git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*" && \
    git config --get remote.origin.fetch

# helper for huggingface-login cli
RUN git config --global credential.helper store
