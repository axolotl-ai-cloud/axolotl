ARG BASE_TAG=main-base
FROM winglian/axolotl-base:$BASE_TAG

ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ARG AXOLOTL_EXTRAS=""

RUN apt-get update && \
    apt-get install -y vim curl

WORKDIR /workspace

RUN pip3 install --force-reinstall "peft @ git+https://github.com/huggingface/peft.git@main" \
            "accelerate @ git+https://github.com/huggingface/accelerate.git@main" \
            "transformers @ git+https://github.com/huggingface/transformers.git@main"

RUN git clone https://github.com/OpenAccess-AI-Collective/axolotl.git
# If AXOLOTL_EXTRAS is set, append it in brackets
RUN cd axolotl && \
    if [ "$AXOLOTL_EXTRAS" != "" ] ; then \
        pip install -e .[$AXOLOTL_EXTRAS]; \
    else \
        pip install -e .; \
    fi

# helper for huggingface-login cli
RUN git config --global credential.helper store
