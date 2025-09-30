ARG BASE_TAG=main-base-uv
FROM axolotlai/axolotl-base-uv:$BASE_TAG

ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ARG AXOLOTL_EXTRAS=""
ARG AXOLOTL_ARGS=""
ARG CUDA="118"
ARG PYTORCH_VERSION="2.1.2"
ARG GIT_REF="refs/heads/main"
ARG GIT_SHA="HEAD"
ARG VENV_PYTHON="/workspace/axolotl-venv/bin/python"

ENV PYTORCH_VERSION=$PYTORCH_VERSION
ENV GIT_REF=$GIT_REF
ENV GIT_SHA=$GIT_SHA
ENV VENV_PYTHON=$VENV_PYTHON

RUN apt-get update && \
    apt-get install -y --allow-change-held-packages vim curl nano libnccl2 libnccl-dev rsync s3fs && \
    rm -rf /var/cache/apt/archives && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone --depth=1 https://github.com/axolotl-ai-cloud/axolotl.git

WORKDIR /workspace/axolotl

# Ensure we are on the expected commit and break Docker cache between revisions
RUN git fetch origin "$GIT_REF" && git checkout "$GIT_SHA"

# If AXOLOTL_EXTRAS is set, append it in brackets
RUN if [ "$AXOLOTL_EXTRAS" != "" ] ; then \
        uv pip install --python "$VENV_PYTHON" --no-build-isolation -e .[ring-flash-attn,optimizers,ray,$AXOLOTL_EXTRAS] $AXOLOTL_ARGS; \
    else \
        uv pip install --python "$VENV_PYTHON" --no-build-isolation -e .[ring-flash-attn,optimizers,ray] $AXOLOTL_ARGS; \
    fi && \
    uv pip install --python "$VENV_PYTHON" --no-build-isolation flash-attn $AXOLOTL_ARGS && \
    "$VENV_PYTHON" scripts/unsloth_install.py | sh && \
    "$VENV_PYTHON" scripts/cutcrossentropy_install.py | sh && \
    uv pip install --python "$VENV_PYTHON" pytest

# fix so that git fetch/pull from remote works with shallow clone
RUN git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*" && \
    git config --get remote.origin.fetch && \
    git config --global credential.helper store

COPY .axolotl-complete.bash /root/.axolotl-complete.bash
RUN chmod +x /root/.axolotl-complete.bash && \
    echo 'source /root/.axolotl-complete.bash' >> ~/.bashrc
