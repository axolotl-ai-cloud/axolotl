FROM axolotlai/axolotl-cloud:main-py3.11-cu124-2.6.0

COPY .runpod/requirements.txt /requirements.txt
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv pip install --system -r /requirements.txt

# Environment settings
ARG BASE_VOLUME="/runpod-volume"
ENV BASE_VOLUME=$BASE_VOLUME
ENV HF_DATASETS_CACHE="${BASE_VOLUME}/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="${BASE_VOLUME}/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="${BASE_VOLUME}/huggingface-cache/hub"

COPY .runpod/src /src

WORKDIR /src
CMD ["python3", "/src/handler.py"]
