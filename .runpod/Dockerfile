FROM runpod/pytorch:3.10-2.0.0-117

COPY .runpod/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt


# Environment settings
ARG BASE_VOLUME="/runpod-volume"
ENV BASE_VOLUME=$BASE_VOLUME
ENV HF_DATASETS_CACHE="${BASE_VOLUME}/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="${BASE_VOLUME}/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="${BASE_VOLUME}/huggingface-cache/hub"

COPY .runpod/src /src

CMD ["python3", "/src/handler.py"]
