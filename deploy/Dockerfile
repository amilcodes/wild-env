# CUDA/PyTorch version is intentionally a build argument: cluster driver policy
# determines the correct wheel index. This image contains no incident data.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.13.0
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python3-pip git tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
RUN python${PYTHON_VERSION} -m pip install --upgrade pip && \
    python${PYTHON_VERSION} -m pip install --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION} && \
    python${PYTHON_VERSION} -m pip install --no-deps . && \
    python${PYTHON_VERSION} -m pip install numpy PyYAML gymnasium pettingzoo

COPY configs ./configs
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["aeolus-train", "--config", "configs/cluster_mappo.yaml"]
