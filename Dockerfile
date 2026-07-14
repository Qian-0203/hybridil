FROM python:3.8-slim-bookworm

ARG INSTALL_MODE=train
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    DEBIAN_FRONTEND=noninteractive \
    FLEXIV_RDK_PATH=/opt/flexiv_rdk/lib_py

WORKDIR /workspace/hybridil

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libusb-1.0-0 \
    libx11-6 \
    libxext6 \
    libxi6 \
    libxrender1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*

COPY train_requirements.txt eval_requirements.txt setup.py ./
COPY docker/constraints-py38.txt docker/constraints-py38.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" \
        -c docker/constraints-py38.txt torch torchvision \
    && if [ "${INSTALL_MODE}" = "train" ]; then \
        python -m pip install -c docker/constraints-py38.txt -r train_requirements.txt; \
    elif [ "${INSTALL_MODE}" = "eval" ]; then \
        python -m pip install -c docker/constraints-py38.txt -r train_requirements.txt \
        && python -m pip install -c docker/constraints-py38.txt -r eval_requirements.txt; \
    else \
        echo "INSTALL_MODE must be 'train' or 'eval'" >&2; exit 1; \
    fi

COPY . .

RUN python -m pip install -e .

CMD ["python", "-c", "import torch, robomimic; print('HybridIL Docker environment ready')"]
