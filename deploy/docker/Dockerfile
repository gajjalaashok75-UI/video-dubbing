# =============================================================================
# Multiva — Docker image
# =============================================================================
# A CUDA base image so the whole pipeline (faster-whisper, NLLB-200, IndicF5,
# Wav2Lip) runs on the GPU instead of the CPU. On a CPU-only machine the same
# image still runs — the engine picks the device itself — it is just slow.
#
# Build from the repository root, so `deploy/docker/...` paths resolve:
#     docker build -f deploy/docker/Dockerfile -t multiva:latest .
# or with compose:
#     docker compose -f deploy/docker/docker-compose.yml up --build
#
# CUDA 12.4 / cuDNN 9 is the pair the cu124 PyTorch wheels below are built
# against, so the runtime libraries and the wheels match.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_ENABLE_MPS_FALLBACK=1

# ffmpeg does every render; libsndfile1 is soundfile's native backend;
# libgl1 + libglib2.0 are what opencv needs even in its headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        ffmpeg \
        libsndfile1 \
        libgl1 \
        libglib2.0-0 \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --upgrade pip setuptools wheel

# --- PyTorch, first and on its own layer ------------------------------------
# Installed before the rest so pip does not drag in the CPU-only wheel from
# PyPI when it resolves the ML stack below.
RUN python3 -m pip install \
        "torch>=2.2.0" "torchaudio>=2.2.0" "torchvision>=0.17.0" \
        --index-url https://download.pytorch.org/whl/cu124

# --- everything else, from the pinned manifest -------------------------------
# requirements.docker.txt is requirements.txt with the CUDA-specific comments
# and without the torch lines already installed above. Copied alone so a source
# edit does not invalidate the dependency layer.
COPY deploy/docker/requirements.docker.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# --- the application ---------------------------------------------------------
COPY . /app

# The interface is a committed build artefact (web/), so no Node build is
# needed here. If web/ were ever missing, this warns instead of failing.
RUN test -f /app/web/index.html \
      && echo "studio interface present" \
      || echo "WARNING: web/ is missing; the API will run but /app/ will 404"

# Runtime tree: uploads, finished renders and the Hugging Face cache all live
# under /data so a mounted volume can persist them across container restarts.
RUN mkdir -p /data/uploads /data/output /data/hf

EXPOSE 8000

# Waits for the health endpoint, then reports; see entrypoint.sh.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

ENTRYPOINT ["/app/deploy/docker/entrypoint.sh"]
CMD ["serve"]
