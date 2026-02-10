# ===========================================================================
#  Text-to-Audio Inference & Evaluation Pipeline
#  Target: NVIDIA H100 80 GB SXM5  |  CUDA 12.1  |  Python 3.10
# ===========================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ── Prevent interactive prompts during apt installs ──────────────────────
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        python3.10-dev \
        git \
        ffmpeg \
        libsndfile1 \
        libsox-dev \
        sox \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# ── Upgrade pip ──────────────────────────────────────────────────────────
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ── PyTorch (CUDA 12.1 wheels) ──────────────────────────────────────────
# Installed first so Docker caches this heavy layer separately
RUN pip install --no-cache-dir \
        torch>=2.1.0 \
        torchaudio>=2.1.0 \
        --index-url https://download.pytorch.org/whl/cu121

# ── xformers (memory-efficient attention, H100 optimised) ───────────────
RUN pip install --no-cache-dir xformers>=0.0.23

# ── Working directory ────────────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements and install Python dependencies ────────────────────
# (torch/torchaudio/xformers already installed above, pip will skip them)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Model-specific packages (uncomment as needed) ───────────────────────
# RUN pip install --no-cache-dir mmaudio         # MMAudio
# RUN pip install --no-cache-dir tangoflux       # TangoFlux
# RUN pip install --no-cache-dir tango           # Tango

# ── Copy project source ─────────────────────────────────────────────────
COPY . .

# ── H100 runtime environment variables ──────────────────────────────────
# Enable TF32 for FP32 operations (2x throughput on H100)
ENV NVIDIA_TF32_OVERRIDE=1
# Ensure CUDA is visible
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# HuggingFace cache inside the container
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

# ── Create cache directories ────────────────────────────────────────────
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch

# ── Default entrypoint ──────────────────────────────────────────────────
# Run all inference by default; override with docker run ... python <script>
ENTRYPOINT ["python"]
CMD ["inference_scripts/run_all_inference.py"]
