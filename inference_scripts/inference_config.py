"""
inference_config.py
===================
Shared configuration for all inference scripts.

Hardware target: NVIDIA H100 80 GB SXM5
  - 80 GB HBM3 VRAM
  - BF16 / FP16 tensor-core support
  - Flash Attention 2 compatible

Default hyperparameter ranges (from thesis experiment details):
  - Diffusion steps   : 100 – 250
  - Guidance scale     : 1.0 – 10.0
  - Noise schedule     : linear / cosine / custom
  - Sampler            : DDPM / DDIM / DPM-Solver / Heun 2nd-order
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path defaults  (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "Dataset" / "Inference_Dataset" / "Audio_samples"
METADATA_CSV = REPO_ROOT / "Dataset" / "Inference_Dataset" / "metadata.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "inference_outputs"


# ---------------------------------------------------------------------------
# H100 hardware profile
# ---------------------------------------------------------------------------
@dataclass
class H100Profile:
    """Hardware constants for the NVIDIA H100 80 GB."""
    device: str = "cuda"
    dtype_str: str = "bfloat16"         # native BF16 tensor cores
    vram_gb: int = 80
    max_batch_size_16k: int = 32        # conservative for 16 kHz 10 s clips
    max_batch_size_44k: int = 8         # for 44.1 kHz stereo clips
    enable_tf32: bool = True            # TF32 for FP32 fallback ops
    enable_flash_attn: bool = True      # Flash Attention 2
    cudnn_benchmark: bool = True
    compile_model: bool = False         # torch.compile (set True for PyTorch 2+)
    gradient_checkpointing: bool = False  # not needed for inference


H100 = H100Profile()


# ---------------------------------------------------------------------------
# Per-model default hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class InferenceHyperparams:
    """Hyperparameters for a single inference run."""
    num_inference_steps: int = 200
    guidance_scale: float = 3.5
    scheduler: str = "ddim"             # ddpm | ddim | dpm_solver | heun
    noise_schedule: str = "cosine"      # linear | cosine | custom
    seed: int = 42
    batch_size: int = 1
    num_waveforms_per_prompt: int = 1
    audio_length_s: Optional[float] = None   # None → model default
    negative_prompt: Optional[str] = None


# Model-specific defaults (best-performing from thesis experiments)
MODEL_DEFAULTS = {
    "audioldm": InferenceHyperparams(
        num_inference_steps=200,
        guidance_scale=2.5,
        scheduler="ddim",
        audio_length_s=10.0,
        batch_size=16,
    ),
    "audioldm2": InferenceHyperparams(
        num_inference_steps=200,
        guidance_scale=3.5,
        scheduler="ddim",
        audio_length_s=10.0,
        batch_size=8,
    ),
    "tango": InferenceHyperparams(
        num_inference_steps=200,
        guidance_scale=3.0,
        scheduler="ddim",
        audio_length_s=10.0,
        batch_size=16,
    ),
    "tango2": InferenceHyperparams(
        num_inference_steps=200,
        guidance_scale=3.0,
        scheduler="ddim",
        audio_length_s=10.0,
        batch_size=16,
    ),
    "ezaudio": InferenceHyperparams(
        num_inference_steps=200,
        guidance_scale=3.0,
        scheduler="ddim",
        audio_length_s=10.0,
        batch_size=16,
    ),
    "stable_audio": InferenceHyperparams(
        num_inference_steps=200,
        guidance_scale=3.5,
        scheduler="dpm_solver",
        audio_length_s=None,   # variable — uses per-sample duration
        batch_size=4,
    ),
    "tangoflux": InferenceHyperparams(
        num_inference_steps=100,
        guidance_scale=4.5,
        scheduler="dpm_solver",
        audio_length_s=None,
        batch_size=4,
    ),
    "mmaudio": InferenceHyperparams(
        num_inference_steps=200,
        guidance_scale=4.5,
        scheduler="ddim",
        audio_length_s=8.0,
        batch_size=8,
    ),
    "thinksound": InferenceHyperparams(
        num_inference_steps=200,
        guidance_scale=3.5,
        scheduler="ddim",
        audio_length_s=10.0,
        batch_size=8,
    ),
    "vintage": InferenceHyperparams(
        num_inference_steps=200,
        guidance_scale=3.0,
        scheduler="ddim",
        audio_length_s=10.0,
        batch_size=16,
    ),
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def setup_hardware():
    """
    Configure PyTorch for optimal H100 inference.
    Call this once at the start of every inference script.
    """
    import torch

    # Enable TF32 for float32 matmuls (2x faster on H100 with minimal precision loss)
    if H100.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # cuDNN auto-tuner (finds fastest conv algorithms for fixed input sizes)
    torch.backends.cudnn.benchmark = H100.cudnn_benchmark

    # Verify GPU
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"GPU: {device_name}  |  VRAM: {vram_gb:.1f} GB")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        print("WARNING: No CUDA device found. Running on CPU.")

    return torch.device(H100.device if torch.cuda.is_available() else "cpu")


def get_torch_dtype():
    """Return the torch dtype matching the H100 profile."""
    import torch
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(H100.dtype_str, torch.bfloat16)


def clear_gpu_cache():
    """Free unused GPU memory between runs."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
