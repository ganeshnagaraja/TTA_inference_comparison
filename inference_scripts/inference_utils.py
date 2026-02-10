"""
inference_utils.py
==================
Shared utilities for running inference across all models.

Provides:
  - Prompt loading from metadata CSV
  - Output directory management
  - Waveform post-processing (normalisation, 16-bit PCM export)
  - Batch iteration helpers
  - Inference timing and logging
  - Result manifest generation
"""

import os
import csv
import json
import time
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Metadata / prompt loading
# ---------------------------------------------------------------------------
def load_prompts(csv_path: str) -> List[Dict]:
    """
    Load inference prompts from the dataset metadata CSV.

    Returns list of dicts:
        sample_number, filename, prompt, duration, num_layers
    """
    records = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append({
                "sample_number": int(row["Sample Number"]),
                "filename": row["Filename"].strip(),
                "prompt": row["Prompt"].strip(),
                "duration": float(row["Duration (s)"]),
                "num_layers": int(row["Num Layers"]),
            })
    return records


def batch_prompts(records: List[Dict], batch_size: int) -> List[List[Dict]]:
    """Split records into batches of batch_size."""
    return [
        records[i:i + batch_size]
        for i in range(0, len(records), batch_size)
    ]


# ---------------------------------------------------------------------------
# Output management
# ---------------------------------------------------------------------------
def setup_output_dir(
    model_name: str,
    output_root: str,
    run_tag: Optional[str] = None,
) -> Path:
    """
    Create output directory for a model's inference results.

    Structure:
        {output_root}/{model_name}/{run_tag}/
            audio/          — generated WAV files
            manifest.json   — metadata + hyperparams
    """
    if run_tag is None:
        run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root) / model_name / run_tag
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Waveform post-processing
# ---------------------------------------------------------------------------
def postprocess_waveform(
    waveform,
    target_sr: Optional[int] = None,
    current_sr: Optional[int] = None,
) -> np.ndarray:
    """
    Post-process generated waveform:
      1. Convert to numpy if tensor
      2. Normalise to [-1, 1]
      3. Resample if needed

    Returns np.ndarray of shape (channels, T).
    """
    # Convert from torch tensor if needed
    if TORCH_AVAILABLE and isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().float().numpy()

    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]    # (1, T)
    elif waveform.ndim == 3:
        waveform = waveform.squeeze(0)        # remove batch dim

    # Normalise
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak

    return waveform


def save_wav_16bit(
    filepath: str,
    waveform: np.ndarray,
    sample_rate: int,
) -> None:
    """Save waveform as 16-bit PCM WAV."""
    filepath = str(filepath)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]

    # Ensure [-1, 1]
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak

    if TORCH_AVAILABLE:
        torchaudio.save(filepath, torch.from_numpy(waveform).float(),
                        sample_rate, bits_per_sample=16)
    elif SOUNDFILE_AVAILABLE:
        sf.write(filepath, waveform.T, sample_rate, subtype="PCM_16")
    else:
        raise RuntimeError("Need torchaudio or soundfile for WAV export.")


# ---------------------------------------------------------------------------
# Timing / logging
# ---------------------------------------------------------------------------
class InferenceTimer:
    """Context manager for timing inference batches."""

    def __init__(self, model_name: str, batch_idx: int, batch_size: int):
        self.model_name = model_name
        self.batch_idx = batch_idx
        self.batch_size = batch_size
        self.start = None
        self.elapsed = None

    def __enter__(self):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start

    def log(self):
        per_sample = self.elapsed / self.batch_size if self.batch_size > 0 else 0
        print(f"  [{self.model_name}] Batch {self.batch_idx}: "
              f"{self.elapsed:.2f}s total, {per_sample:.2f}s/sample")


def log_gpu_memory(tag: str = ""):
    """Print current GPU memory usage."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"  GPU Memory [{tag}]: {alloc:.2f} GB allocated, "
              f"{reserved:.2f} GB reserved")


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------
def save_manifest(
    output_dir: Path,
    model_name: str,
    hyperparams: Dict,
    results: List[Dict],
    total_time_s: float,
    reference_audio_dir: str = "",
) -> None:
    """
    Save a JSON manifest documenting the inference run.

    Includes: model name, hyperparameters, per-sample results, timing,
    and reference_audio_dir so the evaluation script can pair generated
    audio against ground-truth reference files.
    """
    manifest = {
        "model": model_name,
        "hardware": "NVIDIA H100 80GB SXM5",
        "precision": "bfloat16",
        "reference_audio_dir": str(reference_audio_dir),
        "hyperparameters": hyperparams,
        "total_samples": len(results),
        "total_time_s": round(total_time_s, 2),
        "avg_time_per_sample_s": round(total_time_s / max(len(results), 1), 2),
        "results": results,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")
