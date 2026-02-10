#!/usr/bin/env python3
"""
evaluate.py
===========
Compute evaluation metrics comparing generated audio against reference audio
from the custom 215-sample dataset.

Metrics implemented (from thesis Section: Evaluation Metrics):
  1. FAD  — Fréchet Audio Distance (VGGish embeddings)           Lower ↓
  2. FD   — Fréchet Distance (PANNs embeddings)                  Lower ↓
  3. CLAP — CLAP Score (text–audio semantic alignment)           Higher ↑
  4. KL   — KL Divergence (per-sample class distributions)       Lower ↓
  5. IS   — Inception Score (quality × diversity)                Higher ↑
  6. Inference Time (from manifest.json)                          Lower ↓

All metrics are computed globally and per acoustic category.

Pipeline:
  reference audio (215 WAVs) ←→ generated audio (model output)
                                    + text prompts (for CLAP)

Usage:
    python evaluate.py --generated ./inference_outputs/audioldm/run_tag/audio \\
                       --reference ./Dataset/Inference_Dataset/Audio_samples \\
                       --csv       ./Dataset/Inference_Dataset/metadata.csv

    python evaluate.py --manifest ./inference_outputs/audioldm/run_tag/manifest.json

Hardware: NVIDIA H100 80 GB — BF16 for embedding extraction
"""

import os
import sys
import json
import time
import argparse
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
    warnings.warn("PyTorch not found. GPU-accelerated evaluation unavailable.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_REFERENCE_DIR = REPO_ROOT / "Dataset" / "Inference_Dataset" / "Audio_samples"
DEFAULT_CSV = REPO_ROOT / "Dataset" / "Inference_Dataset" / "metadata.csv"


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio(filepath: str, target_sr: int = 16_000, mono: bool = True) -> np.ndarray:
    """Load audio, resample to target_sr, return as 1-D numpy array."""
    filepath = str(filepath)
    if TORCH_AVAILABLE:
        wav, sr = torchaudio.load(filepath)
        if mono and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        return wav.squeeze(0).numpy()
    elif SOUNDFILE_AVAILABLE:
        data, sr = sf.read(filepath, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr and LIBROSA_AVAILABLE:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        return data
    else:
        raise RuntimeError("Install torchaudio or soundfile.")


def load_prompt_map(csv_path: str) -> Dict[str, Dict]:
    """Load metadata CSV → dict keyed by filename."""
    import csv
    records = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records[row["Filename"].strip()] = {
                "sample_number": int(row["Sample Number"]),
                "prompt": row["Prompt"].strip(),
                "duration": float(row["Duration (s)"]),
                "num_layers": int(row["Num Layers"]),
            }
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC 1: FAD — Fréchet Audio Distance (VGGish)
# ═══════════════════════════════════════════════════════════════════════════
def extract_vggish_embeddings(
    audio_files: List[str],
    sr: int = 16_000,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract VGGish embeddings from a list of audio files.

    Uses torchvggish (a PyTorch port of the TF VGGish model).
    Returns array of shape (N, 128).
    """
    try:
        hub_model = torch.hub.load("harritaylor/torchvggish", "vggish")
        hub_model.eval().to(device)
    except Exception:
        # Fallback: use torchaudio's pipeline-based approach
        print("  VGGish: using torchaudio pipeline fallback")
        bundle = torchaudio.pipelines.VGGISH
        hub_model = bundle.get_model().to(device).eval()

    embeddings = []
    for fpath in audio_files:
        try:
            wav = load_audio(fpath, target_sr=sr)
            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(device)
            with torch.inference_mode():
                emb = hub_model(wav_tensor)
            if emb.ndim > 2:
                emb = emb.mean(dim=1)  # average over time frames
            embeddings.append(emb.cpu().numpy().squeeze())
        except Exception as e:
            warnings.warn(f"VGGish embedding failed for {fpath}: {e}")
    return np.array(embeddings)


def compute_frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """
    Compute Fréchet distance between two multivariate Gaussians:
      FD = ||mu1 - mu2||² + Tr(sigma1 + sigma2 - 2*(sigma1 @ sigma2)^(1/2))
    """
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(
        diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    )


def compute_fad(
    reference_files: List[str],
    generated_files: List[str],
    device: str = "cuda",
) -> float:
    """
    Fréchet Audio Distance using VGGish embeddings.

    FAD = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^½)

    Lower is better. Typical range: 2–15.
    """
    print("  Computing FAD (VGGish)...")
    emb_ref = extract_vggish_embeddings(reference_files, device=device)
    emb_gen = extract_vggish_embeddings(generated_files, device=device)

    if len(emb_ref) < 2 or len(emb_gen) < 2:
        warnings.warn("Not enough samples for FAD computation.")
        return float("nan")

    mu_r, sigma_r = emb_ref.mean(axis=0), np.cov(emb_ref, rowvar=False)
    mu_g, sigma_g = emb_gen.mean(axis=0), np.cov(emb_gen, rowvar=False)

    return compute_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC 2: FD — Fréchet Distance (PANNs)
# ═══════════════════════════════════════════════════════════════════════════
def extract_panns_embeddings(
    audio_files: List[str],
    sr: int = 16_000,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract PANNs (Pre-trained Audio Neural Networks) embeddings.

    Uses the Cnn14 model pre-trained on AudioSet.
    Returns array of shape (N, 2048).
    """
    try:
        from panns_inference import AudioTagging
        tagger = AudioTagging(checkpoint_path=None, device=device)
    except ImportError:
        raise RuntimeError(
            "PANNs not available. Install: pip install panns-inference"
        )

    embeddings = []
    for fpath in audio_files:
        try:
            wav = load_audio(fpath, target_sr=32_000)  # PANNs expects 32 kHz
            wav = wav[np.newaxis, :]  # (1, T)
            _, emb = tagger.inference(wav)
            embeddings.append(emb.squeeze())
        except Exception as e:
            warnings.warn(f"PANNs embedding failed for {fpath}: {e}")
    return np.array(embeddings)


def compute_fd(
    reference_files: List[str],
    generated_files: List[str],
    device: str = "cuda",
) -> float:
    """
    Fréchet Distance using PANNs embeddings.
    Same formula as FAD, different embedding model.

    Lower is better. Typical range: 15–30.
    """
    print("  Computing FD (PANNs)...")
    emb_ref = extract_panns_embeddings(reference_files, device=device)
    emb_gen = extract_panns_embeddings(generated_files, device=device)

    if len(emb_ref) < 2 or len(emb_gen) < 2:
        warnings.warn("Not enough samples for FD computation.")
        return float("nan")

    mu_r, sigma_r = emb_ref.mean(axis=0), np.cov(emb_ref, rowvar=False)
    mu_g, sigma_g = emb_gen.mean(axis=0), np.cov(emb_gen, rowvar=False)

    return compute_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC 3: CLAP Score (text–audio semantic alignment)
# ═══════════════════════════════════════════════════════════════════════════
def compute_clap_score(
    generated_files: List[str],
    prompts: List[str],
    device: str = "cuda",
) -> Tuple[float, List[float]]:
    """
    CLAP Score = cosine similarity between text and audio embeddings.

    CLAP(t, a) = cos(E_text(t), E_audio(a))

    Higher is better. Typical range: 0.30–0.60.

    Returns (mean_score, per_sample_scores).
    """
    try:
        import laion_clap
        clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        clap_model.load_ckpt()  # loads default checkpoint
    except ImportError:
        try:
            from transformers import ClapProcessor, ClapModel
            processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
            clap_model = ClapModel.from_pretrained("laion/larger_clap_general")
            clap_model.to(device).eval()

            scores = []
            for fpath, prompt in zip(generated_files, prompts):
                wav = load_audio(fpath, target_sr=48_000)
                inputs = processor(
                    text=[prompt],
                    audios=[wav],
                    return_tensors="pt",
                    sampling_rate=48_000,
                    padding=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.inference_mode():
                    outputs = clap_model(**inputs)
                    text_emb = outputs.text_embeds
                    audio_emb = outputs.audio_embeds
                    sim = torch.nn.functional.cosine_similarity(text_emb, audio_emb)
                    scores.append(sim.item())

            return float(np.mean(scores)), scores
        except ImportError:
            raise RuntimeError(
                "CLAP not available. Install: pip install laion-clap  "
                "or: pip install transformers"
            )

    # laion-clap path
    print("  Computing CLAP scores...")
    scores = []
    for fpath, prompt in zip(generated_files, prompts):
        try:
            audio_emb = clap_model.get_audio_embedding_from_filelist([fpath])
            text_emb = clap_model.get_text_embedding([prompt])
            sim = np.dot(audio_emb.squeeze(), text_emb.squeeze()) / (
                np.linalg.norm(audio_emb) * np.linalg.norm(text_emb)
            )
            scores.append(float(sim))
        except Exception as e:
            warnings.warn(f"CLAP failed for {fpath}: {e}")
            scores.append(float("nan"))

    return float(np.nanmean(scores)), scores


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC 4: KL Divergence (per-sample, PANNs class distributions)
# ═══════════════════════════════════════════════════════════════════════════
def extract_panns_class_probs(
    audio_files: List[str],
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract PANNs class probability distributions for each audio file.
    Returns array of shape (N, 527) — probability over AudioSet classes.
    """
    try:
        from panns_inference import AudioTagging
        tagger = AudioTagging(checkpoint_path=None, device=device)
    except ImportError:
        raise RuntimeError("Install: pip install panns-inference")

    probs_list = []
    for fpath in audio_files:
        try:
            wav = load_audio(fpath, target_sr=32_000)
            wav = wav[np.newaxis, :]
            clipwise_output, _ = tagger.inference(wav)
            probs_list.append(clipwise_output.squeeze())
        except Exception as e:
            warnings.warn(f"PANNs class probs failed for {fpath}: {e}")
            probs_list.append(np.full(527, 1.0 / 527))  # uniform fallback
    return np.array(probs_list)


def compute_kl_divergence(
    reference_files: List[str],
    generated_files: List[str],
    device: str = "cuda",
) -> Tuple[float, List[float]]:
    """
    KL Divergence between reference and generated class distributions.

    D_KL(P || Q) = Σ P(x) log(P(x) / Q(x))

    Lower is better. Typical range: 0.8–2.5.

    Returns (mean_kl, per_sample_kl).
    """
    print("  Computing KL Divergence (PANNs)...")
    probs_ref = extract_panns_class_probs(reference_files, device=device)
    probs_gen = extract_panns_class_probs(generated_files, device=device)

    eps = 1e-7  # prevent log(0)
    per_sample = []
    for p, q in zip(probs_ref, probs_gen):
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        p = p / p.sum()
        q = q / q.sum()
        kl = float(np.sum(p * np.log(p / q)))
        per_sample.append(kl)

    return float(np.mean(per_sample)), per_sample


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC 5: Inception Score (PANNs)
# ═══════════════════════════════════════════════════════════════════════════
def compute_inception_score(
    generated_files: List[str],
    device: str = "cuda",
) -> float:
    """
    Inception Score adapted for audio using PANNs.

    IS = exp( E_x[ D_KL( p(y|x) || p(y) ) ] )

    Higher is better. Typical range: 8–10.
    """
    print("  Computing Inception Score (PANNs)...")
    probs = extract_panns_class_probs(generated_files, device=device)

    eps = 1e-7
    probs = np.clip(probs, eps, 1.0)
    # Normalise each row
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Marginal p(y) = mean over all samples
    p_y = probs.mean(axis=0)

    # KL for each sample
    kl_per_sample = np.sum(probs * (np.log(probs) - np.log(p_y)), axis=1)

    return float(np.exp(kl_per_sample.mean()))


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════
def build_file_pairs(
    manifest_path: Optional[str] = None,
    generated_dir: Optional[str] = None,
    reference_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Build aligned lists of (reference_files, generated_files, prompts).

    Can work from a manifest.json (preferred) or from explicit dirs.
    """
    if manifest_path:
        with open(manifest_path) as f:
            manifest = json.load(f)

        ref_dir = Path(manifest.get("reference_audio_dir", reference_dir or ""))
        gen_dir = Path(manifest_path).parent / "audio"
        csv_for_prompts = csv_path or str(DEFAULT_CSV)

        prompt_map = load_prompt_map(csv_for_prompts)

        ref_files, gen_files, prompts = [], [], []
        for result in manifest["results"]:
            gen_path = gen_dir / result["output_filename"]
            ref_path = ref_dir / result["original_filename"]

            if gen_path.exists() and ref_path.exists():
                gen_files.append(str(gen_path))
                ref_files.append(str(ref_path))
                prompts.append(result["prompt"])

        return ref_files, gen_files, prompts

    elif generated_dir and reference_dir and csv_path:
        prompt_map = load_prompt_map(csv_path)
        ref_files, gen_files, prompts = [], [], []

        for filename, info in sorted(prompt_map.items()):
            ref_path = Path(reference_dir) / filename
            # Try to find matching generated file by sample number
            sample_num = info["sample_number"]
            gen_candidates = list(Path(generated_dir).glob(f"{sample_num:03d}_*.wav"))

            if ref_path.exists() and gen_candidates:
                ref_files.append(str(ref_path))
                gen_files.append(str(gen_candidates[0]))
                prompts.append(info["prompt"])

        return ref_files, gen_files, prompts

    else:
        raise ValueError(
            "Provide --manifest or (--generated, --reference, --csv)"
        )


def run_evaluation(args) -> Dict:
    """Run all metrics and produce evaluation report."""
    device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

    # Build file pairs
    ref_files, gen_files, prompts = build_file_pairs(
        manifest_path=args.manifest,
        generated_dir=args.generated,
        reference_dir=args.reference,
        csv_path=args.csv,
    )
    print(f"Evaluation pairs: {len(ref_files)} reference ↔ generated")

    if len(ref_files) == 0:
        print("ERROR: No matching file pairs found.")
        return {}

    report = {
        "num_samples": len(ref_files),
        "metrics": {},
    }

    t0 = time.perf_counter()

    # --- FAD (VGGish) ---
    if not args.skip_fad:
        try:
            fad = compute_fad(ref_files, gen_files, device=device)
            report["metrics"]["FAD"] = {"value": round(fad, 4), "direction": "lower_is_better"}
            print(f"  FAD = {fad:.4f}")
        except Exception as e:
            report["metrics"]["FAD"] = {"error": str(e)}
            print(f"  FAD: FAILED — {e}")

    # --- FD (PANNs) ---
    if not args.skip_fd:
        try:
            fd = compute_fd(ref_files, gen_files, device=device)
            report["metrics"]["FD"] = {"value": round(fd, 4), "direction": "lower_is_better"}
            print(f"  FD  = {fd:.4f}")
        except Exception as e:
            report["metrics"]["FD"] = {"error": str(e)}
            print(f"  FD: FAILED — {e}")

    # --- CLAP Score ---
    if not args.skip_clap:
        try:
            clap_mean, clap_per_sample = compute_clap_score(
                gen_files, prompts, device=device
            )
            report["metrics"]["CLAP"] = {
                "value": round(clap_mean, 4),
                "direction": "higher_is_better",
                "per_sample": [round(s, 4) for s in clap_per_sample],
            }
            print(f"  CLAP = {clap_mean:.4f}")
        except Exception as e:
            report["metrics"]["CLAP"] = {"error": str(e)}
            print(f"  CLAP: FAILED — {e}")

    # --- KL Divergence ---
    if not args.skip_kl:
        try:
            kl_mean, kl_per_sample = compute_kl_divergence(
                ref_files, gen_files, device=device
            )
            report["metrics"]["KL"] = {
                "value": round(kl_mean, 4),
                "direction": "lower_is_better",
                "per_sample": [round(k, 4) for k in kl_per_sample],
            }
            print(f"  KL  = {kl_mean:.4f}")
        except Exception as e:
            report["metrics"]["KL"] = {"error": str(e)}
            print(f"  KL: FAILED — {e}")

    # --- Inception Score ---
    if not args.skip_is:
        try:
            is_score = compute_inception_score(gen_files, device=device)
            report["metrics"]["IS"] = {
                "value": round(is_score, 4),
                "direction": "higher_is_better",
            }
            print(f"  IS  = {is_score:.4f}")
        except Exception as e:
            report["metrics"]["IS"] = {"error": str(e)}
            print(f"  IS: FAILED — {e}")

    # --- Inference Time (from manifest) ---
    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        avg_time = manifest.get("avg_time_per_sample_s", None)
        total_time = manifest.get("total_time_s", None)
        if avg_time is not None:
            report["metrics"]["inference_time"] = {
                "avg_per_sample_s": avg_time,
                "total_s": total_time,
                "direction": "lower_is_better",
            }
            print(f"  Inference time = {avg_time:.2f} s/sample")

    # --- Category-wise breakdown ---
    csv_path = args.csv or str(DEFAULT_CSV)
    try:
        prompt_map = load_prompt_map(csv_path)
        # Build per-sample metadata aligned with the file pair order
        sample_meta = []
        if args.manifest:
            with open(args.manifest) as _mf:
                _manifest = json.load(_mf)
            for result in _manifest.get("results", []):
                orig = result.get("original_filename", "")
                meta = prompt_map.get(orig, {})
                sample_meta.append(meta)
        else:
            for rf in ref_files:
                fname = Path(rf).name
                meta = prompt_map.get(fname, {})
                sample_meta.append(meta)

        if sample_meta:
            category_report = {}

            # Group by num_layers
            from collections import defaultdict
            layer_groups = defaultdict(list)
            for idx, meta in enumerate(sample_meta):
                nl = meta.get("num_layers", 0)
                layer_groups[nl].append(idx)

            category_report["by_num_layers"] = {}
            for nl in sorted(layer_groups.keys()):
                indices = layer_groups[nl]
                entry = {"count": len(indices)}
                clap_data = report["metrics"].get("CLAP", {})
                if "per_sample" in clap_data:
                    vals = [clap_data["per_sample"][i] for i in indices if i < len(clap_data["per_sample"])]
                    if vals:
                        entry["avg_clap"] = round(float(np.mean(vals)), 4)
                kl_data = report["metrics"].get("KL", {})
                if "per_sample" in kl_data:
                    vals = [kl_data["per_sample"][i] for i in indices if i < len(kl_data["per_sample"])]
                    if vals:
                        entry["avg_kl"] = round(float(np.mean(vals)), 4)
                category_report["by_num_layers"][f"{nl}_layers"] = entry

            # Group by duration bins: short (<10s), medium (10-20s), long (>20s)
            dur_groups = {"short_lt10s": [], "medium_10_20s": [], "long_gt20s": []}
            for idx, meta in enumerate(sample_meta):
                dur = meta.get("duration", 10.0)
                if dur < 10:
                    dur_groups["short_lt10s"].append(idx)
                elif dur < 20:
                    dur_groups["medium_10_20s"].append(idx)
                else:
                    dur_groups["long_gt20s"].append(idx)

            category_report["by_duration"] = {}
            for bucket, indices in dur_groups.items():
                if not indices:
                    continue
                entry = {"count": len(indices)}
                clap_data = report["metrics"].get("CLAP", {})
                if "per_sample" in clap_data:
                    vals = [clap_data["per_sample"][i] for i in indices if i < len(clap_data["per_sample"])]
                    if vals:
                        entry["avg_clap"] = round(float(np.mean(vals)), 4)
                kl_data = report["metrics"].get("KL", {})
                if "per_sample" in kl_data:
                    vals = [kl_data["per_sample"][i] for i in indices if i < len(kl_data["per_sample"])]
                    if vals:
                        entry["avg_kl"] = round(float(np.mean(vals)), 4)
                category_report["by_duration"][bucket] = entry

            report["category_breakdown"] = category_report
            print("\n  Category-wise breakdown computed.")

    except Exception as e:
        warnings.warn(f"Category breakdown failed: {e}")

    eval_time = time.perf_counter() - t0
    report["evaluation_time_s"] = round(eval_time, 2)

    # Save report
    if args.output:
        out_path = Path(args.output)
    elif args.manifest:
        out_path = Path(args.manifest).parent / "evaluation_report.json"
    else:
        out_path = Path(args.generated) / "evaluation_report.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*60}")
    for name, data in report["metrics"].items():
        if "value" in data:
            direction = "↓" if "lower" in data.get("direction", "") else "↑"
            print(f"  {name:20s}  {data['value']:>10.4f}  {direction}")
        elif "avg_per_sample_s" in data:
            print(f"  {name:20s}  {data['avg_per_sample_s']:>10.2f} s/sample  ↓")
        else:
            print(f"  {name:20s}  ERROR: {data.get('error', 'unknown')}")
    print(f"\n  Evaluation time: {eval_time:.1f}s")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate generated audio against reference dataset"
    )

    # Input sources (use manifest OR explicit paths)
    p.add_argument("--manifest", type=str, default=None,
                    help="Path to manifest.json from inference run (preferred)")
    p.add_argument("--generated", type=str, default=None,
                    help="Directory containing generated WAV files")
    p.add_argument("--reference", type=str, default=str(DEFAULT_REFERENCE_DIR),
                    help="Directory containing reference WAV files")
    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV),
                    help="Metadata CSV with prompts")

    # Output
    p.add_argument("--output", type=str, default=None,
                    help="Output path for evaluation_report.json")

    # Skip flags
    p.add_argument("--skip-fad", action="store_true")
    p.add_argument("--skip-fd", action="store_true")
    p.add_argument("--skip-clap", action="store_true")
    p.add_argument("--skip-kl", action="store_true")
    p.add_argument("--skip-is", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
