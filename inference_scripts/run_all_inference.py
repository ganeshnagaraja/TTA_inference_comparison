#!/usr/bin/env python3
"""
run_all_inference.py
====================
Master script to run inference across all (or selected) text-to-audio
models on the custom 215-sample dataset.

Runs each model sequentially, clearing GPU memory between models.
Generates a combined summary report at the end.

Hardware: NVIDIA H100 80 GB — BF16 inference throughout.

Usage:
    python run_all_inference.py                                  # all models
    python run_all_inference.py --models audioldm tango tangoflux # selected
    python run_all_inference.py --max-samples 10 --seed 42       # quick test
    python run_all_inference.py --sweep                          # hyperparameter sweep

Supported models:
    audioldm, audioldm2, tango, tango2, ezaudio,
    stable_audio, tangoflux, mmaudio, thinksound, vintage
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

# Resolve paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from inference_config import (
    METADATA_CSV, DEFAULT_OUTPUT_ROOT, MODEL_DEFAULTS,
    setup_hardware, clear_gpu_cache,
)
from inference_utils import load_prompts


# ---------------------------------------------------------------------------
# Model registry → inference scripts
# ---------------------------------------------------------------------------
MODEL_SCRIPTS = {
    "audioldm":     "infer_audioldm.py",
    "audioldm2":    "infer_audioldm2.py",
    "tango":        "infer_tango.py",
    "tango2":       "infer_tango.py",        # same script, --model-id differs
    "ezaudio":      "infer_ezaudio.py",
    "stable_audio": "infer_stable_audio.py",
    "tangoflux":    "infer_tangoflux.py",
    "mmaudio":      "infer_mmaudio.py",
    "thinksound":   "infer_thinksound.py",
    "vintage":      "infer_vintage.py",
}

MODEL_IDS = {
    "audioldm":     "cvssp/audioldm-l-full",
    "audioldm2":    "cvssp/audioldm2-large",
    "tango":        "declare-lab/tango-full-ft-audiocaps",
    "tango2":       "declare-lab/tango2",
    "ezaudio":      "haoheliu/EzAudio-xl",
    "stable_audio": "stabilityai/stable-audio-open-1.0",
    "tangoflux":    "declare-lab/TangoFlux",
    "mmaudio":      "hkchengrex/MMAudio",
    "thinksound":   "thinksound/thinksound-v1",
    "vintage":      "vintage-audio/vintage-v1",
}

ALL_MODELS = list(MODEL_SCRIPTS.keys())


def run_single_model(
    model_name: str,
    csv_path: str,
    output_root: str,
    steps: int = None,
    guidance: float = None,
    batch_size: int = None,
    seed: int = 42,
    max_samples: int = 0,
) -> Dict:
    """
    Launch a single model inference script as a subprocess.

    Returns a dict with model name, status, time, and output path.
    """
    script = SCRIPT_DIR / MODEL_SCRIPTS[model_name]
    model_id = MODEL_IDS[model_name]

    cmd = [
        sys.executable, str(script),
        "--csv", csv_path,
        "--output", output_root,
        "--model-id", model_id,
        "--seed", str(seed),
    ]
    if steps is not None:
        cmd += ["--steps", str(steps)]
    if guidance is not None:
        cmd += ["--guidance", str(guidance)]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]
    if max_samples > 0:
        cmd += ["--max-samples", str(max_samples)]

    print(f"\n{'='*70}")
    print(f"  Running: {model_name}")
    print(f"  Model ID: {model_id}")
    print(f"  Script: {script.name}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per model
        )
        elapsed = time.perf_counter() - t0

        # Print stdout/stderr
        if result.stdout:
            # Print last 20 lines of stdout
            lines = result.stdout.strip().split("\n")
            for line in lines[-20:]:
                print(f"  {line}")

        if result.returncode != 0:
            print(f"  ERROR (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-10:]:
                    print(f"  STDERR: {line}")
            return {
                "model": model_name,
                "status": "failed",
                "error": result.stderr[-500:] if result.stderr else "unknown",
                "time_s": round(elapsed, 1),
            }

        return {
            "model": model_name,
            "status": "success",
            "time_s": round(elapsed, 1),
        }

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"  TIMEOUT after {elapsed:.0f}s")
        return {
            "model": model_name,
            "status": "timeout",
            "time_s": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  EXCEPTION: {e}")
        return {
            "model": model_name,
            "status": "exception",
            "error": str(e),
            "time_s": round(elapsed, 1),
        }


def run_hyperparameter_sweep(
    models: List[str],
    csv_path: str,
    output_root: str,
    seed: int = 42,
    max_samples: int = 0,
) -> List[Dict]:
    """
    Run a hyperparameter sweep across step counts and guidance scales.

    Sweeps:
        steps    : [100, 200]
        guidance : [2.5, 3.5, 5.0]
    """
    step_values = [100, 200]
    guidance_values = [2.5, 3.5, 5.0]

    all_results = []
    for model_name in models:
        for steps in step_values:
            for guidance in guidance_values:
                print(f"\n--- Sweep: {model_name} | steps={steps} | cfg={guidance} ---")
                result = run_single_model(
                    model_name, csv_path, output_root,
                    steps=steps, guidance=guidance,
                    seed=seed, max_samples=max_samples,
                )
                result["sweep_steps"] = steps
                result["sweep_guidance"] = guidance
                all_results.append(result)

                # Clear GPU between sweep runs
                clear_gpu_cache()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run text-to-audio inference across all models"
    )
    parser.add_argument("--csv", default=str(METADATA_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sweep", action="store_true",
                        help="Run hyperparameter sweep instead of single run")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        print("-" * 70)
        for name in ALL_MODELS:
            defaults = MODEL_DEFAULTS.get(name)
            print(f"  {name:15s}  {MODEL_IDS[name]:45s}  "
                  f"steps={defaults.num_inference_steps if defaults else '?'}  "
                  f"cfg={defaults.guidance_scale if defaults else '?'}")
        return

    print("=" * 70)
    print("  Text-to-Audio Inference Pipeline")
    print("  Hardware: NVIDIA H100 80 GB — BF16")
    print("=" * 70)
    print(f"  Models     : {', '.join(args.models)}")
    print(f"  CSV        : {args.csv}")
    print(f"  Output     : {args.output}")
    print(f"  Seed       : {args.seed}")
    print(f"  Max samples: {'all' if args.max_samples <= 0 else args.max_samples}")
    if args.sweep:
        print(f"  Mode       : Hyperparameter sweep")
    print()

    # Verify GPU
    setup_hardware()

    total_start = time.perf_counter()

    if args.sweep:
        results = run_hyperparameter_sweep(
            args.models, args.csv, args.output,
            seed=args.seed, max_samples=args.max_samples,
        )
    else:
        results = []
        for model_name in args.models:
            result = run_single_model(
                model_name, args.csv, args.output,
                steps=args.steps, guidance=args.guidance,
                batch_size=args.batch_size,
                seed=args.seed, max_samples=args.max_samples,
            )
            results.append(result)
            clear_gpu_cache()

    total_time = time.perf_counter() - total_start

    # Summary report
    print(f"\n{'='*70}")
    print("  INFERENCE SUMMARY")
    print(f"{'='*70}")
    for r in results:
        status_icon = "OK" if r["status"] == "success" else "FAIL"
        print(f"  [{status_icon:4s}]  {r['model']:15s}  {r['time_s']:7.1f}s  "
              f"{r.get('sweep_steps', ''):>5}  {r.get('sweep_guidance', ''):>5}")

    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save summary
    summary_path = Path(args.output) / "inference_summary.json"
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "total_time_s": round(total_time, 1),
            "models_run": len(results),
            "results": results,
        }, f, indent=2)
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
