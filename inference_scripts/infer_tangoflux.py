#!/usr/bin/env python3
"""
infer_tangoflux.py
==================
Run TangoFlux inference on the custom 215-sample dataset.

Model : TangoFlux  (declare-lab/TangoFlux)
Arch  : FluxTransformer (rectified flow matching) + DAC codec
        (64-d latent @ 21.5 Hz, same as Stable Audio Open)
Text  : FLAN-T5 + timing conditioning + CRPO alignment
Audio : 44.1 kHz stereo, variable-length up to 30 seconds
GPU   : NVIDIA H100 80 GB — BF16, batch size 4

Usage:
    python infer_tangoflux.py
    python infer_tangoflux.py --steps 100 --guidance 4.5
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference_config import (
    METADATA_CSV, DEFAULT_OUTPUT_ROOT, MODEL_DEFAULTS, AUDIO_DIR,
    setup_hardware, get_torch_dtype, clear_gpu_cache,
)
from inference_utils import (
    load_prompts, setup_output_dir,
    postprocess_waveform, save_wav_16bit,
    InferenceTimer, log_gpu_memory, save_manifest,
)

MODEL_NAME = "tangoflux"
HF_MODEL_ID = "declare-lab/TangoFlux"


def load_pipeline(model_id: str, device: torch.device, dtype: torch.dtype):
    """
    Load TangoFlux pipeline.

    TangoFlux may be loaded via:
      1. The official tangoflux package
      2. A diffusers-compatible StableAudioPipeline wrapper
    """
    try:
        # Try official TangoFlux package
        from tangoflux import TangoFlux
        print(f"Loading TangoFlux (native): {model_id}")
        model = TangoFlux(model_id, device=str(device))
        return model, "native"
    except ImportError:
        try:
            from diffusers import StableAudioPipeline
            print(f"Loading TangoFlux via diffusers: {model_id}")
            pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=dtype)
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            return pipe, "diffusers"
        except Exception as e:
            raise RuntimeError(
                f"Cannot load TangoFlux. Install:\n"
                f"  pip install tangoflux  (official)\n"
                f"  or pip install diffusers  (HuggingFace)\n"
                f"Error: {e}"
            )


def run_inference(args):
    device = setup_hardware()
    dtype = get_torch_dtype()

    defaults = MODEL_DEFAULTS[MODEL_NAME]
    steps = args.steps or defaults.num_inference_steps
    guidance = args.guidance or defaults.guidance_scale
    seed = args.seed or defaults.seed

    prompts = load_prompts(args.csv)
    if args.max_samples > 0:
        prompts = prompts[:args.max_samples]
    print(f"Prompts loaded: {len(prompts)}")

    run_tag = f"steps{steps}_cfg{guidance}_seed{seed}"
    output_dir = setup_output_dir(MODEL_NAME, args.output, run_tag)

    pipe_or_model, backend = load_pipeline(args.model_id, device, dtype)
    log_gpu_memory("after model load")

    generator = torch.Generator(device=device).manual_seed(seed)

    results = []
    total_start = time.perf_counter()

    # Variable-length: process one at a time
    for i, record in enumerate(prompts):
        audio_length = min(record["duration"], 30.0)  # TangoFlux max

        with InferenceTimer(MODEL_NAME, i, 1) as timer:
            with torch.inference_mode():
                if backend == "native":
                    audio = pipe_or_model.generate(
                        record["prompt"],
                        steps=steps,
                        guidance=guidance,
                        duration=audio_length,
                    )
                    audios = [audio]
                else:
                    output = pipe_or_model(
                        prompt=record["prompt"],
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        audio_end_in_s=audio_length,
                        num_waveforms_per_prompt=1,
                        generator=generator,
                    )
                    audios = output.audios

        if (i + 1) % 20 == 0 or i == 0:
            timer.log()

        waveform = audios[0]
        waveform = postprocess_waveform(waveform)
        out_filename = f"{record['sample_number']:03d}_{MODEL_NAME}.wav"
        out_path = output_dir / "audio" / out_filename
        save_wav_16bit(str(out_path), waveform, sample_rate=44_100)

        results.append({
            "sample_number": record["sample_number"],
            "original_filename": record["filename"],
            "prompt": record["prompt"],
            "output_filename": out_filename,
            "audio_length_s": audio_length,
            "inference_time_s": round(timer.elapsed, 2),
        })

        if (i + 1) % 20 == 0:
            clear_gpu_cache()

    total_time = time.perf_counter() - total_start
    print(f"\nTotal: {len(results)} samples in {total_time:.1f}s")

    save_manifest(output_dir, MODEL_NAME, {
        "model_id": args.model_id,
        "backend": backend,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "audio_length_s": "variable (per-sample, max 30s)",
        "seed": seed,
        "dtype": str(dtype),
        "sample_rate": 44_100,
        "channels": "stereo",
    }, results, total_time, reference_audio_dir=str(AUDIO_DIR))

    del pipe_or_model
    clear_gpu_cache()
    return results


def parse_args():
    p = argparse.ArgumentParser(description="TangoFlux inference")
    p.add_argument("--csv", default=str(METADATA_CSV))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--model-id", default=HF_MODEL_ID)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
