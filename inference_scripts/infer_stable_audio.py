#!/usr/bin/env python3
"""
infer_stable_audio.py
=====================
Run Stable Audio Open inference on the custom 215-sample dataset.

Model : Stable Audio Open  (stabilityai/stable-audio-open-1.0)
Arch  : Timing-conditioned Latent Diffusion with DAC-derived codec
        (64-d latent @ 21.5 Hz), fully convolutional encoder/decoder
Text  : T5 encoder + learned timing embeddings (seconds_start, seconds_total)
Audio : 44.1 kHz stereo, variable-length up to ~47 seconds
GPU   : NVIDIA H100 80 GB — BF16, batch size 4 (stereo 44.1 kHz = large tensors)

Key feature: variable-length generation via timing conditioning.
Each sample uses its original duration from the metadata.

Usage:
    python infer_stable_audio.py
    python infer_stable_audio.py --steps 200 --guidance 3.5 --batch-size 4
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
    load_prompts, batch_prompts, setup_output_dir,
    postprocess_waveform, save_wav_16bit,
    InferenceTimer, log_gpu_memory, save_manifest,
)

MODEL_NAME = "stable_audio"
HF_MODEL_ID = "stabilityai/stable-audio-open-1.0"


def load_pipeline(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load the Stable Audio Open pipeline."""
    from diffusers import StableAudioPipeline

    print(f"Loading Stable Audio Open: {model_id}")
    pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("  xformers enabled")
    except Exception:
        print("  xformers not available")

    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_inference(args):
    device = setup_hardware()
    dtype = get_torch_dtype()

    defaults = MODEL_DEFAULTS[MODEL_NAME]
    steps = args.steps or defaults.num_inference_steps
    guidance = args.guidance or defaults.guidance_scale
    batch_size = args.batch_size or defaults.batch_size
    seed = args.seed or defaults.seed

    prompts = load_prompts(args.csv)
    if args.max_samples > 0:
        prompts = prompts[:args.max_samples]
    print(f"Prompts loaded: {len(prompts)}")

    run_tag = f"steps{steps}_cfg{guidance}_seed{seed}"
    output_dir = setup_output_dir(MODEL_NAME, args.output, run_tag)

    pipe = load_pipeline(args.model_id, device, dtype)
    log_gpu_memory("after model load")

    generator = torch.Generator(device=device).manual_seed(seed)

    # Stable Audio uses per-sample duration; process one at a time
    # or batch samples with the same duration
    results = []
    total_start = time.perf_counter()

    # For variable-length, process individually (durations differ per sample)
    for i, record in enumerate(prompts):
        # Use original sample duration for timing conditioning
        audio_length = min(record["duration"], 47.0)  # cap at model max

        with InferenceTimer(MODEL_NAME, i, 1) as timer:
            with torch.inference_mode():
                output = pipe(
                    prompt=record["prompt"],
                    negative_prompt=None,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    audio_end_in_s=audio_length,
                    num_waveforms_per_prompt=1,
                    generator=generator,
                )
        if (i + 1) % 20 == 0 or i == 0:
            timer.log()

        waveform = output.audios[0]
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
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "audio_length_s": "variable (per-sample duration)",
        "seed": seed,
        "dtype": str(dtype),
        "sample_rate": 44_100,
        "channels": "stereo",
    }, results, total_time, reference_audio_dir=str(AUDIO_DIR))

    del pipe
    clear_gpu_cache()
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Stable Audio Open inference")
    p.add_argument("--csv", default=str(METADATA_CSV))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--model-id", default=HF_MODEL_ID)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
