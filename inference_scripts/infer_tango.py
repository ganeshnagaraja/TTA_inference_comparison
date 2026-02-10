#!/usr/bin/env python3
"""
infer_tango.py
==============
Run Tango / Tango 2 inference on the custom 215-sample dataset.

Model : Tango  (declare-lab/tango)  or  Tango 2  (declare-lab/tango2)
Arch  : Latent Diffusion on mel-spectrogram VAE + HiFi-GAN vocoder
Text  : FLAN-T5-Large (frozen, 780M params)
Audio : 16 kHz mono, 10 s fixed duration
GPU   : NVIDIA H100 80 GB — BF16, batch size 16

Tango 2 uses the same pipeline but was trained with DPO (preference
optimisation via CLAP-scored pairs). Select with --model-id.

Usage:
    python infer_tango.py                                   # Tango
    python infer_tango.py --model-id declare-lab/tango2     # Tango 2
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

MODEL_NAME = "tango"
HF_MODEL_ID = "declare-lab/tango-full-ft-audiocaps"


def load_pipeline(model_id: str, device: torch.device, dtype: torch.dtype):
    """
    Load the Tango pipeline.

    Tango uses a custom pipeline that differs from the diffusers
    AudioLDMPipeline.  If the official 'tango' package is installed,
    we use it directly; otherwise fall back to diffusers AudioLDMPipeline
    which is architecture-compatible.
    """
    try:
        # Try official Tango package first
        from tango import Tango
        print(f"Loading Tango (native package): {model_id}")
        model = Tango(model_id, device=str(device))
        return model, "native"
    except ImportError:
        # Fall back to diffusers
        from diffusers import AudioLDMPipeline
        print(f"Loading Tango via diffusers AudioLDMPipeline: {model_id}")
        pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.set_progress_bar_config(disable=True)
        return pipe, "diffusers"


def run_inference(args):
    device = setup_hardware()
    dtype = get_torch_dtype()

    model_key = "tango2" if "tango2" in args.model_id.lower() else "tango"
    defaults = MODEL_DEFAULTS.get(model_key, MODEL_DEFAULTS["tango"])
    steps = args.steps or defaults.num_inference_steps
    guidance = args.guidance or defaults.guidance_scale
    batch_size = args.batch_size or defaults.batch_size
    audio_length = args.audio_length or defaults.audio_length_s
    seed = args.seed or defaults.seed

    prompts = load_prompts(args.csv)
    if args.max_samples > 0:
        prompts = prompts[:args.max_samples]
    print(f"Prompts loaded: {len(prompts)}")

    run_tag = f"{model_key}_steps{steps}_cfg{guidance}_seed{seed}"
    output_dir = setup_output_dir(model_key, args.output, run_tag)
    print(f"Output: {output_dir}")

    pipe_or_model, backend = load_pipeline(args.model_id, device, dtype)
    log_gpu_memory("after model load")

    generator = torch.Generator(device=device).manual_seed(seed)

    batches = batch_prompts(prompts, batch_size)
    results = []
    total_start = time.perf_counter()

    for bi, batch in enumerate(batches):
        batch_prompts_text = [r["prompt"] for r in batch]

        with InferenceTimer(model_key, bi, len(batch)) as timer:
            with torch.inference_mode():
                if backend == "native":
                    # Tango native: processes one prompt at a time
                    audios = []
                    for prompt_text in batch_prompts_text:
                        audio = pipe_or_model.generate(
                            prompt_text,
                            steps=steps,
                            guidance=guidance,
                        )
                        audios.append(audio)
                else:
                    # diffusers pipeline: supports batched inference
                    output = pipe_or_model(
                        prompt=batch_prompts_text,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        audio_length_in_s=audio_length,
                        num_waveforms_per_prompt=1,
                        generator=generator,
                    )
                    audios = output.audios
        timer.log()

        for j, record in enumerate(batch):
            waveform = audios[j]
            waveform = postprocess_waveform(waveform)
            out_filename = f"{record['sample_number']:03d}_{model_key}.wav"
            out_path = output_dir / "audio" / out_filename
            save_wav_16bit(str(out_path), waveform, sample_rate=16_000)

            results.append({
                "sample_number": record["sample_number"],
                "original_filename": record["filename"],
                "prompt": record["prompt"],
                "output_filename": out_filename,
                "inference_time_s": round(timer.elapsed / len(batch), 2),
            })

        if (bi + 1) % 10 == 0:
            clear_gpu_cache()

    total_time = time.perf_counter() - total_start
    print(f"\nTotal: {len(results)} samples in {total_time:.1f}s")

    save_manifest(output_dir, model_key, {
        "model_id": args.model_id,
        "backend": backend,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "audio_length_s": audio_length,
        "seed": seed,
        "batch_size": batch_size,
        "dtype": str(dtype),
    }, results, total_time, reference_audio_dir=str(AUDIO_DIR))

    del pipe_or_model
    clear_gpu_cache()
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Tango / Tango 2 inference")
    p.add_argument("--csv", default=str(METADATA_CSV))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--model-id", default=HF_MODEL_ID)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--audio-length", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
