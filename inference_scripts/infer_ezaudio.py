#!/usr/bin/env python3
"""
infer_ezaudio.py
================
Run EzAudio inference on the custom 215-sample dataset.

Model : EzAudio  (haoheliu/EzAudio)
Arch  : Latent Diffusion on 1-D waveform VAE (snake activations, DAC-style)
        No mel spectrogram, no separate vocoder — decoder outputs waveform directly
Text  : FLAN-T5-Large (frozen)
Audio : 16 kHz mono, 10 s fixed duration
GPU   : NVIDIA H100 80 GB — BF16, batch size 16

Usage:
    python infer_ezaudio.py
    python infer_ezaudio.py --steps 200 --guidance 3.0 --batch-size 16
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

MODEL_NAME = "ezaudio"
HF_MODEL_ID = "haoheliu/EzAudio-xl"


def load_pipeline(model_id: str, device: torch.device, dtype: torch.dtype):
    """
    Load EzAudio pipeline.

    EzAudio may be available through:
      1. The official EzAudio repository (pip install ezaudio)
      2. A HuggingFace diffusers-compatible pipeline

    We attempt both, with the diffusers approach as the primary path.
    """
    try:
        from diffusers import AudioLDMPipeline
        print(f"Loading EzAudio via diffusers: {model_id}")
        pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.set_progress_bar_config(disable=True)
        return pipe, "diffusers"
    except Exception as e:
        print(f"  diffusers load failed: {e}")
        print("  Attempting native EzAudio package...")

        try:
            # Official EzAudio package
            from ezaudio import EzAudio
            model = EzAudio.from_pretrained(model_id)
            model = model.to(device)
            if dtype == torch.bfloat16:
                model = model.to(dtype=dtype)
            model.eval()
            return model, "native"
        except ImportError:
            raise RuntimeError(
                f"Cannot load EzAudio model. Install either:\n"
                f"  pip install diffusers  (for HuggingFace pipeline)\n"
                f"  pip install ezaudio    (for native package)\n"
                f"Then ensure model weights are available: {model_id}"
            )


def run_inference(args):
    device = setup_hardware()
    dtype = get_torch_dtype()

    defaults = MODEL_DEFAULTS[MODEL_NAME]
    steps = args.steps or defaults.num_inference_steps
    guidance = args.guidance or defaults.guidance_scale
    batch_size = args.batch_size or defaults.batch_size
    audio_length = args.audio_length or defaults.audio_length_s
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

    batches = batch_prompts(prompts, batch_size)
    results = []
    total_start = time.perf_counter()

    for bi, batch in enumerate(batches):
        batch_prompts_text = [r["prompt"] for r in batch]

        with InferenceTimer(MODEL_NAME, bi, len(batch)) as timer:
            with torch.inference_mode():
                if backend == "diffusers":
                    output = pipe_or_model(
                        prompt=batch_prompts_text,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        audio_length_in_s=audio_length,
                        num_waveforms_per_prompt=1,
                        generator=generator,
                    )
                    audios = output.audios
                else:
                    # Native EzAudio: typically single-prompt
                    audios = []
                    for prompt_text in batch_prompts_text:
                        audio = pipe_or_model.generate(
                            prompt_text,
                            steps=steps,
                            cfg_scale=guidance,
                            duration=audio_length,
                        )
                        audios.append(audio)
        timer.log()

        for j, record in enumerate(batch):
            waveform = audios[j]
            waveform = postprocess_waveform(waveform)
            out_filename = f"{record['sample_number']:03d}_{MODEL_NAME}.wav"
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

    save_manifest(output_dir, MODEL_NAME, {
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
    p = argparse.ArgumentParser(description="EzAudio inference")
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
