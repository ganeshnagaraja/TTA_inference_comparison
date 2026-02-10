#!/usr/bin/env python3
"""
infer_mmaudio.py
================
Run MMAudio inference on the custom 215-sample dataset (text-only mode).

Model : MMAudio  (hkchengrex/MMAudio)
Arch  : Multimodal diffusion transformer with Synchformer backbone.
        Jointly conditions on text (CLIP) and visual features (Synchformer).
        For text-only inference, visual branch receives zeros.
Text  : CLIP text encoder (ViT-L/14)
Audio : 16 kHz mono, ~8 s fixed duration
GPU   : NVIDIA H100 80 GB — BF16, batch size 8

Note: MMAudio natively supports video+text conditioning.
      For this benchmark we use text-only mode (visual features zeroed).

Usage:
    python infer_mmaudio.py
    python infer_mmaudio.py --steps 200 --guidance 4.5 --batch-size 8
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

MODEL_NAME = "mmaudio"
HF_MODEL_ID = "hkchengrex/MMAudio"
MMAUDIO_VARIANT = "large_44k_v2"  # or "small_16k", "medium_44k"


def load_pipeline(model_id: str, variant: str, device: torch.device, dtype: torch.dtype):
    """
    Load MMAudio model.

    MMAudio is typically loaded from its own repository:
        pip install mmaudio
    """
    try:
        from mmaudio.eval_utils import (
            ModelConfig, all_model_cfg, generate, setup_eval_logging
        )
        from mmaudio.model.flow_matching import FlowMatching
        from mmaudio.model.networks import MMAudio as MMAudioModel

        print(f"Loading MMAudio: {variant}")
        model_cfg: ModelConfig = all_model_cfg[variant]

        model = MMAudioModel(
            model_cfg.model_name,
            device=device,
            dtype=dtype,
        )

        print(f"  Model loaded: {variant}")
        return model, model_cfg, "native"

    except ImportError:
        print("MMAudio native package not found.")
        print("Install: pip install mmaudio")
        print("Falling back to HuggingFace pipeline wrapper...")

        # Fallback: attempt loading as a generic diffusion pipeline
        try:
            from diffusers import AudioLDMPipeline
            pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=dtype)
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            return pipe, None, "diffusers"
        except Exception as e:
            raise RuntimeError(
                f"Cannot load MMAudio. Install:\n"
                f"  pip install mmaudio  (recommended)\n"
                f"Error: {e}"
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

    model_or_pipe, model_cfg, backend = load_pipeline(
        args.model_id, args.variant, device, dtype
    )
    log_gpu_memory("after model load")

    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    results = []
    total_start = time.perf_counter()

    # MMAudio: process per-sample (text-only mode, no video)
    for i, record in enumerate(prompts):
        with InferenceTimer(MODEL_NAME, i, 1) as timer:
            with torch.inference_mode():
                if backend == "native":
                    from mmaudio.eval_utils import generate
                    audio = generate(
                        model=model_or_pipe,
                        cfg=model_cfg,
                        prompt=record["prompt"],
                        video=None,           # text-only: no video input
                        num_steps=steps,
                        cfg_strength=guidance,
                        duration=audio_length,
                        seed=seed,
                    )
                else:
                    output = model_or_pipe(
                        prompt=record["prompt"],
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        audio_length_in_s=audio_length,
                        generator=generator,
                    )
                    audio = output.audios[0]

        if (i + 1) % 20 == 0 or i == 0:
            timer.log()

        waveform = postprocess_waveform(audio)
        out_filename = f"{record['sample_number']:03d}_{MODEL_NAME}.wav"
        out_path = output_dir / "audio" / out_filename
        # MMAudio outputs at 16 kHz (or 44.1 kHz for _44k variants)
        sr = 44_100 if "44k" in args.variant else 16_000
        save_wav_16bit(str(out_path), waveform, sample_rate=sr)

        results.append({
            "sample_number": record["sample_number"],
            "original_filename": record["filename"],
            "prompt": record["prompt"],
            "output_filename": out_filename,
            "inference_time_s": round(timer.elapsed, 2),
            "mode": "text-only",
        })

        if (i + 1) % 20 == 0:
            clear_gpu_cache()

    total_time = time.perf_counter() - total_start
    print(f"\nTotal: {len(results)} samples in {total_time:.1f}s")

    save_manifest(output_dir, MODEL_NAME, {
        "model_id": args.model_id,
        "variant": args.variant,
        "backend": backend,
        "mode": "text-only (no video)",
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "audio_length_s": audio_length,
        "seed": seed,
        "batch_size": batch_size,
        "dtype": str(dtype),
    }, results, total_time, reference_audio_dir=str(AUDIO_DIR))

    del model_or_pipe
    clear_gpu_cache()
    return results


def parse_args():
    p = argparse.ArgumentParser(description="MMAudio inference (text-only)")
    p.add_argument("--csv", default=str(METADATA_CSV))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--model-id", default=HF_MODEL_ID)
    p.add_argument("--variant", default=MMAUDIO_VARIANT,
                    choices=["small_16k", "medium_44k", "large_44k_v2"])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--audio-length", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
