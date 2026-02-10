#!/usr/bin/env python3
"""
infer_thinksound.py
===================
Run ThinkSound inference on the custom 215-sample dataset.

Model : ThinkSound  (reasoning-augmented text-to-audio generation)
Arch  : AudioLDM-style VAE backbone + chain-of-thought (CoT) reasoning
        layer powered by a multimodal LLM that decomposes complex prompts
        into structured descriptions of sound sources, temporal relations,
        and acoustic properties before generation.
Text  : FLAN-T5-Large (for conditioning after CoT enrichment)
Audio : 16 kHz mono, 10 s fixed duration
GPU   : NVIDIA H100 80 GB — BF16, batch size 8

Key innovation: before generation, each prompt is first expanded via CoT
reasoning into a richer structured caption, improving semantic grounding.

Usage:
    python infer_thinksound.py
    python infer_thinksound.py --steps 200 --guidance 3.5 --use-cot
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

# Import the CoT caption generator from our preprocessor
from preprocess_thinksound import generate_cot_caption

MODEL_NAME = "thinksound"
HF_MODEL_ID = "thinksound/thinksound-v1"  # placeholder


def load_pipeline(model_id: str, device: torch.device, dtype: torch.dtype):
    """
    Load ThinkSound pipeline.

    ThinkSound augments an AudioLDM-style backbone with CoT reasoning.
    The generation backbone is loaded as an AudioLDM pipeline, and the
    CoT reasoning is handled in the preprocessing step.
    """
    try:
        # Try native ThinkSound package
        from thinksound import ThinkSound
        print(f"Loading ThinkSound (native): {model_id}")
        model = ThinkSound.from_pretrained(model_id)
        model = model.to(device)
        if dtype == torch.bfloat16:
            model = model.to(dtype=dtype)
        model.eval()
        return model, "native"
    except ImportError:
        # Fall back to AudioLDM pipeline with CoT preprocessing
        from diffusers import AudioLDMPipeline
        # Use a compatible AudioLDM backbone as the generation engine
        backbone_id = "cvssp/audioldm-l-full"
        print(f"Loading ThinkSound backbone via AudioLDM: {backbone_id}")
        print("  (CoT reasoning applied in preprocessing)")
        pipe = AudioLDMPipeline.from_pretrained(backbone_id, torch_dtype=dtype)
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.set_progress_bar_config(disable=True)
        return pipe, "diffusers_with_cot"


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

    run_tag = f"steps{steps}_cfg{guidance}_cot{args.use_cot}_seed{seed}"
    output_dir = setup_output_dir(MODEL_NAME, args.output, run_tag)

    pipe_or_model, backend = load_pipeline(args.model_id, device, dtype)
    log_gpu_memory("after model load")

    generator = torch.Generator(device=device).manual_seed(seed)

    # Pre-generate CoT captions for all prompts
    if args.use_cot:
        print("Generating chain-of-thought captions...")
        for record in prompts:
            record["cot_caption"] = generate_cot_caption(
                record["prompt"], record["num_layers"]
            )
        print(f"  {len(prompts)} CoT captions generated")

    batches = batch_prompts(prompts, batch_size)
    results = []
    total_start = time.perf_counter()

    for bi, batch in enumerate(batches):
        # Use CoT-enriched captions if enabled
        if args.use_cot:
            batch_texts = [r.get("cot_caption", r["prompt"]) for r in batch]
        else:
            batch_texts = [r["prompt"] for r in batch]

        with InferenceTimer(MODEL_NAME, bi, len(batch)) as timer:
            with torch.inference_mode():
                if backend == "native":
                    audios = []
                    for text in batch_texts:
                        audio = pipe_or_model.generate(
                            text,
                            steps=steps,
                            guidance=guidance,
                            use_cot=args.use_cot,
                        )
                        audios.append(audio)
                else:
                    output = pipe_or_model(
                        prompt=batch_texts,
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
            out_filename = f"{record['sample_number']:03d}_{MODEL_NAME}.wav"
            out_path = output_dir / "audio" / out_filename
            save_wav_16bit(str(out_path), waveform, sample_rate=16_000)

            results.append({
                "sample_number": record["sample_number"],
                "original_filename": record["filename"],
                "prompt": record["prompt"],
                "cot_caption_used": args.use_cot,
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
        "use_cot": args.use_cot,
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
    p = argparse.ArgumentParser(description="ThinkSound inference")
    p.add_argument("--csv", default=str(METADATA_CSV))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--model-id", default=HF_MODEL_ID)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--audio-length", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--use-cot", action="store_true", default=True,
                    help="Use chain-of-thought enriched captions (default: True)")
    p.add_argument("--no-cot", dest="use_cot", action="store_false",
                    help="Use raw prompts without CoT")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
