# Text-to-Audio Inference & Evaluation Pipeline

Inference and evaluation scripts for benchmarking **ten text-to-audio generative models** on a custom evaluation dataset of 215 mixed audio samples.

## Pipeline Overview

```
Text Prompts (metadata.csv)
        │
        ▼
┌─────────────────┐
│  Inference       │  prompt → model → generated WAV
│  (per model)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Evaluation      │  generated WAV  vs  reference WAV
│  (5 metrics)     │  + text–audio alignment (CLAP)
└────────┬────────┘
         │
         ▼
   evaluation_report.json
```

## Repository Structure

```
text_to_audio/
├── README.md
├── requirements.txt
├── .gitignore
│
├── inference_scripts/
│   ├── inference_config.py         # H100 hardware profile & hyperparameter defaults
│   ├── inference_utils.py          # Shared inference utilities
│   ├── infer_audioldm.py           # AudioLDM inference
│   ├── infer_audioldm2.py          # AudioLDM 2 inference
│   ├── infer_tango.py              # Tango / Tango 2 inference
│   ├── infer_ezaudio.py            # EzAudio inference
│   ├── infer_stable_audio.py       # Stable Audio Open inference
│   ├── infer_tangoflux.py          # TangoFlux inference
│   ├── infer_mmaudio.py            # MMAudio inference (text-only mode)
│   ├── infer_thinksound.py         # ThinkSound inference (with CoT reasoning)
│   ├── infer_vintage.py            # VinTAGe inference
│   └── run_all_inference.py        # Master script: run all models sequentially
│
└── evaluation/
    └── evaluate.py                 # Compute FAD, FD, CLAP, KL, IS metrics
```

## Models Covered

| Model | Sample Rate | Audio Encoding | Text Encoder | Duration |
|-------|-----------|----------------|-------------|----------|
| **AudioLDM** | 16 kHz mono | Mel-spectrogram VAE | CLAP (RoBERTa) | 10 s fixed |
| **AudioLDM 2** | 16 kHz mono | AudioMAE (LOA) | CLAP + FLAN-T5 + GPT-2 | 10 s fixed |
| **Tango** | 16 kHz mono | Mel-spectrogram VAE | FLAN-T5-Large | 10 s fixed |
| **Tango 2** | 16 kHz mono | Mel-spectrogram VAE | FLAN-T5-Large + DPO | 10 s fixed |
| **EzAudio** | 16 kHz mono | 1-D Waveform VAE | FLAN-T5-Large | 10 s fixed |
| **Stable Audio Open** | 44.1 kHz stereo | DAC neural codec | T5 + timing embeddings | Variable (up to 47 s) |
| **TangoFlux** | 44.1 kHz stereo | DAC neural codec | FLAN-T5 + timing | Variable (up to 30 s) |
| **MMAudio** | 16 kHz mono | Synchformer | CLIP text encoder | 8 s fixed |
| **ThinkSound** | 16 kHz mono | Mel-spectrogram VAE | FLAN-T5 + CoT reasoning | 10 s fixed |
| **VinTAGe** | 16 kHz mono | Mel-spectrogram VAE | FLAN-T5-Large | 10 s fixed |

## Evaluation Metrics

| Metric | What It Measures | Direction | Typical Range | Embedding Model |
|--------|-----------------|-----------|---------------|-----------------|
| **FAD** | Distributional quality (real vs generated) | Lower ↓ | 2–15 | VGGish |
| **FD** | Distributional quality (alternative embedding) | Lower ↓ | 15–30 | PANNs |
| **CLAP Score** | Text–audio semantic alignment | Higher ↑ | 0.30–0.60 | CLAP |
| **KL Divergence** | Per-sample class-distribution similarity | Lower ↓ | 0.8–2.5 | PANNs |
| **IS** | Sample quality and diversity | Higher ↑ | 8–10 | PANNs |
| Inference Time | Computational efficiency per sample | Lower ↓ | 0.07–30 s | N/A |

## Hardware Requirements

These scripts are optimised for:

- **GPU**: NVIDIA H100 80 GB SXM5
- **Precision**: BF16 (native tensor-core support)
- **CUDA**: 12.1+
- **PyTorch**: 2.1+

Key H100 optimisations applied:

- BF16 inference throughout (2x throughput vs FP32)
- TF32 enabled for any FP32 fallback operations
- cuDNN auto-tuner for optimal convolution algorithms
- xformers memory-efficient attention (when available)
- Batch sizes tuned to 80 GB VRAM capacity

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd text_to_audio

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install PyTorch for your CUDA version first
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Model-specific packages (install as needed)

```bash
# MMAudio
pip install mmaudio

# TangoFlux
pip install tangoflux

# Tango (native package, optional — diffusers fallback available)
pip install tango
```

### 3. Dataset

Download the custom inference dataset from Google Drive:

**[Download Dataset](https://drive.google.com/drive/folders/1xZyTwCLv6Q3sHq-_VA84JP4STswwe7_H?usp=drive_link)**

Place the downloaded files so the directory structure looks like:

```
Dataset/Inference_Dataset/
├── Audio_samples/          # 215 reference WAV files
│   ├── sample_001_*.wav
│   ├── ...
│   └── sample_215_*.wav
└── metadata.csv            # Prompts, durations, layer counts
```

## Running Inference

During inference, each model receives **only the text prompt** as input and generates an audio file. No audio preprocessing is needed.

### Quick start — single model

```bash
cd inference_scripts

# AudioLDM with default hyperparameters
python infer_audioldm.py

# With custom settings
python infer_audioldm.py --steps 200 --guidance 2.5 --batch-size 16 --seed 42

# Test with a small subset
python infer_audioldm.py --max-samples 10
```

### Run all models

```bash
# All models, default hyperparameters
python run_all_inference.py

# Selected models only
python run_all_inference.py --models audioldm tango stable_audio tangoflux

# Quick test (10 samples per model)
python run_all_inference.py --max-samples 10

# Hyperparameter sweep (steps × guidance scale grid)
python run_all_inference.py --sweep --max-samples 50

# List available models
python run_all_inference.py --list-models
```

## Running Evaluation

After inference, evaluate the generated audio against the reference dataset. Each inference run produces a `manifest.json` that links generated files to their reference counterparts.

### From a manifest (recommended)

```bash
python evaluation/evaluate.py \
    --manifest ./inference_outputs/audioldm/steps200_cfg2.5_seed42/manifest.json
```

### From explicit directories

```bash
python evaluation/evaluate.py \
    --generated ./inference_outputs/audioldm/steps200_cfg2.5_seed42/audio \
    --reference ./Dataset/Inference_Dataset/Audio_samples \
    --csv       ./Dataset/Inference_Dataset/metadata.csv
```

### Skip specific metrics

```bash
# Skip FAD and FD (faster evaluation)
python evaluation/evaluate.py --manifest ./path/to/manifest.json --skip-fad --skip-fd

# Only compute CLAP score
python evaluation/evaluate.py --manifest ./path/to/manifest.json \
    --skip-fad --skip-fd --skip-kl --skip-is
```

### Custom output path

```bash
python evaluation/evaluate.py --manifest ./path/to/manifest.json \
    --output ./results/audioldm_eval.json
```

## Hyperparameters

Default hyperparameters per model (tuned from thesis experiments):

| Model | Steps | Guidance Scale | Scheduler | Batch Size |
|-------|-------|---------------|-----------|------------|
| AudioLDM | 200 | 2.5 | DDIM | 16 |
| AudioLDM 2 | 200 | 3.5 | DDIM | 8 |
| Tango / Tango 2 | 200 | 3.0 | DDIM | 16 |
| EzAudio | 200 | 3.0 | DDIM | 16 |
| Stable Audio Open | 200 | 3.5 | DPM-Solver | 4 |
| TangoFlux | 100 | 4.5 | DPM-Solver | 4 |
| MMAudio | 200 | 4.5 | DDIM | 8 |
| ThinkSound | 200 | 3.5 | DDIM | 8 |
| VinTAGe | 200 | 3.0 | DDIM | 16 |

Override any parameter via CLI flags (e.g., `--steps 100 --guidance 5.0`).

## Output Structure

Each inference run produces:

```
inference_outputs/
└── {model_name}/
    └── steps{N}_cfg{G}_seed{S}/
        ├── audio/
        │   ├── 001_{model}.wav
        │   ├── 002_{model}.wav
        │   └── ...
        ├── manifest.json               # Hyperparams, timing, per-sample results
        └── evaluation_report.json      # Metrics (after running evaluate.py)
```

The `manifest.json` records the model, hardware, hyperparameters, reference audio directory, and per-sample inference time for reproducibility. The evaluation script reads this manifest to automatically pair generated and reference audio files.

## Notes

- **Inference input**: Each model takes only the text prompt as input. Audio files are not used during generation — they serve as the ground-truth reference for evaluation.

- **Variable-length models** (Stable Audio Open, TangoFlux) use each sample's original duration from the metadata rather than a fixed length. Samples are capped at the model's architectural maximum (47 s and 30 s respectively).

- **MMAudio** supports multimodal (text + video) conditioning, but runs in text-only mode here since the benchmark dataset contains audio-only samples.

- **ThinkSound** includes a chain-of-thought (CoT) reasoning layer that decomposes prompts into structured scene descriptions before generation. This is enabled by default (`--use-cot`) and can be disabled with `--no-cot` for comparison.

- All outputs are normalised to [-1, 1] and saved as 16-bit PCM WAV files.
