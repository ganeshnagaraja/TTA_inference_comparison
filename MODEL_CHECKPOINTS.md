# Model Checkpoints & Weights - Download Reference

All checkpoint/weight links needed for this text-to-audio inference & evaluation repo.
Download them one by one as needed.

---

## 1. Text-to-Audio Inference Models (HuggingFace)

| # | Model | HuggingFace ID | Link |
|---|-------|---------------|------|
| 1 | AudioLDM | `cvssp/audioldm-l-full` | https://huggingface.co/cvssp/audioldm-l-full |
| 2 | AudioLDM 2 | `cvssp/audioldm2-large` | https://huggingface.co/cvssp/audioldm2-large |
| 3 | Tango | `declare-lab/tango-full-ft-audiocaps` | https://huggingface.co/declare-lab/tango-full-ft-audiocaps |
| 4 | Tango 2 | `declare-lab/tango2` | https://huggingface.co/declare-lab/tango2 |
| 5 | EzAudio | `haoheliu/EzAudio-xl` | https://huggingface.co/haoheliu/EzAudio-xl |
| 6 | Stable Audio Open | `stabilityai/stable-audio-open-1.0` | https://huggingface.co/stabilityai/stable-audio-open-1.0 |
| 7 | TangoFlux | `declare-lab/TangoFlux` | https://huggingface.co/declare-lab/TangoFlux |
| 8 | MMAudio | `hkchengrex/MMAudio` | https://huggingface.co/hkchengrex/MMAudio |
| 9 | ThinkSound | `thinksound/thinksound-v1` | https://huggingface.co/thinksound/thinksound-v1 |
| 10 | VinTAGe | `vintage-audio/vintage-v1` | https://huggingface.co/vintage-audio/vintage-v1 |

> **Note:** Models 9 & 10 are marked as "placeholder" IDs in the scripts. Verify they exist on HuggingFace before downloading.

### Bundled sub-components inside the above models

These text encoders / VAEs / vocoders are **included** inside the HuggingFace repos above
(no separate download needed), listed here for reference:

| TTA Model | Text Encoder | Audio Decoder |
|-----------|-------------|---------------|
| AudioLDM | CLAP (RoBERTa) | Mel-spec VAE + HiFi-GAN |
| AudioLDM 2 | CLAP RoBERTa + FLAN-T5-Large + GPT-2 | AudioMAE |
| Tango / Tango 2 | FLAN-T5-Large (780M, frozen) | Mel-spec VAE + HiFi-GAN |
| EzAudio | FLAN-T5-Large (frozen) | 1-D Waveform VAE (DAC-style) |
| Stable Audio Open | T5 encoder + timing embeddings | DAC neural codec |
| TangoFlux | FLAN-T5 + timing + CRPO | DAC codec + FluxTransformer |
| MMAudio | CLIP text encoder (ViT-L/14) | Synchformer |
| ThinkSound | FLAN-T5-Large | Mel-spec VAE (AudioLDM-style) |
| VinTAGe | FLAN-T5-Large | Mel-spec VAE (flow-matching) |

---

## 2. Evaluation Models (separate downloads)

These are needed for computing metrics (FAD, FD, KL, IS, CLAP score).

### 2a. CLAP (for CLAP Score metric)

**Primary** — auto-downloaded by `laion-clap` pip package (`clap_model.load_ckpt()`):
- Package: `pip install laion-clap>=1.1.4`
- The default checkpoint is downloaded automatically on first use.

**Fallback** — HuggingFace transformers path:
| Model | HuggingFace ID | Link |
|-------|---------------|------|
| LAION CLAP (general) | `laion/larger_clap_general` | https://huggingface.co/laion/larger_clap_general |

### 2b. VGGish (for FAD metric)

**Primary** — via PyTorch Hub:
- Repo: `harritaylor/torchvggish`
- GitHub: https://github.com/harritaylor/torchvggish
- Auto-downloaded on first call to `torch.hub.load("harritaylor/torchvggish", "vggish")`

**Fallback** — via torchaudio built-in:
- `torchaudio.pipelines.VGGISH` (bundled with torchaudio, no separate download)

### 2c. PANNs / Cnn14 (for FD, KL, IS metrics)

- Package: `pip install panns-inference>=0.1.0`
- The Cnn14 AudioSet-pretrained checkpoint is auto-downloaded on first use
  when calling `AudioTagging(checkpoint_path=None, device=device)`

---

## 3. MMAudio Variant Weights

MMAudio supports multiple variants (configured in `infer_mmaudio.py`):

| Variant | Description |
|---------|-------------|
| `large_44k_v2` | **Default** — large model, 44.1kHz |
| `medium_44k` | Medium model, 44.1kHz |
| `small_16k` | Small model, 16kHz |

These are selected internally via `all_model_cfg[variant]` from the `mmaudio` package.
The weights are in the `hkchengrex/MMAudio` HuggingFace repo.

---

## 4. ThinkSound Fallback

When the native `thinksound` package is unavailable, the script falls back to:
- `cvssp/audioldm-l-full` (same as AudioLDM above — no extra download)

---

## 5. Custom Inference Dataset

| Resource | Link |
|----------|------|
| 215-sample inference dataset | https://drive.google.com/drive/folders/1xZyTwCLv6Q3sHq-_VA84JP4STswwe7_H?usp=drive_link |

---

## Quick Download Commands (HuggingFace CLI)

If you have `huggingface-cli` installed, you can download models one at a time:

```bash
# Install HF CLI if needed
pip install huggingface_hub[cli]

# Download one model at a time (example)
huggingface-cli download cvssp/audioldm-l-full --local-dir ./models/audioldm
huggingface-cli download cvssp/audioldm2-large --local-dir ./models/audioldm2
huggingface-cli download declare-lab/tango-full-ft-audiocaps --local-dir ./models/tango
huggingface-cli download declare-lab/tango2 --local-dir ./models/tango2
huggingface-cli download haoheliu/EzAudio-xl --local-dir ./models/ezaudio
huggingface-cli download stabilityai/stable-audio-open-1.0 --local-dir ./models/stable_audio
huggingface-cli download declare-lab/TangoFlux --local-dir ./models/tangoflux
huggingface-cli download hkchengrex/MMAudio --local-dir ./models/mmaudio
huggingface-cli download thinksound/thinksound-v1 --local-dir ./models/thinksound
huggingface-cli download vintage-audio/vintage-v1 --local-dir ./models/vintage

# Evaluation model
huggingface-cli download laion/larger_clap_general --local-dir ./models/clap
```
