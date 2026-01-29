# Kinyarwanda TTS Audio Generation

This directory contains scripts for generating Kinyarwanda audio using Text-to-Speech (TTS) models with voice cloning support. The implementation supports multiple TTS backends with fallback strategies for robustness.

## Architecture

```
corpus/kinyarwanda/*.wav
    ↓ (reference audio)
    └─→ [TTS Voice Profile Extraction]
            ↓
        [Audio Generation Pipeline]
            ↓
        generated_audio/*.wav
            ↓ (24kHz mono)
        [npm run corpus:build]
            ↓
        public/audio/kinyarwanda/*.{opus,mp3}
```

## Supported TTS Methods

1. **Qwen3-TTS** (Primary): Advanced multilingual TTS with voice cloning
   - Models: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` or `1.7B-Base`
   - Requires: GPU (A10G+) or CPU with patience
   - Setup: Custom transformers integration

2. **Facebook MMS-TTS** (Fallback): Efficient multilingual TTS
   - Model: `facebook/mms-tts-kin` (Kinyarwanda-specific)
   - Requires: PyTorch + transformers
   - No voice cloning, but native Kinyarwanda support

## Setup

### 1. Initialize Virtual Environment

```bash
cd scripts/tts
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Validate Setup

```bash
python validate_setup.py
```

Expected output:
```
✓ PyTorch                        2.10.0
✓ Hugging Face Transformers      5.0.0
✓ Librosa                        0.11.0
✓ SoundFile                      0.13.1
...
✅ Environment validation successful!
```

## Quick Start

### Step 1: Prepare Transcription Data

Extract Kinyarwanda text from `lib/phrases.ts`:

```bash
python prepare_transcriptions.py
```

Output: `generated_audio/transcriptions.json`

Status: Shows 87 recorded phrases, 767 missing phrases

### Step 2: Generate Audio (Pilot Mode - Recommended First)

Test with 10 sample phrases:

```bash
python simple_tts_generator.py --mode pilot --sample-count 10
```

This will:
- Try to load the best available TTS model
- Generate audio for 10 diverse phrases
- Save results to `generated_audio/results_pilot.json`
- Create WAV files in `generated_audio/`

### Step 3: Generate Full Batch

Once pilot testing succeeds:

```bash
python simple_tts_generator.py --mode batch
```

This generates audio for all 767 missing phrases.

### Step 4: Build Corpus

Convert generated WAVs to Opus/MP3:

```bash
npm run corpus:build
```

Verify coverage:

```bash
npm run corpus:status
```

Expected: 100% coverage across all categories

## Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `validate_setup.py` | Check environment and dependencies | - | Console report |
| `prepare_transcriptions.py` | Extract phrases and organize by category | `lib/phrases.ts` | `generated_audio/transcriptions.json` |
| `simple_tts_generator.py` | Generate audio with fallback TTS | `transcriptions.json` | `generated_audio/*.wav` |
| `generate_audio.py` | Advanced generation with Qwen3-TTS voice cloning | Voice profile + text | `generated_audio/*.wav` |
| `audio_processor.py` | Utilities for audio processing | Audio files | Normalized audio |
| `utils.py` | Common utilities (path handling, phrase parsing) | Various | Parsed data |

## Audio Specifications

All generated audio follows these specifications:

- **Sample Rate**: 24 kHz (24000 Hz)
- **Channels**: Mono (1)
- **Bit Depth**: 16-bit PCM
- **Duration**: 0.5–8 seconds per phrase
- **Loudness**: Normalized to -20 LUFS
- **Format**: WAV (before conversion to Opus/MP3)

## Advanced Usage

### Using Qwen3-TTS Directly

For advanced voice cloning with Qwen3-TTS:

```bash
python generate_audio.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --mode pilot
```

This requires:
- GPU with 12GB+ VRAM (A10G recommended)
- Installed Qwen3-TTS from Hugging Face

### Custom Reference Audio

To use specific recorded phrases as voice reference:

```python
from generate_audio import KinyarwandaTTSGenerator

gen = KinyarwandaTTSGenerator()
gen.load_model()
gen.extract_voice_profile([1, 2, 3, 4, 5])  # Use phrases 1-5 as reference
results = gen.generate_batch()
```

### Verbose Logging

Enable debug logging for troubleshooting:

```bash
python simple_tts_generator.py --mode pilot --verbose
```

## Troubleshooting

### "Model not found" Error

This usually means the model couldn't be downloaded from Hugging Face.

**Solutions:**
1. Check internet connection
2. Check Hugging Face availability (`huggingface.co`)
3. Pre-download model: `python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/mms-tts-kin')"`
4. Use smaller model: `0.6B` instead of `1.7B` for Qwen3-TTS

### Out of Memory Error

GPU or CPU memory insufficient.

**Solutions:**
1. Use smaller batch sizes (already done in pilot mode)
2. Use 0.6B model instead of 1.7B
3. Reduce sample count: `--sample-count 5`
4. Close other applications
5. Use CPU for testing, then GPU for batch generation

### Audio Quality Issues

Generated audio sounds robotic or has pronunciation errors.

**Solutions:**
1. Use more reference phrases: `extract_voice_profile([1,2,3,4,5,10])`
2. Try different TTS model
3. Check that reference phrases are well-recorded
4. Increase model size (1.7B > 0.6B)

## Output

### Directory Structure

```
scripts/tts/
├── generated_audio/
│   ├── transcriptions.json          # Phrase metadata and status
│   ├── results_pilot.json           # Pilot generation results
│   ├── results_batch.json           # Batch generation results
│   ├── 1.wav, 2.wav, ...           # Generated audio files
│   └── 767.wav
├── models/                          # Cached TTS models (gitignored)
├── venv/                            # Python virtual environment
├── utils.py                         # Shared utilities
├── prepare_transcriptions.py        # Data preparation
├── simple_tts_generator.py         # TTS generation (recommended)
├── generate_audio.py               # Advanced Qwen3-TTS integration
├── audio_processor.py              # Audio processing utilities
└── validate_setup.py               # Environment validation
```

### Result JSON Format

```json
{
  "1": {
    "status": "success",
    "path": "/path/to/generated_audio/1.wav",
    "duration": 1.23,
    "method": "mms-tts"
  },
  "2": {
    "status": "failed",
    "error": "synthesis failed"
  }
}
```

## Integration with Igisha

After generating audio:

1. **Generated audio** → `generated_audio/*.wav` (24kHz mono WAV)
2. **Run build** → `npm run corpus:build`
3. **Output** → `public/audio/kinyarwanda/{id}.opus` and `.mp3`
4. **Verify** → `npm run corpus:status` (should show 100%)

## Performance Metrics

Typical generation times (on CPU):

- **Pilot (10 phrases)**: 2–5 minutes
- **Batch (767 phrases)**: 30–60 minutes
- **With GPU (A10G)**: 10–20 minutes for batch

File sizes:

- **Average WAV**: ~300KB (1–2 seconds at 24kHz)
- **After conversion**: ~50KB Opus, ~100KB MP3
- **Total corpus**: ~50MB Opus, ~100MB MP3

## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Qwen3-TTS Model Card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- [Facebook MMS-TTS](https://huggingface.co/facebook/mms-tts)
- [Librosa Documentation](https://librosa.org/)

## License

Audio generation scripts use:
- Qwen3-TTS: Apache 2.0
- Facebook MMS: CC-BY-NC 4.0 (research use)
- Generated audio: Follows Igisha project license (as user-generated content)
