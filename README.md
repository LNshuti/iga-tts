---
title: English-Swahili Translator
app_file: app.py
sdk: gradio
sdk_version: 4.44.1
---

# English <-> Swahili Bidirectional Speech & Text Translator

Speak or type in English or Swahili, get translations and natural speech output.

## Features

- **Speech-to-Text**: Whisper-based transcription with auto language detection
- **Translation**: MarianMT for accurate en<->sw translation
- **Text-to-Speech**: Natural voice synthesis with XTTS-v2
- **Two Modes**: Audio input or text input
- **Privacy**: All processing happens locally

## Models

| Component | Model |
|-----------|-------|
| STT | faster-whisper (small) |
| EN->SW | Helsinki-NLP/opus-mt-en-sw |
| SW->EN | Helsinki-NLP/opus-mt-sw-en |
| TTS | Coqui XTTS-v2 |

## Performance

- **CPU (free tier)**: 15-40s end-to-end
- **GPU (T4)**: 5-15s end-to-end

## Environment Variables

- `WHISPER_MODEL_SIZE`: tiny, base, small (default), medium
- `DEVICE`: auto (default), cpu, cuda

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

## Deployment

Deployed on Hugging Face Spaces.
