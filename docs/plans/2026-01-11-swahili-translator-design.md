# English-Swahili Bidirectional Speech & Text Translator

**Date:** 2026-01-11
**Status:** Approved
**GitHub Issue:** #4

## Overview

Replace the current Kinyarwanda learning app with a bidirectional English-Swahili translator featuring speech-to-text, translation, and text-to-speech capabilities.

## Architecture

### Core Pipeline

```
Audio Input → faster-whisper STT → MarianMT Translation → XTTS-v2 TTS → Audio Output
Text Input  → MarianMT Translation → XTTS-v2 TTS → Audio Output
```

### Models

| Component | Model | Size | Notes |
|-----------|-------|------|-------|
| STT | faster-whisper small | ~500MB | Good Swahili support, fast on CPU |
| EN→SW | Helsinki-NLP/opus-mt-en-sw | ~300MB | Direct translation |
| SW→EN | Helsinki-NLP/opus-mt-sw-en | ~300MB | Direct translation |
| TTS | Coqui XTTS-v2 | ~1.5GB | Multilingual, includes Swahili |

**Total RAM:** ~2.5-3GB

## UI Design

### Tab 1: "🎤 Speak → Translate → Speak"

- **Input:** Audio recorder (microphone or file upload)
- **Controls:**
  - Source language: `auto`, `en`, `sw` (default: auto)
  - Target language: `en`, `sw` (default: sw)
  - "Transcribe + Translate + Speak" button
- **Output:**
  - Detected language
  - Transcript (source)
  - Translation (target)
  - Audio player

### Tab 2: "✍️ Type → Translate → Speak"

- **Input:** Text area
- **Controls:**
  - Source language: `en`, `sw` (default: en)
  - Target language: `en`, `sw` (default: sw)
  - "Translate + Speak" button
- **Output:**
  - Translation text
  - Audio player
- **Examples:** Pre-filled phrases

### Additional Elements

- Collapsible "Advanced Settings & Info" accordion
- Privacy notice (all processing local)
- Tips and troubleshooting

## Implementation

### Files to Modify

| File | Action | Description |
|------|--------|-------------|
| `app.py` | Replace | New Swahili translator |
| `requirements.txt` | Update | Add new dependencies |
| `README.md` | Update | New description |
| `config.py` | Simplify | Remove unused, add Whisper config |

### Files to Remove

- `phrases.py`, `corpus.py`, `corpus_db.py`, `Corpus.txt`
- `ab_test_logging.py`, `bayesian_optimizer.py`, `variant_manager.py`
- `feedback_storage.py`, `audio_encryption.py`
- `preprocess_corpus.py`

### New Dependencies

```
faster-whisper>=0.10.0
TTS>=0.22.0
soundfile>=0.12.0
```

### Environment Variables

- `WHISPER_MODEL_SIZE`: tiny, base, small (default), medium
- `DEVICE`: auto, cpu, cuda

## Error Handling

| Scenario | Handling |
|----------|----------|
| STT model fails | Log error, show message |
| Translation fails | Return error with ❌ prefix |
| TTS model fails | Log warning, no audio output |
| No audio input | Return "No audio input" message |

### Graceful Degradation

- TTS fails → translation still works (text only)
- STT fails → Text tab still works
- All models load with try/except

## Performance (Free CPU Tier)

- STT: 2-5s for 10s audio
- Translation: <1s
- TTS: 10-30s per sentence
- **Total:** 15-40s end-to-end

## Deployment

1. Update local files
2. Test with `python app.py`
3. Push to HuggingFace Space
