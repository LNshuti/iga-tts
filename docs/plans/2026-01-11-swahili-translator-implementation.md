# Swahili Translator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Kinyarwanda learning app with a bidirectional English-Swahili speech and text translator.

**Architecture:** Single-file Gradio app with three ML pipelines: faster-whisper for STT, MarianMT for translation, and XTTS-v2 for TTS. Models load at startup with graceful degradation if any fail.

**Tech Stack:** Gradio, faster-whisper, transformers (MarianMT), Coqui TTS, PyTorch, soundfile

---

## Task 1: Update Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements.txt**

Replace contents with:

```txt
transformers>=4.43.0
accelerate>=0.33.0
torch
gradio>=4.44.1
sentencepiece
sacremoses
numpy
soundfile>=0.12.0
faster-whisper>=0.10.0
TTS>=0.22.0
```

**Step 2: Commit**

```bash
git add requirements.txt
git commit -m "feat: update dependencies for Swahili translator"
```

---

## Task 2: Simplify Configuration

**Files:**
- Modify: `config.py`

**Step 1: Replace config.py**

```python
"""
Configuration for English-Swahili Translator.
"""
import os
import logging

class Config:
    """Application configuration."""

    # Whisper Settings
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small")

    # Device Settings
    DEVICE: str = os.getenv("DEVICE", "auto")

    # Server Settings
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "7860"))

    @classmethod
    def get_device(cls) -> str:
        """Resolve device setting."""
        if cls.DEVICE == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return cls.DEVICE


# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("swahili_translator")
```

**Step 2: Commit**

```bash
git add config.py
git commit -m "refactor: simplify config for Swahili translator"
```

---

## Task 3: Remove Unused Files

**Files:**
- Delete: `phrases.py`, `corpus.py`, `corpus_db.py`, `Corpus.txt`
- Delete: `ab_test_logging.py`, `bayesian_optimizer.py`, `variant_manager.py`
- Delete: `feedback_storage.py`, `audio_encryption.py`, `audio_utils.py`
- Delete: `preprocess_corpus.py`, `translation.py`, `tts.py`

**Step 1: Remove files**

```bash
rm -f phrases.py corpus.py corpus_db.py Corpus.txt
rm -f ab_test_logging.py bayesian_optimizer.py variant_manager.py
rm -f feedback_storage.py audio_encryption.py audio_utils.py
rm -f preprocess_corpus.py translation.py tts.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "chore: remove unused Kinyarwanda app files"
```

---

## Task 4: Write Main Application - Model Loading

**Files:**
- Modify: `app.py`

**Step 1: Write model loading section**

Replace `app.py` with:

```python
#!/usr/bin/env python3
"""
English <-> Swahili Bidirectional Speech & Text Translator
Deployed on Hugging Face Spaces with Gradio UI
"""

import os
import tempfile
import numpy as np
import gradio as gr
import torch
from config import Config, logger

# Resolve device
DEVICE = Config.get_device()
WHISPER_MODEL_SIZE = Config.WHISPER_MODEL_SIZE

logger.info(f"Device: {DEVICE}, Whisper: {WHISPER_MODEL_SIZE}")

# Language mapping
LANG_CODES = {
    "en": "English",
    "sw": "Kiswahili"
}

# Global model references
whisper_model = None
translator_en_sw = None
translator_sw_en = None
tts_model = None


def load_models():
    """Load all models on startup with error handling."""
    global whisper_model, translator_en_sw, translator_sw_en, tts_model

    # 1. Load faster-whisper for STT
    try:
        logger.info(f"Loading faster-whisper model: {WHISPER_MODEL_SIZE}")
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=DEVICE,
            compute_type="float16" if DEVICE == "cuda" else "int8"
        )
        logger.info("Whisper model loaded")
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")
        whisper_model = None

    # 2. Load MarianMT translation models
    try:
        logger.info("Loading translation models (MarianMT)...")
        from transformers import MarianMTModel, MarianTokenizer

        translator_en_sw = {
            "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-sw"),
            "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-sw").to(DEVICE)
        }

        translator_sw_en = {
            "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-sw-en"),
            "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-sw-en").to(DEVICE)
        }
        logger.info("Translation models loaded")
    except Exception as e:
        logger.error(f"Failed to load translation models: {e}")
        translator_en_sw = None
        translator_sw_en = None

    # 3. Load Coqui TTS (XTTS-v2)
    try:
        logger.info("Loading TTS model (XTTS-v2)...")
        from TTS.api import TTS
        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
        logger.info("TTS model loaded")
    except Exception as e:
        logger.error(f"Failed to load TTS: {e}")
        tts_model = None
```

**Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add model loading for Swahili translator"
```

---

## Task 5: Write Core Functions - STT

**Files:**
- Modify: `app.py` (append after model loading)

**Step 1: Add STT function**

Append to `app.py`:

```python


def stt(audio_path, lang_hint="auto"):
    """
    Speech-to-text using faster-whisper.

    Args:
        audio_path: Path to audio file or tuple (sample_rate, numpy_array)
        lang_hint: "auto", "en", or "sw"

    Returns:
        (detected_lang, transcript_text)
    """
    if whisper_model is None:
        return "error", "STT model not loaded. Please check logs."

    try:
        import soundfile as sf

        # Handle Gradio audio input (tuple of sample_rate, data)
        if isinstance(audio_path, tuple):
            sr, audio_data = audio_path
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_data, sr)
                audio_path = tmp.name

        # Transcribe
        language = None if lang_hint == "auto" else lang_hint
        segments, info = whisper_model.transcribe(
            audio_path,
            language=language,
            beam_size=1,
            vad_filter=True
        )

        transcript = " ".join([seg.text for seg in segments]).strip()
        detected_lang = info.language if lang_hint == "auto" else lang_hint

        # Clean up temp file
        if isinstance(audio_path, str) and audio_path.startswith("/tmp"):
            try:
                os.remove(audio_path)
            except:
                pass

        logger.info(f"STT: {detected_lang} -> '{transcript[:50]}...'")
        return detected_lang, transcript

    except Exception as e:
        logger.error(f"STT error: {e}")
        return "error", f"Transcription failed: {str(e)}"
```

**Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add STT function"
```

---

## Task 6: Write Core Functions - Translation

**Files:**
- Modify: `app.py` (append)

**Step 1: Add translation function**

Append to `app.py`:

```python


def translate_text(text, src_lang, tgt_lang):
    """
    Translate text using MarianMT.

    Args:
        text: Source text
        src_lang: "en" or "sw"
        tgt_lang: "en" or "sw"

    Returns:
        Translated text string
    """
    if not text or not text.strip():
        return ""

    if src_lang == tgt_lang:
        return text

    # Select correct model
    if src_lang == "en" and tgt_lang == "sw":
        translator = translator_en_sw
    elif src_lang == "sw" and tgt_lang == "en":
        translator = translator_sw_en
    else:
        return f"Unsupported language pair: {src_lang} -> {tgt_lang}"

    if translator is None:
        return "Translation model not loaded."

    try:
        tokenizer = translator["tokenizer"]
        model = translator["model"]

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            translated = model.generate(**inputs, max_length=512)

        translation = tokenizer.decode(translated[0], skip_special_tokens=True)

        logger.info(f"Translation ({src_lang}->{tgt_lang}): '{text[:30]}...' -> '{translation[:30]}...'")
        return translation

    except Exception as e:
        logger.error(f"Translation error: {e}")
        return f"Translation failed: {str(e)}"
```

**Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add translation function"
```

---

## Task 7: Write Core Functions - TTS

**Files:**
- Modify: `app.py` (append)

**Step 1: Add TTS function**

Append to `app.py`:

```python


def tts_synthesize(text, lang):
    """
    Synthesize speech using XTTS-v2.

    Args:
        text: Text to synthesize
        lang: "en" or "sw"

    Returns:
        (sample_rate, audio_numpy_array) or None on error
    """
    if tts_model is None:
        logger.warning("TTS model not available")
        return None

    if not text or not text.strip():
        return None

    try:
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tts_model.tts_to_file(
                text=text,
                language=lang,
                file_path=tmp.name,
                speaker_wav=None,
                speed=1.0
            )

            audio_data, sr = sf.read(tmp.name)

            try:
                os.remove(tmp.name)
            except:
                pass

            logger.info(f"TTS synthesized: {lang}, {len(audio_data)} samples")
            return (sr, audio_data.astype(np.float32))

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None
```

**Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add TTS function"
```

---

## Task 8: Write Pipeline Functions

**Files:**
- Modify: `app.py` (append)

**Step 1: Add pipeline functions**

Append to `app.py`:

```python


def pipe_audio_to_audio(audio, src_lang, tgt_lang):
    """
    Full pipeline: Audio -> STT -> Translation -> TTS

    Returns:
        (detected_lang_str, transcript, translation, audio_output)
    """
    if audio is None:
        return "—", "No audio input", "", None

    # Step 1: STT
    detected_lang, transcript = stt(audio, src_lang)

    if detected_lang == "error":
        return "Error", transcript, "", None

    # Use detected language as source
    actual_src = detected_lang if detected_lang in ["en", "sw"] else "en"

    # Step 2: Translation
    translation = translate_text(transcript, actual_src, tgt_lang)

    # Step 3: TTS
    audio_output = tts_synthesize(translation, tgt_lang)

    detected_lang_str = LANG_CODES.get(actual_src, actual_src)

    return detected_lang_str, transcript, translation, audio_output


def pipe_text_to_audio(text, src_lang, tgt_lang):
    """
    Pipeline: Text -> Translation -> TTS

    Returns:
        (translation, audio_output)
    """
    if not text or not text.strip():
        return "No text input", None

    # Step 1: Translation
    translation = translate_text(text, src_lang, tgt_lang)

    # Step 2: TTS
    audio_output = tts_synthesize(translation, tgt_lang)

    return translation, audio_output
```

**Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add pipeline functions"
```

---

## Task 9: Write Gradio UI

**Files:**
- Modify: `app.py` (append)

**Step 1: Add Gradio UI**

Append to `app.py`:

```python


def build_ui():
    """Build the Gradio interface."""

    with gr.Blocks(title="English <-> Swahili Translator", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # English <-> Swahili Speech & Text Translator

        Speak or type in English or Swahili, get translations and natural speech output.

        **Privacy Notice**: All processing happens locally on this server. No data is sent to third parties.
        """)

        with gr.Tabs():
            # Tab A: Audio -> Audio
            with gr.Tab("Speak -> Translate -> Speak"):
                gr.Markdown("### Record or upload audio, get transcript, translation, and synthesized speech")

                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="numpy",
                            label="Input Audio"
                        )

                        with gr.Row():
                            src_lang_audio = gr.Dropdown(
                                choices=["auto", "en", "sw"],
                                value="auto",
                                label="Source Language"
                            )
                            tgt_lang_audio = gr.Dropdown(
                                choices=["en", "sw"],
                                value="sw",
                                label="Target Language"
                            )

                        audio_button = gr.Button("Transcribe + Translate + Speak", variant="primary")

                    with gr.Column():
                        detected_lang_out = gr.Textbox(label="Detected Language", interactive=False)
                        transcript_out = gr.Textbox(label="Transcript (Source)", lines=3, interactive=False)
                        translation_out = gr.Textbox(label="Translation (Target)", lines=3, interactive=False)
                        audio_output = gr.Audio(label="Synthesized Speech", type="numpy")

                audio_button.click(
                    fn=pipe_audio_to_audio,
                    inputs=[audio_input, src_lang_audio, tgt_lang_audio],
                    outputs=[detected_lang_out, transcript_out, translation_out, audio_output]
                )

            # Tab B: Text -> Audio
            with gr.Tab("Type -> Translate -> Speak"):
                gr.Markdown("### Enter text, get translation and synthesized speech")

                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Input Text",
                            lines=4,
                            placeholder="Type your message in English or Swahili..."
                        )

                        with gr.Row():
                            src_lang_text = gr.Dropdown(
                                choices=["en", "sw"],
                                value="en",
                                label="Source Language"
                            )
                            tgt_lang_text = gr.Dropdown(
                                choices=["en", "sw"],
                                value="sw",
                                label="Target Language"
                            )

                        text_button = gr.Button("Translate + Speak", variant="primary")

                    with gr.Column():
                        translation_text_out = gr.Textbox(label="Translation", lines=4, interactive=False)
                        audio_text_output = gr.Audio(label="Synthesized Speech", type="numpy")

                text_button.click(
                    fn=pipe_text_to_audio,
                    inputs=[text_input, src_lang_text, tgt_lang_text],
                    outputs=[translation_text_out, audio_text_output]
                )

                gr.Examples(
                    examples=[
                        ["Hello, how are you today?", "en", "sw"],
                        ["Where is the nearest hospital?", "en", "sw"],
                        ["Habari za asubuhi", "sw", "en"],
                        ["Ninahitaji msaada", "sw", "en"],
                        ["Thank you very much", "en", "sw"]
                    ],
                    inputs=[text_input, src_lang_text, tgt_lang_text],
                    label="Example Phrases"
                )

        # Advanced Settings
        with gr.Accordion("Advanced Settings & Info", open=False):
            gr.Markdown(f"""
            **Current Configuration:**
            - Whisper Model: `{WHISPER_MODEL_SIZE}`
            - Device: `{DEVICE}`
            - Translation: MarianMT (Helsinki-NLP opus-mt)
            - TTS: Coqui XTTS-v2

            **Performance Notes:**
            - CPU: Expect 15-40s end-to-end latency
            - GPU (T4): Expect 5-15s end-to-end latency

            **Supported Languages:**
            - English (en)
            - Kiswahili / Swahili (sw)
            """)

        gr.Markdown("""
        ---
        **Tips:**
        - For best STT results, speak clearly and minimize background noise
        - Translation quality is optimized for conversational text
        - TTS synthesis may take 10-20 seconds on CPU
        """)

    return app
```

**Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add Gradio UI"
```

---

## Task 10: Write Main Entry Point

**Files:**
- Modify: `app.py` (append)

**Step 1: Add main entry point**

Append to `app.py`:

```python


if __name__ == "__main__":
    logger.info("Starting English <-> Swahili Translator...")

    # Load models
    load_models()

    # Check critical models
    if whisper_model is None or translator_en_sw is None or translator_sw_en is None:
        logger.error("Critical models failed to load. App may have limited functionality.")

    if tts_model is None:
        logger.warning("TTS model failed to load. Audio output will be unavailable.")

    # Build and launch UI
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=Config.SERVER_PORT,
        share=False
    )
```

**Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add main entry point"
```

---

## Task 11: Update README

**Files:**
- Modify: `README.md`

**Step 1: Replace README.md**

```markdown
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
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for Swahili translator"
```

---

## Task 12: Test Locally

**Step 1: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 2: Run app**

```bash
python app.py
```

Expected: App starts, models load (may take a few minutes), Gradio UI available at http://localhost:7860

**Step 3: Test text translation**

In the "Type -> Translate -> Speak" tab:
- Input: "Hello, how are you?"
- Source: en
- Target: sw
- Click "Translate + Speak"

Expected: Translation appears (e.g., "Habari, u hali gani?"), audio plays

---

## Task 13: Deploy to HuggingFace

**Step 1: Verify HuggingFace remote**

```bash
git remote -v
```

Expected: Shows HuggingFace Spaces remote

**Step 2: Push to HuggingFace**

```bash
git push
```

Expected: Code pushed, Space rebuilds automatically

**Step 3: Verify deployment**

Visit your HuggingFace Space URL and test the translator.

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Update dependencies |
| 2 | Simplify configuration |
| 3 | Remove unused files |
| 4-10 | Write main application |
| 11 | Update README |
| 12 | Test locally |
| 13 | Deploy to HuggingFace |
