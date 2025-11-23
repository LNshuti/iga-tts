"""
Iga TTS - Learn Kinyarwanda
Enterprise-grade multilingual learning app with translation and text-to-speech.
"""
import gradio as gr
from typing import Tuple, Optional, Any
import traceback

from translation import translate
from tts import synthesize
from corpus import get_corpus
from config import Config, logger, TranslationError, TTSError

# Validate configuration on startup
if not Config.validate():
    raise RuntimeError("Configuration validation failed")

logger.info("Starting Iga TTS application")
logger.info(f"TTS Model: {Config.TTS_MODEL}")
logger.info(f"Device: {Config.DEVICE or 'CPU'}")

LANGS = ["en", "fr", "rw"]
LANG_LABELS = {"en": "English", "fr": "Français", "rw": "Kinyarwanda"}

MODES = {
    "Rwanda Mode (Kinyarwanda → EN/FR)": ("rw", "en"),
    "Diaspora Mode (EN/FR → Kinyarwanda)": ("en", "rw"),
}


def do_translate(text: str, src: str, tgt: str) -> str:
    """
    Translate text with error handling.

    Returns error message on failure instead of crashing.
    """
    try:
        if not text or not text.strip():
            return ""

        result = translate(text, src, tgt)
        logger.info(f"Translation successful: {text[:30]}... -> {result[:30]}...")
        return result

    except TranslationError as e:
        error_msg = f"Translation error: {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}"

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Translation failed: {e}", exc_info=True)
        return f"❌ {error_msg}"


def do_tts(text: str) -> Optional[Tuple[int, Any]]:
    """
    Generate speech with error handling.

    Returns None on failure with error logging.
    """
    try:
        if not text or not text.strip():
            return None

        # Don't synthesize error messages
        if text.startswith("❌"):
            logger.warning("Skipping TTS for error message")
            return None

        sr, audio = synthesize(text)
        logger.info(f"TTS synthesis successful for: {text[:30]}...")
        return (sr, audio)

    except TTSError as e:
        logger.error(f"TTS error: {e}")
        gr.Warning(f"Speech synthesis failed: {str(e)}")
        return None

    except Exception as e:
        logger.error(f"TTS failed: {e}", exc_info=True)
        gr.Warning(f"Unexpected error during speech synthesis: {str(e)}")
        return None


def on_mode_change(mode_label: str) -> Tuple[str, str]:
    """Update source and target languages based on mode."""
    if "Rwanda Mode" in mode_label:
        return "rw", "en"
    return "en", "rw"


# Load corpus on startup
try:
    corpus = get_corpus()
    PHRASE_PACKS = corpus.get_by_category_dict()
    logger.info(f"Loaded corpus with {len(corpus.phrases)} phrases")
    logger.info(f"Categories: {corpus.categories}")
except Exception as e:
    logger.error(f"Failed to load corpus: {e}", exc_info=True)
    # Fallback to hardcoded phrases
    from phrases import PHRASE_PACKS
    logger.warning("Using fallback hardcoded phrases")


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🌍 Learn Kinyarwanda — Iga TTS

        **Gamified language learning with AI-powered translation and text-to-speech.**

        Switch modes, translate phrases, and listen to natural pronunciation.
        """
    )

    with gr.Tab("📚 Learn"):
        mode = gr.Dropdown(
            choices=list(MODES.keys()),
            value=list(MODES.keys())[0],
            label="Learning Mode",
            info="Choose your learning context"
        )

        with gr.Row():
            src = gr.Dropdown(
                choices=LANGS,
                value="rw",
                label="Source Language",
                info="Auto-updated by mode"
            )
            tgt = gr.Dropdown(
                choices=LANGS,
                value="en",
                label="Target Language",
                info="Auto-updated by mode"
            )

        mode.change(on_mode_change, inputs=mode, outputs=[src, tgt], api_name=False)

        with gr.Row():
            inp = gr.Textbox(
                label="Input Text",
                placeholder="Type a phrase to translate…",
                lines=2
            )
            out = gr.Textbox(
                label="Translated Text",
                lines=2
            )

        with gr.Row():
            btn_translate = gr.Button("🔄 Translate", variant="primary")
            btn_tts = gr.Button("🔊 Speak Translation")

        audio = gr.Audio(label="Synthesized Speech", type="numpy")

        # Simple XP/Streak mock (non-persistent)
        xp = gr.State(0)
        streak = gr.State(1)
        xp_display = gr.Markdown("**XP:** 0 | **Streak:** 1 day")

        def on_translate(text: str, src_lang: str, tgt_lang: str, xp_val: int, streak_val: int):
            """Handle translation with XP tracking."""
            tr = do_translate(text, src_lang, tgt_lang)
            # Award XP only for successful translations (not error messages)
            if tr and not tr.startswith("❌"):
                xp_val += 5
            return tr, f"**XP:** {xp_val} | **Streak:** {streak_val} day{'s' if streak_val > 1 else ''}", xp_val, streak_val

        btn_translate.click(
            on_translate,
            inputs=[inp, src, tgt, xp, streak],
            outputs=[out, xp_display, xp, streak]
        )
        btn_tts.click(do_tts, inputs=[out], outputs=[audio])

    with gr.Tab("📦 Phrase Packs"):
        gr.Markdown("**Quick-start with curated phrases from the corpus**")

        pack = gr.Dropdown(
            choices=list(PHRASE_PACKS.keys()),
            value=list(PHRASE_PACKS.keys())[0] if PHRASE_PACKS else None,
            label="Category"
        )
        lang_for_pack = gr.Dropdown(
            choices=LANGS,
            value="en",
            label="Phrase Language"
        )
        phrase = gr.Dropdown(
            choices=[],
            label="Select Phrase"
        )

        with gr.Row():
            tgt_pack = gr.Dropdown(
                choices=LANGS,
                value="rw",
                label="Translate To"
            )

        translated = gr.Textbox(label="Translation", lines=2)

        with gr.Row():
            btn_translate2 = gr.Button("🔄 Translate", variant="primary")
            btn_tts2 = gr.Button("🔊 Speak")

        audio2 = gr.Audio(label="Synthesized Speech", type="numpy")

        def update_phrases(cat: str, lang: str):
            """Update available phrases based on category and language."""
            try:
                if cat in PHRASE_PACKS and lang in PHRASE_PACKS[cat]:
                    choices = PHRASE_PACKS[cat][lang]
                    if choices:
                        return gr.update(choices=choices, value=choices[0])
                return gr.update(choices=[], value=None)
            except Exception as e:
                logger.error(f"Failed to update phrases: {e}")
                return gr.update(choices=[], value=None)

        pack.change(update_phrases, inputs=[pack, lang_for_pack], outputs=[phrase], api_name=False)
        lang_for_pack.change(update_phrases, inputs=[pack, lang_for_pack], outputs=[phrase], api_name=False)

        def translate_pack_phrase(lang: str, phr: str, tgt_lang: str) -> str:
            """Translate selected phrase."""
            if not phr:
                return ""
            return do_translate(phr, lang, tgt_lang)

        btn_translate2.click(
            translate_pack_phrase,
            inputs=[lang_for_pack, phrase, tgt_pack],
            outputs=[translated]
        )
        btn_tts2.click(do_tts, inputs=[translated], outputs=[audio2])

    with gr.Tab("ℹ️ About"):
        gr.Markdown(
            """
            ## About Iga TTS

            **Iga** (Kinyarwanda for "learn") is an AI-powered language learning platform.

            ### Features
            - 🔄 **Offline Translation** — MarianMT models for en↔rw, fr↔rw
            - 🔊 **Text-to-Speech** — Natural pronunciation with Bark
            - 🎯 **Two Learning Modes** — Rwanda Mode & Diaspora Mode
            - 📦 **Phrase Packs** — Curated phrases by category
            - 🎮 **Gamification** — XP and streaks (prototype)

            ### Technology
            - **Translation:** Helsinki-NLP MarianMT
            - **TTS:** Suno Bark (small model)
            - **Framework:** Gradio + Transformers
            - **Deployment:** Hugging Face Spaces

            ### Limitations
            - Kinyarwanda pronunciation may be imperfect (Bark limitation)
            - French↔Kinyarwanda uses English bridge translation
            - XP/streaks are non-persistent (this is a prototype)

            ---
            """
        )

        # Show corpus stats if available
        try:
            stats = corpus.get_stats()
            gr.Markdown(
                f"""
                ### Corpus Statistics
                - **Total Phrases:** {stats['total_phrases']}
                - **Categories:** {', '.join(stats['categories'])}
                - **Languages:** {', '.join(stats['languages'])}
                """
            )
        except:
            pass

    demo.queue(max_size=Config.QUEUE_MAX_SIZE)

if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=Config.SERVER_PORT,
        share=Config.SHARE,
        show_api=False  # Disable API docs to avoid gradio_client bug
    )
