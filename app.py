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
from feedback_storage import get_feedback_storage
from config import Config, logger, TranslationError, TTSError
import numpy as np
from datetime import datetime

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

# Domain metadata with emojis
DOMAIN_EMOJIS = {
    "Greetings": "👋",
    "Travel": "✈️",
    "Food": "🍔",
    "Work": "💼",
    "Health": "⚕️",
    "Education": "🎓",
    "Social": "👥",
    "Emotions": "❤️",
    "Numbers": "🔢",
    "Shopping": "🛒",
    "Time": "⏰",
    "Family": "👨‍👩‍👧‍👦",
    "Questions": "❓",
    "Activities": "⚽",
    "General": "📝",
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
        gr.Markdown("""
        **Organized by Domain** — Browse curated phrases grouped by real-world contexts.
        """)

        with gr.Row():
            # Domain selection with emoji labels
            domain_choices = [f"{DOMAIN_EMOJIS.get(d, '📝')} {d}" for d in sorted(PHRASE_PACKS.keys())]
            domain_choice_map = {label: domain for label, domain in zip(domain_choices, sorted(PHRASE_PACKS.keys()))}

            pack = gr.Dropdown(
                choices=domain_choices,
                value=domain_choices[0] if domain_choices else None,
                label="📚 Learning Domain",
                info="Choose a domain to explore phrases"
            )

        with gr.Row():
            lang_for_pack = gr.Dropdown(
                choices=LANGS,
                value="en",
                label="Phrase Language"
            )
            tgt_pack = gr.Dropdown(
                choices=LANGS,
                value="rw",
                label="Translate To"
            )

        phrase = gr.Dropdown(
            choices=[],
            label="Select a Phrase from This Domain",
            interactive=True
        )

        translated = gr.Textbox(label="Translation", lines=2, interactive=False)

        with gr.Row():
            btn_translate2 = gr.Button("🔄 Translate", variant="primary")
            btn_tts2 = gr.Button("🔊 Speak Translation")

        audio2 = gr.Audio(label="Synthesized Speech", type="numpy")

        def update_phrases_with_emoji(pack_with_emoji: str, lang: str):
            """Update available phrases based on domain and language."""
            try:
                # Extract domain from emoji label
                domain = domain_choice_map.get(pack_with_emoji, "General")

                if domain in PHRASE_PACKS and lang in PHRASE_PACKS[domain]:
                    choices = PHRASE_PACKS[domain][lang]
                    if choices:
                        return gr.update(choices=choices, value=choices[0])
                return gr.update(choices=[], value=None)
            except Exception as e:
                logger.error(f"Failed to update phrases: {e}")
                return gr.update(choices=[], value=None)

        pack.change(update_phrases_with_emoji, inputs=[pack, lang_for_pack], outputs=[phrase], api_name=False)
        lang_for_pack.change(update_phrases_with_emoji, inputs=[pack, lang_for_pack], outputs=[phrase], api_name=False)

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

    with gr.Tab("🎤 Feedback"):
        gr.Markdown("""
        ## Share Your Feedback

        Record your pronunciation practice or general feedback about the app.
        Your recordings are encrypted and securely stored for learning analytics.

        **Privacy Notice:** Audio is encrypted with AES-256 before storage.
        """)

        with gr.Row():
            feedback_type = gr.Radio(
                choices=["Pronunciation Practice", "General Feedback"],
                value="Pronunciation Practice",
                label="Feedback Type"
            )

        with gr.Row():
            feedback_domain = gr.Dropdown(
                choices=sorted(corpus.categories) if corpus.categories else ["General"],
                value="General",
                label="Learning Domain (Optional)"
            )

        feedback_phrase = gr.Textbox(
            label="Phrase Being Practiced (Optional)",
            placeholder="Enter the phrase you're practicing...",
            lines=2
        )

        feedback_notes = gr.Textbox(
            label="Additional Notes (Optional)",
            placeholder="Any comments or context for this recording?",
            lines=2
        )

        gr.Markdown("### Record Audio")
        audio_input = gr.Audio(
            label="🎙️ Click to Record",
            type="numpy",
            sources=["microphone"]
        )

        feedback_status = gr.Textbox(
            label="Status",
            interactive=False,
            lines=2
        )

        def process_feedback(
            audio_data: Tuple[int, np.ndarray],
            feedback_type_val: str,
            domain_val: str,
            phrase_val: str,
            notes_val: str
        ) -> str:
            """Process and store feedback."""
            try:
                if audio_data is None:
                    return "⚠️ No audio recorded. Please record something."

                sample_rate, audio_array = audio_data

                # Convert to bytes (16-bit PCM)
                audio_int16 = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

                # Calculate duration and quality score
                duration = len(audio_array) / sample_rate

                # Simple quality metric: check for clipping/distortion
                max_val = np.max(np.abs(audio_array))
                quality_score = min(1.0, max(0.0, 1.0 - (max(0, max_val - 0.95) * 2)))

                # Map feedback type
                feedback_type_mapped = "pronunciation" if "Pronunciation" in feedback_type_val else "general"

                # Submit to storage
                feedback_storage = get_feedback_storage()
                feedback_id = feedback_storage.submit_feedback(
                    audio_bytes=audio_bytes,
                    feedback_type=feedback_type_mapped,
                    domain=domain_val if domain_val != "General" else None,
                    phrase_text=phrase_val if phrase_val.strip() else None,
                    duration_seconds=duration,
                    audio_quality_score=quality_score,
                    notes=notes_val if notes_val.strip() else None
                )

                return (
                    f"✅ **Feedback Recorded Successfully!**\n\n"
                    f"**Recording ID:** {feedback_id}\n"
                    f"**Duration:** {duration:.1f} seconds\n"
                    f"**Quality Score:** {quality_score:.1%}\n"
                    f"**Type:** {feedback_type_mapped}\n"
                    f"**Domain:** {domain_val}\n\n"
                    f"Thank you for your feedback!"
                )

            except Exception as e:
                logger.error(f"Failed to process feedback: {e}")
                return f"❌ Error: {str(e)}"

        btn_submit_feedback = gr.Button("💾 Submit Feedback", variant="primary")
        btn_submit_feedback.click(
            process_feedback,
            inputs=[audio_input, feedback_type, feedback_domain, feedback_phrase, feedback_notes],
            outputs=[feedback_status]
        )

    with gr.Tab("🗂️ Learning Domains"):
        gr.Markdown("""
        ## 📚 Domain-Based Learning

        Phrases are intelligently organized into domains for context-based learning.
        Each domain focuses on a real-world topic with curated vocabulary and examples.
        """)

        # Get corpus stats
        try:
            corpus_obj = get_corpus()
            stats = corpus_obj.get_stats()

            # Display domain summary
            domain_info = []
            for domain in sorted(stats.get("categories", [])):
                count = stats.get("by_domain", {}).get(domain, 0)
                emoji = DOMAIN_EMOJIS.get(domain, "📝")
                domain_info.append(f"{emoji} **{domain}** — {count} phrases")

            gr.Markdown("### Available Domains\n\n" + "\n\n".join(domain_info))

            # Overall statistics
            gr.Markdown(f"""
            ### Corpus Statistics

            - **Total Phrases**: {stats['total_phrases']}
            - **Domains**: {len(stats['categories'])}
            - **Languages**: {', '.join(stats['languages'])}
            - **Difficulty Distribution**:
              - Beginner: {stats['by_difficulty'].get('beginner', 0)} phrases
              - Intermediate: {stats['by_difficulty'].get('intermediate', 0)} phrases
              - Advanced: {stats['by_difficulty'].get('advanced', 0)} phrases
            """)
        except Exception as e:
            logger.error(f"Failed to load domain statistics: {e}")
            gr.Markdown("⚠️ Could not load domain statistics")

    with gr.Tab("ℹ️ About"):
        gr.Markdown(
            """
            ## About Iga TTS

            **Iga** (Kinyarwanda for "learn") is an AI-powered language learning platform.

            ### Features
            - 🔄 **Offline Translation** — MarianMT models for en↔rw, fr↔rw
            - 🔊 **Text-to-Speech** — Natural pronunciation with Bark
            - 🎯 **Two Learning Modes** — Rwanda Mode & Diaspora Mode
            - 🗂️ **Domain-Based Learning** — 15 semantic domains (Travel, Food, Work, etc.)
            - 📦 **Phrase Packs** — 1,000+ curated phrases organized by context
            - 🎮 **Gamification** — XP and streaks (prototype)
            - 📚 **Spaced Repetition Ready** — Infrastructure for SRS scheduling

            ### Technology
            - **Translation:** Helsinki-NLP MarianMT
            - **TTS:** Suno Bark (small model)
            - **Framework:** Gradio + Transformers
            - **Corpus:** DuckDB with intelligent domain classification
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
        share=Config.SHARE
    )



