import gradio as gr
from translation import translate
from tts import synthesize
from phrases import PHRASE_PACKS

LANGS = ["en", "fr", "rw"]
LANG_LABELS = {"en": "English", "fr": "Français", "rw": "Kinyarwanda"}

MODES = {
    "Rwanda Mode (Kinyarwanda -> EN/FR)": ("rw", "en"),
    "Diaspora Mode (EN/FR -> Kinyarwanda)": ("en", "rw"),
}


def do_translate(text, src, tgt):
    return translate(text, src, tgt)


def do_tts(text):
    if not text:
        return None
    sr, audio = synthesize(text)
    return (sr, audio)


def on_mode_change(mode_label):
    if "Rwanda Mode" in mode_label:
        return "rw", "en"
    return "en", "rw"

with gr.Blocks(title="Learn Kinyarwanda — Igisha.org Prototype") as demo:
    gr.Markdown("Learn Kinyarwanda – Gamified Language Learning Prototype. Switch modes, translate, and listen.")

    with gr.Tab("Learn"):
        mode = gr.Dropdown(list(MODES.keys()), value=list(MODES.keys())[0], label="Mode")
        src = gr.Dropdown(LANGS, value="rw", label="Source language", info="Auto-updated by mode")
        tgt = gr.Dropdown(LANGS, value="en", label="Target language", info="Auto-updated by mode")
        mode.change(on_mode_change, inputs=mode, outputs=[src, tgt])

        with gr.Row():
            inp = gr.Textbox(label="Input text", placeholder="Type a phrase…")
            out = gr.Textbox(label="Translated text")
        with gr.Row():
            btn_translate = gr.Button("Translate")
            btn_tts = gr.Button("Speak Target")
        audio = gr.Audio(label="Synthesized Audio", type="numpy")

        # Simple XP/Streak mock
        xp = gr.State(0)
        streak = gr.State(1)
        xp_display = gr.Markdown("XP: 0 | Streak: 1")

        def on_translate(text, src_lang, tgt_lang, xp_val, streak_val):
            tr = do_translate(text, src_lang, tgt_lang)
            xp_val += 5 if tr else 0
            return tr, f"XP: {xp_val} | Streak: {streak_val}", xp_val, streak_val

        btn_translate.click(on_translate, inputs=[inp, src, tgt, xp, streak], outputs=[out, xp_display, xp, streak])
        btn_tts.click(do_tts, inputs=[out], outputs=[audio])

    with gr.Tab("Phrase Packs"):
        pack = gr.Dropdown(list(PHRASE_PACKS.keys()), value="Greetings", label="Category")
        lang_for_pack = gr.Dropdown(LANGS, value="en", label="Pack language")
        phrase = gr.Dropdown([], label="Phrase")
        translated = gr.Textbox(label="Translated")
        audio2 = gr.Audio(label="Synthesized Audio", type="numpy")
        btn_translate2 = gr.Button("Translate")
        btn_tts2 = gr.Button("Speak")

        def update_phrases(cat, lang):
            return gr.update(choices=PHRASE_PACKS[cat][lang], value=PHRASE_PACKS[cat][lang][0])

        pack.change(update_phrases, inputs=[pack, lang_for_pack], outputs=[phrase])
        lang_for_pack.change(update_phrases, inputs=[pack, lang_for_pack], outputs=[phrase])

        def translate_pack_phrase(cat, lang, phr, tgt_lang):
            return do_translate(phr, lang, tgt_lang)

        btn_translate2.click(translate_pack_phrase, inputs=[pack, lang_for_pack, phrase, tgt], outputs=[translated])
        btn_tts2.click(do_tts, inputs=[translated], outputs=[audio2])

    with gr.Tab("About"):
        gr.Markdown(
            "Igisha.org – Learn Kinyarwanda. This prototype uses Transformers (Bark) for TTS and MarianMT for translation."
        )

    demo.queue(max_size=32)

if __name__ == "__main__":
    demo.launch()
