#!/usr/bin/env python3
"""
Simple Audio Recording & Trimming Tool
Record audio from microphone, trim it, and download as .wav file
"""

import os
import re
import tempfile
import zipfile
import gradio as gr
import soundfile as sf


def update_sliders(audio):
    """
    Update slider ranges based on recorded audio duration.

    Args:
        audio: Tuple of (sample_rate, audio_data) from Gradio Audio component

    Returns:
        Updated slider components and info text
    """
    if audio is None:
        return (
            gr.Slider(maximum=1, value=0, interactive=False),
            gr.Slider(maximum=1, value=1, interactive=False),
            "No audio recorded yet"
        )

    sample_rate, audio_data = audio
    duration = len(audio_data) / sample_rate

    return (
        gr.Slider(maximum=duration, value=0, interactive=True),
        gr.Slider(maximum=duration, value=duration, interactive=True),
        f"Total duration: {duration:.2f}s"
    )


def trim_audio(audio, start_time, end_time):
    """
    Trim audio between start_time and end_time.

    Args:
        audio: Tuple of (sample_rate, audio_data) from Gradio Audio component
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        Path to trimmed .wav file
    """
    if audio is None:
        return None

    sample_rate, audio_data = audio

    # Convert times to sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Validate range
    if start_sample >= end_sample:
        return None

    # Trim audio
    trimmed_data = audio_data[start_sample:end_sample]

    # Save to temporary file
    output_path = tempfile.mktemp(suffix=".wav")
    sf.write(output_path, trimmed_data, sample_rate)

    return output_path


def update_selection_info(audio, start_time, end_time):
    """
    Update the selection info text.

    Returns:
        Info string showing selected duration
    """
    if audio is None:
        return "No audio recorded yet"

    sample_rate, audio_data = audio
    total_duration = len(audio_data) / sample_rate
    selected_duration = end_time - start_time

    return f"Selected: {selected_duration:.2f}s / Total: {total_duration:.2f}s"


def parse_batch_list(text):
    """Parse a batch list text into a list of word entries.

    Expected format has lines like:
        ID: 140
        Kinyarwanda: Inzu
        English: House

    Args:
        text: Raw batch list text with ID/Kinyarwanda/English line groups

    Returns:
        Tuple of (entries list, status message)
    """
    if not text or not text.strip():
        return [], "No batch list provided"

    entries = []
    current = {}

    for line in text.strip().split("\n"):
        line = line.strip()

        id_match = re.search(r"ID:\s*(\d+)", line)
        if id_match:
            current["id"] = int(id_match.group(1))
            continue

        kiny_match = re.match(r"Kinyarwanda:\s*(.+)", line)
        if kiny_match:
            current["kinyarwanda"] = kiny_match.group(1).strip()
            continue

        eng_match = re.match(r"English:\s*(.+)", line)
        if eng_match:
            current["english"] = eng_match.group(1).strip()
            if "id" in current and "kinyarwanda" in current:
                entries.append(dict(current))
                current = {}
            continue

    if not entries:
        return [], "Could not parse any entries. Check format."

    count = len(entries)
    return entries, f"Parsed {count} {'word' if count == 1 else 'words'}"


def get_current_word_display(entries, current_index, trims):
    """Return display string for the current word in the wizard."""
    if not entries:
        return "No words loaded. Paste a batch list and click Parse."

    if current_index >= len(entries):
        confirmed = len(trims)
        return f"All done! {confirmed}/{len(entries)} words trimmed. Click Export."

    entry = entries[current_index]
    confirmed = len(trims)
    total = len(entries)
    trimmed_mark = " [TRIMMED]" if entry["id"] in trims else ""

    return (
        f"**Word {current_index + 1} of {total}** (confirmed: {confirmed}) — "
        f"ID: {entry['id']} — {entry['kinyarwanda']} ({entry['english']}){trimmed_mark}"
    )


def confirm_and_next(audio, start_time, end_time, entries, trims, current_index):
    """Save trim points for current word, advance to next."""
    if audio is None or not entries:
        return trims, current_index, "No audio or entries loaded"

    if current_index >= len(entries):
        return trims, current_index, get_current_word_display(entries, current_index, trims)

    entry = entries[current_index]
    trims[entry["id"]] = (start_time, end_time)

    new_index = current_index + 1
    return trims, new_index, get_current_word_display(entries, new_index, trims)


def go_back(entries, trims, current_index):
    """Go back to previous word."""
    if current_index <= 0:
        return current_index, get_current_word_display(entries, 0, trims), None, None

    new_index = current_index - 1
    entry = entries[new_index]

    if entry["id"] in trims:
        start, end = trims[entry["id"]]
        return new_index, get_current_word_display(entries, new_index, trims), start, end

    return new_index, get_current_word_display(entries, new_index, trims), None, None


def preview_trim(audio, start_time, end_time):
    """Return trimmed audio for playback preview."""
    if audio is None:
        return None

    sample_rate, audio_data = audio
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    if start_sample >= end_sample:
        return None

    return (sample_rate, audio_data[start_sample:end_sample])


def export_all(audio, entries, trims):
    """Export all confirmed trims as a zip of {ID}.wav files."""
    if audio is None or not entries or not trims:
        return None

    sample_rate, audio_data = audio
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "trimmed_words.zip")

    with zipfile.ZipFile(zip_path, "w") as zf:
        for entry in entries:
            word_id = entry["id"]
            if word_id not in trims:
                continue

            start_time, end_time = trims[word_id]
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            if start_sample >= end_sample:
                continue

            trimmed = audio_data[start_sample:end_sample]
            wav_path = os.path.join(tmp_dir, f"{word_id}.wav")
            sf.write(wav_path, trimmed, sample_rate)
            zf.write(wav_path, f"{word_id}.wav")
            os.remove(wav_path)

    return zip_path


def build_ui():
    """Build the Gradio interface."""

    with gr.Blocks(title="Audio Recording & Trimming Tool") as app:
        gr.Markdown("# Audio Recording & Trimming Tool")

        with gr.Tabs():
            with gr.Tab("Single Trim"):
                gr.Markdown("Record audio, trim it, and download as a .wav file.")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 1. Record Audio")
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="Record Audio",
                            buttons=[]
                        )

                        gr.Markdown("### 2. Trim Audio")
                        info_text = gr.Textbox(
                            value="No audio recorded yet",
                            label="Audio Info",
                            interactive=False
                        )

                        start_slider = gr.Slider(
                            minimum=0, maximum=1, value=0, step=0.01,
                            label="Start Time (seconds)", interactive=False
                        )

                        end_slider = gr.Slider(
                            minimum=0, maximum=1, value=1, step=0.01,
                            label="End Time (seconds)", interactive=False
                        )

                        selection_info = gr.Textbox(
                            value="No audio recorded yet",
                            label="Selection", interactive=False
                        )

                        trim_button = gr.Button("Trim & Download", variant="primary", size="lg")

                    with gr.Column():
                        gr.Markdown("### 3. Download")
                        audio_output = gr.Audio(
                            label="Trimmed Audio (Right-click to download)",
                            type="filepath"
                        )

                # Single trim event handlers
                audio_input.change(
                    fn=update_sliders,
                    inputs=[audio_input],
                    outputs=[start_slider, end_slider, info_text]
                )
                start_slider.change(
                    fn=update_selection_info,
                    inputs=[audio_input, start_slider, end_slider],
                    outputs=[selection_info]
                )
                end_slider.change(
                    fn=update_selection_info,
                    inputs=[audio_input, start_slider, end_slider],
                    outputs=[selection_info]
                )
                trim_button.click(
                    fn=trim_audio,
                    inputs=[audio_input, start_slider, end_slider],
                    outputs=[audio_output]
                )

            with gr.Tab("Batch Trim"):
                gr.Markdown("Record all words in one take, paste your batch list, then trim each word.")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1. Paste Batch List")
                        batch_text = gr.Textbox(
                            label="Batch List",
                            placeholder="Paste your batch list here...",
                            lines=10
                        )
                        parse_button = gr.Button("Parse List", variant="secondary")
                        parse_status = gr.Textbox(label="Parse Status", interactive=False)

                        gr.Markdown("#### 2. Record All Words")
                        batch_audio = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="Record All Words",
                            buttons=[]
                        )
                        batch_audio_info = gr.Textbox(
                            value="No audio recorded yet",
                            label="Recording Info",
                            interactive=False
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 3. Trim Each Word")
                        word_display = gr.Markdown(
                            "No words loaded. Paste a batch list and click Parse."
                        )

                        batch_start = gr.Slider(
                            minimum=0, maximum=1, value=0, step=0.01,
                            label="Start Time (seconds)", interactive=False
                        )
                        batch_end = gr.Slider(
                            minimum=0, maximum=1, value=1, step=0.01,
                            label="End Time (seconds)", interactive=False
                        )
                        batch_selection = gr.Textbox(
                            value="No audio recorded yet",
                            label="Selection", interactive=False
                        )

                        preview_audio = gr.Audio(label="Preview", type="numpy")

                        with gr.Row():
                            back_button = gr.Button("Back", variant="secondary")
                            preview_button = gr.Button("Preview", variant="secondary")
                            confirm_button = gr.Button("Confirm & Next", variant="primary")

                        gr.Markdown("#### 4. Export")
                        export_button = gr.Button("Export All as Zip", variant="primary", size="lg")
                        zip_output = gr.File(label="Download Zip")

                # Gradio State for batch wizard
                entries_state = gr.State([])
                trims_state = gr.State({})
                index_state = gr.State(0)

                # Batch event handlers
                def on_parse(text):
                    entries, msg = parse_batch_list(text)
                    display = get_current_word_display(entries, 0, {})
                    return entries, {}, 0, msg, display

                parse_button.click(
                    fn=on_parse,
                    inputs=[batch_text],
                    outputs=[entries_state, trims_state, index_state, parse_status, word_display]
                )

                batch_audio.change(
                    fn=update_sliders,
                    inputs=[batch_audio],
                    outputs=[batch_start, batch_end, batch_audio_info]
                )

                batch_start.change(
                    fn=update_selection_info,
                    inputs=[batch_audio, batch_start, batch_end],
                    outputs=[batch_selection]
                )
                batch_end.change(
                    fn=update_selection_info,
                    inputs=[batch_audio, batch_start, batch_end],
                    outputs=[batch_selection]
                )

                preview_button.click(
                    fn=preview_trim,
                    inputs=[batch_audio, batch_start, batch_end],
                    outputs=[preview_audio]
                )

                def on_confirm(audio, start, end, entries, trims, idx):
                    trims, new_idx, display = confirm_and_next(
                        audio, start, end, entries, trims, idx
                    )
                    return trims, new_idx, display

                confirm_button.click(
                    fn=on_confirm,
                    inputs=[batch_audio, batch_start, batch_end,
                            entries_state, trims_state, index_state],
                    outputs=[trims_state, index_state, word_display]
                )

                def on_back(entries, trims, idx):
                    new_idx, display, start, end = go_back(entries, trims, idx)
                    if start is not None:
                        return new_idx, display, gr.Slider(value=start), gr.Slider(value=end)
                    return new_idx, display, gr.Slider(), gr.Slider()

                back_button.click(
                    fn=on_back,
                    inputs=[entries_state, trims_state, index_state],
                    outputs=[index_state, word_display, batch_start, batch_end]
                )

                export_button.click(
                    fn=export_all,
                    inputs=[batch_audio, entries_state, trims_state],
                    outputs=[zip_output]
                )

    return app


if __name__ == "__main__":
    print("Starting Audio Recording & Trimming Tool...")

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft()
    )
