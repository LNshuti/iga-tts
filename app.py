#!/usr/bin/env python3
"""
Simple Audio Recording & Trimming Tool
Record audio from microphone, trim it, and download as .wav file
"""

import tempfile
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


def build_ui():
    """Build the Gradio interface."""

    with gr.Blocks(title="Audio Recording & Trimming Tool") as app:
        gr.Markdown("""
        # Audio Recording & Trimming Tool

        Record audio from your microphone, trim it to the desired length, and download as a .wav file.
        """)

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
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.01,
                    label="Start Time (seconds)",
                    interactive=False
                )

                end_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=1,
                    step=0.01,
                    label="End Time (seconds)",
                    interactive=False
                )

                selection_info = gr.Textbox(
                    value="No audio recorded yet",
                    label="Selection",
                    interactive=False
                )

                trim_button = gr.Button("Trim & Download", variant="primary", size="lg")

            with gr.Column():
                gr.Markdown("### 3. Download")
                audio_output = gr.Audio(
                    label="Trimmed Audio (Right-click to download)",
                    type="filepath"
                )

        gr.Markdown("""
        ---
        **Instructions:**
        1. Click the microphone icon to start recording
        2. Speak or play audio
        3. Click stop when finished
        4. Adjust the start and end sliders to select the portion you want to keep
        5. Click "Trim & Download" to generate the trimmed audio
        6. Right-click on the audio player to download the .wav file
        """)

        # Event handlers
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

    return app


if __name__ == "__main__":
    print("Starting Audio Recording & Trimming Tool...")

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )
