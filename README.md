---
title: iga-tts
app_file: app.py
sdk: gradio
sdk_version: 6.6.0
---
# Audio Recording & Trimming Tool

A simple web-based tool for recording audio from your microphone, trimming it to the desired length, and downloading as a .wav file.

## Features

- **Record Audio**: Click to record audio directly from your microphone
- **Trim Audio**: Use dual sliders to select the exact start and end times
- **Real-time Feedback**: See total duration and selected duration as you adjust
- **Download**: Export trimmed audio as a .wav file
- **Clean Interface**: Simple, intuitive Gradio UI

## How to Use

1. Click the microphone icon to start recording
2. Speak or play audio into your microphone
3. Click stop when finished
4. Adjust the start and end sliders to select the portion you want to keep
5. Click "Trim & Download" to generate the trimmed audio
6. Right-click on the audio player to download the .wav file

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

The app will launch at `http://localhost:7860`

## Deployment

Can be deployed on Hugging Face Spaces or any platform supporting Gradio apps.

```bash
gradio deploy
```

Our deployment can be accessed at `https://leoncensh-iga-tts.hf.space`

## Dependencies
- **gradio**: Web UI framework
- **soundfile**: Audio file I/O
- **numpy**: Audio data processing
