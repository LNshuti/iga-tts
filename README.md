# **Iga**: Multilingual translator between English, French, and Kinyarwanda. 

- We use sythentic audio for target phrases using a Transformers TTS model (default: suno/bark-small for speed)
- The app has two learning modes (Rwanda/Burundi Mode and Diaspora Mode)
- The app is deployed to Hugging Face Spaces with a Gradio UI and an auto-exposed Inference API
- Includes example code to use the TTS from the Transformers library directly

Features
- Rwanda Mode and Diaspora Mode switching
- Offline translation with MarianMT (en↔rw; fr bridged via en)
- Text-to-speech with suno/bark or suno/bark-small
- Phrase packs for quick demos
- Auto-exposed API via Space

Performance 
- CPU (free tier): set TTS_MODEL=suno/bark-small to reduce latency
- GPU (T4/A10G): set TTS_MODEL=suno/bark for higher quality
- Use queueing (enabled) to stabilize concurrent requests

Limitations
- Bark’s Kinyarwanda pronunciation may be imperfect; this is a prototype
- MarianMT quality for fr↔rw via en bridge varies
- Real progress tracking, user auth, and analytics are not included

Next Steps
- Add microphone input + pronunciation feedback (Whisper or equivalent)
- Add XP/streak persistence (Supabase/Firebase)
- Add authentic native voice via custom TTS (dataset + training)
- Improve fr↔rw translation with dedicated models or custom fine-tuning
