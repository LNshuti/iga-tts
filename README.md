# Multilingual web app 

Iga translates knowledge between English, French, and Kinyarwanda (rw)

- We use sythentic audio for target phrase using a Transformers TTS model (default: suno/bark-small for speed)

- The app has two learning modes (Rwanda/Burundi Mode and Diaspora Mode)

- The app is deployed to Hugging Face Spaces with a Gradio UI and an auto-exposed Inference API

- Includes example code to use the TTS from the Transformers library directly

Features
- Rwanda Mode and Diaspora Mode switching
- Offline translation with MarianMT (en↔rw; fr bridged via en)
- Text-to-speech with suno/bark or suno/bark-small
- Phrase packs for quick demos
- Auto-exposed API via Space

Run Locally
1) python -m venv .venv && source .venv/bin/activate (Windows: .venv\\Scripts\\activate)
2) pip install -r requirements.txt
3) (Optional) export TTS_MODEL=suno/bark  # if you have a GPU
4) python app.py  # Opens http://127.0.0.1:7860

Deploy to Hugging Face Spaces
1) Create a new Space: SDK=Gradio; choose CPU or GPU hardware
2) Push this repo to the Space (or drag-and-drop files in the UI)
3) Optional Space metadata in README frontmatter (title, emoji); app_file is app.py by default
4) First build will download models; subsequent runs use cache

Performance Tips
- CPU (free tier): set TTS_MODEL=suno/bark-small to reduce latency
- GPU (T4/A10G): set TTS_MODEL=suno/bark for higher quality
- Use queueing (enabled) to stabilize concurrent requests

Limitations
- Bark’s Kinyarwanda pronunciation may be imperfect; this is a prototype
- MarianMT quality for fr↔rw via en bridge varies
- Real progress tracking, user auth, and analytics are not included here

Use via Transformers (Python)
High-level pipeline
from transformers import pipeline
pipe = pipeline("text-to-speech", model="suno/bark-small")
res = pipe("Muraho neza")
# res["audio"] -> numpy array, res["sampling_rate"] -> int

Low-level API
from transformers import AutoProcessor, AutoModelForTextToWaveform
import soundfile as sf

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModelForTextToWaveform.from_pretrained("suno/bark-small")

inputs = processor(text="Muraho neza", return_tensors="pt")
with torch.no_grad():
    audio = model.generate(**inputs)
# audio is a tensor; convert and save
arr = audio.cpu().numpy().squeeze()
sf.write("out.wav", arr, 24000)

Use Space Inference API (after deployment)
- Gradio auto-exposes an endpoint for the translate/speak functions.
- Example (Python requests) for TTS using the translated text:
import requests
import json

SPACE_URL = "https://hf.space/your-username/learn-kinyarwanda"  # replace
API = f"{SPACE_URL}/run/predict"  # check your Space's API docs tab for exact path
payload = {"data": ["Muraho neza"]}
res = requests.post(API, json=payload)
print(res.json())

Next Steps
- Add microphone input + pronunciation feedback (Whisper or equivalent)
- Add XP/streak persistence (Supabase/Firebase)
- Add authentic native voice via custom TTS (dataset + training)
- Improve fr↔rw translation with dedicated models or custom fine-tuning
