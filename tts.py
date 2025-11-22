import os
import numpy as np
from transformers import pipeline

_TTS_MODEL = os.getenv("TTS_MODEL", "suno/bark-small")
_DEVICE = os.getenv("DEVICE", None)  # e.g. "cuda" if GPU available

_pipe = None

def get_tts_pipeline():
    global _pipe
    if _pipe is None:
        kwargs = {"model": _TTS_MODEL}
        if _DEVICE:
            kwargs["device"] = _DEVICE
        _pipe = pipeline("text-to-speech", **kwargs)
    return _pipe

def synthesize(text: str):
    pipe = get_tts_pipeline()
    out = pipe(text)
    # Expect keys: out["audio"], out["sampling_rate"] for Bark
    audio = out["audio"]
    sr = out["sampling_rate"]
    # Ensure float32 numpy array for Gradio
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    return sr, audio
