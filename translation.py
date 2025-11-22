from functools import lru_cache
from transformers import pipeline

# MarianMT models
MODELS = {
    ("en", "rw"): "Helsinki-NLP/opus-mt-en-rw",
    ("rw", "en"): "Helsinki-NLP/opus-mt-rw-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
}

@lru_cache(maxsize=8)
def get_translator(src: str, tgt: str):
    model_name = MODELS.get((src, tgt))
    if not model_name:
        return None
    return pipeline("translation", model=model_name)


def translate(text: str, src: str, tgt: str) -> str:
    text = (text or "").strip()
    if not text or src == tgt:
        return text

    # Direct translator
    direct = get_translator(src, tgt)
    if direct:
        out = direct(text, max_length=400)
        return out[0]["translation_text"].strip()

    # Bridge via English if needed (fr<->rw)
    if src == "fr" and tgt == "rw":
        fr_en = get_translator("fr", "en")
        en_rw = get_translator("en", "rw")
        mid = fr_en(text, max_length=400)[0]["translation_text"]
        return en_rw(mid, max_length=400)[0]["translation_text"].strip()
    if src == "rw" and tgt == "fr":
        rw_en = get_translator("rw", "en")
        en_fr = get_translator("en", "fr")
        mid = rw_en(text, max_length=400)[0]["translation_text"]
        return en_fr(mid, max_length=400)[0]["translation_text"].strip()

    # Fallback: return original if unsupported
    return text
