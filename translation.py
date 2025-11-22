"""
Translation module using MarianMT models for multilingual translation.
Supports direct and bridge translation between English, French, and Kinyarwanda.
"""
from functools import lru_cache
from typing import Optional, Dict, Tuple
from transformers import pipeline, Pipeline
import re

from config import Config, logger, TranslationError

# MarianMT models
MODELS: Dict[Tuple[str, str], str] = {
    ("en", "rw"): "Helsinki-NLP/opus-mt-en-rw",
    ("rw", "en"): "Helsinki-NLP/opus-mt-rw-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
}

# Supported language codes
SUPPORTED_LANGUAGES = {"en", "fr", "rw"}


def validate_text(text: str, max_length: int = 5000) -> str:
    """
    Validate and sanitize input text.

    Args:
        text: Input text to validate
        max_length: Maximum allowed text length

    Returns:
        str: Sanitized text

    Raises:
        TranslationError: If text is invalid
    """
    if not text:
        raise TranslationError("Input text cannot be empty")

    text = text.strip()

    if len(text) > max_length:
        logger.warning(f"Text truncated from {len(text)} to {max_length} characters")
        text = text[:max_length]

    # Remove potentially harmful characters (control characters)
    # This is more permissive to support multilingual Unicode text
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    if not text.strip():
        raise TranslationError("Input text is empty after sanitization")

    return text


def validate_language_code(lang: str) -> str:
    """
    Validate language code.

    Args:
        lang: Language code to validate

    Returns:
        str: Validated language code

    Raises:
        TranslationError: If language code is invalid
    """
    lang = lang.lower().strip()
    if lang not in SUPPORTED_LANGUAGES:
        raise TranslationError(
            f"Unsupported language: {lang}. Supported: {SUPPORTED_LANGUAGES}"
        )
    return lang


@lru_cache(maxsize=Config.TRANSLATION_CACHE_SIZE)
def get_translator(src: str, tgt: str) -> Optional[Pipeline]:
    """
    Get or create a cached translation pipeline for a language pair.

    Args:
        src: Source language code
        tgt: Target language code

    Returns:
        Optional[Pipeline]: Translation pipeline or None if model not available

    Raises:
        TranslationError: If pipeline creation fails
    """
    model_name = MODELS.get((src, tgt))
    if not model_name:
        logger.debug(f"No direct model for {src} -> {tgt}")
        return None

    try:
        logger.info(f"Loading translation model: {model_name}")
        return pipeline("translation", model=model_name)
    except Exception as e:
        logger.error(f"Failed to load translation model {model_name}: {e}")
        raise TranslationError(f"Failed to load translation model: {e}")


def translate(text: str, src: str, tgt: str) -> str:
    """
    Translate text from source to target language with comprehensive error handling.

    Supports direct translation and bridge translation via English for fr<->rw pairs.

    Args:
        text: Input text to translate
        src: Source language code (en, fr, or rw)
        tgt: Target language code (en, fr, or rw)

    Returns:
        str: Translated text

    Raises:
        TranslationError: If translation fails
    """
    try:
        # Validate inputs
        src = validate_language_code(src)
        tgt = validate_language_code(tgt)
        text = validate_text(text)

        # Same language - no translation needed
        if src == tgt:
            logger.debug(f"Source and target are the same ({src}), returning original")
            return text

        logger.info(f"Translating from {src} to {tgt}: {text[:50]}...")

        # Direct translator
        direct = get_translator(src, tgt)
        if direct:
            result = direct(text, max_length=Config.MAX_TRANSLATION_LENGTH)
            translated = result[0]["translation_text"].strip()
            logger.info(f"Direct translation successful: {translated[:50]}...")
            return translated

        # Bridge via English if needed (fr<->rw)
        if src == "fr" and tgt == "rw":
            logger.info("Using bridge translation: fr -> en -> rw")
            fr_en = get_translator("fr", "en")
            en_rw = get_translator("en", "rw")

            if not fr_en or not en_rw:
                raise TranslationError("Bridge translation models not available")

            mid = fr_en(text, max_length=Config.MAX_TRANSLATION_LENGTH)[0]["translation_text"]
            result = en_rw(mid, max_length=Config.MAX_TRANSLATION_LENGTH)[0]["translation_text"].strip()
            logger.info(f"Bridge translation successful: {result[:50]}...")
            return result

        if src == "rw" and tgt == "fr":
            logger.info("Using bridge translation: rw -> en -> fr")
            rw_en = get_translator("rw", "en")
            en_fr = get_translator("en", "fr")

            if not rw_en or not en_fr:
                raise TranslationError("Bridge translation models not available")

            mid = rw_en(text, max_length=Config.MAX_TRANSLATION_LENGTH)[0]["translation_text"]
            result = en_fr(mid, max_length=Config.MAX_TRANSLATION_LENGTH)[0]["translation_text"].strip()
            logger.info(f"Bridge translation successful: {result[:50]}...")
            return result

        # Unsupported language pair
        raise TranslationError(f"Translation from {src} to {tgt} is not supported")

    except TranslationError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during translation: {e}", exc_info=True)
        raise TranslationError(f"Translation failed: {str(e)}")
