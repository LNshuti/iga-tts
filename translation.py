"""
Translation module using MarianMT models for multilingual translation.
Supports direct and bridge translation between English, French, and Kinyarwanda.
Enhanced with NLLB-200 support for improved fr↔rw translation quality.
"""
from functools import lru_cache
from typing import Optional, Dict, Tuple, Any
from transformers import pipeline, Pipeline
import re
from difflib import SequenceMatcher

from config import Config, logger, TranslationError

# MarianMT models
MODELS: Dict[Tuple[str, str], str] = {
    ("en", "rw"): "Helsinki-NLP/opus-mt-en-rw",
    ("rw", "en"): "Helsinki-NLP/opus-mt-rw-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
}

# NLLB model configuration
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_LANG_CODES = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "rw": "kin_Latn",
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


class NLLBModelLoader:
    """Loads and manages NLLB-200 translation model for multilingual translation."""

    def __init__(self):
        """Initialize NLLB model loader."""
        self._translator = None
        self._model_name = NLLB_MODEL
        self._lang_codes = NLLB_LANG_CODES

    def supports_language_pair(self, src: str, tgt: str) -> bool:
        """
        Check if NLLB supports a language pair.

        Args:
            src: Source language code
            tgt: Target language code

        Returns:
            bool: True if both languages are in NLLB_LANG_CODES
        """
        return src in self._lang_codes and tgt in self._lang_codes

    def get_nllb_lang_code(self, lang: str) -> str:
        """
        Get NLLB language code for a given language.

        Args:
            lang: Simple language code (en, fr, rw)

        Returns:
            str: NLLB language code (eng_Latn, fra_Latn, kin_Latn)
        """
        return self._lang_codes.get(lang, lang)

    def load_translator(self) -> Optional[Pipeline]:
        """
        Load or return cached NLLB translator pipeline.

        Returns:
            Optional[Pipeline]: Translation pipeline or None if loading fails
        """
        if self._translator is not None:
            return self._translator

        try:
            logger.info(f"Loading NLLB model: {self._model_name}")
            self._translator = pipeline(
                "translation",
                model=self._model_name,
                device=0 if (Config.DEVICE == "cuda") else -1
            )
            logger.info("NLLB model loaded successfully")
            return self._translator
        except Exception as e:
            logger.error(f"Failed to load NLLB model: {e}")
            return None


class TranslationQualityScorer:
    """Evaluates translation quality using multiple metrics."""

    def calculate_edit_distance(self, s1: str, s2: str) -> int:
        """
        Calculate edit distance (Levenshtein) between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            int: Edit distance
        """
        if len(s1) < len(s2):
            return self.calculate_edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def calculate_similarity_score(self, s1: str, s2: str) -> float:
        """
        Calculate string similarity score (0-1) using SequenceMatcher.

        Args:
            s1: First string
            s2: Second string

        Returns:
            float: Similarity score between 0 and 1
        """
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


class TranslationModelManager:
    """Manages selection of best translation model for each language pair."""

    def __init__(self):
        """Initialize translation model manager."""
        self.nllb_loader = NLLBModelLoader()
        self._prefer_nllb_for = {("fr", "rw"), ("rw", "fr")}

    def get_best_translator(self, src: str, tgt: str) -> Optional[Pipeline]:
        """
        Get best translator for a language pair (NLLB if available, else MarianMT).

        Args:
            src: Source language code
            tgt: Target language code

        Returns:
            Optional[Pipeline]: Best available translator pipeline
        """
        # Try NLLB for fr↔rw pairs
        if (src, tgt) in self._prefer_nllb_for:
            if self.nllb_loader.supports_language_pair(src, tgt):
                translator = self.nllb_loader.load_translator()
                if translator is not None:
                    logger.info(f"Using NLLB for {src} -> {tgt}")
                    return translator

        # Fall back to MarianMT
        logger.debug(f"Using MarianMT for {src} -> {tgt}")
        return get_translator(src, tgt)


class ABTestResult:
    """Result of A/B test comparing two translation models."""

    def __init__(self, text: str, src: str, tgt: str,
                 model_a_translation: str, model_b_translation: str,
                 model_a_name: str, model_b_name: str,
                 quality_score_a: float, quality_score_b: float):
        """
        Initialize A/B test result.

        Args:
            text: Original text
            src: Source language
            tgt: Target language
            model_a_translation: Translation from model A
            model_b_translation: Translation from model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            quality_score_a: Quality score for model A (0-1)
            quality_score_b: Quality score for model B (0-1)
        """
        self.text = text
        self.src = src
        self.tgt = tgt
        self.model_a_translation = model_a_translation
        self.model_b_translation = model_b_translation
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.quality_score_a = quality_score_a
        self.quality_score_b = quality_score_b

    def winner(self) -> str:
        """
        Get name of model with higher quality score.

        Returns:
            str: Name of winning model
        """
        if self.quality_score_b > self.quality_score_a:
            return self.model_b_name
        elif self.quality_score_a > self.quality_score_b:
            return self.model_a_name
        else:
            return "TIE"

    def score_difference(self) -> float:
        """
        Get absolute difference in quality scores.

        Returns:
            float: Absolute score difference
        """
        return abs(self.quality_score_a - self.quality_score_b)


class ABTestComparisonRunner:
    """Runs A/B tests comparing translation models."""

    def __init__(self):
        """Initialize A/B test comparison runner."""
        self.model_manager = TranslationModelManager()
        self.quality_scorer = TranslationQualityScorer()
        self.nllb_loader = NLLBModelLoader()

    def run_comparison(self, text: str, src: str, tgt: str) -> Optional[ABTestResult]:
        """
        Run A/B comparison between NLLB and MarianMT bridge for fr↔rw.

        Args:
            text: Text to translate
            src: Source language code
            tgt: Target language code

        Returns:
            Optional[ABTestResult]: Comparison result or None if comparison not applicable
        """
        # Only compare for fr↔rw pairs
        if (src, tgt) not in {("fr", "rw"), ("rw", "fr")}:
            return None

        try:
            # Get NLLB translation
            nllb_translator = self.nllb_loader.load_translator()
            if nllb_translator is None:
                return None

            nllb_translation = _translate_with_pipeline(nllb_translator, text)

            # Get MarianMT bridge translation
            if src == "fr" and tgt == "rw":
                fr_en = get_translator("fr", "en")
                en_rw = get_translator("en", "rw")
                if not fr_en or not en_rw:
                    return None
                mid = _translate_with_pipeline(fr_en, text)
                marian_translation = _translate_with_pipeline(en_rw, mid)
            else:  # rw -> fr
                rw_en = get_translator("rw", "en")
                en_fr = get_translator("en", "fr")
                if not rw_en or not en_fr:
                    return None
                mid = _translate_with_pipeline(rw_en, text)
                marian_translation = _translate_with_pipeline(en_fr, mid)

            # Calculate quality scores
            quality_a = self.quality_scorer.calculate_similarity_score(
                text.lower(), marian_translation.lower()
            )
            quality_b = self.quality_scorer.calculate_similarity_score(
                text.lower(), nllb_translation.lower()
            )

            return ABTestResult(
                text=text,
                src=src,
                tgt=tgt,
                model_a_translation=marian_translation,
                model_b_translation=nllb_translation,
                model_a_name="MarianMT (Bridge)",
                model_b_name="NLLB-200",
                quality_score_a=quality_a,
                quality_score_b=quality_b
            )
        except Exception as e:
            logger.error(f"A/B test comparison failed: {e}")
            return None


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


# Global model manager instance
_model_manager = None


def get_model_manager() -> TranslationModelManager:
    """Get or create global translation model manager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = TranslationModelManager()
    return _model_manager


def _translate_with_pipeline(pipeline_obj: Pipeline, text: str) -> str:
    """
    Execute translation with a pipeline, handling NLLB-specific requirements.

    Args:
        pipeline_obj: Translation pipeline
        text: Text to translate

    Returns:
        str: Translated text
    """
    # Check if this is NLLB model (uses different output format)
    model_name = getattr(pipeline_obj.model.config, 'model_type', '')
    if 'nllb' in model_name.lower() or 'autoencoder' in model_name.lower():
        # NLLB returns different format
        result = pipeline_obj(text, max_length=Config.MAX_TRANSLATION_LENGTH)
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'translation_text' in result[0]:
                return result[0]['translation_text'].strip()
        return str(result).strip()
    else:
        # MarianMT format
        result = pipeline_obj(text, max_length=Config.MAX_TRANSLATION_LENGTH)
        return result[0]["translation_text"].strip()


def translate(text: str, src: str, tgt: str) -> str:
    """
    Translate text from source to target language with comprehensive error handling.

    Uses NLLB-200 for fr<->rw pairs for improved quality, falls back to MarianMT.
    Supports direct translation and bridge translation via English for other pairs.

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

        # Get best translator for this language pair
        manager = get_model_manager()
        translator = manager.get_best_translator(src, tgt)

        if translator:
            translated = _translate_with_pipeline(translator, text)
            logger.info(f"Translation successful: {translated[:50]}...")
            return translated

        # Bridge via English if needed (fr<->rw)
        if src == "fr" and tgt == "rw":
            logger.info("Using bridge translation: fr -> en -> rw")
            fr_en = get_translator("fr", "en")
            en_rw = get_translator("en", "rw")

            if not fr_en or not en_rw:
                raise TranslationError("Bridge translation models not available")

            mid = _translate_with_pipeline(fr_en, text)
            result = _translate_with_pipeline(en_rw, mid)
            logger.info(f"Bridge translation successful: {result[:50]}...")
            return result

        if src == "rw" and tgt == "fr":
            logger.info("Using bridge translation: rw -> en -> fr")
            rw_en = get_translator("rw", "en")
            en_fr = get_translator("en", "fr")

            if not rw_en or not en_fr:
                raise TranslationError("Bridge translation models not available")

            mid = _translate_with_pipeline(rw_en, text)
            result = _translate_with_pipeline(en_fr, mid)
            logger.info(f"Bridge translation successful: {result[:50]}...")
            return result

        # Unsupported language pair
        raise TranslationError(f"Translation from {src} to {tgt} is not supported")

    except TranslationError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during translation: {e}", exc_info=True)
        raise TranslationError(f"Translation failed: {str(e)}")
