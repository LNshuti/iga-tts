"""
Text-to-Speech module using Transformers (Bark) for multilingual audio synthesis.
Provides robust error handling and audio format validation for Gradio compatibility.
"""
import numpy as np
from typing import Tuple, Optional
from transformers import pipeline, Pipeline
import re

from config import Config, logger, TTSError

# Global TTS pipeline cache
_pipe: Optional[Pipeline] = None


def validate_tts_text(text: str, max_length: int = 1000) -> str:
    """
    Validate and sanitize text for TTS.

    Args:
        text: Input text for speech synthesis
        max_length: Maximum allowed text length (Bark has limits)

    Returns:
        str: Sanitized text

    Raises:
        TTSError: If text is invalid
    """
    if not text:
        raise TTSError("TTS input text cannot be empty")

    text = text.strip()

    if len(text) > max_length:
        logger.warning(f"TTS text truncated from {len(text)} to {max_length} characters")
        text = text[:max_length]

    # Remove control characters but preserve multilingual text
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    if not text:
        raise TTSError("TTS text is empty after sanitization")

    return text


def get_tts_pipeline() -> Pipeline:
    """
    Get or create a cached TTS pipeline.

    Returns:
        Pipeline: Text-to-speech pipeline

    Raises:
        TTSError: If pipeline creation fails
    """
    global _pipe

    if _pipe is None:
        try:
            logger.info(f"Loading TTS model: {Config.TTS_MODEL}")
            kwargs = {"model": Config.TTS_MODEL}

            if Config.DEVICE:
                logger.info(f"Using device: {Config.DEVICE}")
                kwargs["device"] = Config.DEVICE

            _pipe = pipeline("text-to-speech", **kwargs)
            logger.info("TTS pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}", exc_info=True)
            raise TTSError(f"Failed to initialize TTS pipeline: {e}")

    return _pipe


def synthesize(text: str) -> Tuple[int, np.ndarray]:
    """
    Synthesize speech from text with comprehensive error handling.

    Args:
        text: Input text to synthesize

    Returns:
        Tuple[int, np.ndarray]: (sampling_rate, audio_array)
            - sampling_rate: Audio sampling rate in Hz
            - audio_array: Float32 numpy array of audio samples

    Raises:
        TTSError: If synthesis fails
    """
    try:
        # Validate input
        text = validate_tts_text(text)
        logger.info(f"Synthesizing speech for: {text[:50]}...")

        # Get pipeline
        pipe = get_tts_pipeline()

        # Generate speech
        out = pipe(text)

        # Validate output format
        if not isinstance(out, dict):
            raise TTSError(f"Unexpected TTS output type: {type(out)}")

        if "audio" not in out or "sampling_rate" not in out:
            raise TTSError(f"Missing required keys in TTS output: {out.keys()}")

        audio = out["audio"]
        sr = out["sampling_rate"]

        # Validate sampling rate
        if not isinstance(sr, int) or sr <= 0:
            raise TTSError(f"Invalid sampling rate: {sr}")

        # Ensure float32 numpy array for Gradio compatibility
        if not isinstance(audio, np.ndarray):
            logger.debug(f"Converting audio from {type(audio)} to numpy array")
            audio = np.array(audio, dtype=np.float32)
        elif audio.dtype != np.float32:
            logger.debug(f"Converting audio from {audio.dtype} to float32")
            audio = audio.astype(np.float32)

        # Validate audio shape and values
        if audio.size == 0:
            raise TTSError("Generated audio is empty")

        if len(audio.shape) > 2:
            raise TTSError(f"Invalid audio shape: {audio.shape}")

        # Check for NaN or Inf values
        if np.isnan(audio).any() or np.isinf(audio).any():
            logger.warning("Audio contains NaN or Inf values, clipping")
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        # Normalize if values exceed [-1, 1] range
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            logger.debug(f"Normalizing audio (max={max_val})")
            audio = audio / max_val

        logger.info(f"TTS synthesis successful: {audio.shape}, {sr}Hz")
        return sr, audio

    except TTSError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during TTS synthesis: {e}", exc_info=True)
        raise TTSError(f"Speech synthesis failed: {str(e)}")
