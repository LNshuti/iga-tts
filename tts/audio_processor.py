#!/usr/bin/env python3
"""
Audio processing utilities for TTS

Handles audio format conversion, normalization, and validation.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
from scipy import signal

logger = logging.getLogger(__name__)

# Audio specifications
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
TARGET_DTYPE = np.float32
MIN_DURATION = 0.5  # seconds
MAX_DURATION = 8.0  # seconds
TARGET_LOUDNESS = -20.0  # LUFS


def load_audio(
    filepath: Path,
    sr: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with resampling to target sample rate

    Args:
        filepath: Path to audio file
        sr: Target sample rate (default 24000)
        mono: Convert to mono if True

    Returns:
        tuple: (audio array, sample rate)
    """
    try:
        audio, original_sr = librosa.load(str(filepath), sr=sr, mono=mono)
        logger.debug(f"Loaded {filepath.name}: {len(audio)} samples @ {sr}Hz")
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        raise


def normalize_loudness(
    audio: np.ndarray,
    target_loudness: float = TARGET_LOUDNESS,
    sr: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """
    Normalize audio loudness to target LUFS (Loudness Units relative to Full Scale)

    Args:
        audio: Audio array
        target_loudness: Target loudness in LUFS (default -20)
        sr: Sample rate

    Returns:
        np.ndarray: Normalized audio
    """
    try:
        # Calculate current loudness using pyloudnorm approach (simplified)
        # RMS-based loudness estimation
        rms = np.sqrt(np.mean(audio ** 2))

        if rms < 1e-6:
            logger.warning("Audio appears to be silent")
            return audio

        # Convert RMS to dB
        db = 20 * np.log10(rms + 1e-10)

        # Calculate gain needed
        target_rms = 10 ** (target_loudness / 20)
        gain = target_rms / (rms + 1e-10)

        # Apply gain
        normalized = audio * gain

        # Soft clip to prevent clipping
        normalized = np.tanh(normalized * 0.95) / np.tanh(0.95)

        logger.debug(f"Normalized audio: {db:.1f}dB → {target_loudness}LUFS")
        return normalized

    except Exception as e:
        logger.warning(f"Error normalizing loudness: {e}")
        return audio


def trim_silence(
    audio: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    top_db: float = 40,
    ref: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Trim leading and trailing silence

    Args:
        audio: Audio array
        sr: Sample rate
        top_db: Threshold for silence in dB
        ref: Reference power (if None, use max)

    Returns:
        np.ndarray: Trimmed audio
    """
    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db, ref=ref)
        trimmed_ms = (len(trimmed) / sr) * 1000
        original_ms = (len(audio) / sr) * 1000
        logger.debug(f"Trimmed silence: {original_ms:.0f}ms → {trimmed_ms:.0f}ms")
        return trimmed
    except Exception as e:
        logger.warning(f"Error trimming silence: {e}")
        return audio


def validate_duration(
    audio: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    min_duration: float = MIN_DURATION,
    max_duration: float = MAX_DURATION,
) -> bool:
    """
    Check if audio duration is within acceptable range

    Args:
        audio: Audio array
        sr: Sample rate
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds

    Returns:
        bool: True if duration is valid
    """
    duration = len(audio) / sr
    valid = min_duration <= duration <= max_duration

    if not valid:
        logger.warning(
            f"Invalid duration: {duration:.2f}s "
            f"(expected {min_duration}–{max_duration}s)"
        )

    return valid


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """
    Resample audio to target sample rate

    Args:
        audio: Audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        tuple: (resampled audio, target sample rate)
    """
    if orig_sr == target_sr:
        return audio, target_sr

    logger.debug(f"Resampling {orig_sr}Hz → {target_sr}Hz")
    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return resampled, target_sr


def process_audio(
    filepath: Path,
    sr: int = TARGET_SAMPLE_RATE,
    normalize: bool = True,
    trim: bool = True,
    validate: bool = True,
) -> Optional[np.ndarray]:
    """
    Process audio file: load, resample, normalize, trim, validate

    Args:
        filepath: Path to audio file
        sr: Target sample rate
        normalize: Normalize loudness if True
        trim: Trim silence if True
        validate: Validate duration if True

    Returns:
        np.ndarray: Processed audio, or None if processing failed
    """
    try:
        # Load
        audio, _ = load_audio(filepath, sr=sr)

        # Normalize loudness
        if normalize:
            audio = normalize_loudness(audio, sr=sr)

        # Trim silence
        if trim:
            audio = trim_silence(audio, sr=sr)

        # Validate duration
        if validate and not validate_duration(audio, sr=sr):
            logger.warning(f"Audio duration invalid for {filepath.name}")
            return None

        return audio

    except Exception as e:
        logger.error(f"Error processing audio {filepath}: {e}")
        return None


def save_audio(
    audio: np.ndarray,
    filepath: Path,
    sr: int = TARGET_SAMPLE_RATE,
    subtype: str = 'PCM_16',
) -> bool:
    """
    Save audio to WAV file

    Args:
        audio: Audio array
        filepath: Output file path
        sr: Sample rate
        subtype: Audio subtype (default PCM_16)

    Returns:
        bool: True if successful
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Clip to [-1, 1] to prevent clipping
        audio = np.clip(audio, -1.0, 1.0)

        sf.write(str(filepath), audio, sr, subtype=subtype)
        logger.debug(f"Saved audio: {filepath.name}")
        return True

    except Exception as e:
        logger.error(f"Error saving audio {filepath}: {e}")
        return False


def get_audio_duration(audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> float:
    """Get audio duration in seconds"""
    return len(audio) / sr


def concatenate_audio(audio_list: list[np.ndarray], sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Concatenate multiple audio arrays with silence gap

    Args:
        audio_list: List of audio arrays
        sr: Sample rate

    Returns:
        np.ndarray: Concatenated audio
    """
    if not audio_list:
        return np.array([])

    # 100ms silence between clips
    silence = np.zeros(int(0.1 * sr))

    concatenated = []
    for audio in audio_list:
        concatenated.append(audio)
        concatenated.append(silence)

    return np.concatenate(concatenated)
