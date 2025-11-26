"""
Audio utility functions for WAV conversion and download.
Provides numpy-to-WAV conversion, sample rate normalization, and temp file handling.
"""
import io
import wave
import tempfile
import os
from datetime import datetime
from typing import Tuple, Optional
import numpy as np

from config import logger

# Target sample rate for corpus compatibility (ASR/TTS standard)
TARGET_SAMPLE_RATE = 16000


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Resample audio to target sample rate using linear interpolation.

    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate (default 16000 Hz)

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    # Calculate resampling ratio
    ratio = target_sr / orig_sr
    new_length = int(len(audio) * ratio)

    # Linear interpolation (sufficient for 16kHz downsampling)
    indices = np.linspace(0, len(audio) - 1, new_length)
    resampled = np.interp(indices, np.arange(len(audio)), audio)

    return resampled.astype(audio.dtype)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono by averaging channels.

    Args:
        audio: Input audio array (1D or 2D)

    Returns:
        Mono audio array (1D)
    """
    if len(audio.shape) == 1:
        return audio
    if len(audio.shape) == 2:
        # Shape is (samples, channels) or (channels, samples)
        if audio.shape[0] == 2:  # (channels, samples)
            return np.mean(audio, axis=0)
        elif audio.shape[1] == 2:  # (samples, channels)
            return np.mean(audio, axis=1)
    return audio


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize audio to target peak level, preventing clipping.

    Args:
        audio: Input audio array
        target_peak: Target peak level (0-1)

    Returns:
        Normalized audio array
    """
    if audio.size == 0:
        return audio

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio * (target_peak / max_val)
    return audio


def numpy_to_wav(audio: np.ndarray, sr: int, normalize: bool = True) -> bytes:
    """
    Convert numpy audio array to WAV bytes.

    Args:
        audio: Float32 numpy array (range -1 to 1)
        sr: Sample rate
        normalize: Whether to normalize to prevent clipping

    Returns:
        WAV file as bytes
    """
    # Ensure float32 and mono
    audio = audio.astype(np.float32)
    audio = to_mono(audio)

    # Normalize to prevent clipping
    if normalize:
        audio = normalize_audio(audio)

    # Convert to 16-bit PCM
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    # Write to WAV buffer
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sr)
        wav.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()


def numpy_to_wav_16k(audio: np.ndarray, orig_sr: int, normalize: bool = True) -> bytes:
    """
    Convert numpy audio to 16kHz mono WAV bytes.
    Standard format for ASR/TTS corpus.

    Args:
        audio: Float32 numpy array
        orig_sr: Original sample rate
        normalize: Whether to normalize

    Returns:
        16kHz mono WAV file as bytes
    """
    audio = to_mono(audio.astype(np.float32))
    audio = resample_audio(audio, orig_sr, TARGET_SAMPLE_RATE)
    return numpy_to_wav(audio, TARGET_SAMPLE_RATE, normalize)


def pcm_bytes_to_wav(pcm_bytes: bytes, sr: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    """
    Convert raw PCM bytes to WAV format.
    Used for decrypted audio from storage.

    Args:
        pcm_bytes: Raw 16-bit PCM audio bytes
        sr: Sample rate
        channels: Number of channels
        sampwidth: Sample width in bytes (2 for 16-bit)

    Returns:
        WAV file as bytes
    """
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sampwidth)
        wav.setframerate(sr)
        wav.writeframes(pcm_bytes)

    buffer.seek(0)
    return buffer.read()


def save_temp_wav(wav_bytes: bytes, feedback_id: Optional[int] = None) -> str:
    """
    Save WAV bytes to a temporary file for Gradio download.

    Args:
        wav_bytes: WAV file as bytes
        feedback_id: Optional feedback ID for filename

    Returns:
        Path to temporary WAV file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if feedback_id:
        filename = f"iga_feedback_{feedback_id}_{timestamp}.wav"
    else:
        filename = f"iga_recording_{timestamp}.wav"

    # Create temp file with proper suffix
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)

    with open(filepath, 'wb') as f:
        f.write(wav_bytes)

    logger.info(f"Saved temporary WAV file: {filepath}")
    return filepath


def cleanup_temp_wav(filepath: str) -> bool:
    """
    Clean up temporary WAV file.

    Args:
        filepath: Path to temporary file

    Returns:
        True if deleted, False otherwise
    """
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up temp file: {filepath}")
            return True
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {filepath}: {e}")
    return False


def validate_audio_for_download(audio_data: Tuple[int, np.ndarray]) -> Tuple[bool, str]:
    """
    Validate audio data before download.

    Args:
        audio_data: Tuple of (sample_rate, numpy_array)

    Returns:
        (is_valid, error_or_warning_message)
    """
    if audio_data is None:
        return False, "No audio recorded. Please record audio first."

    try:
        sr, audio = audio_data

        if sr <= 0:
            return False, "Invalid sample rate."

        if audio is None or audio.size == 0:
            return False, "Audio is empty. Please record again."

        # Check duration first (hard fail)
        duration = len(audio) / sr
        if duration < 0.5:
            return False, "Recording too short (minimum 0.5 seconds)."

        if duration > 300:
            return False, "Recording too long (maximum 5 minutes)."

        # Check for excessive clipping (warning only, not a failure)
        warning_msg = ""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            clipping_ratio = np.mean(np.abs(audio) > 0.99)
            if clipping_ratio > 0.05:
                warning_msg = f"Warning: {clipping_ratio:.1%} clipping detected."

        return True, warning_msg

    except Exception as e:
        return False, f"Validation error: {str(e)}"
