"""
Configuration and logging setup for Iga TTS application.
Enterprise-grade configuration management with comprehensive logging.
"""
import os
import logging
from typing import Optional
from pathlib import Path

# Application Configuration
class Config:
    """Centralized configuration for the application."""

    # Model Settings
    TTS_MODEL: str = os.getenv("TTS_MODEL", "suno/bark-small")
    DEVICE: Optional[str] = os.getenv("DEVICE", None)  # "cuda", "mps", or None for CPU

    # Translation Settings
    MAX_TRANSLATION_LENGTH: int = int(os.getenv("MAX_TRANSLATION_LENGTH", "400"))
    TRANSLATION_CACHE_SIZE: int = int(os.getenv("TRANSLATION_CACHE_SIZE", "8"))

    # Corpus Settings
    CORPUS_FILE: str = os.getenv("CORPUS_FILE", "Corpus.txt")
    CORPUS_ENCODING: str = os.getenv("CORPUS_ENCODING", "utf-8")

    # Gradio Settings
    QUEUE_MAX_SIZE: int = int(os.getenv("QUEUE_MAX_SIZE", "32"))
    SHARE: bool = os.getenv("SHARE", "false").lower() == "true"
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "7860"))

    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", None)

    # Feature Flags
    ENABLE_ANALYTICS: bool = os.getenv("ENABLE_ANALYTICS", "false").lower() == "true"
    ENABLE_PHRASE_SUGGESTIONS: bool = os.getenv("ENABLE_PHRASE_SUGGESTIONS", "true").lower() == "true"

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings."""
        try:
            assert cls.MAX_TRANSLATION_LENGTH > 0, "MAX_TRANSLATION_LENGTH must be positive"
            assert cls.TRANSLATION_CACHE_SIZE > 0, "TRANSLATION_CACHE_SIZE must be positive"
            assert cls.QUEUE_MAX_SIZE > 0, "QUEUE_MAX_SIZE must be positive"
            assert cls.SERVER_PORT > 0, "SERVER_PORT must be positive"
            return True
        except AssertionError as e:
            logging.error(f"Configuration validation failed: {e}")
            return False


# Logging Setup
def setup_logging() -> logging.Logger:
    """
    Setup comprehensive logging with file and console handlers.

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger("iga_tts")
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with detailed formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler if configured
    if Config.LOG_FILE:
        try:
            log_path = Path(Config.LOG_FILE)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(Config.LOG_FILE)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")

    return logger


# Initialize logger
logger = setup_logging()


# Custom Exceptions
class IgaTTSError(Exception):
    """Base exception for Iga TTS application."""
    pass


class TranslationError(IgaTTSError):
    """Exception raised for translation failures."""
    pass


class TTSError(IgaTTSError):
    """Exception raised for TTS failures."""
    pass


class CorpusError(IgaTTSError):
    """Exception raised for corpus loading failures."""
    pass


class ConfigurationError(IgaTTSError):
    """Exception raised for configuration issues."""
    pass
