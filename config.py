"""
Configuration for English-Swahili Translator.
"""
import os
import logging

class Config:
    """Application configuration."""

    # Whisper Settings
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small")

    # Device Settings
    DEVICE: str = os.getenv("DEVICE", "auto")

    # Server Settings
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "7860"))

    @classmethod
    def get_device(cls) -> str:
        """Resolve device setting."""
        if cls.DEVICE == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return cls.DEVICE


# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("swahili_translator")
