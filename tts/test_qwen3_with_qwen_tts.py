#!/usr/bin/env python3
"""
Test Qwen3-TTS using the proper qwen-tts library
"""
import os
import logging
import torch
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_load():
    """Test if Qwen3-TTS model loads with qwen-tts library"""
    try:
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {device}")

        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float32,  # Use float32 for MPS (bfloat16 may not be available)
        )

        logger.info("✓ Model loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_model_load()
    sys.exit(0 if success else 1)
