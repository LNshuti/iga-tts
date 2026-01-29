#!/usr/bin/env python3
"""
Minimal test case for Qwen3-TTS model loading
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_load():
    """Test if Qwen3-TTS model can be loaded"""
    try:
        import torch
        from transformers import AutoProcessor, AutoModel

        logger.info("PyTorch version:", torch.__version__)
        logger.info("Device: mps" if torch.backends.mps.is_available() else "Device: cpu")

        model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

        logger.info(f"Loading processor from {model_name}")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        logger.info("✓ Processor loaded")

        logger.info(f"Loading model {model_name}")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        logger.info("✓ Model loaded")

        # Try to move to MPS
        if torch.backends.mps.is_available():
            logger.info("Moving to MPS device")
            model = model.to("mps")
            logger.info("✓ Model on MPS")

        logger.info("✓✓✓ SUCCESS - Qwen3-TTS loads correctly!")
        return True

    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_load()
    sys.exit(0 if success else 1)
