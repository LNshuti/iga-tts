#!/usr/bin/env python3
"""
Test Qwen3-TTS voice cloning with a single phrase
"""
import os
import sys
import logging
from pathlib import Path
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

from utils import get_corpus_path, get_missing_phrase_ids, get_generated_audio_path, parse_phrases_ts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_voice_clone():
    """Test voice cloning from reference audio"""
    try:
        # Load model
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Loading Qwen3-TTS on device: {device}")

        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map=device,
            dtype=torch.float32,
        )
        logger.info("✓ Model loaded")

        # Get phrases
        phrases_dict = parse_phrases_ts()  # Returns Dict[int, str] = {id: kinyarwanda_text}
        missing_ids = get_missing_phrase_ids()

        # Find a reference phrase (recorded audio)
        recorded_ids = [pid for pid in phrases_dict.keys() if pid not in missing_ids]
        if not recorded_ids:
            logger.error("No recorded phrases found for voice cloning")
            return False

        reference_id = recorded_ids[0]
        reference_text = phrases_dict[reference_id]
        reference_path = get_corpus_path(reference_id)

        if not reference_path.exists():
            logger.error(f"Reference audio not found: {reference_path}")
            return False

        logger.info(f"Using reference phrase #{reference_id}: '{reference_text}'")
        logger.info(f"Reference audio: {reference_path}")

        # Select target phrase to generate
        target_id = missing_ids[0] if missing_ids else recorded_ids[-1]
        target_text = phrases_dict[target_id]

        logger.info(f"Generating audio for phrase #{target_id}: '{target_text}'")
        logger.info(f"Target text: {target_text}")

        # Generate audio using voice cloning
        logger.info("Creating voice clone prompt...")
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=str(reference_path),
            ref_text=reference_text,
        )
        logger.info("✓ Voice clone prompt created")

        logger.info("Generating audio...")
        wavs, sr = model.generate_voice_clone(
            text=target_text,
            language="English",
            voice_clone_prompt=prompt_items,
        )
        logger.info(f"✓ Audio generated (sample rate: {sr}Hz, duration: {len(wavs[0])/sr:.2f}s)")

        # Save output
        output_path = get_generated_audio_path(target_id, "mp3")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(output_path), wavs[0], sr)
        logger.info(f"✓ Saved to: {output_path}")

        return True

    except Exception as e:
        logger.error(f"✗ Voice cloning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_voice_clone()
    sys.exit(0 if success else 1)
