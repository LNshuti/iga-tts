#!/usr/bin/env python3
"""
Simple TTS Audio Generator for Kinyarwanda

Provides multiple fallback approaches for TTS generation:
1. Multilingual MMS-TTS (Facebook/Meta - supports Kinyarwanda)
2. Qwen3-TTS (if available with voice cloning)
3. Placeholder generation for testing

This script prioritizes available models and degrades gracefully.
"""

# Suppress deprecation warnings from torch/transformers to prevent module loading issues
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*torch.jit.script.*')

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import argparse

import numpy as np
import soundfile as sf

from utils import (
    parse_phrases_ts,
    get_recorded_phrase_ids,
    get_missing_phrase_ids,
    ensure_output_dir,
    get_generated_audio_path,
)
from audio_processor import (
    load_audio,
    normalize_loudness,
    trim_silence,
    save_audio,
    TARGET_SAMPLE_RATE,
    concatenate_audio,
)

logger = logging.getLogger(__name__)

# Model configurations
MMS_MODEL = "facebook/mms-tts-kin"  # Kinyarwanda-specific TTS
QWEN_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


class SimpleKinyarwandaTTS:
    """Simple TTS generator with fallback strategies"""

    def __init__(self):
        self.device = self._detect_device()
        self.processor = None
        self.model = None
        self.model_name = None
        self.method = None  # Track which method is being used

    def _detect_device(self) -> str:
        """Detect available compute device"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def try_load_mms_tts(self) -> bool:
        """Try to load Facebook's MMS-TTS (has Kinyarwanda support)"""
        try:
            logger.info("Attempting to load Facebook MMS-TTS (Kinyarwanda)...")
            from transformers import pipeline

            # MMS-TTS pipeline
            self.model = pipeline(
                "text-to-speech",
                model=MMS_MODEL,
                device=0 if self.device == "cuda" else -1,
            )
            self.model_name = MMS_MODEL
            self.method = "mms-tts"
            logger.info("✓ Loaded Facebook MMS-TTS")
            return True

        except Exception as e:
            logger.debug(f"MMS-TTS load failed: {e}")
            return False

    def try_load_qwen_tts(self) -> bool:
        """Try to load Qwen3-TTS with voice cloning"""
        try:
            logger.info("Attempting to load Qwen3-TTS...")
            from transformers import AutoProcessor, AutoModel

            self.processor = AutoProcessor.from_pretrained(
                QWEN_MODEL,
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                QWEN_MODEL,
                trust_remote_code=True,
            )

            if self.device == "cuda":
                self.model = self.model.cuda()

            self.model_name = QWEN_MODEL
            self.method = "qwen3-tts"
            logger.info("✓ Loaded Qwen3-TTS")
            return True

        except Exception as e:
            logger.debug(f"Qwen3-TTS load failed: {e}")
            return False

    def try_load_gpt_solovev(self) -> bool:
        """Try to load GPT-SoVits or similar alternative"""
        try:
            logger.info("Attempting to load alternative TTS model...")
            from transformers import pipeline

            # Use a general multilingual TTS
            self.model = pipeline(
                "text-to-speech",
                model="facebook/mms-tts",  # Falls back to default if specific not available
                device=0 if self.device == "cuda" else -1,
            )
            self.model_name = "facebook/mms-tts"
            self.method = "mms-tts-general"
            logger.info("✓ Loaded general MMS-TTS")
            return True

        except Exception as e:
            logger.debug(f"Alternative TTS load failed: {e}")
            return False

    def load_any_model(self) -> bool:
        """Try to load any available TTS model"""
        logger.info("Attempting to load TTS model (trying multiple strategies)...")

        strategies = [
            self.try_load_mms_tts,
            self.try_load_qwen_tts,
            self.try_load_gpt_solovev,
        ]

        for strategy in strategies:
            if strategy():
                logger.info(f"Successfully loaded model using: {strategy.__name__}")
                return True

        logger.error("Could not load any TTS model")
        return False

    def synthesize_mms_tts(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using Facebook MMS-TTS"""
        try:
            output = self.model(text, forward_params={"speaker_embeddings": None})
            audio = np.array(output["audio"], dtype=np.float32)

            if audio.ndim > 1:
                audio = audio.squeeze()

            # Resample if needed
            if output.get("sampling_rate", 16000) != TARGET_SAMPLE_RATE:
                import librosa
                audio = librosa.resample(
                    audio,
                    orig_sr=output.get("sampling_rate", 16000),
                    target_sr=TARGET_SAMPLE_RATE
                )

            return audio

        except Exception as e:
            logger.error(f"MMS-TTS synthesis error: {e}")
            return None

    def synthesize_qwen_tts(self, text: str, reference_audio: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Synthesize using Qwen3-TTS with optional voice cloning"""
        try:
            import torch

            inputs = self.processor(
                text=text,
                voice=reference_audio,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
            )

            if self.device == "cuda":
                inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model.generate(**inputs)

            audio = output[0].cpu().numpy()

            if audio.ndim > 1:
                audio = audio.squeeze()

            return audio

        except Exception as e:
            logger.error(f"Qwen3-TTS synthesis error: {e}")
            return None

    def synthesize(self, text: str, reference_audio: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Synthesize audio using loaded model"""
        if not self.model:
            logger.error("No model loaded")
            return None

        if self.method == "mms-tts" or self.method == "mms-tts-general":
            return self.synthesize_mms_tts(text)
        elif self.method == "qwen3-tts":
            return self.synthesize_qwen_tts(text, reference_audio)
        else:
            logger.error(f"Unknown synthesis method: {self.method}")
            return None


def generate_with_fallback(
    generator: SimpleKinyarwandaTTS,
    phrase_ids: List[int],
    mode: str = "batch",
    sample_count: int = 10,
) -> Dict[int, Dict]:
    """Generate audio with fallback strategies"""
    all_phrases = parse_phrases_ts()
    results = {}

    if mode == "pilot":
        phrase_ids = phrase_ids[:sample_count]

    logger.info(f"Generating audio for {len(phrase_ids)} phrases using {generator.method}")

    for i, phrase_id in enumerate(phrase_ids, 1):
        if phrase_id not in all_phrases:
            continue

        text = all_phrases[phrase_id]

        if (i - 1) % max(1, len(phrase_ids) // 10) == 0:
            logger.info(f"Progress: {i}/{len(phrase_ids)}")

        try:
            # Synthesize
            audio = generator.synthesize(text)

            if audio is None:
                results[phrase_id] = {"status": "failed", "error": "synthesis failed"}
                continue

            # Process audio
            audio = normalize_loudness(audio)
            audio = trim_silence(audio)

            # Save
            output_path = get_generated_audio_path(phrase_id)
            if save_audio(audio, output_path):
                results[phrase_id] = {
                    "status": "success",
                    "path": str(output_path),
                    "duration": len(audio) / TARGET_SAMPLE_RATE,
                    "method": generator.method,
                }
            else:
                results[phrase_id] = {"status": "failed", "error": "save failed"}

        except Exception as e:
            logger.error(f"Error synthesizing phrase {phrase_id}: {e}")
            results[phrase_id] = {"status": "failed", "error": str(e)}

    return results


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Simple Kinyarwanda TTS audio generator"
    )
    parser.add_argument("--mode", choices=["pilot", "batch"], default="pilot",
                       help="Generation mode")
    parser.add_argument("--sample-count", type=int, default=10,
                       help="Number of samples for pilot mode")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║     Simple Kinyarwanda TTS Audio Generator                       ║")
    print("╚═══════════════════════════════════════════════════════════════════╝\n")

    try:
        # Load model
        generator = SimpleKinyarwandaTTS()
        if not generator.load_any_model():
            logger.error("Failed to load any TTS model")
            logger.info("\nTo fix this:")
            logger.info("  1. Ensure PyTorch is installed: pip install torch")
            logger.info("  2. Ensure transformers is installed: pip install transformers")
            logger.info("  3. Check internet connection (models are downloaded from HuggingFace)")
            return 1

        logger.info(f"Using method: {generator.method}")

        # Get phrases to generate
        missing_ids = get_missing_phrase_ids()
        if not missing_ids:
            logger.error("No missing phrases found")
            return 1

        # Generate audio
        results = generate_with_fallback(
            generator,
            missing_ids,
            mode=args.mode,
            sample_count=args.sample_count,
        )

        # Save results
        ensure_output_dir()
        results_path = ensure_output_dir() / f"results_{args.mode}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Summary
        success = sum(1 for r in results.values() if r["status"] == "success")
        failed = sum(1 for r in results.values() if r["status"] == "failed")

        print(f"\n{'='*70}")
        print("GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Success:    {success}")
        print(f"Failed:     {failed}")
        print(f"Total:      {len(results)}")
        print(f"{'='*70}\n")

        return 0 if failed == 0 else 1

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
