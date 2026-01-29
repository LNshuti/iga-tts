#!/usr/bin/env python3
"""
Kinyarwanda TTS Audio Generation Script

Uses Qwen3-TTS with voice cloning to generate audio for missing phrases.
Supports both pilot testing and batch generation modes.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import soundfile as sf
import librosa

from utils import (
    parse_phrases_ts,
    get_recorded_phrase_ids,
    get_missing_phrase_ids,
    ensure_output_dir,
    ensure_models_dir,
    get_corpus_path,
    get_generated_audio_path,
    CORPUS_DIR,
    TTS_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Target audio specs
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
TARGET_DURATION_MIN = 0.5  # seconds
TARGET_DURATION_MAX = 8.0  # seconds


class KinyarwandaTTSGenerator:
    """Voice cloning and audio generation for Kinyarwanda"""

    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"):
        """
        Initialize the TTS generator

        Args:
            model_name: HuggingFace model identifier for Qwen3-TTS
        """
        self.model_name = model_name
        self.device = self._detect_device()
        self.model = None
        self.processor = None
        self.voice_profile = None

        logger.info(f"Device detected: {self.device}")
        logger.info(f"Model: {model_name}")

    def _detect_device(self) -> str:
        """Detect available compute device"""
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                logger.info("Metal Performance Shaders (MPS) available")
            else:
                device = "cpu"
            return device
        except Exception as e:
            logger.warning(f"Could not detect device: {e}. Using CPU.")
            return "cpu"

    def load_model(self) -> bool:
        """
        Load Qwen3-TTS model from HuggingFace

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading model: {self.model_name}")

            from transformers import AutoProcessor, AutoModel

            # Set HF cache to models directory
            os.environ["HF_HOME"] = str(ensure_models_dir())

            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Move to device
            if self.device == "cuda":
                self.model = self.model.cuda()
            elif self.device == "mps":
                self.model = self.model.to("mps")

            logger.info("✓ Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            return False

    def extract_voice_profile(self, reference_phrase_ids: List[int]) -> bool:
        """
        Extract voice profile from reference recorded audio

        Args:
            reference_phrase_ids: List of phrase IDs to use as reference

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not reference_phrase_ids:
                logger.warning("No reference phrases provided, using first recorded phrase")
                recorded = get_recorded_phrase_ids()
                reference_phrase_ids = [recorded[0]] if recorded else []

            if not reference_phrase_ids:
                logger.error("No recorded phrases available for voice profile")
                return False

            logger.info(f"Extracting voice profile from {len(reference_phrase_ids)} reference phrases")

            reference_audio = []

            for phrase_id in reference_phrase_ids:
                wav_path = get_corpus_path(phrase_id)
                if not wav_path.exists():
                    logger.warning(f"Reference file not found: {wav_path}")
                    continue

                try:
                    # Load and resample to target rate
                    audio, sr = librosa.load(str(wav_path), sr=TARGET_SAMPLE_RATE, mono=True)

                    # Ensure duration is reasonable (0.5-8 seconds)
                    if len(audio) / sr < TARGET_DURATION_MIN:
                        logger.debug(f"Reference audio {phrase_id} too short, padding")
                        # Pad with silence
                        pad_samples = int((TARGET_DURATION_MIN - len(audio) / sr) * sr)
                        audio = np.pad(audio, (0, pad_samples), mode='constant')

                    if len(audio) / sr > TARGET_DURATION_MAX:
                        logger.debug(f"Reference audio {phrase_id} too long, truncating")
                        max_samples = int(TARGET_DURATION_MAX * sr)
                        audio = audio[:max_samples]

                    reference_audio.append(audio)

                except Exception as e:
                    logger.warning(f"Error loading reference {phrase_id}: {e}")
                    continue

            if not reference_audio:
                logger.error("No valid reference audio extracted")
                return False

            # Concatenate reference audio
            self.voice_profile = np.concatenate(reference_audio)

            logger.info(f"✓ Voice profile extracted ({len(self.voice_profile) / TARGET_SAMPLE_RATE:.1f}s)")
            return True

        except Exception as e:
            logger.error(f"✗ Error extracting voice profile: {e}")
            return False

    def synthesize_audio(
        self,
        text: str,
        phrase_id: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Synthesize audio for a given text using voice cloning

        Args:
            text: Kinyarwanda text to synthesize
            phrase_id: Optional phrase ID for logging

        Returns:
            np.ndarray: Audio samples, or None if synthesis failed
        """
        try:
            if not self.model or not self.processor:
                logger.error("Model not loaded. Call load_model() first.")
                return None

            if self.voice_profile is None:
                logger.error("Voice profile not extracted. Call extract_voice_profile() first.")
                return None

            # Prepare input
            inputs = self.processor(
                text=text,
                voice=self.voice_profile,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
            )

            # Move to device if needed
            if self.device == "cuda":
                inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}

            # Generate audio
            import torch
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    do_sample=False,
                    top_k=250,
                    top_p=0.0,
                    temperature=1.0,
                    repetition_penalty=1.0,
                    max_new_tokens=2048,
                )

            # Extract audio (output is typically [batch, channels, samples])
            audio = output[0].cpu().numpy()

            # Handle shape - flatten if needed
            if audio.ndim > 1:
                audio = audio.squeeze()

            # Normalize audio
            if audio.max() > 0:
                audio = audio / audio.max() * 0.95

            # Ensure shape is (samples,)
            if audio.ndim != 1:
                logger.warning(f"Unexpected audio shape {audio.shape}, reshaping")
                audio = audio.reshape(-1)

            return audio

        except Exception as e:
            logger.error(f"✗ Error synthesizing audio for phrase {phrase_id}: {e}")
            return None

    def save_audio(
        self,
        audio: np.ndarray,
        phrase_id: int,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Save synthesized audio to WAV file

        Args:
            audio: Audio samples
            phrase_id: Phrase ID for filename
            output_dir: Output directory (uses default if not specified)

        Returns:
            Path: Output file path, or None if save failed
        """
        try:
            if output_dir is None:
                output_dir = ensure_output_dir()

            output_path = output_dir / f"{phrase_id}.wav"

            # Ensure audio is float32
            audio = audio.astype(np.float32)

            # Save using soundfile
            sf.write(
                str(output_path),
                audio,
                TARGET_SAMPLE_RATE,
                subtype='PCM_16'
            )

            logger.debug(f"✓ Saved: {output_path} ({len(audio) / TARGET_SAMPLE_RATE:.2f}s)")
            return output_path

        except Exception as e:
            logger.error(f"✗ Error saving audio for phrase {phrase_id}: {e}")
            return None

    def generate_pilot(self, num_phrases: int = 10) -> Dict[int, Dict]:
        """
        Generate pilot audio for a small set of phrases to test quality

        Args:
            num_phrases: Number of phrases to generate (default 10)

        Returns:
            dict: Results with success/failure for each phrase
        """
        if not self.model:
            logger.error("Model not loaded")
            return {}

        if not self.voice_profile is not None:
            logger.error("Voice profile not extracted")
            return {}

        all_phrases = parse_phrases_ts()
        missing_ids = get_missing_phrase_ids()

        # Select diverse phrases from different categories
        selected_ids = missing_ids[:num_phrases]

        logger.info(f"Generating pilot audio for {len(selected_ids)} phrases")

        results = {}

        for i, phrase_id in enumerate(selected_ids, 1):
            if phrase_id not in all_phrases:
                continue

            text = all_phrases[phrase_id]

            logger.info(f"[{i}/{len(selected_ids)}] Synthesizing phrase {phrase_id}: {text[:40]}")

            # Synthesize
            audio = self.synthesize_audio(text, phrase_id)

            if audio is None:
                results[phrase_id] = {"status": "failed", "error": "synthesis failed"}
                continue

            # Save
            output_path = self.save_audio(audio, phrase_id)

            if output_path:
                results[phrase_id] = {
                    "status": "success",
                    "path": str(output_path),
                    "duration": len(audio) / TARGET_SAMPLE_RATE,
                    "text": text,
                }
            else:
                results[phrase_id] = {"status": "failed", "error": "save failed"}

        return results

    def generate_batch(self, phrase_ids: Optional[List[int]] = None) -> Dict[int, Dict]:
        """
        Generate audio for a batch of phrases

        Args:
            phrase_ids: List of phrase IDs to generate (uses missing phrases if not specified)

        Returns:
            dict: Results with success/failure for each phrase
        """
        if not self.model:
            logger.error("Model not loaded")
            return {}

        if self.voice_profile is None:
            logger.error("Voice profile not extracted")
            return {}

        all_phrases = parse_phrases_ts()

        if phrase_ids is None:
            phrase_ids = get_missing_phrase_ids()

        logger.info(f"Batch generating audio for {len(phrase_ids)} phrases")

        results = {}
        success_count = 0
        failed_count = 0

        for i, phrase_id in enumerate(phrase_ids, 1):
            if phrase_id not in all_phrases:
                logger.debug(f"Phrase {phrase_id} not found in phrases.ts")
                continue

            text = all_phrases[phrase_id]

            if (i - 1) % 10 == 0:
                logger.info(f"Progress: {i}/{len(phrase_ids)} phrases")

            # Synthesize
            audio = self.synthesize_audio(text, phrase_id)

            if audio is None:
                results[phrase_id] = {"status": "failed", "error": "synthesis failed"}
                failed_count += 1
                continue

            # Save
            output_path = self.save_audio(audio, phrase_id)

            if output_path:
                results[phrase_id] = {
                    "status": "success",
                    "path": str(output_path),
                    "duration": len(audio) / TARGET_SAMPLE_RATE,
                }
                success_count += 1
            else:
                results[phrase_id] = {"status": "failed", "error": "save failed"}
                failed_count += 1

        logger.info(f"\nBatch generation complete: {success_count} success, {failed_count} failed")

        return results


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Generate Kinyarwanda audio using Qwen3-TTS voice cloning"
    )
    parser.add_argument(
        "--mode",
        choices=["pilot", "batch"],
        default="pilot",
        help="Generation mode: pilot (test 10 phrases) or batch (all missing)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--pilot-count",
        type=int,
        default=10,
        help="Number of phrases to generate in pilot mode"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║     Kinyarwanda TTS Voice Cloning Audio Generation                ║")
    print("╚═══════════════════════════════════════════════════════════════════╝\n")

    try:
        # Initialize generator
        generator = KinyarwandaTTSGenerator(model_name=args.model)

        # Load model
        if not generator.load_model():
            logger.error("Failed to load model. Ensure you have GPU and sufficient VRAM.")
            logger.info("Tip: Use CPU with smaller model (0.6B instead of 1.7B)")
            return 1

        # Extract voice profile from recorded audio
        recorded_ids = get_recorded_phrase_ids()
        if not recorded_ids:
            logger.error("No recorded phrases found for voice profile")
            return 1

        # Use first 5 recorded phrases as reference
        reference_ids = recorded_ids[:5]
        if not generator.extract_voice_profile(reference_ids):
            logger.error("Failed to extract voice profile")
            return 1

        # Generate audio
        if args.mode == "pilot":
            logger.info(f"Running in PILOT mode ({args.pilot_count} phrases)")
            results = generator.generate_pilot(num_phrases=args.pilot_count)
        else:
            logger.info("Running in BATCH mode (all missing phrases)")
            results = generator.generate_batch()

        # Save results summary
        summary_path = ensure_output_dir() / f"generation_results_{args.mode}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Results saved to: {summary_path}")

        # Display summary
        success = sum(1 for r in results.values() if r["status"] == "success")
        failed = sum(1 for r in results.values() if r["status"] == "failed")

        print(f"\n{'='*70}")
        print("GENERATION RESULTS")
        print(f"{'='*70}")
        print(f"Mode:       {args.mode}")
        print(f"Success:    {success}")
        print(f"Failed:     {failed}")
        print(f"Total:      {len(results)}")
        print(f"{'='*70}\n")

        if args.mode == "pilot" and success > 0:
            print("✅ Pilot generation successful! Audio samples:")
            for phrase_id, result in list(results.items())[:5]:
                if result["status"] == "success":
                    print(f"  • {phrase_id}: {result['path']}")

        return 0 if failed == 0 else 1

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
