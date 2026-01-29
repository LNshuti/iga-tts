"""
Utility functions for Kinyarwanda TTS scripts
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus" / "kinyarwanda"
PHRASES_FILE = PROJECT_ROOT / "lib" / "phrases.ts"
TTS_DIR = Path(__file__).parent
OUTPUT_DIR = TTS_DIR / "generated_audio"
MODELS_DIR = TTS_DIR / "models"


def parse_phrases_ts() -> Dict[int, str]:
    """
    Parse lib/phrases.ts to extract phrase IDs and Kinyarwanda text

    Returns:
        dict: {phrase_id: kinyarwanda_text}

    Raises:
        FileNotFoundError: If phrases.ts not found
        ValueError: If phrases array not found in file
    """
    if not PHRASES_FILE.exists():
        raise FileNotFoundError(f"Phrases file not found: {PHRASES_FILE}")

    logger.info(f"Parsing phrases from: {PHRASES_FILE}")

    with open(PHRASES_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Look for phrases array - handle both export and const declarations
    match = re.search(
        r'(?:const _phrases|export const phrases):\s*Phrase\[\]\s*=\s*\[([\s\S]*?)^\s*\];',
        content,
        re.MULTILINE
    )

    if not match:
        # Try alternative pattern for array closing
        match = re.search(
            r'(?:const _phrases|export const phrases):\s*Phrase\[\]\s*=\s*\[([\s\S]*?)\s*\];',
            content
        )

    if not match:
        raise ValueError("Could not find phrases array in phrases.ts")

    phrases_str = match.group(1)
    phrases = {}

    # Parse each phrase object - more robust regex
    # Matches: id: NUMBER, ... kinyarwanda: "TEXT"
    pattern = r'id:\s*(\d+)[^}]*?kinyarwanda:\s*["`]([^"`]+)["`]'

    for phrase_match in re.finditer(pattern, phrases_str):
        phrase_id = int(phrase_match.group(1))
        kinyarwanda = phrase_match.group(2)
        phrases[phrase_id] = kinyarwanda

    logger.info(f"✓ Parsed {len(phrases)} phrases from phrases.ts")
    return phrases


def get_recorded_phrase_ids() -> List[int]:
    """
    Get list of phrase IDs that already have audio

    Returns:
        list: Sorted list of phrase IDs with WAV files
    """
    if not CORPUS_DIR.exists():
        logger.warning(f"Corpus directory not found: {CORPUS_DIR}")
        return []

    wav_files = list(CORPUS_DIR.glob("*.wav"))
    phrase_ids = []

    for wav_file in wav_files:
        try:
            phrase_id = int(wav_file.stem)
            phrase_ids.append(phrase_id)
        except ValueError:
            logger.debug(f"Skipping non-numeric filename: {wav_file.name}")
            continue

    return sorted(phrase_ids)


def get_missing_phrase_ids() -> List[int]:
    """
    Get list of phrase IDs that need audio generation

    Returns:
        list: Sorted list of missing phrase IDs
    """
    all_phrases = parse_phrases_ts()
    recorded_ids = set(get_recorded_phrase_ids())
    missing_ids = [pid for pid in all_phrases.keys() if pid not in recorded_ids]

    logger.info(f"✓ Found {len(missing_ids)} missing phrases out of {len(all_phrases)}")
    return sorted(missing_ids)


def ensure_output_dir() -> Path:
    """
    Create output directory for generated audio if it doesn't exist

    Returns:
        Path: Path to output directory
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directory ready: {OUTPUT_DIR}")
    return OUTPUT_DIR


def ensure_models_dir() -> Path:
    """
    Create models directory for cached TTS models

    Returns:
        Path: Path to models directory
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def get_reference_audio_phrases() -> Dict[int, Tuple[int, str]]:
    """
    Get recorded phrases suitable for voice profile extraction.
    Prefers common phrases (greetings, basic words) for reference.

    Returns:
        dict: {phrase_id: (duration_estimate, kinyarwanda_text)}
    """
    all_phrases = parse_phrases_ts()
    recorded_ids = set(get_recorded_phrase_ids())

    # Prioritize categories for voice profile extraction
    priority_ids = {
        1, 2, 3, 4, 5, 10, 20, 21, 22, 23, 24, 25, 26, 27,  # Greetings & basics
        47, 48, 49,  # Time expressions
    }

    reference_phrases = {}
    for pid in priority_ids:
        if pid in recorded_ids and pid in all_phrases:
            reference_phrases[pid] = all_phrases[pid]

    logger.info(f"✓ Selected {len(reference_phrases)} reference phrases for voice profile")
    return reference_phrases


def get_corpus_path(phrase_id: int) -> Path:
    """Get the corpus WAV path for a phrase ID"""
    return CORPUS_DIR / f"{phrase_id}.wav"


def get_generated_audio_path(phrase_id: int) -> Path:
    """Get the output WAV path for a generated phrase"""
    return OUTPUT_DIR / f"{phrase_id}.wav"


def load_json_dataset(filepath: Path) -> Dict:
    """
    Load JSON dataset file

    Args:
        filepath: Path to JSON file

    Returns:
        dict: Parsed JSON data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_dataset(data: Dict, filepath: Path) -> None:
    """
    Save data to JSON file

    Args:
        data: Dictionary to save
        filepath: Path to output file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Saved dataset to: {filepath}")


def get_coverage_stats() -> Dict:
    """
    Calculate corpus coverage statistics

    Returns:
        dict: Coverage metrics by category
    """
    all_phrases = parse_phrases_ts()
    recorded_ids = set(get_recorded_phrase_ids())

    return {
        "total_phrases": len(all_phrases),
        "recorded_count": len(recorded_ids),
        "missing_count": len(all_phrases) - len(recorded_ids),
        "coverage_percent": round(len(recorded_ids) / len(all_phrases) * 100, 2) if all_phrases else 0
    }
