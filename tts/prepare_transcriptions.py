#!/usr/bin/env python3
"""
Transcription Data Preparation Script

Prepares a JSON dataset mapping phrase IDs to Kinyarwanda text for TTS generation.
Extracts transcriptions from phrases.ts and organizes by category.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

from utils import (
    parse_phrases_ts,
    get_recorded_phrase_ids,
    get_missing_phrase_ids,
    get_coverage_stats,
    ensure_output_dir,
    save_json_dataset,
    PHRASES_FILE,
)


def extract_phrase_categories() -> Dict[int, str]:
    """
    Extract phrase category assignments from phrases.ts

    Returns:
        dict: {phrase_id: category_name}
    """
    all_phrases = parse_phrases_ts()
    categories = {}

    # Read phrases.ts to extract category field for each phrase
    with open(PHRASES_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract phrases array and parse category field
    import re
    phrase_pattern = r'id:\s*(\d+)[^}]*?category:\s*["`]([^"`]+)["`]'

    for match in re.finditer(phrase_pattern, content):
        phrase_id = int(match.group(1))
        category = match.group(2)
        categories[phrase_id] = category

    return categories


def create_transcription_dataset() -> Dict:
    """
    Create a comprehensive dataset with all phrases and their recording status

    Returns:
        dict: Dataset containing recorded, missing, and reference phrases organized by category
    """
    print("📖 Parsing phrases.ts...")
    all_phrases = parse_phrases_ts()
    print(f"✓ Found {len(all_phrases)} total phrases")

    print("\n🎤 Checking recorded audio...")
    recorded_ids = set(get_recorded_phrase_ids())
    print(f"✓ Found {len(recorded_ids)} recorded phrases")

    print("\n📋 Identifying missing phrases...")
    missing_ids = get_missing_phrase_ids()
    print(f"✓ Found {len(missing_ids)} missing phrases")

    print("\n📂 Extracting categories...")
    categories = extract_phrase_categories()
    print(f"✓ Found categories for {len(categories)} phrases")

    # Organize by category
    recorded_by_category = {}
    missing_by_category = {}

    for phrase_id, text in all_phrases.items():
        category = categories.get(phrase_id, "Unknown")

        if phrase_id in recorded_ids:
            if category not in recorded_by_category:
                recorded_by_category[category] = {}
            recorded_by_category[category][phrase_id] = text
        else:
            if category not in missing_by_category:
                missing_by_category[category] = {}
            missing_by_category[category][phrase_id] = text

    # Create comprehensive dataset
    dataset = {
        "metadata": {
            "total_phrases": len(all_phrases),
            "recorded_count": len(recorded_ids),
            "missing_count": len(missing_ids),
            "coverage_percent": round(len(recorded_ids) / len(all_phrases) * 100, 2),
            "language": "rw",
            "target_sample_rate": 24000,
            "channels": 1,
        },
        "recorded_by_category": recorded_by_category,
        "missing_by_category": missing_by_category,
        # Flat dictionaries for convenience
        "recorded": {pid: all_phrases[pid] for pid in recorded_ids if pid in all_phrases},
        "missing": {pid: all_phrases[pid] for pid in missing_ids},
    }

    return dataset


def save_transcription_dataset(dataset: Dict, output_path: Path) -> None:
    """Save transcription dataset to JSON"""
    save_json_dataset(dataset, output_path)


def display_summary(dataset: Dict) -> None:
    """Display dataset summary with category breakdowns"""
    meta = dataset["metadata"]

    print("\n" + "=" * 70)
    print("TRANSCRIPTION DATASET SUMMARY")
    print("=" * 70)
    print(f"Total phrases:        {meta['total_phrases']}")
    print(f"Recorded:             {meta['recorded_count']} ({meta['coverage_percent']}%)")
    print(f"Missing:              {meta['missing_count']}")
    print(f"Language:             {meta['language']}")
    print(f"Target sample rate:   {meta['target_sample_rate']} Hz")
    print(f"Channels:             {meta['channels']}")
    print("=" * 70)

    # Show category breakdown
    print("\nRecorded by category:")
    for category, phrases in sorted(dataset["recorded_by_category"].items()):
        print(f"  • {category}: {len(phrases)} phrases")

    print("\nMissing by category:")
    for category, phrases in sorted(dataset["missing_by_category"].items()):
        print(f"  • {category}: {len(phrases)} phrases")

    # Show sample missing phrases
    print("\nFirst 15 missing phrases (sample):")
    count = 0
    for category in sorted(dataset["missing_by_category"].keys()):
        for pid, text in sorted(dataset["missing_by_category"][category].items())[:3]:
            print(f"  ID {pid:5d} ({category:25s}): {text[:50]}")
            count += 1
            if count >= 15:
                break
        if count >= 15:
            break

    if dataset["metadata"]["missing_count"] > 15:
        print(f"  ... and {dataset['metadata']['missing_count'] - 15} more")


def main():
    """Main execution"""
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║     Kinyarwanda TTS Transcription Data Preparation                ║")
    print("╚═══════════════════════════════════════════════════════════════════╝\n")

    try:
        # Ensure output directory exists
        output_dir = ensure_output_dir()

        # Create dataset
        print("Creating transcription dataset...")
        dataset = create_transcription_dataset()

        # Save to JSON
        output_path = output_dir / "transcriptions.json"
        save_transcription_dataset(dataset, output_path)

        # Display summary
        display_summary(dataset)

        print("\n✅ Transcription data prepared successfully!")
        print(f"\n📊 Dataset saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Extract voice profiles from {dataset['metadata']['recorded_count']} recorded phrases")
        print(f"  2. Generate audio for {dataset['metadata']['missing_count']} missing phrases")
        print(f"  3. Run: npm run corpus:build")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
