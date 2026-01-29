#!/usr/bin/env python3
"""
TTS Environment Setup Script
Installs dependencies and downloads Qwen3-TTS models
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")

    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - FAILED")
        print(f"Error: {e}")
        return False

def main():
    """Main setup routine"""
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Kinyarwanda TTS Environment Setup              ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        sys.exit(1)
    if sys.version_info >= (3, 12):
        print("⚠️  Warning: Python 3.12+ may have compatibility issues with torch")
        print("   Recommended: Python 3.8-3.11")
    print(f"✓ Python version: {sys.version.split()[0]}")

    # Install requirements
    if not run_command(
        "pip install -r requirements.txt",
        "Installing Python dependencies"
    ):
        sys.exit(1)

    # Download Qwen3-TTS model (if not cached)
    print("\n📦 Checking Qwen3-TTS model...")
    if not run_command(
        "python -c \"from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-Base', cache_dir='./models')\"",
        "Downloading Qwen3-TTS-0.6B model (~2.5GB)"
    ):
        print("⚠️  Model download failed. Will retry during execution.")

    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Run: npm run tts:prepare-data")
    print("2. Run: npm run tts:generate")
    print("3. Run: npm run corpus:build")

if __name__ == "__main__":
    main()
