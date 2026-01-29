#!/usr/bin/env python3
"""
Validate TTS environment and dependencies

Checks that all required libraries are installed and accessible.
"""

import sys
import importlib
from pathlib import Path


def check_import(module_name: str, package_name: str = None) -> bool:
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name

    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {package_name:30s} {version}")
        return True
    except ImportError as e:
        print(f"✗ {package_name:30s} {str(e)}")
        return False


def check_pytorch():
    """Check PyTorch and CUDA availability"""
    try:
        import torch
        print(f"✓ {'PyTorch':30s} {torch.__version__}")

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  └─ CUDA available: {device_name}")
        else:
            print(f"  └─ CUDA not available (CPU mode)")

        return True
    except ImportError as e:
        print(f"✗ {'PyTorch':30s} {str(e)}")
        return False


def check_paths():
    """Check required file paths"""
    print("\nPath Checks:")

    paths = {
        "Corpus dir": Path(__file__).parent.parent.parent / "corpus" / "kinyarwanda",
        "Phrases.ts": Path(__file__).parent.parent.parent / "lib" / "phrases.ts",
        "Corpus status script": Path(__file__).parent.parent / "corpus-status.js",
    }

    all_exist = True
    for name, path in paths.items():
        if path.exists():
            print(f"✓ {name:30s} {path}")
        else:
            print(f"✗ {name:30s} NOT FOUND: {path}")
            all_exist = False

    return all_exist


def main():
    """Main validation"""
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║     Kinyarwanda TTS Environment Validation                        ║")
    print("╚═══════════════════════════════════════════════════════════════════╝\n")

    print("Python Packages:")

    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("tqdm", "tqdm"),
    ]

    all_installed = True
    for module, name in required_packages:
        if not check_import(module, name):
            all_installed = False

    print()
    check_pytorch()

    print()
    paths_ok = check_paths()

    print("\n" + "=" * 70)
    if all_installed and paths_ok:
        print("✅ Environment validation successful!")
        print("\nNext steps:")
        print("  1. Run: python scripts/tts/prepare_transcriptions.py")
        print("  2. Run: python scripts/tts/generate_audio.py --mode pilot")
        print("  3. Run: npm run corpus:build")
        return 0
    else:
        print("❌ Environment validation failed!")
        print("\nFix missing packages with:")
        print("  source scripts/tts/venv/bin/activate")
        print("  pip install -r scripts/tts/requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
