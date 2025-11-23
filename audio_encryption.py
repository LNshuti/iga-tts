"""
Secure audio encryption and decryption for feedback recordings.
Uses AES-256-GCM for authenticated encryption.
"""
import os
import hashlib
import base64
from pathlib import Path
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from config import logger, Config


class AudioEncryption:
    """Handles AES-256-GCM encryption for audio files."""

    def __init__(self, master_key: str = None):
        """Initialize with optional master key. If not provided, uses config."""
        self.master_key = (master_key or os.environ.get("AUDIO_ENCRYPTION_KEY", "default-key")).encode()
        self._key_hash = hashlib.sha256(self.master_key).hexdigest()

    def encrypt_audio(self, audio_data: bytes) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt audio data using AES-256-GCM.

        Returns: (encrypted_data, nonce, tag)
        """
        # Generate random 96-bit nonce
        nonce = os.urandom(12)

        # Derive 32-byte key from master key using SHA256 (simple but secure for this use)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            self.master_key,
            b"audio_feedback_salt",
            100000,
            dklen=32
        )

        # Encrypt with AES-256-GCM
        cipher = AESGCM(key)
        ciphertext = cipher.encrypt(nonce, audio_data, None)

        # GCM appends 16-byte tag
        encrypted_data = ciphertext[:-16]
        tag = ciphertext[-16:]

        return encrypted_data, nonce, tag

    def decrypt_audio(self, encrypted_data: bytes, nonce: bytes, tag: bytes) -> bytes:
        """
        Decrypt audio data using AES-256-GCM.

        Args:
            encrypted_data: The encrypted audio bytes
            nonce: The 96-bit nonce used during encryption
            tag: The 128-bit authentication tag

        Returns: Decrypted audio data
        """
        # Derive key from master key (must match encryption)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            self.master_key,
            b"audio_feedback_salt",
            100000,
            dklen=32
        )

        # Decrypt with AES-256-GCM
        cipher = AESGCM(key)
        ciphertext = encrypted_data + tag

        try:
            audio_data = cipher.decrypt(nonce, ciphertext, None)
            return audio_data
        except Exception as e:
            logger.error(f"Failed to decrypt audio: {e}")
            raise ValueError("Audio decryption failed - corrupted or invalid data")

    def get_key_hash(self) -> str:
        """Get hash of encryption key for rotation tracking."""
        return self._key_hash

    @staticmethod
    def create_file_hash(user_session_id: str, timestamp: str) -> str:
        """Create deterministic filename from user session and timestamp."""
        data = f"{user_session_id}:{timestamp}".encode()
        return hashlib.sha256(data).hexdigest()[:16]


class AudioFileManager:
    """Manages audio file storage with encryption."""

    def __init__(self, recordings_dir: str = "feedback_recordings"):
        """Initialize with recordings directory path."""
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.encryption = AudioEncryption()

        logger.info(f"Audio file manager initialized at {self.recordings_dir}")

    def save_encrypted_audio(
        self,
        audio_bytes: bytes,
        user_session_id: str,
        timestamp: str,
    ) -> Tuple[str, str, str]:
        """
        Save encrypted audio file and return path with encryption metadata.

        Args:
            audio_bytes: Raw audio data
            user_session_id: User session identifier
            timestamp: ISO format timestamp

        Returns: (file_path, nonce_b64, tag_b64) for storage in DB
        """
        # Encrypt audio
        encrypted_data, nonce, tag = self.encryption.encrypt_audio(audio_bytes)

        # Create filename
        file_hash = AudioEncryption.create_file_hash(user_session_id, timestamp)
        filename = f"{file_hash}.encrypted"
        file_path = self.recordings_dir / filename

        # Save encrypted file
        with open(file_path, "wb") as f:
            f.write(encrypted_data)

        logger.info(f"Saved encrypted audio to {file_path}")

        # Encode nonce and tag as base64 for DB storage
        nonce_b64 = base64.b64encode(nonce).decode()
        tag_b64 = base64.b64encode(tag).decode()

        return str(file_path), nonce_b64, tag_b64

    def load_encrypted_audio(
        self,
        file_path: str,
        nonce_b64: str,
        tag_b64: str,
    ) -> bytes:
        """
        Load and decrypt audio file.

        Args:
            file_path: Path to encrypted audio file
            nonce_b64: Base64-encoded nonce
            tag_b64: Base64-encoded authentication tag

        Returns: Decrypted audio bytes
        """
        # Read encrypted file
        with open(file_path, "rb") as f:
            encrypted_data = f.read()

        # Decode nonce and tag
        nonce = base64.b64decode(nonce_b64)
        tag = base64.b64decode(tag_b64)

        # Decrypt
        audio_bytes = self.encryption.decrypt_audio(encrypted_data, nonce, tag)

        logger.info(f"Decrypted audio from {file_path}")
        return audio_bytes

    def delete_audio(self, file_path: str) -> bool:
        """Securely delete audio file by overwriting before deletion."""
        try:
            path = Path(file_path)
            if path.exists():
                # Overwrite with random data before deletion (secure erase)
                size = path.stat().st_size
                with open(path, "wb") as f:
                    f.write(os.urandom(size))
                path.unlink()
                logger.info(f"Securely deleted audio file {file_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete audio file {file_path}: {e}")
        return False
