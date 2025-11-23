"""
Audio feedback recording and storage using DuckDB.
Manages user feedback submissions with encryption.
"""
import duckdb
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import uuid

from config import Config, logger, CorpusError
from audio_encryption import AudioFileManager


@dataclass
class FeedbackRecord:
    """Represents a feedback recording."""
    id: int
    user_session_id: str
    feedback_type: str  # 'pronunciation' or 'general'
    domain: Optional[str]
    phrase_text: Optional[str]
    duration_seconds: float
    audio_quality_score: float
    created_at: str
    notes: Optional[str]
    file_path: Optional[str] = None
    audio_data: Optional[bytes] = None


class FeedbackStorage:
    """Manages audio feedback storage in DuckDB."""

    def __init__(self, db_file: str = "corpus.duckdb"):
        """Initialize with DuckDB connection."""
        self.db_file = db_file
        self.db: Optional[duckdb.DuckDBPyConnection] = None
        self.audio_manager = AudioFileManager()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        try:
            self.db = duckdb.connect(self.db_file)

            # Create feedback_recordings table if it doesn't exist
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS feedback_recordings (
                id INTEGER PRIMARY KEY,
                user_session_id VARCHAR NOT NULL,
                feedback_type VARCHAR NOT NULL,
                domain VARCHAR,
                phrase_text VARCHAR,
                recorded_audio_path VARCHAR NOT NULL,
                nonce_b64 VARCHAR NOT NULL,
                tag_b64 VARCHAR NOT NULL,
                encryption_key_hash VARCHAR,
                duration_seconds FLOAT,
                audio_quality_score FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT NOW(),
                notes VARCHAR
            )
            """)

            # Create indices
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback_recordings(user_session_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_feedback_domain ON feedback_recordings(domain)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback_recordings(feedback_type)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback_recordings(created_at)")

            self._initialized = True
            logger.info(f"Initialized feedback storage in {self.db_file}")

        except Exception as e:
            logger.error(f"Failed to initialize feedback storage: {e}")
            raise CorpusError(f"Failed to initialize feedback storage: {e}")

    def submit_feedback(
        self,
        audio_bytes: bytes,
        feedback_type: str,
        domain: Optional[str] = None,
        phrase_text: Optional[str] = None,
        duration_seconds: float = 0.0,
        audio_quality_score: float = 0.0,
        notes: Optional[str] = None,
        user_session_id: Optional[str] = None,
    ) -> int:
        """
        Submit audio feedback for storage.

        Args:
            audio_bytes: Raw audio data from recording
            feedback_type: 'pronunciation' or 'general'
            domain: Learning domain (optional)
            phrase_text: The phrase being practiced (optional)
            duration_seconds: Recording duration
            audio_quality_score: Quality metric (0-1)
            notes: User notes (optional)
            user_session_id: User session ID (auto-generated if not provided)

        Returns: Feedback record ID
        """
        if not self._initialized:
            self.initialize()

        # Generate session ID if not provided
        if not user_session_id:
            user_session_id = str(uuid.uuid4())[:8]

        timestamp = datetime.utcnow().isoformat()

        try:
            # Save encrypted audio file
            file_path, nonce_b64, tag_b64 = self.audio_manager.save_encrypted_audio(
                audio_bytes,
                user_session_id,
                timestamp
            )

            # Get encryption key hash
            key_hash = self.audio_manager.encryption.get_key_hash()

            # Get next ID
            max_id_result = self.db.execute("SELECT MAX(id) FROM feedback_recordings").fetchone()
            next_id = (max_id_result[0] or 0) + 1 if max_id_result else 1

            # Insert record into database
            self.db.execute(
                """
                INSERT INTO feedback_recordings (
                    id, user_session_id, feedback_type, domain, phrase_text,
                    recorded_audio_path, nonce_b64, tag_b64, encryption_key_hash,
                    duration_seconds, audio_quality_score, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    next_id,
                    user_session_id,
                    feedback_type,
                    domain,
                    phrase_text,
                    file_path,
                    nonce_b64,
                    tag_b64,
                    key_hash,
                    duration_seconds,
                    audio_quality_score,
                    notes,
                    timestamp,
                ]
            )

            self.db.commit()

            # Get the last inserted ID
            result = self.db.execute("SELECT MAX(id) FROM feedback_recordings").fetchone()
            feedback_id = result[0] if result and result[0] else None

            logger.info(f"Stored feedback record {feedback_id} from session {user_session_id}")
            return feedback_id

        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            raise CorpusError(f"Failed to submit feedback: {e}")

    def get_feedback(self, feedback_id: int) -> Optional[FeedbackRecord]:
        """Retrieve feedback record by ID."""
        if not self._initialized:
            self.initialize()

        try:
            result = self.db.execute(
                """
                SELECT id, user_session_id, feedback_type, domain, phrase_text,
                       duration_seconds, audio_quality_score, created_at, notes,
                       recorded_audio_path, nonce_b64, tag_b64
                FROM feedback_recordings
                WHERE id = ?
                """,
                [feedback_id]
            ).fetchone()

            if not result:
                return None

            record = FeedbackRecord(
                id=result[0],
                user_session_id=result[1],
                feedback_type=result[2],
                domain=result[3],
                phrase_text=result[4],
                duration_seconds=result[5],
                audio_quality_score=result[6],
                created_at=result[7],
                notes=result[8],
                file_path=result[9]
            )

            # Load and decrypt audio
            try:
                audio_data = self.audio_manager.load_encrypted_audio(
                    result[9],  # file_path
                    result[10],  # nonce_b64
                    result[11]   # tag_b64
                )
                record.audio_data = audio_data
            except Exception as e:
                logger.warning(f"Could not load audio for feedback {feedback_id}: {e}")

            return record

        except Exception as e:
            logger.error(f"Failed to retrieve feedback {feedback_id}: {e}")
            return None

    def get_feedback_by_domain(self, domain: str, limit: int = 100) -> List[FeedbackRecord]:
        """Get feedback recordings for a specific domain."""
        if not self._initialized:
            self.initialize()

        try:
            results = self.db.execute(
                """
                SELECT id, user_session_id, feedback_type, domain, phrase_text,
                       duration_seconds, audio_quality_score, created_at, notes,
                       recorded_audio_path
                FROM feedback_recordings
                WHERE domain = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                [domain, limit]
            ).fetchall()

            return [
                FeedbackRecord(
                    id=row[0],
                    user_session_id=row[1],
                    feedback_type=row[2],
                    domain=row[3],
                    phrase_text=row[4],
                    duration_seconds=row[5],
                    audio_quality_score=row[6],
                    created_at=row[7],
                    notes=row[8],
                    file_path=row[9]
                )
                for row in results
            ]

        except Exception as e:
            logger.error(f"Failed to retrieve feedback for domain {domain}: {e}")
            return []

    def get_statistics(self) -> Dict:
        """Get feedback collection statistics."""
        if not self._initialized:
            self.initialize()

        try:
            total = self.db.execute("SELECT COUNT(*) FROM feedback_recordings").fetchone()[0]

            by_type = self.db.execute(
                "SELECT feedback_type, COUNT(*) FROM feedback_recordings GROUP BY feedback_type"
            ).fetchall()

            by_domain = self.db.execute(
                "SELECT domain, COUNT(*) FROM feedback_recordings WHERE domain IS NOT NULL GROUP BY domain"
            ).fetchall()

            avg_duration = self.db.execute(
                "SELECT AVG(duration_seconds) FROM feedback_recordings"
            ).fetchone()[0] or 0.0

            return {
                "total_recordings": total,
                "by_type": {row[0]: row[1] for row in by_type},
                "by_domain": {row[0]: row[1] for row in by_domain},
                "average_duration_seconds": float(avg_duration),
            }

        except Exception as e:
            logger.error(f"Failed to get feedback statistics: {e}")
            return {}

    def close(self) -> None:
        """Close database connection."""
        if self.db:
            self.db.close()


# Global instance
_feedback_instance: Optional[FeedbackStorage] = None


def get_feedback_storage() -> FeedbackStorage:
    """Get or create global feedback storage instance."""
    global _feedback_instance
    if _feedback_instance is None:
        _feedback_instance = FeedbackStorage()
        _feedback_instance.initialize()
    return _feedback_instance
