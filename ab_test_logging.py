"""
A/B test metric logging and data collection using DuckDB.

Manages event logging, schema creation, and data queries for Bayesian optimization.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import csv

import duckdb

logger = logging.getLogger(__name__)


class ABTestLogger:
    """DuckDB-based logging for A/B test metrics."""

    def __init__(self, db_path: str = "ab_test.db"):
        """
        Initialize DuckDB connection and schema.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path
        self.connection = None
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize DuckDB connection and create schema if missing."""
        try:
            self.connection = duckdb.connect(self.db_path)
            self._create_schema()
            logger.info(f"Initialized DuckDB at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB: {e}")
            raise

    def _create_schema(self) -> None:
        """Create DuckDB tables if they don't exist."""
        try:
            # Users table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR PRIMARY KEY,
                    current_variant_id VARCHAR,
                    created_at TIMESTAMP,
                    last_active TIMESTAMP
                )
            """)

            # Phrase attempts table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS phrase_attempts (
                    user_id VARCHAR,
                    variant_id VARCHAR,
                    phrase VARCHAR,
                    duration_ms INTEGER,
                    success BOOLEAN,
                    timestamp TIMESTAMP
                )
            """)

            # Sessions table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id VARCHAR,
                    variant_id VARCHAR,
                    total_xp INTEGER,
                    session_duration_min FLOAT,
                    num_phrases INTEGER,
                    completed_at TIMESTAMP
                )
            """)

            # Feedback table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    user_id VARCHAR,
                    variant_id VARCHAR,
                    rating INTEGER,
                    timestamp TIMESTAMP
                )
            """)

            logger.info("DuckDB schema created/verified")

        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise

    def log_phrase_attempt(
        self,
        user_id: str,
        variant_id: str,
        phrase: str,
        duration_ms: int,
        success: bool,
    ) -> None:
        """
        Log individual phrase attempt.

        Args:
            user_id: Anonymized user session ID
            variant_id: JSON-encoded variant parameters
            phrase: The phrase text
            duration_ms: Time to complete phrase in milliseconds
            success: Whether user succeeded
        """
        try:
            self.connection.execute(
                """
                INSERT INTO phrase_attempts
                (user_id, variant_id, phrase, duration_ms, success, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [user_id, variant_id, phrase, duration_ms, success, datetime.utcnow()],
            )
        except Exception as e:
            logger.error(f"Failed to log phrase attempt: {e}")

    def log_session_end(
        self,
        user_id: str,
        variant_id: str,
        total_xp: int,
        session_duration_min: float,
        num_phrases: int,
    ) -> None:
        """
        Log session summary.

        Args:
            user_id: Anonymized user session ID
            variant_id: JSON-encoded variant parameters
            total_xp: Total XP earned in session
            session_duration_min: Session duration in minutes
            num_phrases: Number of phrases completed
        """
        try:
            self.connection.execute(
                """
                INSERT INTO sessions
                (user_id, variant_id, total_xp, session_duration_min, num_phrases, completed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    user_id,
                    variant_id,
                    total_xp,
                    session_duration_min,
                    num_phrases,
                    datetime.utcnow(),
                ],
            )
        except Exception as e:
            logger.error(f"Failed to log session end: {e}")

    def log_user_feedback(
        self, user_id: str, variant_id: str, rating: int
    ) -> None:
        """
        Log explicit user satisfaction rating.

        Args:
            user_id: Anonymized user session ID
            variant_id: JSON-encoded variant parameters
            rating: User rating (1-5 scale)
        """
        if not (1 <= rating <= 5):
            logger.warning(f"Invalid rating {rating}, ignoring")
            return

        try:
            self.connection.execute(
                """
                INSERT INTO feedback
                (user_id, variant_id, rating, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                [user_id, variant_id, rating, datetime.utcnow()],
            )
        except Exception as e:
            logger.error(f"Failed to log user feedback: {e}")

    def ensure_user_exists(self, user_id: str) -> None:
        """Create user entry if it doesn't exist."""
        try:
            self.connection.execute(
                """
                INSERT INTO users (user_id, created_at, last_active)
                VALUES (?, ?, ?)
                ON CONFLICT (user_id) DO UPDATE SET last_active = ?
                """,
                [user_id, datetime.utcnow(), datetime.utcnow(), datetime.utcnow()],
            )
        except Exception as e:
            logger.error(f"Failed to ensure user exists: {e}")

    def update_user_variant(self, user_id: str, variant_id: str) -> None:
        """Update user's current variant assignment."""
        try:
            self.connection.execute(
                """
                UPDATE users
                SET current_variant_id = ?, last_active = ?
                WHERE user_id = ?
                """,
                [variant_id, datetime.utcnow(), user_id],
            )
        except Exception as e:
            logger.error(f"Failed to update user variant: {e}")

    def get_user_variant(self, user_id: str) -> Optional[str]:
        """Get user's current variant assignment."""
        try:
            result = self.connection.execute(
                """
                SELECT current_variant_id FROM users WHERE user_id = ?
                """,
                [user_id],
            ).fetchall()

            if result:
                return result[0][0]
            return None

        except Exception as e:
            logger.error(f"Failed to get user variant: {e}")
            return None

    def get_metrics_for_optimization(self) -> List[Dict]:
        """
        Get aggregated metrics per variant for Bayesian optimization.

        Returns:
            List of metric dictionaries:
            {
                variant_id: str,
                engagement_score: float,  # normalized 0-1
                retention_score: float,   # normalized 0-1
                satisfaction_score: float, # normalized 0-1
                count: int
            }
        """
        try:
            # Aggregate metrics per variant
            query = """
            WITH variant_stats AS (
                SELECT
                    variant_id,
                    COUNT(DISTINCT user_id) as num_users,
                    COUNT(*) as num_phrases,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_ms) as avg_duration_ms
                FROM phrase_attempts
                GROUP BY variant_id
            ),
            session_stats AS (
                SELECT
                    variant_id,
                    COUNT(*) as num_sessions,
                    AVG(total_xp) as avg_xp,
                    AVG(session_duration_min) as avg_session_min,
                    COUNT(DISTINCT user_id) as returning_users
                FROM sessions
                GROUP BY variant_id
            ),
            feedback_stats AS (
                SELECT
                    variant_id,
                    AVG(rating) as avg_rating,
                    COUNT(*) as num_ratings
                FROM feedback
                GROUP BY variant_id
            )
            SELECT
                v.variant_id,
                CASE
                    WHEN s.num_sessions > 0 THEN
                        (COALESCE(s.num_sessions, 0) / 100.0) *
                        (COALESCE(s.avg_session_min, 0) / 30.0)
                    ELSE 0.0
                END as engagement_score,
                CASE
                    WHEN s.num_sessions > 0 THEN
                        (COALESCE(s.avg_xp, 0) / 1000.0) *
                        (COALESCE(s.returning_users, 0) / COALESCE(v.num_users, 1))
                    ELSE 0.0
                END as retention_score,
                COALESCE(f.avg_rating / 5.0, 0.0) as satisfaction_score,
                COALESCE(v.num_phrases, 0) as count
            FROM variant_stats v
            LEFT JOIN session_stats s ON v.variant_id = s.variant_id
            LEFT JOIN feedback_stats f ON v.variant_id = f.variant_id
            WHERE v.num_phrases > 0
            """

            results = self.connection.execute(query).fetchall()
            columns = ["variant_id", "engagement_score", "retention_score",
                      "satisfaction_score", "count"]

            metrics = [
                {
                    "variant_id": row[0],
                    "engagement_score": row[1],
                    "retention_score": row[2],
                    "satisfaction_score": row[3],
                    "count": row[4],
                }
                for row in results
            ]

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []

    def get_metrics_summary(self) -> List[Dict]:
        """Get summary of metrics per variant for display."""
        try:
            metrics = self.get_metrics_for_optimization()
            summary = []

            for m in metrics:
                summary.append({
                    "Variant": m["variant_id"][:50],  # Truncate for display
                    "Engagement": f"{m['engagement_score']:.3f}",
                    "Retention": f"{m['retention_score']:.3f}",
                    "Satisfaction": f"{m['satisfaction_score']:.3f}",
                    "Sample Size": m["count"],
                })

            return summary

        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return []

    def get_median_performance(self) -> float:
        """Get median overall performance across all variants."""
        try:
            metrics = self.get_metrics_for_optimization()

            if not metrics:
                return 0.5

            # Compute scalarized objective for each variant
            objectives = []
            for m in metrics:
                obj = (
                    0.4 * m["engagement_score"]
                    + 0.4 * m["retention_score"]
                    + 0.2 * m["satisfaction_score"]
                )
                objectives.append(obj)

            # Return median
            objectives.sort()
            return objectives[len(objectives) // 2]

        except Exception as e:
            logger.error(f"Failed to get median performance: {e}")
            return 0.5

    def export_to_csv(self, path: str) -> None:
        """Export all metrics to CSV file."""
        try:
            metrics = self.get_metrics_for_optimization()

            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "variant_id",
                        "engagement_score",
                        "retention_score",
                        "satisfaction_score",
                        "count",
                    ],
                )
                writer.writeheader()
                writer.writerows(metrics)

            logger.info(f"Exported metrics to {path}")

        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")

    def get_experiment_state(self) -> Optional[Dict]:
        """Get serialized experiment state for persistence (future enhancement)."""
        # Placeholder for future serialization
        return None

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
