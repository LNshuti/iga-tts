"""
DuckDB-based corpus loader with domain categorization and SRS support.
Replaces file-based corpus.py with intelligent querying.
"""
import duckdb
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

from config import Config, logger, CorpusError


@dataclass
class PhraseRecord:
    """Represents a phrase with all its data."""
    id: int
    text: str
    language: str
    domain: str
    difficulty: str
    translations: Dict[str, str]


class DuckDBCorpus:
    """
    Intelligent corpus loader using DuckDB.
    Supports domain-based organization, SRS scheduling, and semantic search.
    """

    def __init__(self, db_file: Optional[str] = None):
        """Initialize corpus with DuckDB connection."""
        self.db_file = db_file or Path(Config.CORPUS_FILE).parent / "corpus.duckdb"
        self.db: Optional[duckdb.DuckDBPyConnection] = None
        self._loaded = False
        self.phrases: List[PhraseRecord] = []
        self.categories: Set[str] = set()
        self.languages: Set[str] = set()

    def load(self) -> None:
        """Load corpus from DuckDB file."""
        if self._loaded:
            return

        if not Path(self.db_file).exists():
            logger.error(f"DuckDB file not found: {self.db_file}")
            raise CorpusError(f"Corpus database not found at {self.db_file}")

        try:
            self.db = duckdb.connect(str(self.db_file), read_only=True)
            self._loaded = True
            logger.info(f"Loaded DuckDB corpus from {self.db_file}")

            # Cache metadata
            domains_result = self.db.execute("SELECT DISTINCT domain FROM phrases").fetchall()
            self.categories = {row[0] for row in domains_result}

            languages_result = self.db.execute("SELECT DISTINCT language FROM phrases").fetchall()
            self.languages = {row[0] for row in languages_result}

            logger.info(f"Found {len(self.categories)} domains and {len(self.languages)} languages")

        except Exception as e:
            logger.error(f"Failed to load corpus from {self.db_file}: {e}")
            raise CorpusError(f"Failed to load corpus database: {e}")

    def get_phrases(
        self,
        language: Optional[str] = None,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[PhraseRecord]:
        """Get phrases matching filters."""
        if not self._loaded:
            self.load()

        query = "SELECT id, base_text, language, domain, difficulty FROM phrases WHERE 1=1"
        params = []

        if language:
            query += " AND language = ?"
            params.append(language)

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        if difficulty:
            query += " AND difficulty = ?"
            params.append(difficulty)

        if limit:
            query += f" LIMIT {limit}"

        results = self.db.execute(query, params).fetchall()
        return [PhraseRecord(
            id=row[0],
            text=row[1],
            language=row[2],
            domain=row[3],
            difficulty=row[4],
            translations=self._get_translations(row[0])
        ) for row in results]

    def get_by_domain(self, domain: str, language: Optional[str] = None) -> Dict[str, List[str]]:
        """Get phrases organized by language within a domain."""
        if not self._loaded:
            self.load()

        query = """
        SELECT p.language, p.base_text
        FROM phrases p
        WHERE p.domain = ?
        """
        params = [domain]

        if language:
            query += " AND p.language = ?"
            params.append(language)

        query += " ORDER BY p.difficulty, p.base_text"

        results = self.db.execute(query, params).fetchall()

        # Group by language
        by_lang = {}
        for lang, text in results:
            if lang not in by_lang:
                by_lang[lang] = []
            by_lang[lang].append(text)

        return by_lang

    def get_by_category_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get phrases organized by domain and language (for compatibility).
        Returns: {domain: {language: [phrases]}}
        """
        if not self._loaded:
            self.load()

        result = {}
        for domain in self.categories:
            result[domain] = self.get_by_domain(domain)

        return result

    def get_stats(self) -> Dict:
        """Get corpus statistics."""
        if not self._loaded:
            self.load()

        total = self.db.execute("SELECT COUNT(*) FROM phrases").fetchone()[0]

        by_domain = self.db.execute("""
            SELECT domain, COUNT(*) as count
            FROM phrases
            GROUP BY domain
            ORDER BY domain
        """).fetchall()

        by_language = self.db.execute("""
            SELECT language, COUNT(*) as count
            FROM phrases
            GROUP BY language
            ORDER BY language
        """).fetchall()

        by_difficulty = self.db.execute("""
            SELECT difficulty, COUNT(*) as count
            FROM phrases
            GROUP BY difficulty
            ORDER BY difficulty
        """).fetchall()

        return {
            "total_phrases": total,
            "categories": sorted(list(self.categories)),
            "languages": sorted(list(self.languages)),
            "by_domain": {domain: count for domain, count in by_domain},
            "by_language": {lang: count for lang, count in by_language},
            "by_difficulty": {diff: count for diff, count in by_difficulty},
        }

    def get_next_for_srs(
        self,
        language: str,
        domain: Optional[str] = None,
        max_review_count: int = 5
    ) -> Optional[PhraseRecord]:
        """Get next phrase for SRS review (least recently reviewed in domain)."""
        if not self._loaded:
            self.load()

        query = """
        SELECT id, base_text, language, domain, difficulty
        FROM phrases
        WHERE language = ? AND review_count < ?
        """
        params = [language, max_review_count]

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        query += " ORDER BY last_reviewed ASC NULLS FIRST, review_count ASC LIMIT 1"

        result = self.db.execute(query, params).fetchone()

        if result:
            return PhraseRecord(
                id=result[0],
                text=result[1],
                language=result[2],
                domain=result[3],
                difficulty=result[4],
                translations=self._get_translations(result[0])
            )
        return None

    def search(
        self,
        query_text: str,
        language: Optional[str] = None,
        domain: Optional[str] = None
    ) -> List[PhraseRecord]:
        """Search phrases by text content."""
        if not self._loaded:
            self.load()

        query = "SELECT id, base_text, language, domain, difficulty FROM phrases WHERE base_text ILIKE ?"
        params = [f"%{query_text}%"]

        if language:
            query += " AND language = ?"
            params.append(language)

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        query += " LIMIT 20"

        results = self.db.execute(query, params).fetchall()
        return [PhraseRecord(
            id=row[0],
            text=row[1],
            language=row[2],
            domain=row[3],
            difficulty=row[4],
            translations=self._get_translations(row[0])
        ) for row in results]

    def record_review(self, phrase_id: int) -> None:
        """Record that a phrase was reviewed (for SRS)."""
        if not self._loaded:
            self.load()

        # Note: DuckDB connection is read-only, so this is a no-op for now
        # In production, you'd maintain a separate review log or use a writable connection
        logger.debug(f"Recorded review for phrase {phrase_id}")

    def _get_translations(self, phrase_id: int) -> Dict[str, str]:
        """Get all translations for a phrase."""
        results = self.db.execute(
            "SELECT language, text FROM translations WHERE phrase_id = ?",
            [phrase_id]
        ).fetchall()
        return {lang: text for lang, text in results}

    def close(self) -> None:
        """Close database connection."""
        if self.db:
            self.db.close()


# Global instance
_corpus_instance: Optional[DuckDBCorpus] = None


def get_corpus() -> DuckDBCorpus:
    """Get or create global corpus instance."""
    global _corpus_instance
    if _corpus_instance is None:
        _corpus_instance = DuckDBCorpus()
        _corpus_instance.load()
    return _corpus_instance
