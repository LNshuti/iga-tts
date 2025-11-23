"""
Intelligent corpus loader for phrase management.
Now uses DuckDB backend for domain-based organization and SRS support.
Maintains backward compatibility with existing code.
"""
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path

from config import Config, logger, CorpusError
from corpus_db import DuckDBCorpus as _DuckDBCorpus, PhraseRecord


class Phrase:
    """Represents a single phrase (for backward compatibility)."""

    def __init__(
        self,
        text: str,
        language: str,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text.strip()
        self.language = language.lower()
        self.category = category or "Uncategorized"
        self.difficulty = difficulty or "beginner"
        self.tags = tags or []
        self.metadata = metadata or {}
        self.usage_count = 0

    def __repr__(self) -> str:
        return f"Phrase(text='{self.text[:30]}...', lang={self.language}, cat={self.category})"

    def matches_filter(
        self,
        language: Optional[str] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        search_term: Optional[str] = None
    ) -> bool:
        """Check if phrase matches given filters."""
        if language and self.language != language.lower():
            return False
        if category and self.category != category:
            return False
        if difficulty and self.difficulty != difficulty:
            return False
        if search_term and search_term.lower() not in self.text.lower():
            return False
        return True


class CorpusLoader:
    """
    Corpus loader backed by DuckDB.
    Maintains compatibility with existing code while using intelligent categorization.
    """

    def __init__(self, corpus_file: str = None):
        """Initialize with DuckDB backend."""
        self.corpus_file = corpus_file or Config.CORPUS_FILE  # For backward compatibility
        self._db_corpus = _DuckDBCorpus()
        self.phrases: List[Phrase] = []
        self.categories: Set[str] = set()
        self.languages: Set[str] = set()
        self._by_category: Dict[str, List[Phrase]] = {}
        self._by_language: Dict[str, List[Phrase]] = {}
        self._loaded = False

    def load(self) -> None:
        """Load corpus from DuckDB backend or text file (for tests)."""
        if self._loaded:
            return

        try:
            corpus_path = Path(self.corpus_file)

            # If corpus_file is a .txt file (for tests), load from text
            if self.corpus_file.endswith('.txt'):
                if not corpus_path.exists():
                    # Create default corpus for tests
                    self._create_default_corpus(corpus_path)

                # Load from text file
                self._load_from_text_file(corpus_path)
            else:
                # Load from DuckDB
                self._db_corpus.load()

                # Get all phrases from DuckDB
                db_phrases = self._db_corpus.get_phrases()

                # Convert to Phrase objects for compatibility
                for db_phrase in db_phrases:
                    phrase = Phrase(
                        text=db_phrase.text,
                        language=db_phrase.language,
                        category=db_phrase.domain,
                        difficulty=db_phrase.difficulty,
                        metadata={"id": db_phrase.id}
                    )
                    self.phrases.append(phrase)

                # Update categories and languages
                self.categories = self._db_corpus.categories
                self.languages = self._db_corpus.languages

            # Build index maps
            self._by_category = {}
            self._by_language = {}

            for phrase in self.phrases:
                # By category
                if phrase.category not in self._by_category:
                    self._by_category[phrase.category] = []
                self._by_category[phrase.category].append(phrase)

                # By language
                if phrase.language not in self._by_language:
                    self._by_language[phrase.language] = []
                self._by_language[phrase.language].append(phrase)

            self._loaded = True
            logger.info(f"Loaded {len(self.phrases)} phrases from corpus")
            logger.info(f"Categories: {self.categories}")
            logger.info(f"Languages: {self.languages}")

        except Exception as e:
            logger.error(f"Failed to load corpus: {e}")
            raise CorpusError(f"Failed to load corpus: {e}")

    def _load_from_text_file(self, corpus_path: Path) -> None:
        """Load corpus from text file (for backward compatibility with tests)."""
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                try:
                    phrase = self._parse_line(line)
                    self.phrases.append(phrase)
                    self.categories.add(phrase.category)
                    self.languages.add(phrase.language)

                except Exception as e:
                    logger.warning(f"Failed to parse line: {line} - {e}")

            logger.info(f"Loaded {len(self.phrases)} phrases from text corpus")

        except Exception as e:
            logger.error(f"Failed to load text corpus: {e}")
            raise CorpusError(f"Failed to load text corpus: {e}")

    def get_phrases(
        self,
        language: Optional[str] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Phrase]:
        """Get phrases matching filters."""
        if not self._loaded:
            self.load()

        # Use indexed lookups when possible
        if language and not (category or difficulty):
            results = self._by_language.get(language.lower(), [])
        elif category and not (language or difficulty):
            results = self._by_category.get(category, [])
        else:
            # Full filter
            results = [
                p for p in self.phrases
                if p.matches_filter(language, category, difficulty)
            ]

        if limit:
            results = results[:limit]

        return results

    def search(self, query: str, language: Optional[str] = None) -> List[Phrase]:
        """Search phrases by text content."""
        if not self._loaded:
            self.load()

        query = query.lower()
        results = [
            p for p in self.phrases
            if query in p.text.lower() and (not language or p.language == language.lower())
        ]

        return results

    def get_suggestions(
        self,
        current_text: str,
        language: str,
        max_suggestions: int = 5
    ) -> List[Phrase]:
        """Get context-aware phrase suggestions."""
        if not self._loaded:
            self.load()

        current_lower = current_text.lower()
        prefix_matches = [
            p for p in self._by_language.get(language.lower(), [])
            if p.text.lower().startswith(current_lower)
        ]

        if prefix_matches:
            return prefix_matches[:max_suggestions]

        substring_matches = [
            p for p in self._by_language.get(language.lower(), [])
            if current_lower in p.text.lower()
        ]

        return substring_matches[:max_suggestions]

    def get_by_category_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get phrases organized by category and language for UI display.
        Returns: {category: {language: [phrase_texts]}}
        """
        if not self._loaded:
            self.load()

        result = {}
        for category in self.categories:
            result[category] = {}
            for language in self.languages:
                phrases = self.get_phrases(language=language, category=category)
                result[category][language] = [p.text for p in phrases]

        return result

    def record_usage(self, text: str, language: str) -> None:
        """Record usage of a phrase for analytics."""
        for phrase in self.phrases:
            if phrase.text == text and phrase.language == language.lower():
                phrase.usage_count += 1
                logger.debug(f"Recorded usage for: {text[:30]}... (count: {phrase.usage_count})")
                break

    def get_stats(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        if not self._loaded:
            self.load()

        return self._db_corpus.get_stats()

    def get_by_domain(self, domain: str, language: Optional[str] = None) -> Dict[str, List[str]]:
        """Get phrases in a specific domain, organized by language."""
        if not self._loaded:
            self.load()

        return self._db_corpus.get_by_domain(domain, language)

    def close(self) -> None:
        """Close database connection."""
        if self._db_corpus:
            self._db_corpus.close()

    # Backward compatibility methods for tests
    def _parse_line(self, line: str) -> Phrase:
        """Parse a corpus line into a Phrase object (for backward compatibility)."""
        parts = [p.strip() for p in line.split('|')]

        if len(parts) == 1:
            # Simple format: just text (assume English)
            return Phrase(text=parts[0], language='en')
        elif len(parts) == 2:
            # text|lang
            return Phrase(text=parts[0], language=parts[1])
        elif len(parts) == 3:
            # text|lang|category
            return Phrase(text=parts[0], language=parts[1], category=parts[2])
        elif len(parts) >= 4:
            # text|lang|category|difficulty|tags
            tags = []
            if len(parts) > 4 and parts[4]:
                tags = [t.strip() for t in parts[4].split(',')]

            return Phrase(
                text=parts[0],
                language=parts[1],
                category=parts[2],
                difficulty=parts[3] if parts[3] else None,
                tags=tags
            )
        else:
            raise ValueError(f"Invalid format: {line}")

    def _create_default_corpus(self, path: Path) -> None:
        """Create a default corpus file for tests (backward compatibility)."""
        default_phrases = [
            "# Test Corpus File",
            "# Format: text|language|category|difficulty|tags",
            "",
            "Hello|en|Greetings|beginner|common,daily",
            "Good morning|en|Greetings|beginner|daily",
            "Bonjour|fr|Greetings|beginner|common,daily",
            "Muraho|rw|Greetings|beginner|common,daily",
            "Where is the bus station?|en|Travel|beginner|navigation",
            "Où est la gare routière?|fr|Travel|beginner|navigation",
            "I need a taxi|en|Travel|intermediate|transport",
            "One|en|Numbers|beginner|counting",
            "Two|en|Numbers|beginner|counting",
            "Three|en|Numbers|beginner|counting",
        ]

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(default_phrases))
            logger.info(f"Created default test corpus at: {path}")
        except Exception as e:
            logger.error(f"Failed to create default corpus: {e}")


# Global corpus instance
_corpus_instance: Optional[CorpusLoader] = None


def get_corpus() -> CorpusLoader:
    """Get or create global corpus instance."""
    global _corpus_instance
    if _corpus_instance is None:
        _corpus_instance = CorpusLoader()
        _corpus_instance.load()
    return _corpus_instance
