"""
Intelligent corpus loader for phrase management.
Provides categorization, search, filtering, and context-aware suggestions.
"""
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
import re

from config import Config, logger, CorpusError


class Phrase:
    """Represents a single phrase in the corpus."""

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
    Intelligent corpus loader with advanced features:
    - Automatic categorization
    - Search and filtering
    - Usage tracking
    - Context-aware suggestions
    """

    def __init__(self, corpus_file: str = None):
        self.corpus_file = corpus_file or Config.CORPUS_FILE
        self.phrases: List[Phrase] = []
        self.categories: Set[str] = set()
        self.languages: Set[str] = set()
        self._by_category: Dict[str, List[Phrase]] = defaultdict(list)
        self._by_language: Dict[str, List[Phrase]] = defaultdict(list)
        self._loaded = False

    def load(self) -> None:
        """
        Load corpus from file with intelligent parsing.

        File format supports multiple formats:
        1. Simple: text
        2. With language: text|lang
        3. With category: text|lang|category
        4. Full: text|lang|category|difficulty|tag1,tag2
        """
        corpus_path = Path(self.corpus_file)

        if not corpus_path.exists():
            logger.warning(f"Corpus file not found: {self.corpus_file}")
            logger.info("Creating empty corpus with default phrases")
            self._create_default_corpus(corpus_path)

        try:
            with open(corpus_path, 'r', encoding=Config.CORPUS_ENCODING) as f:
                lines = f.readlines()

            logger.info(f"Loading corpus from: {self.corpus_file}")

            for line_num, line in enumerate(lines, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                try:
                    phrase = self._parse_line(line)
                    self.phrases.append(phrase)
                    self.categories.add(phrase.category)
                    self.languages.add(phrase.language)
                    self._by_category[phrase.category].append(phrase)
                    self._by_language[phrase.language].append(phrase)

                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num}: {line} - {e}")

            self._loaded = True
            logger.info(
                f"Loaded {len(self.phrases)} phrases across "
                f"{len(self.categories)} categories and "
                f"{len(self.languages)} languages"
            )

        except Exception as e:
            logger.error(f"Failed to load corpus: {e}", exc_info=True)
            raise CorpusError(f"Failed to load corpus from {self.corpus_file}: {e}")

    def _parse_line(self, line: str) -> Phrase:
        """Parse a single corpus line into a Phrase object."""
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
        """Create a default corpus file with example phrases."""
        default_phrases = [
            "# Iga TTS Corpus File",
            "# Format: text|language|category|difficulty|tags",
            "# Languages: en (English), fr (Français), rw (Kinyarwanda)",
            "# Difficulty: beginner, intermediate, advanced",
            "",
            "# Greetings",
            "Hello|en|Greetings|beginner|common,daily",
            "Good morning|en|Greetings|beginner|daily",
            "How are you?|en|Greetings|beginner|common,daily",
            "Bonjour|fr|Greetings|beginner|common,daily",
            "Bonsoir|fr|Greetings|beginner|daily",
            "Comment ça va?|fr|Greetings|beginner|common,daily",
            "Muraho|rw|Greetings|beginner|common,daily",
            "Mwaramutse|rw|Greetings|beginner|daily",
            "Amakuru?|rw|Greetings|beginner|common,daily",
            "",
            "# Travel",
            "Where is the bus station?|en|Travel|beginner|navigation,transport",
            "Où est la gare routière?|fr|Travel|beginner|navigation,transport",
            "Gare ya bisi iri he?|rw|Travel|beginner|navigation,transport",
            "I need a taxi|en|Travel|beginner|transport",
            "J'ai besoin d'un taxi|fr|Travel|beginner|transport",
            "Nkeneye taxi|rw|Travel|beginner|transport",
            "",
            "# Numbers",
            "One|en|Numbers|beginner|counting,basic",
            "Two|en|Numbers|beginner|counting,basic",
            "Three|en|Numbers|beginner|counting,basic",
            "Un|fr|Numbers|beginner|counting,basic",
            "Deux|fr|Numbers|beginner|counting,basic",
            "Trois|fr|Numbers|beginner|counting,basic",
            "Rimwe|rw|Numbers|beginner|counting,basic",
            "Kabiri|rw|Numbers|beginner|counting,basic",
            "Gatatu|rw|Numbers|beginner|counting,basic",
            "",
            "# Questions",
            "What is your name?|en|Questions|beginner|common,introductions",
            "Comment vous appelez-vous?|fr|Questions|beginner|common,introductions",
            "Witwa nde?|rw|Questions|beginner|common,introductions",
            "How much does this cost?|en|Questions|beginner|shopping,money",
            "Combien ça coûte?|fr|Questions|beginner|shopping,money",
            "Ni angahe?|rw|Questions|beginner|shopping,money",
        ]

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding=Config.CORPUS_ENCODING) as f:
                f.write('\n'.join(default_phrases))
            logger.info(f"Created default corpus at: {path}")
        except Exception as e:
            logger.error(f"Failed to create default corpus: {e}")

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

        # First, try prefix matching
        current_lower = current_text.lower()
        prefix_matches = [
            p for p in self._by_language.get(language.lower(), [])
            if p.text.lower().startswith(current_lower)
        ]

        if prefix_matches:
            return prefix_matches[:max_suggestions]

        # Then try substring matching
        substring_matches = [
            p for p in self._by_language.get(language.lower(), [])
            if current_lower in p.text.lower()
        ]

        return substring_matches[:max_suggestions]

    def get_by_category_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get phrases organized by category and language for legacy compatibility.

        Returns:
            Dict mapping category -> language -> list of phrase texts
        """
        if not self._loaded:
            self.load()

        result = defaultdict(lambda: defaultdict(list))

        for phrase in self.phrases:
            result[phrase.category][phrase.language].append(phrase.text)

        # Ensure all categories have all languages
        for category in result:
            for lang in self.languages:
                if lang not in result[category]:
                    result[category][lang] = []

        return dict(result)

    def record_usage(self, text: str, language: str) -> None:
        """Record usage of a phrase for analytics."""
        for phrase in self.phrases:
            if phrase.text == text and phrase.language == language.lower():
                phrase.usage_count += 1
                logger.debug(f"Recorded usage for: {text[:30]}... (count: {phrase.usage_count})")
                break

    def get_stats(self) -> Dict:
        """Get corpus statistics."""
        if not self._loaded:
            self.load()

        return {
            "total_phrases": len(self.phrases),
            "categories": sorted(list(self.categories)),
            "languages": sorted(list(self.languages)),
            "by_category": {cat: len(phrases) for cat, phrases in self._by_category.items()},
            "by_language": {lang: len(phrases) for lang, phrases in self._by_language.items()},
        }


# Global corpus instance
_corpus_instance: Optional[CorpusLoader] = None


def get_corpus() -> CorpusLoader:
    """Get or create global corpus instance."""
    global _corpus_instance
    if _corpus_instance is None:
        _corpus_instance = CorpusLoader()
        _corpus_instance.load()
    return _corpus_instance
