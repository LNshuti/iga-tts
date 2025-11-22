"""
Comprehensive test suite for Iga TTS application.
Tests all core components: config, translation, TTS, corpus, and app.
"""
import pytest
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


class TestConfig:
    """Test configuration management."""

    def test_config_validation(self):
        """Test that configuration validates correctly."""
        from config import Config

        assert Config.validate() is True
        assert Config.MAX_TRANSLATION_LENGTH > 0
        assert Config.TRANSLATION_CACHE_SIZE > 0
        assert Config.TTS_MODEL is not None

    def test_logger_setup(self):
        """Test that logger is properly configured."""
        from config import logger

        assert logger is not None
        assert logger.name == "iga_tts"

    def test_custom_exceptions(self):
        """Test custom exception classes."""
        from config import TranslationError, TTSError, CorpusError

        with pytest.raises(TranslationError):
            raise TranslationError("Test error")

        with pytest.raises(TTSError):
            raise TTSError("Test error")

        with pytest.raises(CorpusError):
            raise CorpusError("Test error")


class TestTranslation:
    """Test translation module."""

    def test_validate_language_code(self):
        """Test language code validation."""
        from translation import validate_language_code, TranslationError

        assert validate_language_code("en") == "en"
        assert validate_language_code("EN") == "en"
        assert validate_language_code(" fr ") == "fr"

        with pytest.raises(TranslationError):
            validate_language_code("invalid")

    def test_validate_text(self):
        """Test text validation."""
        from translation import validate_text, TranslationError

        assert validate_text("Hello") == "Hello"
        assert validate_text("  Hello  ") == "Hello"

        with pytest.raises(TranslationError):
            validate_text("")

        with pytest.raises(TranslationError):
            validate_text("   ")

    def test_validate_text_truncation(self):
        """Test that long text is truncated."""
        from translation import validate_text

        long_text = "a" * 10000
        result = validate_text(long_text, max_length=100)
        assert len(result) == 100

    def test_translation_same_language(self):
        """Test translation with same source and target."""
        from translation import translate

        result = translate("Hello", "en", "en")
        assert result == "Hello"

    def test_translation_empty_text(self):
        """Test translation with empty text."""
        from translation import translate, TranslationError

        with pytest.raises(TranslationError):
            translate("", "en", "rw")

    def test_translation_invalid_language(self):
        """Test translation with invalid language."""
        from translation import translate, TranslationError

        with pytest.raises(TranslationError):
            translate("Hello", "invalid", "en")


class TestTTS:
    """Test TTS module."""

    def test_validate_tts_text(self):
        """Test TTS text validation."""
        from tts import validate_tts_text, TTSError

        assert validate_tts_text("Hello") == "Hello"
        assert validate_tts_text("  Hello  ") == "Hello"

        with pytest.raises(TTSError):
            validate_tts_text("")

        with pytest.raises(TTSError):
            validate_tts_text("   ")

    def test_validate_tts_text_truncation(self):
        """Test that long TTS text is truncated."""
        from tts import validate_tts_text

        long_text = "a" * 2000
        result = validate_tts_text(long_text, max_length=100)
        assert len(result) == 100


class TestCorpus:
    """Test corpus loader."""

    def test_phrase_creation(self):
        """Test Phrase object creation."""
        from corpus import Phrase

        phrase = Phrase("Hello", "en", "Greetings", "beginner", ["common"])
        assert phrase.text == "Hello"
        assert phrase.language == "en"
        assert phrase.category == "Greetings"
        assert phrase.difficulty == "beginner"
        assert "common" in phrase.tags

    def test_phrase_matches_filter(self):
        """Test phrase filtering."""
        from corpus import Phrase

        phrase = Phrase("Hello", "en", "Greetings", "beginner")

        assert phrase.matches_filter(language="en") is True
        assert phrase.matches_filter(language="fr") is False
        assert phrase.matches_filter(category="Greetings") is True
        assert phrase.matches_filter(category="Travel") is False
        assert phrase.matches_filter(difficulty="beginner") is True
        assert phrase.matches_filter(search_term="Hell") is True
        assert phrase.matches_filter(search_term="Goodbye") is False

    def test_corpus_loader_initialization(self):
        """Test corpus loader initialization."""
        from corpus import CorpusLoader

        loader = CorpusLoader()
        assert loader.corpus_file is not None
        assert len(loader.phrases) == 0
        assert loader._loaded is False

    def test_corpus_parse_line(self):
        """Test parsing different corpus line formats."""
        from corpus import CorpusLoader

        loader = CorpusLoader()

        # Simple format
        phrase1 = loader._parse_line("Hello")
        assert phrase1.text == "Hello"
        assert phrase1.language == "en"

        # With language
        phrase2 = loader._parse_line("Bonjour|fr")
        assert phrase2.text == "Bonjour"
        assert phrase2.language == "fr"

        # With category
        phrase3 = loader._parse_line("Muraho|rw|Greetings")
        assert phrase3.text == "Muraho"
        assert phrase3.language == "rw"
        assert phrase3.category == "Greetings"

        # Full format
        phrase4 = loader._parse_line("Hello|en|Greetings|beginner|common,daily")
        assert phrase4.text == "Hello"
        assert phrase4.language == "en"
        assert phrase4.category == "Greetings"
        assert phrase4.difficulty == "beginner"
        assert "common" in phrase4.tags
        assert "daily" in phrase4.tags

    def test_corpus_load_creates_default(self):
        """Test that loading creates default corpus if file doesn't exist."""
        import tempfile
        from corpus import CorpusLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_file = Path(tmpdir) / "test_corpus.txt"
            loader = CorpusLoader(str(corpus_file))
            loader.load()

            assert corpus_file.exists()
            assert len(loader.phrases) > 0
            assert loader._loaded is True

    def test_corpus_get_phrases(self):
        """Test getting phrases with filters."""
        import tempfile
        from corpus import CorpusLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_file = Path(tmpdir) / "test_corpus.txt"
            loader = CorpusLoader(str(corpus_file))
            loader.load()

            # Get all phrases
            all_phrases = loader.get_phrases()
            assert len(all_phrases) > 0

            # Filter by language
            en_phrases = loader.get_phrases(language="en")
            assert all(p.language == "en" for p in en_phrases)

            # Filter by category
            greeting_phrases = loader.get_phrases(category="Greetings")
            assert all(p.category == "Greetings" for p in greeting_phrases)

            # Limit results
            limited = loader.get_phrases(limit=5)
            assert len(limited) <= 5

    def test_corpus_search(self):
        """Test corpus search functionality."""
        import tempfile
        from corpus import CorpusLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_file = Path(tmpdir) / "test_corpus.txt"
            loader = CorpusLoader(str(corpus_file))
            loader.load()

            # Search for "Hello"
            results = loader.search("Hello")
            assert any("Hello" in p.text for p in results)

            # Search with language filter
            results_en = loader.search("Hello", language="en")
            assert all(p.language == "en" for p in results_en)

    def test_corpus_stats(self):
        """Test corpus statistics."""
        import tempfile
        from corpus import CorpusLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_file = Path(tmpdir) / "test_corpus.txt"
            loader = CorpusLoader(str(corpus_file))
            loader.load()

            stats = loader.get_stats()
            assert "total_phrases" in stats
            assert "categories" in stats
            assert "languages" in stats
            assert stats["total_phrases"] > 0


class TestApp:
    """Test app module."""

    def test_on_mode_change(self):
        """Test mode change functionality."""
        from app import on_mode_change

        # Rwanda Mode
        src, tgt = on_mode_change("Rwanda Mode (Kinyarwanda → EN/FR)")
        assert src == "rw"
        assert tgt == "en"

        # Diaspora Mode
        src, tgt = on_mode_change("Diaspora Mode (EN/FR → Kinyarwanda)")
        assert src == "en"
        assert tgt == "rw"

    def test_do_translate_empty(self):
        """Test translation with empty text."""
        from app import do_translate

        result = do_translate("", "en", "rw")
        assert result == ""

    def test_do_tts_empty(self):
        """Test TTS with empty text."""
        from app import do_tts

        result = do_tts("")
        assert result is None

    def test_do_tts_error_message(self):
        """Test that TTS skips error messages."""
        from app import do_tts

        result = do_tts("❌ Translation error")
        assert result is None


def run_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("Running Iga TTS Test Suite")
    print("=" * 70)

    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-ra"
    ])

    return exit_code


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
