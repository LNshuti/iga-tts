"""
Preprocess Corpus.txt into intelligent domain-categorized DuckDB database.
Uses semantic analysis and keyword heuristics for smart categorization.
"""
import re
import duckdb
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

from config import Config, logger

# Domain keywords for heuristic classification
DOMAIN_KEYWORDS = {
    "Greetings": ["hello", "hi", "greeting", "bonjour", "bonsoir", "muraho", "mwaramutse", "amakuru", "salut", "coucou"],
    "Travel": ["travel", "journey", "airport", "train", "bus", "taxi", "hotel", "map", "gare", "voyage", "transport"],
    "Food": ["food", "eat", "drink", "restaurant", "meal", "coffee", "water", "bread", "manger", "boire", "plat"],
    "Work": ["work", "job", "office", "meeting", "business", "company", "travail", "bureau", "réunion", "employé"],
    "Health": ["health", "doctor", "hospital", "medicine", "pain", "sick", "médecin", "hôpital", "maladie", "douleur"],
    "Education": ["school", "student", "learn", "study", "teacher", "class", "école", "étudiant", "apprendre", "cours"],
    "Social": ["friend", "family", "love", "relationship", "marry", "ami", "famille", "amour", "relation"],
    "Emotions": ["happy", "sad", "angry", "fear", "hope", "believe", "celebrate", "heureux", "triste", "colère", "peur"],
    "Numbers": ["one", "two", "three", "number", "count", "un", "deux", "trois", "nombre", "compter"],
    "Shopping": ["shop", "buy", "sell", "price", "money", "cost", "magasin", "acheter", "vendre", "prix", "argent"],
    "Time": ["time", "day", "week", "month", "year", "morning", "night", "heure", "jour", "semaine", "mois", "année"],
    "Family": ["mother", "father", "brother", "sister", "child", "parent", "mère", "père", "frère", "soeur", "enfant"],
    "Questions": ["what", "how", "why", "where", "when", "who", "quoi", "comment", "pourquoi", "où", "quand"],
    "Activities": ["dance", "play", "swim", "run", "walk", "sing", "danser", "jouer", "nager", "courir", "marcher"],
}

# Difficulty inference keywords
DIFFICULTY_KEYWORDS = {
    "beginner": ["hello", "yes", "no", "thank", "please", "one", "two", "water", "food", "name", "good", "bad"],
    "advanced": ["complicated", "sophisticated", "theoretical", "philosophical", "abstract", "ambiguous"],
}


def extract_phrases_from_corpus(corpus_file: str) -> List[Dict]:
    """
    Parse Corpus.txt into structured phrase records.
    Format: Kinyarwanda, English, Kirundi, French, then examples/context
    """
    phrases = []

    with open(corpus_file, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]

    # Skip header lines (lines 0-6)
    i = 7
    while i < len(lines):
        # Skip empty lines
        while i < len(lines) and not lines[i].strip():
            i += 1

        if i >= len(lines):
            break

        # Pattern: RW phrase, EN translation, KI translation, FR translation, then examples
        if i + 3 < len(lines):
            text_rw = lines[i].strip()
            text_en = lines[i + 1].strip()
            text_ki = lines[i + 2].strip()
            text_fr = lines[i + 3].strip()

            if text_rw and text_en:  # At least RW and EN should exist
                phrase_record = {
                    "text_rw": text_rw,
                    "text_en": text_en,
                    "text_ki": text_ki,
                    "text_fr": text_fr,
                    "examples": [],
                }

                # Collect example lines (lines with – or multiple language indicators)
                j = i + 4
                while j < len(lines) and lines[j].strip() and not (
                    j + 3 < len(lines) and
                    lines[j].strip() and
                    lines[j + 1].strip() and
                    not any(sep in lines[j] for sep in ['–', '-'])
                ):
                    example = lines[j].strip()
                    if example:
                        phrase_record["examples"].append(example)
                    j += 1

                phrases.append(phrase_record)
                i = j
            else:
                i += 1
        else:
            break

    logger.info(f"Extracted {len(phrases)} phrases from corpus")
    return phrases


def classify_domain(text: str) -> str:
    """Classify phrase into domain using keyword matching."""
    text_lower = text.lower()

    # Keyword-based classification
    best_domain = "General"
    max_matches = 0

    for domain, keywords in DOMAIN_KEYWORDS.items():
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        if matches > max_matches:
            max_matches = matches
            best_domain = domain

    return best_domain


def infer_difficulty(text: str) -> str:
    """Infer difficulty level from phrase complexity."""
    text_lower = text.lower()
    word_count = len(text.split())

    # Check for advanced keywords
    advanced_count = sum(1 for keyword in DIFFICULTY_KEYWORDS.get("advanced", [])
                        if keyword.lower() in text_lower)
    if advanced_count > 0:
        return "advanced"

    # Check for beginner keywords
    beginner_count = sum(1 for keyword in DIFFICULTY_KEYWORDS.get("beginner", [])
                        if keyword.lower() in text_lower)
    if beginner_count >= 2:
        return "beginner"

    # Length-based heuristic
    if word_count <= 2:
        return "beginner"
    elif word_count >= 8:
        return "advanced"
    else:
        return "intermediate"


def create_duckdb_database(output_file: str):
    """Create DuckDB database with proper schema."""
    db = duckdb.connect(output_file)

    # Drop existing tables if they exist
    try:
        db.execute("DROP TABLE IF EXISTS translations")
        db.execute("DROP TABLE IF EXISTS phrases")
        db.execute("DROP TABLE IF EXISTS domains")
    except:
        pass

    # Create domains table
    db.execute("""
    CREATE TABLE domains (
        name VARCHAR PRIMARY KEY,
        description VARCHAR,
        emoji VARCHAR
    )
    """)

    # Create phrases table with auto-increment ID
    db.execute("""
    CREATE TABLE phrases (
        id INTEGER PRIMARY KEY,
        base_text VARCHAR NOT NULL,
        language VARCHAR(2) NOT NULL,
        domain VARCHAR NOT NULL,
        difficulty VARCHAR NOT NULL,
        part_of_speech VARCHAR,
        tags VARCHAR,
        frequency_score FLOAT DEFAULT 0.5,
        created_at TIMESTAMP DEFAULT NOW(),
        last_reviewed TIMESTAMP NULL,
        review_count INTEGER DEFAULT 0
    )
    """)

    # Create translations table
    db.execute("""
    CREATE TABLE translations (
        id INTEGER PRIMARY KEY,
        phrase_id INTEGER REFERENCES phrases(id),
        language VARCHAR(2) NOT NULL,
        text VARCHAR NOT NULL,
        example_sentence VARCHAR,
        UNIQUE(phrase_id, language)
    )
    """)

    # Create indices
    db.execute("CREATE INDEX idx_phrases_domain ON phrases(domain)")
    db.execute("CREATE INDEX idx_phrases_language_domain ON phrases(language, domain)")
    db.execute("CREATE INDEX idx_phrases_difficulty ON phrases(difficulty)")
    db.execute("CREATE INDEX idx_phrases_language_difficulty ON phrases(language, difficulty)")
    db.execute("CREATE INDEX idx_translations_phrase_id ON translations(phrase_id)")
    db.execute("CREATE INDEX idx_translations_language ON translations(language)")

    logger.info(f"Created DuckDB database at {output_file}")
    return db


def populate_database(db: duckdb.DuckDBPyConnection, phrases: List[Dict]) -> None:
    """Populate database with preprocessed phrases."""

    # Insert domains
    domains = [
        ("Greetings", "Greetings and polite phrases", "👋"),
        ("Travel", "Travel and transportation", "✈️"),
        ("Food", "Food and dining", "🍔"),
        ("Work", "Work and business", "💼"),
        ("Health", "Health and medical", "⚕️"),
        ("Education", "Education and learning", "🎓"),
        ("Social", "Social interactions", "👥"),
        ("Emotions", "Emotions and feelings", "❤️"),
        ("Numbers", "Numbers and counting", "🔢"),
        ("Shopping", "Shopping and money", "🛒"),
        ("Time", "Time expressions", "⏰"),
        ("Family", "Family relationships", "👨‍👩‍👧‍👦"),
        ("Questions", "Question words", "❓"),
        ("Activities", "Activities and sports", "⚽"),
        ("General", "General phrases", "📝"),
    ]

    for name, description, emoji in domains:
        db.execute(
            "INSERT INTO domains VALUES (?, ?, ?)",
            [name, description, emoji]
        )

    logger.info(f"Inserted {len(domains)} domains")

    # Insert phrases and translations
    inserted_count = 0
    phrase_id = 1
    trans_id = 1

    for phrase in phrases:
        text_rw = phrase.get("text_rw", "").strip()
        text_en = phrase.get("text_en", "").strip()
        text_ki = phrase.get("text_ki", "").strip()
        text_fr = phrase.get("text_fr", "").strip()

        if not text_rw or not text_en:
            continue

        # Classify and infer difficulty from all available text
        all_text = " ".join([text_rw, text_en, text_ki, text_fr])
        domain = classify_domain(all_text)
        difficulty = infer_difficulty(all_text)

        # Insert Kinyarwanda phrase
        try:
            db.execute(
                """INSERT INTO phrases (id, base_text, language, domain, difficulty)
                   VALUES (?, ?, ?, ?, ?)""",
                [phrase_id, text_rw, "rw", domain, difficulty]
            )

            # Insert translations
            if text_en:
                db.execute(
                    "INSERT INTO translations (id, phrase_id, language, text) VALUES (?, ?, ?, ?)",
                    [trans_id, phrase_id, "en", text_en]
                )
                trans_id += 1

            if text_ki:
                db.execute(
                    "INSERT INTO translations (id, phrase_id, language, text) VALUES (?, ?, ?, ?)",
                    [trans_id, phrase_id, "ki", text_ki]
                )
                trans_id += 1

            if text_fr:
                db.execute(
                    "INSERT INTO translations (id, phrase_id, language, text) VALUES (?, ?, ?, ?)",
                    [trans_id, phrase_id, "fr", text_fr]
                )
                trans_id += 1

            phrase_id += 1
            inserted_count += 1

        except Exception as e:
            logger.warning(f"Failed to insert phrase '{text_rw}': {e}")

    db.commit()
    logger.info(f"Inserted {inserted_count} phrases with translations")


def main():
    """Main preprocessing pipeline."""
    corpus_file = Path(Config.CORPUS_FILE)
    db_file = Path(corpus_file.parent) / "corpus.duckdb"

    logger.info("Starting corpus preprocessing pipeline...")
    logger.info(f"Input: {corpus_file}")
    logger.info(f"Output: {db_file}")

    # Extract phrases
    phrases = extract_phrases_from_corpus(str(corpus_file))

    # Create database
    db = create_duckdb_database(str(db_file))

    # Populate database
    populate_database(db, phrases)

    # Verify
    stats = db.execute("SELECT COUNT(*) as phrase_count FROM phrases").fetchone()
    logger.info(f"✓ Preprocessing complete: {stats[0]} phrases in database")

    db.close()


if __name__ == "__main__":
    main()
