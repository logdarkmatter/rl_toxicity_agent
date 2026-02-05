import re
import logging
from typing import Set

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from shared.config import MAX_REPEATED_CHARS
from shared.logging_config import configure_logging
from shared.exceptions.custom_exceptions import TextPreprocessingError

# Configure logging
configure_logging("logs/rl-toxicity-agent.log")


def _ensure_nltk_data() -> None:
    """Download required NLTK data if not available."""
    required_data = [
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet')
    ]

    for path, name in required_data:
        try:
            nltk.data.find(path)
        except LookupError:
            logging.info(f"Downloading NLTK data: {name}")
            nltk.download(name, quiet=True)


def _get_stop_words() -> Set[str]:
    _ensure_nltk_data()
    excluded_negations = {'not', 'no', 'don', 'won', 'wouldn', 'couldn'}
    return set(stopwords.words('english')).union(ENGLISH_STOP_WORDS) - excluded_negations


_ensure_nltk_data()
lemmatizer = WordNetLemmatizer()
stop_words = _get_stop_words()


def preprocess_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    if not isinstance(text, str):
        text = str(text)

    logging.debug(f'Original text: {text}')

    try:
        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Remove stop words
        if remove_stopwords:
            text = " ".join(word for word in text.split() if word not in stop_words)
            logging.debug(f'Text after removing stop words: {text}')

        # Limit repeated characters
        repeated_char_pattern = r'(.)\1{' + str(MAX_REPEATED_CHARS) + r',}'
        text = re.sub(repeated_char_pattern, r'\1' * MAX_REPEATED_CHARS, text)

        # Apply lemmatization
        if lemmatize and text:
            text = " ".join(lemmatizer.lemmatize(word) for word in text.split() if word)

        logging.info("Text preprocessed successfully")
        return text

    except Exception as e:
        logging.error(f"Error preprocessing text: {text[:50]}... - {e}")
        raise TextPreprocessingError(f"Failed to preprocess text: {e}")
