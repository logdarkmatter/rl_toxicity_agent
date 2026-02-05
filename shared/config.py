"""
Configuration file for constants used across the application.

This file contains various constants such as API keys, default thresholds, and other configuration settings
that are used throughout the application.

Attributes:
    MAX_REPEATED_CHARS (int): The maximum number of repeated characters allowed in a word.
    DEFAULT_CLASSIFIER_PATH (str): The default path to the classifier model.
    SBERT_MODEL_NAME (str): The name of the SBERT model to use for text embedding.
    TOXICITY_THRESHOLD (float): The default threshold for toxicity classification.
    EPSILON (float): A small value used for numerical stability.
"""

MAX_REPEATED_CHARS = 25
DEFAULT_CLASSIFIER_PATH = 'model/toxic_classifier.pkl'
DEFAULT_TRAINED_MODEL_PATH = "model/trained/trained_mediator_agent.pkl"
BEST_MODEL_PATH = "model/trained/best_mediator_agent.pkl"
DEFAULT_CHAT_DATA_PATH = "data/chat_messages.csv"
MAX_STEPS_PER_EPISODE = 50  # How long a chat session lasts
SBERT_MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
TOXICITY_THRESHOLD = 0.5  # For future use
EPSILON = 1e-6
