import logging
import pickle
import random
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple

import pandas as pd

from shared.config import DEFAULT_CLASSIFIER_PATH, SBERT_MODEL_NAME, DEFAULT_CHAT_DATA_PATH, TOXICITY_THRESHOLD
from shared.logging_config import configure_logging
from shared.preprocessing import preprocess_text

configure_logging("logs/rl-toxicity-agent.log")


class Action(Enum):
    DoNothing = 0
    WarnSpeaker = 1


@dataclass
class UserConfig:
    propensity: float
    warn_count: int = 0


class ChatEnvironment:
    DEFAULT_USERS = {
        'user_A': UserConfig(propensity=0.3),
        'user_B': UserConfig(propensity=0.6)
    }

    def __init__(self, classifier_path: str = DEFAULT_CLASSIFIER_PATH,
                 users: Optional[Dict[str, UserConfig]] = None,
                 chat_data_path: str = DEFAULT_CHAT_DATA_PATH) -> None:

        logging.debug('Initializing environment...')

        # Load chat data from external file
        self.toxic_messages, self.safe_messages = self._load_chat_data(chat_data_path)

        self.sbert_model = self._load_sbert_model()
        self.classifier = self._load_classifier(classifier_path)

        self._initial_users = deepcopy(users) if users is not None else deepcopy(self.DEFAULT_USERS)
        self.users = deepcopy(self._initial_users)
        self.user_map = {name: i for i, name in enumerate(self.users.keys())}
        self.id_map = {i: name for name, i in self.user_map.items()}

        self.steps_since_last_action = 0
        self.episode_toxicity = []
        self.last_speaker_id = 0

    def _load_chat_data(self, chat_data_path: str) -> Tuple[list, list]:
        """Loads chat messages from a CSV file."""
        try:
            df = pd.read_csv(chat_data_path)
            toxic_messages = df[df['is_toxic'] == 1]['text'].tolist()
            safe_messages = df[df['is_toxic'] == 0]['text'].tolist()
            logging.info(f"Loaded {len(toxic_messages)} toxic and {len(safe_messages)} safe messages.")
            return toxic_messages, safe_messages
        except FileNotFoundError:
            logging.error(f"Chat data file not found at '{chat_data_path}'. Using empty lists.")
            return [], []
        except Exception as e:
            logging.error(f"Error loading chat data: {e}")
            return [], []

    def _load_sbert_model(self):
        try:
            import os
            import ssl
            import certifi
            from sentence_transformers import SentenceTransformer

            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
            os.environ['SSL_CERT_FILE'] = certifi.where()

            model = SentenceTransformer(SBERT_MODEL_NAME)
            logging.info(f"Successfully loaded SBERT model: {SBERT_MODEL_NAME}")
            return model
        except ImportError:
            logging.error("sentence-transformers library not found.")
            return None
        except Exception as e:
            logging.error(f"Error loading SBERT model: {e}")
            return None

    def _load_classifier(self, classifier_path: str) -> Optional[Any]:
        try:
            with open(classifier_path, 'rb') as f:
                loaded_data = pickle.load(f)

                if isinstance(loaded_data, dict) and 'model' in loaded_data:
                    classifier = loaded_data['model']
                else:
                    classifier = loaded_data

            logging.info(f"Successfully loaded classifier from {classifier_path}")
            return classifier

        except FileNotFoundError:
            logging.error(f"Could not find '{classifier_path}'.")
            return None
        except Exception as e:
            logging.error(f"Error loading classifier: {e}")
            return None

    def _get_toxicity(self, text: str) -> float:
        if self.classifier is None:
            raise RuntimeError("Classifier not initialized")

        if self.sbert_model is None:
            raise RuntimeError("SBERT model not initialized")

        try:
            processed_text = preprocess_text(text)
            embeddings = self.sbert_model.encode([processed_text])
            probabilities = self.classifier.predict_proba(embeddings)
            toxic_probability = probabilities[0][1]

            logging.debug(f"Text: '{text}' | Toxicity: {toxic_probability:.4f}")
            return toxic_probability

        except Exception as e:
            logging.error(f"Error during classification: {e}")
            return 0.0

    def _get_state(self) -> Tuple[int, int, int]:
        last_toxicity = self.episode_toxicity[-1] if self.episode_toxicity else 0.0

        if last_toxicity < 0.4:
            toxicity_level = 0  # Safe
        elif last_toxicity < 0.75:
            toxicity_level = 1  # Borderline
        else:
            toxicity_level = 2  # Toxic

        if self.steps_since_last_action < 3:
            time_state = 0
        elif self.steps_since_last_action < 6:
            time_state = 1
        else:
            time_state = 2

        return (self.last_speaker_id, toxicity_level, time_state)

    def reset(self, randomize_propensities: bool = True) -> Tuple[int, int, int]:
        logging.info("--- Reseting Environment ---")

        if randomize_propensities:
            self.users = {}
            for name in self._initial_users.keys():
                random_propensity = random.uniform(0.1, 0.8)
                self.users[name] = UserConfig(propensity=random_propensity)
                logging.info(f"New episode: Randomized propensity for {name} is {random_propensity:.2f}")

        else:
            self.users = deepcopy(self._initial_users)
            for name, config in self.users.items():
                logging.info(f"Evaluation episode: Using fixed propensity for {name} of {config.propensity:.2f}")

        self.steps_since_last_action = 0
        self.episode_toxicity = []
        self.last_speaker_id = random.choice(list(self.user_map.values()))
        return self._get_state()

    def step(self, action: Action) -> Tuple[Tuple[int, int, int], float, bool]:
        last_toxicity = self.episode_toxicity[-1] if self.episode_toxicity else 0.0

        is_toxic = last_toxicity > TOXICITY_THRESHOLD
        reward = 0.0

        if action == Action.DoNothing:
            self.steps_since_last_action += 1
            if is_toxic:
                reward = -10.0  # Penalty for missing a toxic message
            else:
                reward = 1.0  # Small reward for correctly doing nothing
        else:  # Agent issued a warning (Action.WarnSpeaker)
            self.steps_since_last_action = 0
            if is_toxic:
                reward = 10.0  # High reward for correctly warning a toxic message
            else:
                reward = -5.0  # Penalty for an unnecessary warning

            last_speaker_name = self.id_map[self.last_speaker_id]
            user_to_warn = self.users[last_speaker_name]
            user_to_warn.propensity *= 0.7
            user_to_warn.warn_count += 1
            logging.info(f"Agent warned {last_speaker_name}")

        # C. Simulate the NEXT user message to generate the next state
        speaker = random.choice(list(self.users.keys()))
        self.last_speaker_id = self.user_map[speaker]
        user_config = self.users[speaker]

        if random.random() < user_config.propensity:
            msg = random.choice(self.toxic_messages) if self.toxic_messages else "toxic fallback"
        else:
            msg = random.choice(self.safe_messages) if self.safe_messages else "safe fallback"

        current_toxicity = self._get_toxicity(msg)
        self.episode_toxicity.append(current_toxicity)

        logging.debug(f"Speaker: {speaker} | Msg: {msg} | Tox: {current_toxicity:.2f}")

        # D. Return Next State
        next_state = self._get_state()
        done = False

        return next_state, reward, done


if __name__ == "__main__":
    env = ChatEnvironment(classifier_path=DEFAULT_CLASSIFIER_PATH)

    initial_state = env.reset()
    logging.debug(f"Initial State: {initial_state}")

    for _ in range(5):
        random_action = random.choice(list(Action))
        state, reward, _ = env.step(random_action)
        logging.debug(f"Action: {random_action.name} -> State: {state} | Reward: {reward:.2f}")
