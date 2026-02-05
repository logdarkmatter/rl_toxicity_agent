import logging
import pickle
import random
from collections import defaultdict
from typing import Tuple

import numpy as np


class QLearningAgent:
    def __init__(self,
                 n_actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 exploration_rate: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):

        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

        logging.info("QLearningAgent initialized.")

    def choose_action(self, state: Tuple[int, int, int], use_epsilon: bool = True) -> int:
        if use_epsilon and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        return int(np.argmax(self.q_table[state]))

    def learn(self,
              state: Tuple[int, int, int],
              action_idx: int,
              reward: float,
              next_state: Tuple[int, int, int],
              done: bool):

        current_q = self.q_table[state][action_idx]
        best_next_q = np.max(self.q_table[next_state]) if not done else 0
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - current_q

        self.q_table[state][action_idx] += self.lr * td_error

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath: str = "q_table.pkl"):
        model_data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logging.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str = "q_table.pkl"):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
            self.q_table.update(model_data["q_table"])
            self.epsilon = model_data["epsilon"]
            logging.info(f"Model loaded from {filepath}")
        except FileNotFoundError:
            logging.warning("Model file not found. Starting from scratch.")