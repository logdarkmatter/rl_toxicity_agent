import argparse
import logging
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment import ChatEnvironment, Action, UserConfig
from qlearningagent import QLearningAgent
from shared.config import DEFAULT_TRAINED_MODEL_PATH, MAX_STEPS_PER_EPISODE, BEST_MODEL_PATH

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def run_training_session():
    print("--- Starting Reinforcement Learning Training ---")

    # 1. Initialize Environment and Agent
    env = ChatEnvironment()
    agent = QLearningAgent(n_actions=len(Action))

    # 2. Training Hyperparameters
    N_EPISODES = 1000

    # 3. Metrics for Report
    history = {
        "episode": [],
        "total_reward": [],
        "avg_toxicity": [],
        "epsilon": [],
        "propensity_a": [],
        "propensity_b": []
    }

    # --- Checkpointing variables to save the best model ---
    best_rolling_reward = -np.inf
    rolling_window = 50
    best_model_saved = False
    best_episode = -1

    # --- THE TRAINING LOOP ---
    for episode in range(N_EPISODES):
        state = env.reset(randomize_propensities=True)
        total_reward = 0
        episode_toxicity_scores = []

        history["propensity_a"].append(env.users['user_A'].propensity)
        history["propensity_b"].append(env.users['user_B'].propensity)

        for step in range(MAX_STEPS_PER_EPISODE):
            action_idx = agent.choose_action(state)
            action_enum = Action(action_idx)
            next_state, reward, done = env.step(action_enum)
            agent.learn(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            if len(env.episode_toxicity) > 0:
                episode_toxicity_scores.append(env.episode_toxicity[-1])

        agent.update_epsilon()
        avg_tox = np.mean(episode_toxicity_scores) if episode_toxicity_scores else 0
        history["episode"].append(episode)
        history["total_reward"].append(total_reward)
        history["avg_toxicity"].append(avg_tox)
        history["epsilon"].append(agent.epsilon)

        # --- Checkpointing Logic: Save the best model based on rolling reward ---
        if episode >= rolling_window:
            current_rolling_reward = np.mean(history["total_reward"][-rolling_window:])
            if current_rolling_reward > best_rolling_reward:
                best_rolling_reward = current_rolling_reward
                best_episode = episode + 1  # Store the episode number
                agent.save_model(BEST_MODEL_PATH)
                best_model_saved = True
                print(f"** New best model saved at episode {episode + 1} "
                      f"with rolling reward: {current_rolling_reward:.2f} **")

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {total_reward:6.2f} | "
                  f"Avg Tox: {avg_tox:.4f} | "
                  f"Epsilon: {agent.epsilon:.4f}")

    print("--- Training Completed ---")
    # Overwrite the default model with the best one found during training
    if best_model_saved:
        shutil.copy(BEST_MODEL_PATH, DEFAULT_TRAINED_MODEL_PATH)
        print(f"Final model '{DEFAULT_TRAINED_MODEL_PATH}' updated with the best performing agent.")
    else:
        # If no "best" model was ever saved, save the final one as default
        agent.save_model(DEFAULT_TRAINED_MODEL_PATH)
        print(f"Final model saved to '{DEFAULT_TRAINED_MODEL_PATH}'")

    df = pd.DataFrame(history)
    df.to_csv("results/training_results.csv", index=False)
    print("Results saved to 'results/training_results.csv'")
    return df, best_episode


def plot_results(df, best_episode: int):
    rolling_window = 50

    plt.figure(figsize=(12, 5))

    # Plot 1: Reward over time
    plt.subplot(1, 2, 1)
    plt.plot(df['total_reward'], alpha=0.3, color='gray', label='Raw Reward')
    plt.plot(df['total_reward'].rolling(rolling_window).mean(), color='blue', linewidth=2, label='Reward Trend')

    # Add a vertical line to mark the best model checkpoint
    if best_episode != -1:
        plt.axvline(x=best_episode, color='green', linestyle='--', linewidth=2,
                    label=f'Best Model (Ep. {best_episode})')

    plt.title("Agent Performance (Total Reward)")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Higher is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Toxicity over time
    plt.subplot(1, 2, 2)
    plt.plot(df['avg_toxicity'], alpha=0.3, color='gray', label='Raw Toxicity')
    plt.plot(df['avg_toxicity'].rolling(rolling_window).mean(), color='red', linewidth=2, label='Toxicity Trend')
    plt.title("Chat Toxicity Level")
    plt.xlabel("Episode")
    plt.ylabel("Avg Toxicity (Lower is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/results_plot.png")
    print("Plots saved to 'results/plots/results_plot.png'")
    plt.show()


def run_evaluation_session(model_path: str, n_episodes: int, propensity_a: float, propensity_b: float):
    """
    Runs a test session with a trained agent.
    The agent will not learn or explore.
    """
    print(f"\n--- Starting Evaluation Session for model: {model_path} ---")
    print(f"Custom User Propensities: UserA={propensity_a}, UserB={propensity_b}")

    # Create a custom user configuration for the environment
    custom_users = {
        'user_A': UserConfig(propensity=propensity_a),
        'user_B': UserConfig(propensity=propensity_b)
    }

    env = ChatEnvironment(users=custom_users)
    agent = QLearningAgent(n_actions=len(Action))
    agent.load_model(model_path)

    total_rewards = []
    avg_toxicities = []
    actions_taken = {action.name: 0 for action in Action}

    for episode in range(n_episodes):
        state = env.reset(randomize_propensities=False)
        total_reward = 0
        episode_toxicity_scores = []
        for step in range(MAX_STEPS_PER_EPISODE):
            action_idx = agent.choose_action(state, use_epsilon=False)
            action_enum = Action(action_idx)
            actions_taken[action_enum.name] += 1
            next_state, reward, done = env.step(action_enum)
            state = next_state
            total_reward += reward
            if len(env.episode_toxicity) > 0:
                episode_toxicity_scores.append(env.episode_toxicity[-1])

        total_rewards.append(total_reward)
        avg_tox = np.mean(episode_toxicity_scores) if episode_toxicity_scores else 0
        avg_toxicities.append(avg_tox)

    print("\n--- Evaluation Report ---")
    print(f"Model: '{model_path}'")
    print(f"Ran for {n_episodes} episodes.")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Toxicity: {np.mean(avg_toxicities):.4f}")
    print("\nAction Distribution:")
    for action, count in actions_taken.items():
        print(f"- {action}: {count} times")
    print("--- Evaluation Completed ---")


if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Train or evaluate the Mediator Bot.")
        parser.add_argument(
            "--mode",
            type=str,
            choices=["train", "eval"],
            default="train",
            help="Set to 'train' to run a full training session, or 'eval' to test an existing model."
        )
        parser.add_argument(
            "--model-path",
            type=str,
            default=DEFAULT_TRAINED_MODEL_PATH,
            help="Path to the model file for evaluation."
        )
        parser.add_argument(
            "--eval-episodes",
            type=int,
            default=1000,
            help="Number of episodes to run during an evaluation session."
        )
        parser.add_argument(
            "--propensity-a",
            type=float,
            help="Set toxicity propensity for User A during evaluation (e.g., 0.7)."
        )
        parser.add_argument(
            "--propensity-b",
            type=float,
            help="Set toxicity propensity for User B during evaluation (e.g., 0.2)."
        )
        args = parser.parse_args()

        prop_a = args.propensity_a if args.propensity_a is not None else 0.3
        prop_b = args.propensity_b if args.propensity_b is not None else 0.6

        if args.mode == "train":
            df_results, best_episode = run_training_session()
            plot_results(df_results, best_episode)

            print("\n--- Evaluating Final (Best) Model ---")
            run_evaluation_session(
                model_path=DEFAULT_TRAINED_MODEL_PATH,
                n_episodes=args.eval_episodes,
                propensity_a=prop_a,
                propensity_b=prop_b
            )
        elif args.mode == "eval":
            run_evaluation_session(
                model_path=args.model_path,
                n_episodes=args.eval_episodes,
                propensity_a=prop_a,
                propensity_b=prop_b
            )
