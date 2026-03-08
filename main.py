import os
import sys
import argparse
import logging
import shutil

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from environment import ChatEnvironment, Action, UserConfig
from qlearningagent import QLearningAgent
from shared.config import DEFAULT_TRAINED_MODEL_PATH, MAX_STEPS_PER_EPISODE, BEST_MODEL_PATH

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
                best_episode = episode + 1
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
    if best_model_saved:
        shutil.copy(BEST_MODEL_PATH, DEFAULT_TRAINED_MODEL_PATH)
        print(f"Final model '{DEFAULT_TRAINED_MODEL_PATH}' updated with the best performing agent.")
    else:
        agent.save_model(DEFAULT_TRAINED_MODEL_PATH)
        print(f"Final model saved to '{DEFAULT_TRAINED_MODEL_PATH}'")

    df = pd.DataFrame(history)
    df.to_csv("results/training_results.csv", index=False)
    print("Results saved to 'results/training_results.csv'")
    return df, best_episode


def plot_results(df, best_episode: int):
    rolling_window = 50

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df['total_reward'], alpha=0.3, color='gray', label='Raw Reward')
    plt.plot(df['total_reward'].rolling(rolling_window).mean(), color='blue', linewidth=2, label='Reward Trend')

    if best_episode != -1:
        plt.axvline(x=best_episode, color='green', linestyle='--', linewidth=2,
                    label=f'Best Model (Ep. {best_episode})')

    plt.title("Agent Performance (Total Reward)")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Higher is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)

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
    print(f"\n--- Starting Evaluation Session for model: {model_path} ---")
    print(f"Custom User Propensities: UserA={propensity_a}, UserB={propensity_b}")

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


def plot_from_csv(csv_path: str = "results/training_results.csv", out_dir: str = "results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        print(f"File `{csv_path}` not found. Generate logs first or provide the correct path.")
        return

    df = pd.read_csv(csv_path)
    print("Columns available:", list(df.columns))

    if "episode" in df.columns:
        df.set_index("episode", inplace=True)

    required_metrics = ["total_reward", "avg_toxicity"]
    if not all(col in df.columns for col in required_metrics):
        print(f"Missing one of the required columns: {required_metrics}")
        return

    window = 50
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward (MA)', color=color)
    ax1.plot(df.index, df['total_reward'].rolling(window).mean(), color=color, linewidth=2, label='Reward (MA)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Avg Toxicity (MA)', color=color)
    ax2.plot(df.index, df['avg_toxicity'].rolling(window).mean(), color=color, linewidth=2, linestyle='--', label='Toxicity (MA)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Training Progress: Reward vs Toxicity (Rolling Window {window})')
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_progress_dual.png"))
    plt.close()

    # 2. Epsilon Decay
    if "epsilon" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['epsilon'], color='purple', linewidth=2)
        plt.title('Epsilon Decay over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "epsilon_decay.png"))
        plt.close()

    # 3. Scatter Plot: Reward vs Toxicity
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='avg_toxicity', y='total_reward', data=df, alpha=0.5, edgecolor=None)
    plt.title('Correlation: Total Reward vs Avg Toxicity')
    plt.xlabel('Average Toxicity')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "correlation_reward_toxicity.png"))
    plt.close()

    # 4. Histograms of Metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df['total_reward'], bins=30, kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Distribution of Total Rewards')

    sns.histplot(df['avg_toxicity'], bins=30, kde=True, ax=axes[1], color='red')
    axes[1].set_title('Distribution of Average Toxicity')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distributions.png"))
    plt.close()

    # 5. Heatmap: Toxicity by Propensity Bins (if propensities exist)
    if "propensity_a" in df.columns and "propensity_b" in df.columns:
        bins = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
        labels = [f"{i:.1f}" for i in bins[:-1]]

        df['prop_a_bin'] = pd.cut(df['propensity_a'], bins=bins, labels=labels, include_lowest=True)
        df['prop_b_bin'] = pd.cut(df['propensity_b'], bins=bins, labels=labels, include_lowest=True)

        pivot_table = df.pivot_table(index='prop_a_bin', columns='prop_b_bin', values='avg_toxicity', aggfunc='mean')

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Avg Toxicity'})
        plt.title('Heatmap: Avg Toxicity by User Propensities')
        plt.xlabel('User B Propensity')
        plt.ylabel('User A Propensity')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "heatmap_toxicity_propensities.png"))
        plt.close()

        # 6. Scatter with Hue for Propensities
        plt.figure(figsize=(10, 6))
        df['total_propensity'] = df['propensity_a'] + df['propensity_b']

        scatter = plt.scatter(df['avg_toxicity'], df['total_reward'],
                            c=df['total_propensity'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Sum of Propensities (A+B)')
        plt.title('Reward vs Toxicity (Colored by User Propensity)')
        plt.xlabel('Average Toxicity')
        plt.ylabel('Total Reward')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "scatter_reward_toxicity_propensity.png"))
        plt.close()

    print(f"All plots saved to directory: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, evaluate or plot results for the Mediator Bot.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "plot"],
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
    parser.add_argument(
        "--csv-path",
        type=str,
        default="results/training_results.csv",
        help="Path to evaluation CSV used when --mode plot is selected."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/plots",
        help="Directory to save plots when --mode plot is selected."
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
    elif args.mode == "plot":
        plot_from_csv(csv_path=args.csv_path, out_dir=args.out_dir)
        sys.exit(0)
