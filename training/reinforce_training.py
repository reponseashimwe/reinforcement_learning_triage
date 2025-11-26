"""
REINFORCE Training Script
Train REINFORCE (vanilla policy gradient) agent on ClinicEnv with best configuration.

Usage:
    python training/reinforce_training.py --episodes 2000 --seeds 5
"""

import argparse
import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environment.custom_env import ClinicEnv


class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hd), nn.ReLU()])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)

    def get_action(self, obs, deterministic: bool = False):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        logits = self.forward(obs_tensor)
        probs = torch.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
            log_prob = torch.log(probs[0, action] + 1e-9)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action))

        return action, log_prob


class ValueNetwork(nn.Module):
    """Value network used as baseline."""

    def __init__(self, obs_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hd), nn.ReLU()])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs).squeeze(-1)


class REINFORCE:
    """Vanilla policy gradient agent with optional baseline."""

    def __init__(
        self,
        env,
        learning_rate: float,
        gamma: float,
        hidden_dims: List[int],
        use_baseline: bool,
        max_grad_norm: float,
        entropy_coef: float,
    ):
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_baseline = use_baseline

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dims)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        if use_baseline:
            self.value_net = ValueNetwork(obs_dim, hidden_dims)
            self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        else:
            self.value_net = None

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns

    def train_episode(self):
        obs, _ = self.env.reset()
        observations = []
        log_probs = []
        rewards = []
        done = False

        while not done:
            observations.append(obs)
            action, log_prob = self.policy.get_action(obs)
            next_obs, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc

            rewards.append(reward)
            log_probs.append(log_prob)
            obs = next_obs

        returns = self.compute_returns(rewards)
        log_probs = torch.stack(log_probs)
        obs_tensor = torch.FloatTensor(np.array(observations))

        if self.use_baseline and self.value_net is not None:
            values = self.value_net(obs_tensor)
            advantages = returns - values.detach()

            value_loss = torch.nn.functional.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
        else:
            advantages = returns

        policy_loss = -(log_probs * advantages).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def predict(self, obs, deterministic: bool = True):
        action, _ = self.policy.get_action(obs, deterministic=deterministic)
        return action, None

    def save(self, path: str):
        state = {"policy": self.policy.state_dict()}
        if self.value_net:
            state["value"] = self.value_net.state_dict()
        torch.save(state, path)


def evaluate_agent(model, env, num_episodes: int = 20) -> Dict[str, float]:
    """Evaluate trained agent and return metrics."""
    episode_rewards = []
    episode_lengths = []
    triage_accuracies = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        correct = 0
        total = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if "correct_action" in info:
                total += 1
                if action == info["correct_action"]:
                    correct += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if total > 0:
            triage_accuracies.append(100.0 * correct / total)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_triage_accuracy": float(np.mean(triage_accuracies)) if triage_accuracies else 0.0,
        "std_triage_accuracy": float(np.std(triage_accuracies)) if triage_accuracies else 0.0,
    }


def train_reinforce_config(
    config: Dict[str, Any],
    env,
    num_episodes: int = 100,
    seed: int = 42,
):
    """Train REINFORCE with given configuration."""
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    agent = REINFORCE(
        env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        hidden_dims=config.get("hidden_dims", [256, 256]),
        use_baseline=config.get("use_baseline", True),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        entropy_coef=config.get("entropy_coef", 0.01),
    )

    for _ in range(num_episodes):
        agent.train_episode()

    eval_results = evaluate_agent(agent, env, num_episodes=20)
    return agent, eval_results


def train_reinforce(
    config: Dict[str, Any],
    num_episodes: int = 2000,
    seed: int = 42,
    output_dir: str = "models/reinforce",
    verbose: int = 1,
) -> Dict[str, Any]:
    """Train REINFORCE agent with given configuration."""
    print(f"\n{'=' * 60}")
    print("Training REINFORCE Agent")
    print(f"Configuration: {config['id']}")
    print(f"Seed: {seed}, Episodes: {num_episodes}")
    print(f"{'=' * 60}\n")

    env = ClinicEnv(seed=seed, max_steps=500)

    start_time = time.time()
    agent, eval_results = train_reinforce_config(
        config,
        env,
        num_episodes=num_episodes,
        seed=seed,
    )
    training_time = time.time() - start_time

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"reinforce_seed{seed}.pt")
    agent.save(model_path)
    env.close()

    results = {
        "config_id": config["id"],
        "seed": seed,
        "num_episodes": num_episodes,
        "training_time_sec": training_time,
        "final_mean_reward": eval_results["mean_reward"],
        "final_std_reward": eval_results["std_reward"],
        "final_mean_triage_accuracy": eval_results["mean_triage_accuracy"],
        "final_std_triage_accuracy": eval_results["std_triage_accuracy"],
        "model_path": model_path,
    }

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"Final Reward: {results['final_mean_reward']:.2f} ± {results['final_std_reward']:.2f}")
    print(
        f"Final Accuracy: {results['final_mean_triage_accuracy']:.1f}% ± "
        f"{results['final_std_triage_accuracy']:.1f}%"
    )
    print(f"Model saved: {model_path}")
    print(f"{'=' * 60}\n")

    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train REINFORCE agent on ClinicEnv")
    parser.add_argument("--episodes", type=int, default=2000, help="Training episodes per seed (default: 2000)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds to train (default: 5)")
    parser.add_argument("--output_dir", type=str, default="models/reinforce", help="Output directory for models")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0=quiet, 1=info)")
    args = parser.parse_args()

    best_config = {
        "id": "reinforce_no_baseline",
        "description": "Ablation study: No baseline (likely unstable)",
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "hidden_dims": [256, 256],
        "use_baseline": False,
        "max_grad_norm": 0.5,
        "entropy_coef": 0.01,
    }

    seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
    if args.seeds > len(seeds):
        additional = [seeds[-1] + i * 1000 for i in range(1, args.seeds - len(seeds) + 1)]
        seeds.extend(additional)
    seeds_to_use = seeds[: args.seeds]

    print(f"\n{'=' * 80}")
    print("REINFORCE TRAINING - BEST CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"Configuration: {best_config['id']}")
    print(f"Description: {best_config['description']}")
    print(f"Training {args.seeds} seeds × {args.episodes} episodes each")
    print(f"Seeds: {seeds_to_use}")
    print(f"{'=' * 80}\n")

    all_results = []
    for seed in seeds_to_use:
        results = train_reinforce(
            best_config,
            num_episodes=args.episodes,
            seed=seed,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
        all_results.append(results)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": best_config,
                "seeds": all_results,
                "aggregate": {
                    "mean_final_reward": float(np.mean([r["final_mean_reward"] for r in all_results])),
                    "std_final_reward": float(np.std([r["final_mean_reward"] for r in all_results])),
                    "mean_final_accuracy": float(np.mean([r["final_mean_triage_accuracy"] for r in all_results])),
                    "std_final_accuracy": float(np.std([r["final_mean_triage_accuracy"] for r in all_results])),
                },
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 80}")
    print("ALL SEEDS COMPLETE")
    print(f"{'=' * 80}")
    print(f"Aggregate Results ({args.seeds} seeds):")
    print(
        f"  Mean Reward: {np.mean([r['final_mean_reward'] for r in all_results]):.2f} ± "
        f"{np.std([r['final_mean_reward'] for r in all_results]):.2f}"
    )
    print(
        f"  Mean Accuracy: {np.mean([r['final_mean_triage_accuracy'] for r in all_results]):.1f}% ± "
        f"{np.std([r['final_mean_triage_accuracy'] for r in all_results]):.1f}%"
    )
    print(f"Results saved: {results_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()


