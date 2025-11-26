"""
A2C Training Script
Train Advantage Actor-Critic agent on ClinicEnv with best configuration.

Usage:
    python training/a2c_training.py --timesteps 200000 --seeds 5
"""

import argparse
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environment.custom_env import ClinicEnv
from stable_baselines3 import A2C


def evaluate_agent(model, env, num_episodes: int = 20, deterministic: bool = True) -> Dict[str, float]:
    """Evaluate trained agent and return metrics."""
    episode_rewards = []
    episode_lengths = []
    triage_accuracies = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        correct = 0
        total = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
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


def train_a2c_config(
    config: Dict[str, Any],
    env,
    total_timesteps: int = 50000,
    seed: int = 42,
    verbose: int = 0,
):
    """Train A2C with given configuration."""
    env.reset(seed=seed)

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        gae_lambda=config["gae_lambda"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        normalize_advantage=config["normalize_advantage"],
        seed=seed,
        verbose=verbose,
    )

    model.learn(total_timesteps=total_timesteps)
    eval_results = evaluate_agent(model, env, num_episodes=20)
    return model, eval_results


def train_a2c(
    config: Dict[str, Any],
    total_timesteps: int = 200000,
    seed: int = 42,
    output_dir: str = "models/a2c",
    verbose: int = 1,
) -> Dict[str, Any]:
    """Train A2C agent with given configuration."""
    print(f"\n{'=' * 60}")
    print("Training A2C Agent")
    print(f"Configuration: {config['id']}")
    print(f"Seed: {seed}, Timesteps: {total_timesteps}")
    print(f"{'=' * 60}\n")

    env = ClinicEnv(seed=seed, max_steps=500)

    start_time = time.time()
    model, eval_results = train_a2c_config(
        config,
        env,
        total_timesteps=total_timesteps,
        seed=seed,
        verbose=verbose,
    )
    training_time = time.time() - start_time

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"a2c_seed{seed}.zip")
    model.save(model_path)
    env.close()

    results = {
        "config_id": config["id"],
        "seed": seed,
        "total_timesteps": total_timesteps,
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
    parser = argparse.ArgumentParser(description="Train A2C agent on ClinicEnv")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total training timesteps (default: 200000)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds to train (default: 5)")
    parser.add_argument("--output_dir", type=str, default="models/a2c", help="Output directory for models")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0=quiet, 1=info)")
    args = parser.parse_args()

    best_config = {
        "id": "a2c_high_entropy",
        "description": "Higher entropy for exploration",
        "learning_rate": 0.0007,
        "gamma": 0.99,
        "n_steps": 5,
        "gae_lambda": 1.0,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": False,
    }

    seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
    if args.seeds > len(seeds):
        additional = [seeds[-1] + i * 1000 for i in range(1, args.seeds - len(seeds) + 1)]
        seeds.extend(additional)
    seeds_to_use = seeds[: args.seeds]

    print(f"\n{'=' * 80}")
    print("A2C TRAINING - BEST CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"Configuration: {best_config['id']}")
    print(f"Description: {best_config['description']}")
    print(f"Training {args.seeds} seeds × {args.timesteps} timesteps each")
    print(f"Seeds: {seeds_to_use}")
    print(f"{'=' * 80}\n")

    all_results = []
    for seed in seeds_to_use:
        results = train_a2c(
            best_config,
            total_timesteps=args.timesteps,
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

