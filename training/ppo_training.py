"""
PPO Training Script
Train Proximal Policy Optimization agent on ClinicEnv with best configuration.

Usage:
    python training/ppo_training.py --timesteps 200000 --seeds 5
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environment.custom_env import ClinicEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def evaluate_agent(model, env, num_episodes=20, deterministic=True):
    """
    Evaluate model on env. Works with VecEnv/VecNormalize and non-vectorized envs.
    Returns a dict of metrics.
    """
    is_vec = hasattr(env, "num_envs") and getattr(env, "num_envs", None) is not None
    rewards = []
    lengths = []
    accuracies = []

    for _ in range(num_episodes):
        # reset
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        correct = 0
        total = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)

            if is_vec:
                # VecEnv step returns arrays
                obs, reward, done, info = env.step(action)
                reward = float(reward[0])
                done = bool(done[0])
                info = info[0]
                action_scalar = int(np.array(action).reshape(-1)[0])
            else:
                # gymnasium style step (obs, reward, terminated, truncated, info)
                action_scalar = int(np.array(action).reshape(-1)[0])
                obs, reward, terminated, truncated, info = env.step(action_scalar)
                reward = float(reward)
                done = bool(terminated or truncated)

            ep_reward += reward
            ep_len += 1

            if "correct_action" in info:
                total += 1
                if action_scalar == int(info["correct_action"]):
                    correct += 1

        rewards.append(ep_reward)
        lengths.append(ep_len)
        if total > 0:
            accuracies.append(100.0 * correct / total)

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_triage_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "std_triage_accuracy": float(np.std(accuracies)) if accuracies else 0.0
    }


def train_ppo_config(config, total_timesteps=50000, seed=42, project_dir=None, verbose=0):
    """
    Train PPO for one config and return (model, eval_results).
    This function creates its own DummyVecEnv+VecNormalize factory; it DOES NOT accept an env instance.
    Expects config to be a dict with keys like 'id','learning_rate','gamma','n_steps',...
    """
    def make_env():
        return ClinicEnv(seed=seed, max_steps=500)

    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    venv.seed(seed)

    policy_kwargs = dict(net_arch=[256, 256])

    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=policy_kwargs,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        seed=seed,
        verbose=verbose,
        tensorboard_log=(f"{project_dir}/logs/ppo/{config['id']}" if project_dir else None)
    )

    model.learn(total_timesteps=total_timesteps)

    # Freeze normalization stats for evaluation
    venv.training = False
    eval_results = evaluate_agent(model, venv, num_episodes=20)

    # Return model, eval_results, AND venv (for saving normalization stats)
    return model, eval_results, venv


def train_ppo(
    config: Dict[str, Any],
    total_timesteps: int = 200000,
    seed: int = 42,
    output_dir: str = "models/ppo",
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Train PPO agent with given configuration.
    
    Args:
        config: PPO hyperparameter configuration
        total_timesteps: Total training timesteps
        seed: Random seed
        output_dir: Directory to save model
        verbose: Verbosity level
        
    Returns:
        Training results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Training PPO Agent")
    print(f"Configuration: {config['id']}")
    print(f"Seed: {seed}, Timesteps: {total_timesteps}")
    print(f"{'='*60}\n")
    
    # Train using the notebook's function
    start_time = time.time()
    model, eval_results, venv = train_ppo_config(
        config,
        total_timesteps=total_timesteps,
        seed=seed,
        project_dir=output_dir,
        verbose=verbose
    )
    training_time = time.time() - start_time
    
    # Save model AND VecNormalize stats (CRITICAL!)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"ppo_seed{seed}.zip")
    model.save(model_path)
    
    # Save VecNormalize statistics
    vecnorm_path = os.path.join(output_dir, f"ppo_seed{seed}_vecnormalize.pkl")
    venv.save(vecnorm_path)
    print(f"✓ Saved VecNormalize stats: {vecnorm_path}")
    
    # Compile results
    results = {
        'config_id': config['id'],
        'seed': seed,
        'total_timesteps': total_timesteps,
        'training_time_sec': training_time,
        'final_mean_reward': eval_results["mean_reward"],
        'final_std_reward': eval_results["std_reward"],
        'final_mean_triage_accuracy': eval_results["mean_triage_accuracy"],
        'final_std_triage_accuracy': eval_results["std_triage_accuracy"],
        'model_path': model_path
    }
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final Reward: {results['final_mean_reward']:.2f} ± {results['final_std_reward']:.2f}")
    print(f"Final Accuracy: {results['final_mean_triage_accuracy']:.1f}% ± {results['final_std_triage_accuracy']:.1f}%")
    print(f"Model saved: {model_path}")
    print(f"{'='*60}\n")
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PPO agent on ClinicEnv")
    
    parser.add_argument('--timesteps', type=int, default=200000,
                        help='Total training timesteps (default: 200000)')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of random seeds to train (default: 5)')
    parser.add_argument('--output_dir', type=str, default='models/ppo',
                        help='Output directory for models')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0=quiet, 1=info)')
    
    args = parser.parse_args()
    
    # Best PPO configuration from sweep (ppo_10_mid_exploration) - 506.39 reward, 94.80% accuracy
    best_config = {
        "id": "ppo_10_mid_exploration",
        "description": "Moderate LR, mid-range n_steps, slightly higher exploration",
        "learning_rate": 2e-4,
        "gamma": 0.99,
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 8,
        "gae_lambda": 0.95,
        "clip_range": 0.15,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5
    }
    
    # Define seed list - expand if more seeds needed
    SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
    
    # Slice seeds based on num_seeds requested
    if args.seeds > len(SEEDS):
        # Expand seed list if needed (add more seeds)
        additional_seeds = [SEEDS[-1] + i * 1000 for i in range(1, args.seeds - len(SEEDS) + 1)]
        SEEDS = SEEDS + additional_seeds
    
    seeds_to_use = SEEDS[:args.seeds]
    
    print(f"\n{'='*80}")
    print(f"PPO TRAINING - BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"Configuration: {best_config['id']}")
    print(f"Description: {best_config['description']}")
    print(f"Training {args.seeds} seeds × {args.timesteps} timesteps each")
    print(f"Seeds: {seeds_to_use}")
    print(f"{'='*80}\n")
    
    # Train multiple seeds
    all_results = []
    for seed_idx, seed in enumerate(seeds_to_use):
        results = train_ppo(
            best_config,
            total_timesteps=args.timesteps,
            seed=seed,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        all_results.append(results)
    
    # Save aggregate results
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'config': best_config,
            'seeds': all_results,
            'aggregate': {
                'mean_final_reward': float(np.mean([r['final_mean_reward'] for r in all_results])),
                'std_final_reward': float(np.std([r['final_mean_reward'] for r in all_results])),
                'mean_final_accuracy': float(np.mean([r['final_mean_triage_accuracy'] for r in all_results])),
                'std_final_accuracy': float(np.std([r['final_mean_triage_accuracy'] for r in all_results]))
            }
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"ALL SEEDS COMPLETE")
    print(f"{'='*80}")
    print(f"Aggregate Results ({args.seeds} seeds):")
    print(f"  Mean Reward: {np.mean([r['final_mean_reward'] for r in all_results]):.2f} ± {np.std([r['final_mean_reward'] for r in all_results]):.2f}")
    print(f"  Mean Accuracy: {np.mean([r['final_mean_triage_accuracy'] for r in all_results]):.1f}% ± {np.std([r['final_mean_triage_accuracy'] for r in all_results]):.1f}%")
    print(f"Results saved: {results_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
