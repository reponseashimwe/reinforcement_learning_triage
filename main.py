"""
Main Entry Point for Best Performing Model

This script loads and runs the best-performing trained model
for the Dermatology Clinic Triage environment.

Usage:
    python main.py --model_type ppo --model_path models/ppo/best_model.zip
    python main.py --model_type reinforce --model_path models/reinforce/best_model.pt
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environment.custom_env import ClinicEnv
from environment.rendering import render_episode_to_video
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def load_sb3_model(model_type: str, model_path: str, env):
    """
    Load Stable Baselines3 model (DQN, PPO, A2C).

    Args:
        model_type: Model type ('dqn', 'ppo', 'a2c')
        model_path: Path to saved model
        env: Environment instance

    Returns:
        Loaded model (and VecEnv if PPO with normalization)
    """
    if model_type.lower() == 'dqn':
        from sb3_contrib import DQN
        model = DQN.load(model_path, env=env)
        return model, env
    elif model_type.lower() == 'ppo':
        from stable_baselines3 import PPO
        
        # Check if VecNormalize stats exist
        vecnormalize_path = model_path.replace('.zip', '_vecnormalize.pkl')
        if os.path.exists(vecnormalize_path):
            print(f"  Loading VecNormalize stats from {vecnormalize_path}")
            # Create vectorized environment
            venv = DummyVecEnv([lambda: env])
            venv = VecNormalize.load(vecnormalize_path, venv)
            venv.training = False  # Disable training mode
            venv.norm_reward = False
            
            model = PPO.load(model_path)
            model.set_env(venv)
            return model, venv
        else:
            print("  Warning: VecNormalize stats not found, using raw environment")
            model = PPO.load(model_path, env=env)
            return model, env
    elif model_type.lower() == 'a2c':
        from stable_baselines3 import A2C
        model = A2C.load(model_path, env=env)
        return model, env
    else:
        raise ValueError(f"Unknown SB3 model type: {model_type}")



def load_reinforce_model(model_path: str, env):
    """
    Load custom REINFORCE model.

    Args:
        model_path: Path to saved model
        env: Environment instance

    Returns:
        Loaded REINFORCE agent
    """
    from training.reinforce_training import REINFORCE

    agent = REINFORCE(env=env)
    agent.load(model_path)

    return agent


def evaluate_model(model, env, num_episodes: int = 10, model_type: str = 'sb3'):
    """
    Evaluate model performance.

    Args:
        model: Trained model
        env: Environment instance (can be VecEnv or regular env)
        num_episodes: Number of episodes to evaluate
        model_type: 'sb3' or 'reinforce'

    Returns:
        Dictionary of evaluation metrics
    """
    is_vec = hasattr(env, 'num_envs')
    episode_rewards = []
    episode_lengths = []
    triage_accuracies = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        correct_triages = 0
        total_triages = 0

        while not done:
            # Get action from model
            if model_type == 'sb3':
                action, _ = model.predict(obs, deterministic=True)
            else:  # REINFORCE
                action, _ = model.policy.get_action(obs, deterministic=True)

            # Step environment
            if is_vec:
                obs, reward, done, info = env.step(action)
                reward = float(reward[0])
                done = bool(done[0])
                info = info[0]
                action = int(np.array(action).reshape(-1)[0])
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Track triage accuracy
            if 'correct_action' in info:
                total_triages += 1
                if action == info['correct_action']:
                    correct_triages += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if total_triages > 0:
            triage_accuracies.append(100.0 * correct_triages / total_triages)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "mean_triage_accuracy": np.mean(triage_accuracies) if triage_accuracies else 0.0,
        "std_triage_accuracy": np.std(triage_accuracies) if triage_accuracies else 0.0
    }


def run_demo(model, env, model_type: str = 'sb3', render_mode: str = 'human'):
    """
    Run a demo episode with the trained model.

    Args:
        model: Trained model
        env: Environment instance
        model_type: 'sb3' or 'reinforce'
        render_mode: Rendering mode ('human', 'rgb_array', 'ansi')
    """
    obs, info = env.reset()
    done = False
    episode_reward = 0.0
    step_count = 0

    print("\n" + "="*60)
    print("RUNNING DEMO EPISODE")
    print("="*60 + "\n")

    while not done:
        # Render if in human mode
        if render_mode == 'ansi':
            print(env._render_ansi())

        # Get action from model
        if model_type == 'sb3':
            action, _ = model.predict(obs, deterministic=True)
        else:  # REINFORCE
            action, _ = model.policy.get_action(obs, deterministic=True)

        # Print action
        action_name = env.ACTION_NAMES[action]
        correct_action = info.get('correct_action', -1)
        correct_name = env.ACTION_NAMES[correct_action] if correct_action >= 0 else "N/A"

        print(f"Step {step_count + 1}: Action = {action_name} | Optimal = {correct_name}")

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        step_count += 1

        print(f"  Reward: {reward:.2f} | Cumulative: {episode_reward:.2f}\n")

    print("="*60)
    print(f"DEMO COMPLETE")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Episode Length: {step_count}")

    if 'episode_stats' in info:
        stats = info['episode_stats']
        total = stats.get('correct_triages', 0) + stats.get('incorrect_triages', 0)
        if total > 0:
            accuracy = 100.0 * stats.get('correct_triages', 0) / total
            print(f"Triage Accuracy: {accuracy:.1f}%")

    print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run best-performing RL model for Clinic Triage"
    )

    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=['dqn', 'ppo', 'a2c', 'reinforce'],
        help='Type of model to load'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to saved model file'
    )

    parser.add_argument(
        '--num_eval_episodes',
        type=int,
        default=10,
        help='Number of episodes for evaluation'
    )

    parser.add_argument(
        '--run_demo',
        action='store_true',
        help='Run a single demo episode with verbose output'
    )

    parser.add_argument(
        '--render_mode',
        type=str,
        default='ansi',
        choices=['human', 'rgb_array', 'ansi'],
        help='Rendering mode for demo'
    )

    parser.add_argument(
        '--save_video',
        type=str,
        default=None,
        help='Path to save demo video (requires pygame)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    # Create environment
    print(f"Creating environment with seed {args.seed}...")
    env = ClinicEnv(seed=args.seed, max_steps=500)

    # Load model
    print(f"Loading {args.model_type.upper()} model from {args.model_path}...")

    if args.model_type == 'reinforce':
        model = load_reinforce_model(args.model_path, env)
        eval_env = env
        model_framework = 'reinforce'
    else:
        model, eval_env = load_sb3_model(args.model_type, args.model_path, env)
        model_framework = 'sb3'

    print("Model loaded successfully!\n")

    # Run demo if requested
    if args.run_demo:
        run_demo(model, env, model_type=model_framework, render_mode=args.render_mode)

    # Evaluate model
    print(f"Evaluating model over {args.num_eval_episodes} episodes...")
    eval_results = evaluate_model(
        model, eval_env,
        num_episodes=args.num_eval_episodes,
        model_type=model_framework
    )

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Mean Reward:          {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Mean Episode Length:  {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
    print(f"Triage Accuracy:      {eval_results['mean_triage_accuracy']:.1f}% ± {eval_results['std_triage_accuracy']:.1f}%")
    print("="*60 + "\n")

    # Save video if requested
    if args.save_video:
        print(f"Generating demo video: {args.save_video}")

        def policy_func(obs):
            if model_framework == 'sb3':
                action, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = model.policy.get_action(obs, deterministic=True)
            return action

        render_episode_to_video(
            env, policy_func,
            args.save_video,
            num_episodes=1,
            max_steps=500
        )

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
