"""
Generate Demo Videos for Report
Creates random agent and trained agent demonstration videos.

Usage:
    python generate_videos.py
"""

import os
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environment.custom_env import ClinicEnv
from environment.rendering import render_episode_to_video
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def generate_random_agent_video(output_path: str = "demos/random_demo.mp4", duration_seconds: int = 30):
    """Generate video of random agent."""
    print("\n" + "="*60)
    print("GENERATING RANDOM AGENT VIDEO")
    print("="*60)
    
    os.makedirs("demos", exist_ok=True)
    
    env = ClinicEnv(seed=42, max_steps=500)
    
    def random_policy(obs):
        return env.action_space.sample()
    
    print(f"Recording random agent for ~{duration_seconds} seconds...")
    render_episode_to_video(
        env,
        random_policy,
        output_path,
        num_episodes=1,
        max_steps=min(500, duration_seconds * 6)
    )
    
    env.close()
    print(f"✓ Random agent video saved: {output_path}")
    print("="*60 + "\n")


def generate_trained_agent_video(
    model_path: str = "models/ppo/ppo_short_horizon_sweep.zip",
    output_path: str = "demos/ppo_demo.mp4",
    duration_seconds: int = 30
):
    """Generate video of trained PPO agent."""
    print("\n" + "="*60)
    print("GENERATING TRAINED AGENT VIDEO (PPO)")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        print("Please train the model first or specify correct path.")
        return
    
    os.makedirs("demos", exist_ok=True)
    
    # Create base environment for rendering
    base_env = ClinicEnv(seed=42, max_steps=500)
    
    # Create vectorized environment with normalization
    # This is CRITICAL - the model was trained with VecNormalize!
    def make_env():
        return ClinicEnv(seed=42, max_steps=500)
    
    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Load VecNormalize statistics (CRITICAL!)
    vecnorm_path = model_path.replace('.zip', '_vecnormalize.pkl')
    if os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from {vecnorm_path}...")
        venv = VecNormalize.load(vecnorm_path, venv)
        print("✓ Loaded normalization statistics!")
    else:
        print(f"⚠️  WARNING: No VecNormalize stats found at {vecnorm_path}")
        print("   Model performance will be SEVERELY degraded!")
        print("   Please retrain the model with the updated training script.")
        return
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=venv)
    
    # Disable training mode - freeze normalization stats
    venv.training = False
    
    def trained_policy(obs):
        # CRITICAL: Normalize observation through VecNormalize before prediction
        # The model was trained on normalized observations!
        obs_normalized = venv.normalize_obs(obs)
        action, _ = model.predict(obs_normalized, deterministic=True)
        # Handle both scalar and array returns
        if isinstance(action, np.ndarray):
            return int(action[0]) if action.ndim > 0 else int(action)
        return int(action)
    
    print(f"Recording trained agent for ~{duration_seconds} seconds...")
    render_episode_to_video(
        base_env,
        trained_policy,
        output_path,
        num_episodes=1,
        max_steps=min(500, duration_seconds * 6)
    )
    
    base_env.close()
    venv.close()
    print(f"✓ Trained agent video saved: {output_path}")
    print("="*60 + "\n")


def main():
    """Generate all demo videos."""
    print("\n" + "="*80)
    print("VIDEO GENERATION FOR REPORT")
    print("="*80 + "\n")
    
    generate_random_agent_video(
        output_path="demos/random_demo.mp4",
        duration_seconds=30
    )
    
    generate_trained_agent_video(
        model_path="models/ppo/ppo_short_horizon_sweep.zip",
        output_path="demos/ppo_demo.mp4",
        duration_seconds=30
    )
    
    print("\n" + "="*80)
    print("✅ ALL VIDEOS GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  - demos/random_demo.mp4 (Random agent baseline)")
    print("  - demos/ppo_demo.mp4 (Best trained agent)")
    print("\nNext steps:")
    print("  1. Review videos to ensure quality")
    print("  2. Record full presentation video (≤3 min)")
    print("  3. Upload videos and add links to report")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

