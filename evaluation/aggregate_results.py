"""
Aggregate Results & Comparison Script
Compare all 4 RL algorithms and generate visualization plots for the report.

Usage:
    python evaluation/aggregate_results.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_training_results(algorithm: str, models_dir: str = "models") -> Dict[str, Any]:
    """
    Load training results from JSON file.
    
    Args:
        algorithm: Algorithm name ('dqn', 'ppo', 'a2c', 'reinforce')
        models_dir: Base models directory
        
    Returns:
        Training results dictionary
    """
    results_path = os.path.join(models_dir, algorithm, "training_results.json")
    
    if not os.path.exists(results_path):
        print(f"⚠️  Warning: Results not found for {algorithm.upper()}: {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def create_comparison_table(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create comparison table of all algorithms.
    
    Args:
        results_dict: Dictionary mapping algorithm names to results
        
    Returns:
        Comparison DataFrame
    """
    data = []
    
    for algo, results in results_dict.items():
        if results is None:
            continue
            
        aggregate = results.get('aggregate', {})
        config = results.get('config', {})
        
        row = {
            'Algorithm': algo.upper(),
            'Config ID': config.get('id', 'N/A'),
            'Mean Reward': aggregate.get('mean_final_reward', 0.0),
            'Std Reward': aggregate.get('std_final_reward', 0.0),
            'Mean Accuracy (%)': aggregate.get('mean_final_accuracy', 0.0),
            'Std Accuracy (%)': aggregate.get('std_final_accuracy', 0.0),
            'Learning Rate': config.get('learning_rate', 'N/A')
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values('Mean Reward', ascending=False).reset_index(drop=True)
    
    return df


def plot_algorithm_comparison(results_dict: Dict[str, Dict], output_dir: str = "evaluation/plots"):
    """
    Generate comprehensive comparison plots.
    
    Args:
        results_dict: Dictionary mapping algorithm names to results
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    algorithms = []
    mean_rewards = []
    std_rewards = []
    mean_accuracies = []
    std_accuracies = []
    
    for algo, results in results_dict.items():
        if results is None:
            continue
            
        aggregate = results.get('aggregate', {})
        algorithms.append(algo.upper())
        mean_rewards.append(aggregate.get('mean_final_reward', 0.0))
        std_rewards.append(aggregate.get('std_final_reward', 0.0))
        mean_accuracies.append(aggregate.get('mean_final_accuracy', 0.0))
        std_accuracies.append(aggregate.get('std_final_accuracy', 0.0))
    
    # ========== Plot 1: Reward Comparison ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Mean Reward with error bars
    colors = sns.color_palette("husl", len(algorithms))
    bars1 = ax1.bar(algorithms, mean_rewards, yerr=std_rewards, 
                    capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Episode Reward', fontsize=14, fontweight='bold')
    ax1.set_title('Algorithm Performance Comparison\n(Mean Reward over 5 Seeds)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', labelsize=12)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars1, mean_rewards, std_rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 5,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Subplot 2: Triage Accuracy
    bars2 = ax2.bar(algorithms, mean_accuracies, yerr=std_accuracies,
                    capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Triage Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Triage Accuracy Comparison\n(Mean Accuracy over 5 Seeds)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_ylim(0, 100)
    
    # Add value labels
    for bar, mean, std in zip(bars2, mean_accuracies, std_accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{mean:.1f}%±{std:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "algorithm_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'algorithm_comparison.png')}")
    plt.close()
    
    # ========== Plot 2: Learning Curves (if available) ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (algo, results) in enumerate(results_dict.items()):
        if results is None or idx >= 4:
            continue
            
        ax = axes[idx]
        
        # Extract learning curves from first seed
        seeds = results.get('seeds', [])
        if seeds and 'eval_history' in seeds[0]:
            eval_history = seeds[0]['eval_history']
            timesteps = [e['timestep'] for e in eval_history]
            rewards = [e['mean_reward'] for e in eval_history]
            
            ax.plot(timesteps, rewards, linewidth=2, marker='o', markersize=4, 
                   label=f'{algo.upper()} (Seed 42)', color=colors[idx])
            ax.set_xlabel('Training Timesteps', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
            ax.set_title(f'{algo.upper()} Learning Curve', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, linestyle='--')
            ax.legend(fontsize=10)
        elif seeds and 'training_history' in seeds[0]:
            # REINFORCE uses episodes instead of timesteps
            history = seeds[0]['training_history']
            episodes = history.get('eval_episodes', [])
            rewards = history.get('eval_rewards', [])
            
            ax.plot(episodes, rewards, linewidth=2, marker='o', markersize=4,
                   label=f'{algo.upper()} (Seed 42)', color=colors[idx])
            ax.set_xlabel('Training Episodes', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
            ax.set_title(f'{algo.upper()} Learning Curve', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, linestyle='--')
            ax.legend(fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No learning curve data', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{algo.upper()}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'learning_curves.png')}")
    plt.close()
    
    # ========== Plot 3: Combined Comparison ==========
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    # Normalize rewards to 0-100 scale for better comparison
    max_reward = max(mean_rewards) if mean_rewards else 1
    normalized_rewards = [r / max_reward * 100 for r in mean_rewards]
    
    bars1 = ax.bar(x - width/2, normalized_rewards, width, 
                   label='Normalized Reward (0-100)', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, mean_accuracies, width,
                   label='Triage Accuracy (%)', alpha=0.8, color='coral')
    
    ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Combined Performance Comparison\n(Normalized Reward vs Triage Accuracy)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'combined_comparison.png')}")
    plt.close()


def print_summary(comparison_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    if len(comparison_df) > 0:
        best_algo = comparison_df.iloc[0]['Algorithm']
        best_reward = comparison_df.iloc[0]['Mean Reward']
        best_accuracy = comparison_df.iloc[0]['Mean Accuracy (%)']
        
        print(f"\n🏆 BEST PERFORMING ALGORITHM: {best_algo}")
        print(f"   Mean Reward: {best_reward:.2f}")
        print(f"   Triage Accuracy: {best_accuracy:.1f}%")
        print("="*80 + "\n")


def main():
    """Main function."""
    print("\n" + "="*80)
    print("REINFORCEMENT LEARNING SUMMATIVE - RESULTS AGGREGATION")
    print("="*80 + "\n")
    
    # Load results for all algorithms
    algorithms = ['dqn', 'ppo', 'a2c', 'reinforce']
    results_dict = {}
    
    print("Loading training results...")
    for algo in algorithms:
        results = load_training_results(algo)
        results_dict[algo] = results
        if results:
            aggregate = results.get('aggregate', {})
            print(f"✓ {algo.upper()}: Reward={aggregate.get('mean_final_reward', 0):.2f}, "
                  f"Accuracy={aggregate.get('mean_final_accuracy', 0):.1f}%")
    
    # Create comparison table
    print("\nCreating comparison table...")
    comparison_df = create_comparison_table(results_dict)
    
    # Save comparison table
    os.makedirs("evaluation", exist_ok=True)
    comparison_df.to_csv("evaluation/algorithm_comparison.csv", index=False)
    print(f"✓ Saved: evaluation/algorithm_comparison.csv")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_algorithm_comparison(results_dict)
    
    # Print summary
    print_summary(comparison_df)
    
    print("✅ Results aggregation complete!\n")


if __name__ == "__main__":
    main()
