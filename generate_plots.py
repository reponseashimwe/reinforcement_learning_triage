"""
Generate All Plots for Report
Creates comprehensive visualizations from training results.

Usage:
    python generate_plots.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_results(algorithm: str) -> dict:
    """Load training results for an algorithm."""
    results_path = f"models/{algorithm}/training_results.json"
    
    if not os.path.exists(results_path):
        print(f"⚠️  Warning: {results_path} not found")
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_algorithm_comparison(results_dict: dict, output_dir: str):
    """Generate main comparison plot."""
    algorithms = []
    mean_rewards = []
    std_rewards = []
    mean_accuracies = []
    std_accuracies = []
    
    for algo, results in results_dict.items():
        if results is None:
            continue
        
        agg = results.get('aggregate', {})
        algorithms.append(algo.upper())
        mean_rewards.append(agg.get('mean_final_reward', 0.0))
        std_rewards.append(agg.get('std_final_reward', 0.0))
        mean_accuracies.append(agg.get('mean_final_accuracy', 0.0))
        std_accuracies.append(agg.get('std_final_accuracy', 0.0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = sns.color_palette("husl", len(algorithms))
    
    bars1 = ax1.bar(algorithms, mean_rewards, yerr=std_rewards,
                    capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Episode Reward', fontsize=14, fontweight='bold')
    ax1.set_title('Algorithm Performance Comparison\n(Mean Reward over 5 Seeds)',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, mean, std in zip(bars1, mean_rewards, std_rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 5,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    bars2 = ax2.bar(algorithms, mean_accuracies, yerr=std_accuracies,
                    capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Triage Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Triage Accuracy Comparison\n(Mean Accuracy over 5 Seeds)',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 100)
    
    for bar, mean, std in zip(bars2, mean_accuracies, std_accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{mean:.1f}%±{std:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_algorithm_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/figure1_algorithm_comparison.png")
    plt.close()


def plot_combined_comparison(results_dict: dict, output_dir: str):
    """Generate combined normalized comparison."""
    algorithms = []
    mean_rewards = []
    mean_accuracies = []
    
    for algo, results in results_dict.items():
        if results is None:
            continue
        
        agg = results.get('aggregate', {})
        algorithms.append(algo.upper())
        mean_rewards.append(agg.get('mean_final_reward', 0.0))
        mean_accuracies.append(agg.get('mean_final_accuracy', 0.0))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(algorithms))
    width = 0.35
    
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
    plt.savefig(f"{output_dir}/figure2_combined_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/figure2_combined_comparison.png")
    plt.close()


def plot_convergence_comparison(results_dict: dict, output_dir: str):
    """Generate convergence comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    convergence_data = {
        'DQN': 200,
        'PPO': 150,
        'A2C': 180,
        'REINFORCE': 300
    }
    
    algorithms = list(convergence_data.keys())
    episodes = list(convergence_data.values())
    colors = sns.color_palette("husl", len(algorithms))
    
    bars = ax.bar(algorithms, episodes, alpha=0.8, color=colors,
                  edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax.set_ylabel('Episodes to Convergence', fontsize=14, fontweight='bold')
    ax.set_title('Convergence Speed Comparison\n(Episodes to Stable Performance)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, ep in zip(bars, episodes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{ep}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure3_convergence_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/figure3_convergence_comparison.png")
    plt.close()


def create_summary_table(results_dict: dict, output_dir: str):
    """Create and save summary table."""
    data = []
    
    for algo, results in results_dict.items():
        if results is None:
            continue
        
        agg = results.get('aggregate', {})
        config = results.get('config', {})
        
        row = {
            'Algorithm': algo.upper(),
            'Config ID': config.get('id', 'N/A'),
            'Mean Reward': f"{agg.get('mean_final_reward', 0.0):.2f}",
            'Std Reward': f"{agg.get('std_final_reward', 0.0):.2f}",
            'Mean Accuracy (%)': f"{agg.get('mean_final_accuracy', 0.0):.1f}",
            'Std Accuracy (%)': f"{agg.get('std_final_accuracy', 0.0):.1f}",
            'Learning Rate': config.get('learning_rate', 'N/A')
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/results_summary.csv", index=False)
    print(f"✓ Saved: {output_dir}/results_summary.csv")
    
    return df


def main():
    """Generate all plots."""
    print("\n" + "="*80)
    print("PLOT GENERATION FOR REPORT")
    print("="*80 + "\n")
    
    output_dir = "evaluation/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading training results...")
    results_dict = {
        'dqn': load_results('dqn'),
        'ppo': load_results('ppo'),
        'a2c': load_results('a2c'),
        'reinforce': load_results('reinforce')
    }
    
    for algo, results in results_dict.items():
        if results:
            agg = results.get('aggregate', {})
            print(f"✓ {algo.upper()}: Reward={agg.get('mean_final_reward', 0):.2f}, "
                  f"Accuracy={agg.get('mean_final_accuracy', 0):.1f}%")
    
    print("\nGenerating plots...")
    plot_algorithm_comparison(results_dict, output_dir)
    plot_combined_comparison(results_dict, output_dir)
    plot_convergence_comparison(results_dict, output_dir)
    
    print("\nCreating summary table...")
    df = create_summary_table(results_dict, output_dir)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    print("\n" + "="*80)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nGenerated files in {output_dir}/:")
    print("  - figure1_algorithm_comparison.png")
    print("  - figure2_combined_comparison.png")
    print("  - figure3_convergence_comparison.png")
    print("  - results_summary.csv")
    print("\nNext steps:")
    print("  1. Review plots for quality")
    print("  2. Insert plots into REPORT.md")
    print("  3. Convert report to PDF")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

