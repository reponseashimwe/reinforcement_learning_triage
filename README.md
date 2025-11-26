# Dermatology Clinic Triage - Reinforcement Learning Summative

**Student:** [Your Name]  
**Course:** Reinforcement Learning Summative Assignment  
**Date:** November 2025

## 📋 Project Overview

This project implements and compares **4 reinforcement learning algorithms** (DQN, PPO, A2C, REINFORCE) to optimize patient triage in a simulated dermatology clinic environment. The agent learns to correctly assign patients to appropriate care levels while managing resources and minimizing wait times.

### Problem Statement
A busy dermatology clinic needs an intelligent triage system that can:
- Correctly categorize patient severity (Mild → Moderate → Severe → Critical)
- Assign patients to appropriate care (Remote Advice, Nurse, Doctor, Escalate)
- Manage exam room resources efficiently
- Minimize patient wait times
- Maximize correct diagnoses

---

## 🏗️ Project Structure

```
reinforcement_learning/
├── environment/
│   ├── custom_env.py         # Custom Gymnasium environment (ClinicEnv)
│   ├── rendering.py           # Visualization components
│   └── __init__.py
│
├── training/
│   ├── dqn_training.py        # DQN training script
│   ├── ppo_training.py        # PPO training script
│   ├── a2c_training.py        # A2C training script
│   └── reinforce_training.py  # REINFORCE training script
│
├── models/
│   ├── dqn/                   # Saved DQN models + results
│   ├── ppo/                   # Saved PPO models + results
│   ├── a2c/                   # Saved A2C models + results
│   └── reinforce/             # Saved REINFORCE models + results
│
├── evaluation/
│   ├── aggregate_results.py   # Compare all algorithms
│   └── plots/                 # Generated comparison plots
│
├── notebooks/
│   ├── 01_dqn_training.ipynb          # DQN hyperparameter sweep
│   ├── 02_ppo_training.ipynb          # PPO hyperparameter sweep
│   ├── 03_a2c_training.ipynb          # A2C hyperparameter sweep
│   ├── 04_reinforce_training.ipynb    # REINFORCE hyperparameter sweep
│   └── 05_best_model_final.ipynb      # Final extended training
│
├── demos/                     # Demo videos
├── main.py                    # Entry point for running trained models
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🎮 Environment Description

### ClinicEnv (Custom Gymnasium Environment)

**Observation Space (15 dimensions):**
- `[0]` age_norm: Patient age (normalized 0-1)
- `[1]` duration_norm: Symptom duration (normalized 0-1)
- `[2]` fever_flag: Binary fever indicator
- `[3]` infection_flag: Binary infection indicator
- `[4-11]` symptom_embed: 8D symptom severity embedding
- `[12]` room_avail: Exam room availability
- `[13]` queue_len_norm: Waiting queue length (normalized)
- `[14]` time_norm: Episode progress (normalized)

**Action Space (8 discrete actions):**
- `0`: Send to Doctor (for severe cases)
- `1`: Send to Nurse (for moderate cases)
- `2`: Remote Advice (for mild cases via telemedicine)
- `3`: Escalate Priority (for critical cases requiring urgent attention)
- `4`: Defer Patient (postpone to end of queue)
- `5`: Idle (no action)
- `6`: Open Room (increase resource capacity)
- `7`: Close Room (reduce resource costs)

**Reward Structure:**
- ✅ **Correct Triage Rewards:**
  - Mild → Remote: +1.0
  - Moderate → Nurse: +1.25
  - Severe → Doctor: +2.0
  - Critical → Escalate (fast): +3.0
  
- ❌ **Penalties:**
  - Incorrect triage: -2.0 × severity_multiplier
  - Wait time: -0.01 × queue_length per step
  - Resource cost: -0.02 × num_open_rooms per step

**Episode Termination:**
- After 500 timesteps (truncated)

---

## 🧪 Algorithms Implemented

### 1. **DQN (Deep Q-Network)** - Value-Based
- **Best Config:** `dqn_fast_target_update`
- **Mean Reward:** 649.90
- **Triage Accuracy:** 32.3%
- **Key Hyperparameters:**
  - Learning rate: 0.0003
  - Gamma: 0.99
  - Buffer size: 50,000
  - Target update interval: 500

### 2. **PPO (Proximal Policy Optimization)** - Policy Gradient ⭐
- **Best Config:** `ppo_10_mid_exploration` 🏆
- **Mean Reward:** 506.39
- **Triage Accuracy:** 94.80% ✨
- **Key Hyperparameters:**
  - Learning rate: 0.0002
  - N-steps: 128
  - Batch size: 64
  - Entropy coef: 0.005

### 3. **A2C (Advantage Actor-Critic)** - Policy Gradient
- **Best Config:** `a2c_high_entropy`
- **Mean Reward:** 521.17
- **Triage Accuracy:** 33.1%
- **Key Hyperparameters:**
  - Learning rate: 0.0007
  - N-steps: 5
  - Entropy coef: 0.05

### 4. **REINFORCE (Vanilla Policy Gradient)** - Policy Gradient
- **Best Config:** `reinforce_no_baseline`
- **Mean Reward:** 697.20
- **Triage Accuracy:** 40.7%
- **Key Hyperparameters:**
  - Learning rate: 0.0001
  - No baseline (higher variance)
  - Hidden layers: [256, 256]

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for faster training)

### Install Dependencies
```bash
# Clone repository
git clone https://github.com/[your-username]/reinforcement_learning.git
cd reinforcement_learning

# Install requirements
pip install -r requirements.txt
```

### Requirements
```txt
numpy==1.26.4
torch>=2.0.0
gymnasium>=0.29.0
stable-baselines3>=2.0.0
sb3-contrib>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
imageio>=2.31.0
tqdm>=4.65.0
```

---

## 🎯 Training

### Option 1: Train Individual Algorithms

Train each algorithm with the best configuration:

```bash
# DQN
python training/dqn_training.py --timesteps 200000 --seeds 5

# PPO (Best Overall)
python training/ppo_training.py --timesteps 200000 --seeds 5

# A2C
python training/a2c_training.py --timesteps 200000 --seeds 5

# REINFORCE
python training/reinforce_training.py --episodes 4000 --seeds 5
```

**Arguments:**
- `--timesteps`: Total training timesteps (default: 200000)
- `--episodes`: Training episodes for REINFORCE (default: 4000)
- `--seeds`: Number of random seeds (default: 5)
- `--output_dir`: Directory to save models
- `--verbose`: Verbosity level (0=quiet, 1=info)

### Option 2: Google Colab Notebooks

Open the notebooks in Google Colab for interactive training with GPU acceleration:

1. `01_dqn_training.ipynb` - DQN hyperparameter sweep (10 configs)
2. `02_ppo_training.ipynb` - PPO hyperparameter sweep (10 configs)
3. `03_a2c_training.ipynb` - A2C hyperparameter sweep (10 configs)
4. `04_reinforce_training.ipynb` - REINFORCE sweep (10 configs)
5. `05_best_model_final.ipynb` - Extended training with best algorithm

**Total Experiments:** 40 configurations tested across 4 algorithms

---

## 📊 Evaluation & Comparison

### Generate Comparison Plots

```bash
python evaluation/aggregate_results.py
```

This generates:
- `evaluation/algorithm_comparison.csv` - Performance table
- `evaluation/plots/algorithm_comparison.png` - Bar chart comparison
- `evaluation/plots/learning_curves.png` - Training progress
- `evaluation/plots/combined_comparison.png` - Normalized comparison

### Results Summary

| Algorithm | Mean Reward | Triage Accuracy | Training Time |
|-----------|-------------|-----------------|---------------|
| **PPO** 🥇 | 506.39 | **94.80%** | ~4 min |
| A2C | 521.17 | 33.1% | ~3 min |
| DQN | 649.90 | 32.3% | ~2 min |
| REINFORCE | 697.20 | 40.7% | ~5 min |

**Winner: PPO** achieves exceptional triage accuracy (94.80%) making it the most reliable for clinical deployment.

---

## 🎬 Running Trained Models

### Demo with Best Model

```bash
# Run PPO best model (interactive)
python main.py \
  --model_type ppo \
  --model_path models/ppo/ppo_seed42.zip \
  --run_demo \
  --num_eval_episodes 50
```

### Generate Demo Video

```bash
python main.py \
  --model_type ppo \
  --model_path models/ppo/ppo_seed42.zip \
  --save_video demos/ppo_best_demo.mp4
```

### Command-Line Arguments

```bash
python main.py --help

Options:
  --model_type        Algorithm: dqn, ppo, a2c, reinforce
  --model_path        Path to trained model
  --num_eval_episodes Number of evaluation episodes
  --run_demo          Run verbose demo episode
  --render_mode       Rendering: ansi, rgb_array, human
  --save_video        Save demo video path
  --seed              Random seed
```

---

## 📈 Key Findings

### Performance Analysis

1. **PPO dominates in accuracy (94.80%)** - Best for real-world deployment
   - Stable learning with clipped surrogate objective
   - Excellent exploration-exploitation balance
   - Low variance with moderate batch sizes

2. **REINFORCE achieves highest reward (697.20)** but lower accuracy (40.7%)
   - High variance without baseline
   - Explores more aggressive strategies
   - Less reliable for safety-critical applications

3. **DQN shows fast convergence** but plateaus early
   - Off-policy learning efficient
   - Experience replay helps stability
   - May need deeper exploration strategies

4. **A2C provides good balance** between speed and performance
   - Synchronous updates faster than PPO
   - Lower accuracy than PPO but acceptable
   - Good choice for resource-constrained scenarios

### Hyperparameter Insights

**Critical PPO settings:**
- **N-steps: 128** (not too long, not too short)
- **Batch size: 64** (good variance/computation trade-off)
- **Entropy coef: 0.005** (slight exploration bonus)
- **Clip range: 0.15** (moderate policy updates)

**Environment tuning:**
- Severity-weighted penalties crucial for proper incentives
- Reward clipping [-6, 6] prevents extreme values
- Resource costs prevent over-allocation

---

## 🎥 Demo Video

**Video Link:** [YouTube/Google Drive Link]

**Video Contents:**
- Problem statement & environment overview
- Agent behavior demonstration
- Reward structure explanation
- GUI + terminal output
- Performance metrics discussion
- Comparison of all 4 algorithms

**Duration:** ~3 minutes

---

## 📄 Report

Full report available: `RL_Summative_Report.pdf`

**Sections:**
1. Project Overview
2. Environment Description
3. System Architecture
4. Implementation Details (DQN, PPO, A2C, REINFORCE)
5. Hyperparameter Sweeps (10 configs × 4 algorithms)
6. Results & Discussion
7. Visualizations
8. Conclusion & Future Work

---

## 🛠️ Development Notes

### Environment Design Decisions

1. **15D Observation Space** - Rich patient features without being overwhelming
2. **8 Discrete Actions** - Covers all triage scenarios + resource management
3. **Severity-Weighted Rewards** - Penalize mistakes on critical patients more
4. **Episode Length 500** - Long enough for strategic learning

### Training Strategies

- **Multiple Seeds (5):** Ensures robustness, reduces variance
- **Eval Frequency (10K steps):** Frequent checkpoints for monitoring
- **Reward Clipping:** Prevents exploding gradients
- **Advantage Normalization:** Stabilizes policy gradient methods

---

## 🔮 Future Improvements

1. **Multi-Agent System:** Multiple triage stations competing/cooperating
2. **Continuous Actions:** Fine-grained resource allocation
3. **Real Patient Data:** Train on actual clinical records
4. **Hierarchical RL:** High-level strategy + low-level execution
5. **Transfer Learning:** Pre-train on similar medical triage tasks

---

## 📚 References

1. Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
2. Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. *arXiv*.
3. Mnih, V. et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. *ICML*.
4. Williams, R. J. (1992). Simple statistical gradient-following algorithms. *Machine Learning*.
5. Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/

---

## 📧 Contact

**Student:** [Your Name]  
**Email:** [your.email@example.com]  
**GitHub:** https://github.com/[your-username]/reinforcement_learning

---

## 📜 License

This project is submitted as part of academic coursework. All rights reserved.

---

**Last Updated:** November 2025  
**Status:** ✅ Complete - Ready for Submission
