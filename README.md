# Dermatology Triage Clinic - Reinforcement Learning

A comprehensive reinforcement learning project implementing an intelligent medical triage system for a dermatology clinic. This project compares four state-of-the-art RL algorithms (DQN, PPO, A2C, REINFORCE) on a custom healthcare environment with multi-objective optimization.

## Problem Statement

Healthcare facilities, particularly dermatology clinics, face significant challenges in efficiently triaging patients while maintaining high diagnostic accuracy. Manual triage processes are time-consuming, subject to human error, and struggle to balance multiple competing objectives simultaneously. The core problems include:

1. **Diagnostic Accuracy**: Correctly classifying patient severity (Mild, Moderate, Severe, Critical) based on partial information
2. **Resource Management**: Dynamically allocating limited exam rooms to optimize patient flow
3. **Wait Time Optimization**: Minimizing patient wait times, especially for critical cases
4. **Multi-Objective Optimization**: Balancing accuracy, efficiency, and resource costs in real-time

This project develops an intelligent RL-based triage system that learns optimal policies to address these challenges, demonstrating that automated systems can achieve near-perfect accuracy (99.6% with PPO) while efficiently managing clinic resources.

## Project Overview

This project addresses the challenge of automated medical triage in a resource-constrained dermatology clinic. An RL agent learns to:

-   Triage patients based on symptom severity (Mild, Moderate, Severe, Critical)
-   Manage resources by dynamically opening/closing exam rooms
-   Optimize wait times while maintaining diagnostic accuracy
-   Balance competing objectives (accuracy, efficiency, cost)

### Key Features

-   **Custom Gymnasium Environment**: 15-dimensional observation space, 8 discrete actions
-   **Partial Observability**: Agent must infer severity from noisy symptoms
-   **Multi-Objective Rewards**: Balances triage accuracy, wait time, and resource costs
-   **Extensive Hyperparameter Tuning**: 40+ configurations across 4 algorithms
-   **Professional Visualization**: Pygame-based rendering with real-time metrics

## Results Summary

| Algorithm     | Mean Reward | Accuracy  | Convergence    | Best For                     |
| ------------- | ----------- | --------- | -------------- | ---------------------------- |
| **PPO**       | 683.91      | **99.6%** | ~150 episodes  | Healthcare (high accuracy)   |
| **REINFORCE** | **697.20**  | 40.74%    | ~300+ episodes | Exploration-heavy tasks      |
| **DQN**       | 649.90      | 32.31%    | ~200 episodes  | Multi-objective optimization |
| **A2C**       | 501.03      | 94.99%    | ~180 episodes  | Fast adaptation              |

**Winner**: PPO achieves the best balance with near-perfect triage accuracy (99.6%) and high reward.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/reponseashimwe/reinforcement_learning_triage.git
cd reinforcement_learning_triage

# Install dependencies
pip install -r requirements.txt
```

### Run Best Model (PPO)

The `main.py` script has sensible defaults and can be run without any arguments:

```bash
# Run with defaults (PPO model, 10 evaluation episodes)
python main.py

# Run simulation with GUI and Terminal verbose output (RECOMMENDED FOR DEMO)
python main.py --run_demo --save_video demos/simulation.mp4 --num_eval_episodes 3

# Run with verbose terminal output only
python main.py --run_demo --render_mode ansi

# Generate video with GUI visualization only
python main.py --save_video demos/my_demo.mp4
```

**For Assignment Video Recording:**
The recommended command shows both GUI (in the video) and terminal verbose outputs simultaneously:

```bash
python main.py --run_demo --save_video demos/simulation.mp4 --num_eval_episodes 3
```

This will:

1. Display step-by-step actions and rewards in the terminal (verbose output)
2. Generate a video file showing the GUI visualization with performance metrics
3. Evaluate the model and print final statistics

**Available Options:**

-   `--model_type`: Model to load (default: `ppo`)
-   `--model_path`: Path to model file (default: `models/ppo/ppo_short_horizon_sweep.zip`)
-   `--run_demo`: Show step-by-step terminal output
-   `--render_mode`: Rendering mode (`ansi`, `human`, `rgb_array`)
-   `--save_video`: Path to save video
-   `--num_eval_episodes`: Number of episodes to evaluate (default: 10)
-   `--seed`: Random seed (default: 42)

### Generate Report Materials

```bash
# Step 1: Extract results from notebooks
python extract_notebook_results.py

# Step 2: Generate comparison plots
python generate_plots.py

# Step 3: Generate demo videos (30 seconds each)
python generate_videos.py
```

**Generated Files:**

-   **Plots**: `evaluation/plots/figure1_*.png`, `figure2_*.png`, `figure3_*.png`
-   **Videos**: `demos/random_demo.mp4` (random agent), `demos/ppo_demo.mp4` (trained agent)

## Project Structure

```
reinforcement_learning/
├── environment/              # Custom Gym environment
│   ├── custom_env.py        # ClinicEnv implementation
│   └── rendering.py         # Pygame visualization
├── training/                # Training scripts
│   ├── dqn_training.py      # Deep Q-Network
│   ├── ppo_training.py      # Proximal Policy Optimization
│   ├── a2c_training.py      # Advantage Actor-Critic
│   └── reinforce_training.py # Policy Gradient
├── models/                  # Trained models (48 total)
│   ├── dqn/                 # 10 DQN configurations
│   ├── ppo/                 # 10 PPO configurations
│   ├── a2c/                 # 10 A2C configurations
│   └── reinforce/           # 18 REINFORCE configurations
├── notebooks/               # Google Colab training notebooks
│   ├── 01_dqn_training.ipynb
│   ├── 02_ppo_training.ipynb
│   ├── 03_a2c_training.ipynb
│   └── 04_reinforce_training.ipynb
├── evaluation/              # Results analysis
│   ├── aggregate_results.py
│   └── plots/              # Generated figures
├── demos/                   # Generated videos
│   ├── random_demo.mp4     # Random agent baseline
│   └── ppo_demo.mp4        # Best trained agent
├── configs/                 # Hyperparameter configurations
├── main.py                  # Entry point for evaluation
├── requirements.txt         # Python dependencies
├── REPORT.md               # Complete project report
└── README.md               # This file
```

## Environment Details

### Observation Space (15 dimensions)

```
[0]      age_norm           [0.0, 1.0]   Patient age (normalized)
[1]      duration_norm      [0.0, 1.0]   Symptom duration
[2]      fever_flag         {0.0, 1.0}   Presence of fever
[3]      infection_flag     {0.0, 1.0}   Signs of infection
[4-11]   symptom_embed      [0.0, 1.0]   8D clinical symptom vector
[12]     room_avail         {0.0, 1.0}   Room availability
[13]     queue_len_norm     [0.0, 1.0]   Queue length
[14]     time_norm          [0.0, 1.0]   Episode progress
```

### Action Space (8 discrete actions)

| Action | Name                | Description                            |
| ------ | ------------------- | -------------------------------------- |
| 0      | `send_doctor`       | Route to dermatologist (Severe cases)  |
| 1      | `send_nurse`        | Route to nurse practitioner (Moderate) |
| 2      | `remote_advice`     | Telemedicine consultation (Mild)       |
| 3      | `escalate_priority` | Mark urgent + doctor (Critical)        |
| 4      | `defer_patient`     | Postpone to end of queue               |
| 5      | `idle`              | Wait/observe                           |
| 6      | `open_room`         | Add exam room capacity                 |
| 7      | `close_room`        | Reduce room capacity                   |

### Reward Structure

```
R(s, a) = R_triage + R_wait + R_resource

R_triage:   +1.0 to +3.0 for correct triage, -1.5 for incorrect
R_wait:     -0.01 × queue_length (per step)
R_resource: -0.05 × num_open_rooms (per step)
```

## Training Details

### Algorithms Implemented

1. **Deep Q-Network (DQN)**

    - Experience replay buffer (50K transitions)
    - Target network with soft updates
    - Epsilon-greedy exploration
    - Best config: `dqn_fast_target_update`

2. **Proximal Policy Optimization (PPO)**

    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Observation normalization (VecNormalize)
    - Best config: `ppo_short_horizon`

3. **Advantage Actor-Critic (A2C)**

    - Synchronous updates
    - Entropy regularization
    - Longer rollouts for credit assignment
    - Best config: `a2c_longer_rollout`

4. **REINFORCE (Policy Gradient)**
    - Monte Carlo returns
    - Optional baseline (value network)
    - High variance but simple
    - Best config: `reinforce_no_baseline`

### Hyperparameter Tuning

Each algorithm was trained with 10 different hyperparameter configurations:

-   **Total configurations**: 40
-   **Training time per config**: ~4-5 hours on Google Colab (GPU)
-   **Total compute time**: ~180 GPU hours
-   **Best configs selected** based on triage accuracy and reward

## Key Findings

### 1. PPO Dominates for Healthcare Applications

-   **99.6% triage accuracy** minimizes misdiagnosis risk
-   Stable training with observation normalization
-   Fast convergence (~150 episodes)
-   Recommended for production deployment

### 2. REINFORCE Shows Surprising Performance

-   **Highest raw reward** (697.20) without baseline
-   High variance but aggressive exploration
-   Sensitive to hyperparameters
-   Interesting for research but unstable

### 3. A2C Requires Longer Rollouts

-   **Dramatic performance difference**: 501.03 (20 steps) vs -41.94 (5 steps)
-   Short rollouts cause instability in long episodes
-   Good accuracy (94.99%) when properly configured

### 4. DQN Excels at Multi-Objective Optimization

-   Balances triage, wait time, and resource costs
-   Lower accuracy but higher overall reward
-   Experience replay helps with rare events

## Visualizations

### Generated Plots

1. **Figure 1**: Algorithm Performance Comparison (Reward + Accuracy)
2. **Figure 2**: Normalized Performance Comparison
3. **Figure 3**: Convergence Speed Analysis

### Demo Videos

-   **Random Agent** (30s): Baseline performance (~12.5% accuracy)
-   **Trained PPO Agent** (30s): Near-perfect triage (99.6% accuracy)

## Development

### Training a New Model

```bash
# Train PPO with custom config
python training/ppo_training.py --timesteps 200000 --seeds 5

# Train DQN
python training/dqn_training.py --timesteps 200000 --seeds 5
```

### Evaluation

```bash
# Evaluate model
python main.py --model_type ppo \
  --model_path models/ppo/ppo_short_horizon_sweep.zip \
  --num_eval_episodes 100

# Run with visualization
python main.py --model_type ppo \
  --model_path models/ppo/ppo_short_horizon_sweep.zip \
  --run_demo --render_mode human
```

### Generate Video

```bash
# Save episode as video
python main.py --model_type ppo \
  --model_path models/ppo/ppo_short_horizon_sweep.zip \
  --save_video demos/my_demo.mp4
```

## Documentation

-   **task.md**: Original assignment requirements
-   **usecase.md**: Detailed environment documentation

## Technologies Used

-   **Python 3.10+**
-   **Gymnasium**: Environment framework
-   **Stable Baselines3**: DQN, PPO, A2C implementations
-   **PyTorch**: Neural networks and REINFORCE implementation
-   **Pygame**: Visualization and rendering
-   **Matplotlib/Seaborn**: Plotting and analysis
-   **ImageIO**: Video generation

## Performance Metrics

### Triage Accuracy by Severity

| Algorithm | Mild  | Moderate | Severe | Critical | Overall |
| --------- | ----- | -------- | ------ | -------- | ------- |
| PPO       | 99.8% | 99.7%    | 99.5%  | 98.9%    | 99.6%   |
| A2C       | 95.2% | 95.1%    | 94.8%  | 93.5%    | 94.99%  |
| REINFORCE | 42.1% | 41.3%    | 39.8%  | 38.2%    | 40.74%  |
| DQN       | 33.5% | 32.9%    | 31.8%  | 30.1%    | 32.31%  |

### Convergence Speed

| Algorithm | Episodes to Converge | Timesteps | Training Time |
| --------- | -------------------- | --------- | ------------- |
| PPO       | 150 ± 20             | 75,000    | ~45 min       |
| A2C       | 180 ± 30             | 90,000    | ~50 min       |
| DQN       | 200 ± 25             | 100,000   | ~60 min       |
| REINFORCE | 300+ ± 50            | 150,000+  | ~90 min       |

## Future Improvements

1. **Environment Enhancements**

    - Time-of-day arrival patterns
    - Patient satisfaction metrics
    - Diagnostic test uncertainty

2. **Algorithm Improvements**

    - Prioritized Experience Replay for DQN
    - Curiosity-driven exploration
    - Multi-task learning (predict + act)

3. **Deployment Considerations**
    - Safety constraints on misdiagnosis rates
    - Human-in-the-loop approval
    - Continual learning from new data

## License

This project was developed as part of an academic assignment. All rights reserved.

## Author

Reponse Ashimwe  
Institution: African Leadership University  
Course: Reinforcement Learning  
Date: November 2025

## Acknowledgments

-   Assignment designed by ALU Faculty
-   Stable Baselines3 library by DLR-RM
-   Gymnasium framework by Farama Foundation
-   Google Colab for GPU compute resources
