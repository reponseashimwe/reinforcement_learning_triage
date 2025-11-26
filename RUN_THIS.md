# Execution Guide

## Setup

```bash
pip install -r requirements.txt
```

---

## Generate Report Materials

### Step 1: Extract Results from Notebooks

```bash
python extract_notebook_results.py
```

Creates JSON files from notebook training data in `models/*/training_results.json`.

### Step 2: Generate Plots

```bash
python generate_plots.py
```

Creates 3 comparison plots in `evaluation/plots/`:
- `figure1_algorithm_comparison.png` - Mean Reward and Triage Accuracy
- `figure2_combined_comparison.png` - Normalized Performance Scatter
- `figure3_convergence_comparison.png` - Learning Curves

### Step 3: Generate Demo Videos

```bash
python generate_videos.py
```

Creates two 30-second demo videos in `demos/`:
- `random_demo.mp4` - Random agent baseline
- `ppo_demo.mp4` - Trained PPO agent

**Note:** Requires `ppo_short_horizon_sweep_vecnormalize.pkl` for proper model performance.

---

## Run Best Model (PPO)

### Option 1: Demo with Verbose Terminal Output

```bash
python main.py \
  --model_type ppo \
  --model_path models/ppo/ppo_short_horizon_sweep.zip \
  --run_demo \
  --render_mode ansi \
  --num_eval_episodes 10
```

Shows step-by-step actions and rewards in the terminal.

### Option 2: Generate Video with GUI Visualization

```bash
python main.py \
  --model_type ppo \
  --model_path models/ppo/ppo_short_horizon_sweep.zip \
  --save_video demos/best_model_demo.mp4 \
  --num_eval_episodes 10
```

Creates a video showing the GUI visualization with performance metrics.

### Option 3: Both Demo + Video

```bash
python main.py \
  --model_type ppo \
  --model_path models/ppo/ppo_short_horizon_sweep.zip \
  --run_demo \
  --render_mode ansi \
  --save_video demos/best_model_demo.mp4 \
  --num_eval_episodes 10
```

Shows terminal output AND generates a video.

---

## Expected Results

| Algorithm | Mean Reward | Accuracy | Training Time |
|-----------|-------------|----------|---------------|
| **PPO** | **683.91** | **99.6%** | 281.8s |
| REINFORCE | 697.20 | 40.74% | 450.2s |
| DQN | 649.90 | 32.31% | 132.7s |
| A2C | 501.03 | 94.99% | 239.8s |

**Winner:** PPO achieves the best balance of high accuracy and reward.

---

## Troubleshooting

### Video shows low accuracy (~20%)

**Problem:** VecNormalize stats file is missing or not loaded properly.

**Solution:** 
1. Check that `models/ppo/ppo_short_horizon_sweep_vecnormalize.pkl` exists
2. If missing, retrain the model using `training/ppo_training.py` (it now saves VecNormalize stats)
3. The updated scripts automatically load VecNormalize when available

### "Module not found" errors

**Solution:** 
```bash
pip install -r requirements.txt
```

### Plots are empty or missing data

**Solution:**
```bash
python extract_notebook_results.py  # Extract data from notebooks first
python generate_plots.py            # Then generate plots
```

---

## File Structure

```
reinforcement_learning/
├── main.py                          # Run best model with GUI/video
├── generate_videos.py               # Generate demo videos
├── generate_plots.py                # Generate comparison plots
├── extract_notebook_results.py     # Extract results from notebooks
├── REPORT.md                        # Final report (5 pages)
├── README.md                        # Project documentation
├── requirements.txt                 # Dependencies
├── environment/
│   ├── custom_env.py               # ClinicEnv implementation
│   └── rendering.py                # Pygame visualization
├── training/
│   ├── dqn_training.py
│   ├── ppo_training.py             # Now saves VecNormalize stats!
│   ├── a2c_training.py
│   └── reinforce_training.py
├── models/
│   ├── ppo/
│   │   ├── ppo_short_horizon_sweep.zip
│   │   └── ppo_short_horizon_sweep_vecnormalize.pkl  # Critical!
│   ├── dqn/
│   ├── a2c/
│   └── reinforce/
├── evaluation/
│   ├── aggregate_results.py
│   └── plots/
│       ├── figure1_algorithm_comparison.png
│       ├── figure2_combined_comparison.png
│       └── figure3_convergence_comparison.png
└── demos/
    ├── random_demo.mp4
    ├── ppo_demo.mp4
    └── best_model_demo.mp4
```

---

## Next Steps

1. ✅ Extract results from notebooks
2. ✅ Generate plots for report
3. ✅ Generate demo videos
4. ✅ Run best model with GUI + terminal output
5. ⏳ Insert plots into REPORT.md
6. ⏳ Record 3-minute presentation video
7. ⏳ Convert REPORT.md to PDF
8. ⏳ Create GitHub repository
9. ⏳ Submit to Canvas
