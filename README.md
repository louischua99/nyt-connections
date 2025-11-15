# NYT Connections Fine-Tuning Research

This codebase investigates fine-tuning strategies for teaching language models to solve NYT Connections puzzles. We examine three research questions: (1) data augmentation effects, (2) reasoning format impact, and (3) curriculum learning benefits.

## Codebase Structure

```
.
├── data/                           # Experimental datasets & evaluation results
│   ├── experiment1/                # Data augmentation configs (baseline, permutation, synthetic, full)
│   ├── experiment2/                # Reasoning format configs (structured, unstructured, mixed, sequential)
│   ├── experiment3/                # Curriculum learning configs (no_warmup, warmup, staged)
│   ├── global_test.jsonl           # Universal test set (104 puzzles)
│   ├── global_validation.jsonl     # Universal validation set (276 puzzles)
│   ├── predictions_test/           # Model predictions on test set
│   ├── predictions_validation/     # Model predictions on validation set
│   ├── evaluation_results_*.csv    # Detailed evaluation metrics
│   ├── evaluation_summary_*.csv    # Aggregate performance by model
│   └── results/                    # Results from validation test for core, reasoning and judge metrics
│
├── data2/                          # Source data (pre-processing)
│   ├── puzzles/                    # Raw puzzle data (NYT + synthetic + pre-connections)
│   └── reasoning/                  # Generated reasoning traces (structured/unstructured)
│
├── scripts/                        # Training and evaluation scripts
│   ├── train_experiment.py         # Main training script (supports all 11 experiments)
│   ├── launch_all_training.sh      # Parallel launch script for all experiments
│   ├── prepare_experiments.py      # Create experiment datasets from source data
│   ├── evaluate_predictions.py     # Evaluate model predictions and compute metrics
│   ├── generate_predictions.py     # Generate predictions from trained models
│   ├── gen_*.py                    # Data generation scripts (reasoning, synthetic puzzles)
│   ├── process_*.py                # Format processing utilities
│   ├── eval_core_reasoning.py      # Evaluation for core metrics (f1, precision, recall) and reasoning quality (average steps and coverage ratio)
│   └── eval_judge.py               # Evaluation for Judge 
│
│
│
├── models/                         # Saved model checkpoints (exp1_*, exp2_*, exp3_*)
├── logs/                           # Training logs
└── main.tex                        # Research paper (LaTeX)
```

## Data Folders

### `data/` - Experimental Datasets
Contains prepared training/validation/test splits for all experiments, along with evaluation results. This is the **primary working directory** for training and evaluation.

**Key files:**
- `experiment{1,2,3}/` - Training datasets for each experiment type
- `global_test.jsonl` - 104 puzzles held out for final evaluation
- `global_validation.jsonl` - 276 puzzles for validation during training
- `predictions_{test,validation}/` - Model outputs in JSON format
- `evaluation_summary_*.csv` - Performance metrics (avg score, perfect puzzles, etc.)

### `data2/` - Source Data
Contains raw puzzle data and generated reasoning traces **before** experimental processing. Used as input to `prepare_experiments.py`.

**Key files:**
- `puzzles/connections.json` - 831 NYT puzzles
- `puzzles/connections_synthetic.json` - 200 synthetic puzzles
- `puzzles/preconn.json` - 800 pre-connection warm-up tasks
- `reasoning/` - Structured and unstructured reasoning traces

## Key Scripts

### Training
- **`train_experiment.py`** - Main training script
  - Supports all 11 experimental configurations
  - Usage: `python scripts/train_experiment.py --experiment exp1_baseline --epochs 8 --lr 2e-4`
  - Configurations: `exp1_{baseline,permutation,synthetic,full}`, `exp2_{structured,unstructured,mixed,sequential}`, `exp3_{no_warmup,warmup,staged}`

- **`launch_all_training.sh`** - Parallel training launcher
  - Runs all 11 experiments across multiple GPUs
  - Logs output to `logs/`
  - Usage: `bash scripts/launch_all_training.sh`

### Data Preparation
- **`prepare_experiments.py`** - Creates experiment datasets
  - Reads from `data2/`
  - Generates train/val splits in `data/experiment{1,2,3}/`
  - Ensures no data leakage between splits

### Evaluation
- **`evaluate_predictions.py`** - Compute metrics from predictions
  - Usage: `python scripts/evaluate_predictions.py --predictions-dir data/predictions_test --output data/evaluation_results_test.csv`
  - Outputs: detailed results CSV, summary CSV, extraction JSON

- **`generate_predictions.py`** - Generate model predictions
  - Usage: `python scripts/generate_predictions.py --model models/exp1_baseline --test-file data/global_test.jsonl`

### Data Generation (Optional - for reproducing source data)
- `gen_reason_struct.py` - Generate structured reasoning traces
- `gen_reason_unstruct.py` - Generate unstructured reasoning traces
- `gen_synthetic_conn.py` - Create synthetic Connections puzzles
- `gen_preconn.py` - Generate pre-connection warm-up tasks

## Quick Start

```bash
# 1. Prepare experimental datasets
python scripts/prepare_experiments.py

# 2. Train a single experiment
python scripts/train_experiment.py --experiment exp1_baseline --epochs 8 --lr 2e-4

# 3. Generate predictions
python scripts/generate_predictions.py --model models/exp1_baseline --test-file data/global_test.jsonl --output data/predictions_test/exp1_baseline.json

# 4. Evaluate predictions
python scripts/evaluate_predictions.py --predictions-dir data/predictions_test --output data/evaluation_results_test.csv
```

## Experiments

### Experiment 1: Data Augmentation
- **baseline**: NYT only, single permutation (673 puzzles)
- **permutation**: NYT only, 3 permutations (2,019 puzzles)
- **synthetic**: NYT + synthetic, single permutation (835 puzzles)
- **full**: NYT + synthetic, 3 permutations (2,505 puzzles)

### Experiment 2: Reasoning Format
- **structured**: Systematic reasoning with explicit steps (500 puzzles)
- **unstructured**: Free-form intuitive reasoning (500 puzzles)
- **mixed**: 50/50 blend of both formats (500 puzzles)
- **sequential**: Two-phase training (unstructured → structured)

### Experiment 3: Curriculum Learning
- **no_warmup**: Direct training on full dataset (2,505 puzzles)
- **warmup**: Pre-connections → full dataset (720 + 2,505)
- **staged**: Pre-connections → synthetic → NYT (720 + 486 + 2,019)

## Results

Best performing model: **exp3_warmup** (37.77% test accuracy)

See `data/evaluation_summary_test.csv` for complete results and `main.tex` for detailed analysis.

## Citation

```bibtex
@article{nyt_connections_2025,
  title={Fine-Tuning Language Models for NYT Connections: A Structured Reasoning Approach},
  author={Ting, Nicholas and Pham, Ngoc Minh and Lee, Weijiang and Chua, Yong Yaw Louis},
  year={2025}
}
```

## License

MIT
