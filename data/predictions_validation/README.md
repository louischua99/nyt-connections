# NYT Connections Predictions

This dataset contains model predictions from fine-tuned language models evaluated on NYT Connections puzzles. These predictions correspond to the experiments described in the [NYT Connections Experiments Dataset](https://huggingface.co/datasets/nickting/nyt-connections-experiments).

## Dataset Overview

This dataset contains prediction outputs from 11 experimental runs testing different training configurations on a shared global validation and test set:

### Evaluation Sets (Shared Across All Experiments)
- **Validation Set**: 276 entries from 92 unique original puzzles
  - 88 puzzle IDs with 3 permutations each (84 IDs × 3 = 252 entries)
  - 4 puzzle IDs containing 2 distinct puzzles with 3 permutations each (4 IDs × 6 = 24 entries)
  - Sources: 222 NYT + 54 Synthetic puzzles

- **Test Set**: 104 entries from 102 unique original puzzles
  - 100 puzzles with 1 permutation (100 entries)
  - 2 puzzles with 2 permutations (4 entries)
  - All puzzles are NYT puzzles

### Training Sets (Vary by Experiment)
Training data composition differs across experiments based on augmentation strategy, reasoning format, and curriculum approach. See experiment descriptions below for details.

- **Prediction Files**: 11 JSON files containing model outputs for each experiment

## File Structure

```
.
├── exp1_baseline.json          # Predictions from baseline (NYT only, perm=1)
├── exp1_full.json              # Predictions from full dataset (NYT + Synthetic, all perms)
├── exp1_permutation.json       # Predictions from permutation training (NYT only, all perms)
├── exp1_synthetic.json         # Predictions from synthetic augmentation (NYT + Synthetic, perm=1)
├── exp2_mixed.json             # Predictions from mixed format training (50/50 structured/unstructured)
├── exp2_sequential.json        # Predictions from sequential format training (unstructured→structured)
├── exp2_structured.json        # Predictions from structured-only training
├── exp2_unstructured.json      # Predictions from unstructured-only training
├── exp3_no_warmup.json         # Predictions without curriculum warmup
├── exp3_staged.json            # Predictions from staged curriculum (Pre-Connections→Synthetic→NYT)
└── exp3_warmup.json            # Predictions from warmup curriculum (Pre-Connections→Full)
```

## Experiments Description

### Experiment 1: Data Augmentation
Evaluates the impact of data augmentation strategies:
- **Baseline**: NYT puzzles only, single permutation
- **Permutation**: NYT puzzles with all permutations
- **Synthetic**: NYT + synthetic puzzles, single permutation
- **Full**: NYT + synthetic puzzles with all permutations

### Experiment 2: Reasoning Format
Compares different reasoning format approaches:
- **Structured**: Chain-of-thought with explicit structure
- **Unstructured**: Free-form reasoning
- **Mixed**: 50% structured, 50% unstructured training
- **Sequential**: Two-phase training (unstructured→structured)

### Experiment 3: Curriculum Learning
Tests curriculum learning strategies:
- **No Warmup**: Direct training on full puzzles
- **Warmup**: Pre-Connections tasks → Full dataset
- **Staged**: Pre-Connections → Synthetic → NYT puzzles

## Prediction Format

Each JSON file contains predictions with metadata for analysis:

```json
{
  "puzzle_id": 95,
  "prediction": "[[word1, word2, word3, word4], ...]",
  "ground_truth": "[[word1, word2, word3, word4], ...]",
  "reasoning": "Model's reasoning process...",
  "metadata": {
    "experiment": "exp1_baseline",
    "model_checkpoint": "...",
    "evaluation_date": "2025-10-22"
  }
}
```

## Related Resources

- **Training Dataset**: [NYT Connections Experiments Dataset](https://huggingface.co/datasets/nickting/nyt-connections-experiments)
- **Global Validation Set**: 276 entries (92 unique original puzzles: 222 NYT + 54 Synthetic)
- **Global Test Set**: 104 entries (102 unique NYT puzzles)
- **Training Data**: Varies by experiment (see experiment descriptions)

## Usage

These predictions can be used for:
- Performance analysis across different training configurations
- Error analysis and failure mode identification
- Comparing augmentation and curriculum learning strategies
- Reasoning quality evaluation

## Citation

If you use these predictions, please cite:

```bibtex
@dataset{nyt_connections_predictions,
  title={NYT Connections Predictions},
  author={nickting},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/nickting/nyt-connections-predictions}
}

@dataset{nyt_connections_experiments,
  title={NYT Connections Experiments Dataset},
  author={nickting},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/nickting/nyt-connections-experiments}
}
```

## License

MIT License
