---
license: mit
task_categories:
- question-answering
- text-generation
language:
- en
size_categories:
- 1K<n<10K
---

# NYT Connections Experiments Dataset

This dataset contains training, validation, and test splits for fine-tuning language models on New York Times Connections puzzles. It includes three experimental configurations examining data augmentation, reasoning format, and curriculum learning.

## Dataset Overview

- **NYT Puzzles**: 831 total (673 training, 74 validation, 84 test)
- **Synthetic Puzzles**: 200 total (162 training, 18 validation, 20 test)
- **Pre-Connections Tasks**: 720 training examples for curriculum learning
- **Validation Set**: 276 entries (222 NYT + 54 synthetic, with 3 permutations per puzzle)
- **Test Set**: 104 entries (84 NYT + 20 synthetic, single permutation per puzzle)

## File Structure

```
.
├── global_test.jsonl                    # Universal test set (104 entries)
├── global_validation.jsonl              # Global validation set (276 entries)
├── test_ids.json                        # Test puzzle ID registry
├── validation_ids.json                  # Validation puzzle ID registry
├── experiment1/                         # Data Augmentation Experiments
│   ├── baseline_train.jsonl            # 673 entries (NYT only, perm=1)
│   ├── permutation_train.jsonl         # 2,019 entries (NYT only, all perms)
│   ├── synthetic_train.jsonl           # 835 entries (NYT + Synthetic, perm=1)
│   ├── full_train.jsonl                # 2,505 entries (NYT + Synthetic, all perms)
│   ├── validation_nyt_perm1.jsonl      # 74 entries (NYT-only validation, perm=1)
│   └── validation_nyt_all_perms.jsonl  # 222 entries (NYT-only validation, all perms)
├── experiment2/                         # Format Comparison Experiments
│   ├── structured_only_train.jsonl     # 500 entries (structured format)
│   ├── unstructured_only_train.jsonl   # 500 entries (unstructured format)
│   ├── mixed_train.jsonl               # 500 entries (50% structured, 50% unstructured)
│   ├── sequential_phase1_unstructured.jsonl  # 250 entries (phase 1)
│   ├── sequential_phase2_structured.jsonl    # 250 entries (phase 2)
│   ├── validation_structured.jsonl     # 74 entries (structured validation)
│   ├── validation_unstructured.jsonl   # 74 entries (unstructured validation)
│   ├── validation_mixed.jsonl          # 74 entries (mixed validation)
│   ├── sampled_ids.json                # 500 puzzle IDs used in experiment 2
│   └── id_splits.json                  # Documentation of ID splits
└── experiment3/                         # Curriculum Learning Experiments
    ├── preconn_warmup.jsonl            # 720 entries (Pre-Connections warmup tasks)
    ├── synthetic_component.jsonl       # 486 entries (Synthetic puzzles)
    ├── nyt_component.jsonl             # 2,019 entries (NYT puzzles)
    └── full_augmented.jsonl            # 2,505 entries (Full dataset)
```

## Validation Strategy

### **Experiment 1: Data Augmentation**

To ensure valid training monitoring, validation sets are matched to training data distribution:

- **baseline_train.jsonl** → Uses `validation_nyt_perm1.jsonl` (74 entries)
  - Matches: NYT-only, single permutation
- **permutation_train.jsonl** → Uses `validation_nyt_all_perms.jsonl` (222 entries)
  - Matches: NYT-only, all permutations
- **synthetic_train.jsonl** & **full_train.jsonl** → Use `global_validation.jsonl` (276 entries)
  - Matches: Mixed NYT + Synthetic distribution

### **Experiment 2: Format Comparison**

Format-specific validation ensures models are evaluated on matching formats:

- **structured_only_train.jsonl** → Uses `validation_structured.jsonl`
- **unstructured_only_train.jsonl** → Uses `validation_unstructured.jsonl`
- **mixed_train.jsonl** → Uses `validation_mixed.jsonl` (50% structured, 50% unstructured)

All validation sets contain the same 74 NYT puzzle IDs in different format representations.

### **Experiment 3: Curriculum Learning**

- **Warmup phases** (Pre-Connections, Synthetic): Validation **disabled** during training
  - Rationale: Task mismatch makes validation on full Connections puzzles uninterpretable
- **Final phases**: Use `global_validation.jsonl` for meaningful monitoring

## Data Format

Each entry follows this structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Solve this NYT Connections puzzle..."
    },
    {
      "role": "assistant",
      "content": "<think>Reasoning process...</think>\n\nAnswer: [groups]"
    }
  ],
  "metadata": {
    "puzzle_id": 95,
    "original_id": 95,
    "permutation": 1,
    "reasoning_length": 2620
  }
}
```

## Data Leakage Prevention

- **ID-based separation**: Train/validation/test splits are separated at the puzzle ID level
- **No overlap**: Validation uses different puzzle IDs than test (74 vs 84 NYT puzzles)
- **Experiment 2 isolation**: The 500 training puzzles are completely separate from the 74 validation and 84 test puzzles
- **Permutations**: Multiple permutations of the same puzzle only appear within the same split

## Citation

If you use this dataset, please cite:

```bibtex
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
