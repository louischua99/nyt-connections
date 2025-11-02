---
license: mit
task_categories:
- question-answering
- text-generation
language:
- en
size_categories:
- 10K<n<100K
---

# NYT Connections Raw Datasets

This repository contains the **raw and formatted reasoning data** for NYT Connections puzzle solving experiments. These files are the source data used to create the experiment splits in [nickting/nyt-connections-experiments](https://huggingface.co/datasets/nickting/nyt-connections-experiments).

## Overview

This dataset includes three types of puzzle data with AI-generated reasoning:

1. **NYT Connections Puzzles** - Authentic New York Times puzzles
2. **Synthetic Connections Puzzles** - Algorithmically generated puzzles
3. **Pre-Connections Tasks** - Curriculum learning warmup tasks

All data includes both raw reasoning output and formatted conversational format ready for fine-tuning.

## Directory Structure

```
data2/
├── puzzles/                    # Raw puzzle definitions (JSON)
│   ├── connections.json        # NYT puzzle data
│   ├── connections_synthetic.json  # Synthetic puzzle data
│   └── preconn.json            # Pre-Connections tasks
└── reasoning/                  # Generated reasoning data (JSONL)
    ├── structured_nyt_train.jsonl              # NYT structured reasoning (raw)
    ├── structured_nyt_test.jsonl
    ├── structured_nyt_train_formatted.jsonl    # NYT structured reasoning (formatted)
    ├── structured_nyt_test_formatted.jsonl
    ├── structured_synthetic_train.jsonl        # Synthetic structured reasoning (raw)
    ├── structured_synthetic_test.jsonl
    ├── structured_synthetic_train_formatted.jsonl  # Synthetic structured reasoning (formatted)
    ├── structured_synthetic_test_formatted.jsonl
    ├── unstructured_nyt.jsonl                  # NYT unstructured reasoning (raw)
    ├── unstructured_nyt_formatted.jsonl        # NYT unstructured reasoning (formatted)
    ├── unstructured_synthetic.jsonl            # Synthetic unstructured reasoning (raw)
    ├── unstructured_synthetic_formatted.jsonl  # Synthetic unstructured reasoning (formatted)
    ├── structured_preconn_train.jsonl          # Pre-Connections reasoning (raw)
    ├── structured_preconn_test.jsonl
    ├── structured_preconn_train_formatted.jsonl  # Pre-Connections reasoning (formatted)
    └── structured_preconn_test_formatted.jsonl
```

## File Descriptions

### Puzzle Files (puzzles/)

**connections.json** - NYT puzzle definitions
- Contains 831 authentic NYT Connections puzzles
- Each puzzle has 16 words grouped into 4 categories
- Includes metadata: difficulty, category types, publication date

**connections_synthetic.json** - Synthetic puzzle definitions
- Contains 200 algorithmically generated puzzles
- Similar structure to NYT puzzles
- Generated with categorical constraints

**preconn.json** - Pre-Connections tasks
- Contains 800 curriculum learning tasks
- Simpler than full Connections puzzles
- Three task types: odd-word-out, find-odd-words, group-formation

### Reasoning Files (reasoning/)

#### Format Types

**Structured Format** - Reasoning in `<think>` tags:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Solve this puzzle: [16 words]"
    },
    {
      "role": "assistant",
      "content": "<think>\nStep 1: Analyze words...\nStep 2: Identify patterns...\n</think>\n\nAnswer: [groups]"
    }
  ]
}
```

**Unstructured Format** - Natural narrative reasoning:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Solve this puzzle: [16 words]"
    },
    {
      "role": "assistant",
      "content": "Looking at these words, I notice... [reasoning]... Therefore, the groups are: [groups]"
    }
  ]
}
```

#### Raw vs Formatted

- **Raw files** (e.g., `structured_nyt_train.jsonl`): Initial AI-generated reasoning
- **Formatted files** (e.g., `structured_nyt_train_formatted.jsonl`): Processed into conversational format with proper metadata

Formatted files include additional metadata:
```json
{
  "messages": [...],
  "metadata": {
    "puzzle_id": 95,
    "original_id": 95,
    "permutation": 1,
    "reasoning_length": 2620
  }
}
```

## Data Statistics

### NYT Puzzles
- Total: 831 puzzles
- Train: 747 puzzles (90%)
- Test: 84 puzzles (10%)
- Formats: Structured (3 permutations) + Unstructured (1 permutation)
- Total entries: ~2,700 structured + ~750 unstructured

### Synthetic Puzzles
- Total: 200 puzzles
- Train: 180 puzzles (90%)
- Test: 20 puzzles (10%)
- Formats: Structured (3 permutations) + Unstructured (1 permutation)
- Total entries: ~540 structured + ~200 unstructured

### Pre-Connections Tasks
- Total: 800 tasks
- Train: 720 tasks (90%)
- Test: 80 tasks (10%)
- Format: Structured only
- Total entries: ~800

## Data Generation Pipeline

1. **Raw Puzzle Data** → Generated/collected puzzle definitions
2. **Reasoning Generation** → AI generates step-by-step solutions
   - `gen_reason_struct.py` - Structured reasoning
   - `gen_reason_unstruct.py` - Unstructured reasoning
   - `gen_reason_preconn.py` - Pre-Connections reasoning
3. **Formatting** → Convert to conversational format with metadata
   - `process_reasoning_format.py` - Format Connections data
   - `process_preconn_format.py` - Format Pre-Connections data
4. **Experiment Splits** → Create train/validation/test splits
   - `prepare_experiments.py` → Outputs to [nickting/nyt-connections-experiments](https://huggingface.co/datasets/nickting/nyt-connections-experiments)

## Usage

### Load Formatted Data for Training

```python
from datasets import load_dataset

# Load NYT structured reasoning
dataset = load_dataset(
    "nickting/nyt-connections-datasets-raw",
    data_files="reasoning/structured_nyt_train_formatted.jsonl"
)

# Load Pre-Connections for curriculum learning
preconn = load_dataset(
    "nickting/nyt-connections-datasets-raw",
    data_files="reasoning/structured_preconn_train_formatted.jsonl"
)
```

### Load Raw Puzzle Definitions

```python
import json

# Load NYT puzzle definitions
with open("puzzles/connections.json") as f:
    nyt_puzzles = json.load(f)
```

## Related Datasets

- **[nickting/nyt-connections-experiments](https://huggingface.co/datasets/nickting/nyt-connections-experiments)** - Experiment-ready train/validation/test splits with proper data leakage prevention

## Data Quality

- **AI-Generated Reasoning**: All reasoning chains are generated by AI, not human-written
- **Validation**: Test splits maintained for evaluation
- **Permutations**: Multiple word orderings to improve model robustness
- **Format Diversity**: Both structured (explicit thinking) and unstructured (narrative) formats

## License

MIT License

## Citation

```bibtex
@dataset{nyt_connections_raw,
  title={NYT Connections Raw Datasets},
  author={nickting},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/nickting/nyt-connections-datasets-raw}
}
```

## Acknowledgments

- New York Times for the original Connections puzzle format
- BigBench for the odd-word-out task inspiration
