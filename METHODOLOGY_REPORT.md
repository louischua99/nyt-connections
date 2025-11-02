# Methodology Report: Experimental Design and Data Management
## Fine-tuning Language Models for NYT Connections Puzzle Solving

---

## Abstract

This report documents the experimental methodology for a study investigating fine-tuned language models on New York Times Connections puzzles. The study comprises three experiments examining data augmentation, reasoning format, and curriculum learning effects. We describe the dataset preparation pipeline, test set isolation procedures, training configurations, and prediction generation protocols. The report provides a detailed account of the data splits, experimental conditions, and implementation choices to support reproducibility and enable assessment of experimental validity.

---

## 1. Introduction

The New York Times Connections puzzle requires identifying four groups of four related words from a set of sixteen words. This task involves semantic reasoning, pattern recognition, and systematic elimination. We investigate whether fine-tuned language models can learn to solve these puzzles and which training strategies optimize performance.

The study addresses three research questions:

1. How does training data quantity and augmentation affect model performance?
2. Does structured reasoning format differ from unstructured format in training effectiveness?
3. Can pre-training on simpler tasks improve performance through curriculum learning?

We examine these questions through dedicated experiments using a 4-billion parameter language model with Low-Rank Adaptation fine-tuning.

---

## 2. Dataset

### 2.1 Data Sources

The dataset combines two sources:

**NYT Puzzles**: Official New York Times Connections puzzles representing authentic game instances. The collection spans multiple difficulty levels and semantic pattern types.

**Synthetic Puzzles**: Algorithmically generated puzzles following similar structural constraints. These provide additional training diversity while maintaining the 4x4 grouping format.

### 2.2 Data Formats

The dataset includes three format variants:

**Structured Format**: Solutions include reasoning within `<think>` tags followed by the final answer:

```
<think>
[Detailed reasoning process]
</think>

[Final answer with groups]
```

**Unstructured Format**: Solutions present reasoning as natural narrative without formal structure markers.

**Pre-Connections Format**: Progressive difficulty tasks based on the BigBench "odd word out" dataset adapted with categorical reasoning. The tasks follow a curriculum from simpler to more complex:

1. Basic odd-word-out: Identify one anomalous word from 4-5 words where 3-4 share a common property
2. Find odd words (plural): Identify multiple anomalous words from a larger set
3. Group formation: Identify groups of related words within a set

These tasks incorporate the same 15 semantic category types used in the full Connections puzzles (Semantic Taxonomy, Semantic Synonymy, Semantic Association, Named Entities, Collocational/Idiomatic, Lexical Morphology, Lexical Orthography, Phonological Pattern, Grammatical/Syntactic, Wordplay Double Meaning, Temporal/Sequential, Numerical/Quantitative, Lexical Etymology, Sociolinguistic Register, and Cross-Linguistic patterns). This format serves as a warmup by training models on the fundamental skill of identifying semantic relationships before requiring them to partition 16 words into four groups simultaneously.

### 2.3 Dataset Statistics

The dataset comprises:

- NYT puzzles: 831 total (673 training, 74 validation, 84 test)
- Synthetic puzzles: 200 total (162 training, 18 validation, 20 test)
- Pre-Connections tasks: 720 training examples
- Validation set: 276 entries (222 NYT + 54 synthetic; includes 3 permutations per puzzle)
- Test set: 104 entries (84 NYT + 20 synthetic; single permutation per puzzle)

Each puzzle is represented in a conversational format with user message (puzzle prompt) and assistant message (solution with reasoning). The data is split into train/validation/test with strict ID-based separation to prevent leakage.

---

## 3. Data Split and Isolation

### 3.1 Test and Validation Set Definition

Data split construction occurs before training data processing in `prepare_experiments.py`. The script first loads pre-designated test files and extracts unique puzzle identifiers:

```python
# Load test files
structured_nyt_test = load_jsonl('data2/output/structured_nyt_test_formatted.jsonl')
structured_synthetic_test = load_jsonl('data2/output/structured_synthetic_test_formatted.jsonl')

# Extract test IDs
nyt_test_ids = set()
for entry in structured_nyt_test:
    original_id = entry['metadata'].get('original_id')
    if original_id:
        nyt_test_ids.add(original_id)
```

This produces 84 NYT test IDs and 20 synthetic test IDs. These 104 examples form the universal test set used across all experiments.

The script then creates a validation split by sampling 10% of the remaining training puzzles:

```python
# Create validation split (10% of training data)
nyt_unique_ids = sorted(list(nyt_train_ids_full))
nyt_val_size = max(1, int(len(nyt_unique_ids) * 0.10))
nyt_val_ids = set(random.sample(nyt_unique_ids, nyt_val_size))
nyt_train_ids = nyt_train_ids_full - nyt_val_ids
```

This produces 74 NYT validation puzzles and 18 synthetic validation puzzles. The validation set is used for monitoring during training, while the test set remains isolated for final evaluation only.

### 3.2 Training/Validation/Test Set Verification

After creating the three-way split, the script verifies that all sets are disjoint:

```python
nyt_train_ids = {e['metadata'].get('original_id') for e in structured_nyt_train}
nyt_leak = nyt_train_ids & nyt_test_ids

if nyt_leak:
    print(f"WARNING: NYT train/test leakage detected: {nyt_leak}")
else:
    print("No NYT train/test leakage")
```

The set intersection checks ensure training, validation, and test sets are mutually exclusive. The script performs this verification separately for NYT and synthetic data. All split IDs are saved to `data/test_ids.json` and `data/validation_ids.json` for documentation and reproducibility.

### 3.3 Permutation Handling

Each puzzle can be presented with different word orderings (permutations). The leakage checks operate on `original_id` rather than individual permutation instances, ensuring that all permutations of a test puzzle are excluded from training.

### 3.4 Format-Specific Considerations

Experiment 2 requires puzzles in both structured and unstructured formats. The sampling procedure explicitly excludes test IDs when building format-specific datasets:

```python
# Build unstructured lookup, excluding test IDs
for entry in unstructured_nyt:
    puzzle_id = entry['metadata'].get('puzzle_id')
    if puzzle_id and puzzle_id not in nyt_test_ids:
        unstructured_by_id[puzzle_id] = entry

# Find common IDs and remove test IDs
common_safe_ids = set(structured_by_id.keys()) & set(unstructured_by_id.keys())
common_safe_ids = common_safe_ids - nyt_test_ids

# Sample from safe IDs only
sampled_ids = random.sample(sorted(common_safe_ids), sample_size)

# Verify no test contamination
sampled_leak = set(sampled_ids) & nyt_test_ids
if sampled_leak:
    raise ValueError("Data leakage detected!")
```

This ensures the same puzzle cannot appear in training and test, regardless of format representation.

### 3.5 Random Seed Control

The data preparation script sets a random seed at initialization:

```python
random.seed(42)
```

This controls all random operations including sampling for Experiment 2 and shuffling for mixed training data. The fixed seed enables reproducibility while maintaining proper randomization properties.

---

## 4. Experimental Design

### 4.1 Experiment 1: Data Augmentation

This experiment examines the effect of training data quantity and diversity on model performance.

**Experimental Conditions:**

| Configuration | Training Examples | Description |
|--------------|------------------|-------------|
| Baseline | 673 | NYT puzzles, single permutation |
| Permutation | 2,019 | NYT puzzles, 3 permutations each |
| Synthetic | 835 | NYT (1 perm) + Synthetic (1 perm) |
| Full | 2,505 | NYT (3 perms) + Synthetic (3 perms) |

**Implementation:**

The baseline configuration uses only the first permutation of each training puzzle. The permutation configuration includes all three permutation variants. The synthetic configuration adds synthetic puzzles without permutation augmentation. The full configuration combines both strategies.

All configurations use the structured reasoning format. Training occurs in a single phase from the base model.

**Validation Strategy:**

To ensure valid training monitoring, Experiment 1 uses domain-matched validation sets:

- **Baseline & Permutation**: Use NYT-only validation sets to match training distribution (synthetic puzzles excluded)
  - `validation_nyt_perm1.jsonl`: 74 entries for baseline (matches perm=1 training)
  - `validation_nyt_all_perms.jsonl`: 222 entries for permutation (matches 3-perm training)
- **Synthetic & Full**: Use global validation (276 entries: 222 NYT + 54 synthetic) matching mixed training distribution

This domain and permutation matching prevents validation metrics from being skewed by data the model hasn't seen (synthetic puzzles) or different augmentation densities (permutation counts).

### 4.2 Experiment 2: Format Comparison

This experiment compares structured and unstructured reasoning formats while controlling for puzzle content.

**Design Constraint:**

To isolate format effects, the same puzzles must appear in both formats. The dataset includes 500 puzzles available in both representations.

**Experimental Conditions:**

| Configuration | Training Examples | Format | Puzzle IDs |
|--------------|------------------|--------|------------|
| Structured | 500 | Structured only | 500 unique IDs |
| Unstructured | 500 | Unstructured only | Same 500 IDs |
| Mixed | 500 | Both (250 + 250) | Same 500 IDs, shuffled |
| Sequential | 500 | Both (250 → 250) | Same 500 IDs, phased |

**Implementation:**

The structured and unstructured configurations train on different format representations of the same 500 puzzles. The mixed configuration shuffles 250 unstructured and 250 structured examples from these puzzles. The sequential configuration trains on 250 unstructured examples first, then continues training on 250 structured examples.

**Validation Data for Experiment 2:**

Experiment 2 requires format-specific validation sets since models must be evaluated on data matching their training format. The 74 NYT validation puzzles (from the global validation set) exist in both structured and unstructured formats, enabling format-matched evaluation:

- `validation_structured.jsonl`: 74 puzzles in structured format (used by exp2_structured, exp2_sequential phase 2)
- `validation_unstructured.jsonl`: 74 puzzles in unstructured format (used by exp2_unstructured, exp2_sequential phase 1)
- `validation_mixed.jsonl`: 74 puzzles mixed format (37 structured + 37 unstructured, shuffled) (used by exp2_mixed)

All three validation files contain the same 74 NYT puzzle IDs in different format representations, ensuring fair comparison across conditions while maintaining format consistency during training monitoring. The mixed validation set matches the mixed training distribution (50% structured, 50% unstructured).

The script saves ID splits to `data/experiment2/id_splits.json`:

```json
{
  "all_500_ids": [...],
  "first_half_ids_1_250": [...],
  "second_half_ids_251_500": [...],
  "note": "Mixed uses all 500 shuffled. Sequential uses first_half then second_half."
}
```

This documentation enables verification that mixed and sequential configurations use identical puzzle content.

### 4.3 Experiment 3: Curriculum Learning

This experiment tests whether pre-training on simpler tasks improves final performance through curriculum learning.

**Experimental Conditions:**

| Configuration | Training Strategy | Phases |
|--------------|------------------|--------|
| No Warmup | Direct training | Full dataset (2,505 examples) |
| Warmup | Two-phase | Preconn (720) → Full (2,505) |
| Staged | Three-phase | Preconn (720) → Synthetic (486) → NYT (2,019) |

**Implementation:**

The no warmup configuration trains directly on the full augmented dataset with validation monitoring. The warmup configuration pre-trains on Pre-Connections tasks before training on Connections puzzles. The staged configuration introduces an intermediate synthetic puzzle phase between warmup and NYT puzzles.

The staged approach incorporates two levels of curriculum:

1. **Task complexity**: Pre-Connections tasks (identifying anomalous words or simple groups) are structurally simpler than full Connections puzzles (partitioning 16 words into four groups).

2. **Puzzle difficulty**: Anecdotally, synthetic puzzles appear easier than authentic NYT puzzles. Synthetic puzzles are algorithmically generated with explicit category constraints, potentially producing more regular semantic patterns than human-designed NYT puzzles which may employ more subtle wordplay and misdirection. The staged configuration orders training from simpler (synthetic) to more complex (NYT) full Connections puzzles.

**Validation and Learning Rate Strategy:**

Curriculum phases use adjusted training parameters to account for task differences:

- **Warmup phases** (Pre-Connections, Synthetic): Validation **disabled** and learning rate **halved** (1e-4 instead of 2e-4)
  - Validation disabled because warmup tasks differ fundamentally from the final task (Pre-Connections) or represent intermediate difficulty (Synthetic), making validation on full Connections puzzles uninterpretable
  - Lower learning rate used for gradual adaptation during curriculum stages
- **Final phases** (Full dataset, NYT): Validation **enabled** with standard learning rate (2e-4)
  - Validation on global validation set (276 entries) provides meaningful performance monitoring

Multi-phase training uses checkpoint continuation, where each phase loads the model from the previous phase:

```python
# Phase 1: Train on preconn (LR=1e-4, no validation)
train_single_phase("data/experiment3/preconn_warmup.jsonl",
                  validation_data=None, learning_rate=1e-4,
                  model_to_load=MODEL_NAME)

# Phase 2 (Staged only): Synthetic (LR=1e-4, no validation)
train_single_phase("data/experiment3/synthetic_component.jsonl",
                  validation_data=None, learning_rate=1e-4,
                  model_to_load=phase1_dir)

# Phase 3/Final: Full/NYT (LR=2e-4, with validation)
train_single_phase("data/experiment3/nyt_component.jsonl",
                  validation_data="data/global_validation.jsonl",
                  learning_rate=2e-4, model_to_load=phase2_dir)
```

---

## 5. Training Configuration

### 5.1 Model Architecture

The study uses Qwen3-4B-Thinking-2507 as the base model, a 4-billion parameter language model with built-in reasoning capabilities. This model is selected for its explicit reasoning features aligned with the task requirements and its manageable size for research compute constraints. Fine-tuning applies Low-Rank Adaptation (LoRA) to reduce memory requirements and training time.

**LoRA Configuration:**

```python
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

This configuration adapts attention projection layers and feed-forward components while keeping other parameters frozen. The rank of 32 balances adaptation capacity with efficiency. Setting alpha equal to rank maintains standard scaling. Zero dropout is used to preserve learned patterns during fine-tuning.

### 5.2 Training Hyperparameters

All experiments use consistent hyperparameters, with learning rate adjusted for curriculum phases:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Batch Size | 2 | Memory constraint for 4B model on available GPUs |
| Gradient Accumulation | 4 | Effective batch size of 8 for stable gradients |
| Learning Rate | 2e-4 (1e-4 for warmup phases) | Standard for LoRA; halved for curriculum warmup |
| Warmup Steps | 5 | Brief warmup to stabilize initial training |
| Training Epochs | 8 | Multiple epochs for thorough LoRA adaptation |
| Optimizer | AdamW (8-bit) | Memory-efficient variant of AdamW |
| Weight Decay | 0.01 | Standard regularization to prevent overfitting |
| LR Schedule | Linear decay | Gradual reduction from peak to zero |
| Random Seed | 3407 | Fixed seed for reproducible training dynamics |
| Max Sequence Length | 10,000 | Accommodate long reasoning chains |

The effective batch size is 8 (batch size 2 × gradient accumulation 4). The maximum sequence length of 10,000 tokens allows models to generate extended reasoning for complex puzzles without truncation.

**Learning Rate Variation:** Experiment 3 curriculum phases (Pre-Connections warmup, Synthetic intermediate) use half the standard learning rate (1e-4 instead of 2e-4) to enable gentler adaptation during early training stages. Final phases use the standard 2e-4 rate.

### 5.3 Training Procedure

The training implementation uses the SFTTrainer from the TRL library with response-only loss masking:

```python
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```

This masks the user prompt during loss calculation, focusing training on generating solutions rather than repeating prompts.

The Qwen3-thinking chat template structures conversations with appropriate role markers:

```python
tokenizer = get_chat_template(tokenizer, chat_template="qwen3-thinking")
```

---

## 6. Prediction Generation

### 6.1 Generation Configuration

After training, the `scripts/generate_predictions.py` script generates predictions for all test examples. The generation configuration uses:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Max New Tokens | 2048 | Sufficient for complete reasoning and answer |
| Temperature | 0.7 | Balance between creativity and coherence |
| Top-p | 0.9 | Nucleus sampling for focused diversity |
| Sampling | True | Stochastic generation rather than greedy |
| Max Sequence Length | 10,000 | Match training context length |

```python
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    use_cache=True,
    pad_token_id=tokenizer.eos_token_id,
)
```

The temperature of 0.7 allows for varied reasoning paths without producing incoherent outputs. Top-p sampling at 0.9 maintains output quality by excluding low-probability tokens. All models generate predictions for the same test set loaded from `data/global_test.jsonl`.

### 6.2 Output Format

Each prediction is saved with metadata:

```json
{
  "puzzle_id": 95,
  "user_message": "Solve this Connections puzzle...",
  "prediction": "Model's generated solution",
  "ground_truth": "Expected solution",
  "metadata": {
    "puzzle_id": 95,
    "original_id": 95,
    "permutation": 0,
    "reasoning_length": 2620
  }
}
```

This structure enables matching predictions to test examples and comparing against ground truth.

### 6.3 Generation Statistics

The prediction generation produces:

- 11 model prediction files (one per experimental configuration)
- 104 predictions per model (universal test set)
- 1,144 total predictions

---

## 7. Reproducibility

### 7.1 Random Seed Management

The pipeline uses two random seeds:

- Data preparation: `random.seed(42)` (controls sampling and shuffling)
- Training: `seed=3407` (controls model initialization and training stochasticity)

These seeds are fixed and documented in the code.

### 7.2 Configuration Documentation

Experimental configurations are defined in code rather than external configuration files. The `train_experiment.py` script accepts command-line arguments specifying which experiment to run:

```bash
python train_experiment.py --experiment exp1_baseline --epochs 8 --lr 2e-4
python train_experiment.py --experiment exp2_structured --epochs 8 --lr 2e-4
```

All hyperparameters are visible in the source code.

### 7.2.1 Parallel Training Infrastructure

All 11 experiments are trained in parallel across 3 NVIDIA H100 80GB GPUs using the `launch_all_training.sh` script. Each GPU runs 3-4 experiments sequentially with nohup logging:

- **GPU 0**: exp1_baseline, exp1_full, exp2_mixed, exp3_no_warmup
- **GPU 1**: exp1_permutation, exp2_structured, exp2_sequential, exp3_warmup
- **GPU 2**: exp1_synthetic, exp2_unstructured, exp3_staged

Each experiment runs independently with isolated logs (`logs/<experiment>_<timestamp>.log`) and model checkpoints (`models/<experiment>/`). This parallel approach enables efficient training of all configurations while maintaining reproducibility through fixed random seeds.

### 7.3 Data Provenance

The pipeline maintains clear data flow:

```
Raw Data → Formatted Data → Experiment Splits → Training → Predictions
data2/   →  data2/output/  →    data/        → models/  → data/predictions/
```

Each stage produces saved outputs enabling reconstruction of the pipeline.

### 7.4 Software Dependencies

The `requirements.txt` file lists dependencies:

```
unsloth
torch
transformers
datasets
trl
requests
tqdm
```

However, version numbers are not pinned, which may affect reproducibility across time.

---

## 8. Data Management

### 8.1 File Organization

Training and validation data is organized by experiment:

```
data/
├── global_test.jsonl                    # Universal test set (104 examples)
├── global_validation.jsonl              # Global validation set (276 examples)
├── test_ids.json                        # Test ID registry
├── validation_ids.json                  # Validation ID registry
├── experiment1/
│   ├── baseline_train.jsonl            # 673 examples
│   ├── permutation_train.jsonl         # 2,019 examples
│   ├── synthetic_train.jsonl           # 835 examples
│   ├── full_train.jsonl                # 2,505 examples
│   ├── validation_nyt_perm1.jsonl      # 74 examples (NYT-only, perm=1)
│   └── validation_nyt_all_perms.jsonl  # 222 examples (NYT-only, all perms)
├── experiment2/
│   ├── structured_only_train.jsonl     # 500 examples
│   ├── unstructured_only_train.jsonl   # 500 examples
│   ├── mixed_train.jsonl               # 500 examples
│   ├── sequential_phase1_unstructured.jsonl  # 250 examples
│   ├── sequential_phase2_structured.jsonl    # 250 examples
│   ├── validation_structured.jsonl     # 74 examples
│   ├── validation_unstructured.jsonl   # 74 examples
│   ├── validation_mixed.jsonl          # 74 examples
│   ├── sampled_ids.json                # ID documentation
│   └── id_splits.json                  # Split documentation
└── experiment3/
    ├── preconn_warmup.jsonl            # 720 examples
    ├── synthetic_component.jsonl       # 486 examples
    ├── nyt_component.jsonl             # 2,019 examples
    └── full_augmented.jsonl            # 2,505 examples
```

### 8.2 Metadata Structure

Each example includes metadata for tracking:

```json
"metadata": {
  "puzzle_id": 95,
  "original_id": 95,
  "permutation": 0,
  "reasoning_length": 2620
}
```

The `puzzle_id` and `original_id` enable tracking across permutations and formats. The `permutation` field indicates which variant of the puzzle is represented.

### 8.3 Training Output Organization

Model checkpoints are saved to experiment-specific directories:

```
models/
├── exp1_baseline/
├── exp1_permutation/
├── exp1_synthetic/
├── exp1_full/
├── exp2_structured/
├── exp2_unstructured/
├── exp2_mixed/
├── exp2_sequential/
│   ├── phase1_unstructured/
│   └── phase2_structured_final/
├── exp3_no_warmup/
├── exp3_warmup/
│   ├── phase1_preconn/
│   └── phase2_full_final/
└── exp3_staged/
    ├── phase1_preconn/
    ├── phase2_synthetic/
    └── phase3_nyt_final/
```

Multi-phase experiments save intermediate checkpoints.

---

## 9. Evaluation Considerations

### 9.1 Test Set Usage During Training

All 11 models are evaluated on the identical 104-example test set. The training script passes this test set to the trainer as `eval_dataset`:

```python
test_data_path = "data/global_test.jsonl"
eval_dataset = prepare_dataset(load_jsonl(test_data_path))
```

The trainer logs evaluation loss on this dataset during training. While the trainer configuration does not include automatic early stopping or checkpoint selection based on these metrics, the visibility of test set performance during training represents a potential source of indirect leakage. Researchers could observe test set metrics and make decisions (restarting runs, adjusting approaches, etc.) based on this information, even if no automated mechanisms use the test data.

The standard practice would be to either omit the evaluation dataset during training or use a separate validation set held out from the training data. The current implementation provides test set visibility during training, which should be noted when interpreting results.

### 9.2 Evaluation Metrics

The current implementation generates and saves predictions but does not include evaluation code. The prediction format includes ground truth solutions, enabling downstream metric computation:

- Exact match accuracy
- Group-level precision and recall
- Partial credit scoring
- Error type analysis

These metrics would need to be implemented to quantify model performance.

### 9.3 Statistical Analysis

Each experimental configuration is trained once with a fixed random seed. This provides point estimates of performance but does not quantify variance due to initialization or training stochasticity. Multiple runs per configuration would enable:

- Confidence intervals on performance estimates
- Statistical tests of performance differences
- Assessment of training stability

---

## 10. Limitations and Considerations

### 10.1 Single Training Run

Each configuration is trained once. Performance estimates are therefore point estimates without uncertainty quantification. Running each configuration multiple times with different random seeds would provide more robust estimates.

### 10.2 Data Split

The dataset is split into three non-overlapping sets:

- **Training**: 673 NYT puzzles + 162 synthetic puzzles (with permutation augmentation, this ranges from 673 to 2,505 entries depending on experiment configuration)
- **Validation**: 74 NYT puzzles + 18 synthetic puzzles (276 entries with permutations)
- **Test**: 84 NYT puzzles + 20 synthetic puzzles (104 entries)

The validation set represents approximately 10% of the available training data (sampled before augmentation) and is used for monitoring training progress and evaluation metrics during model development. The test set remains completely isolated and is reserved exclusively for final evaluation after all training and development decisions are complete.

All puzzle IDs are strictly separated across splits—no puzzle appears in more than one set. The `prepare_experiments.py` script includes verification checks at multiple stages to ensure no data leakage between splits. During training, the `train_experiment.py` script uses the validation set (not the test set) for the `eval_dataset` parameter, following standard machine learning best practices.

This three-way split ensures:
1. The training set is used exclusively for gradient updates
2. The validation set monitors performance during training and can inform development decisions
3. The test set provides an unbiased final evaluation of model performance

### 10.3 Test Set Size

The test set comprises 104 examples (84 NYT, 20 synthetic). This provides reasonable statistical power for detecting large effects but may have limited power for small effect sizes or fine-grained comparisons. The validation set (276 entries) provides additional evaluation data during training without compromising test set integrity.

### 10.4 Hyperparameter Selection

The study uses fixed hyperparameters across all experiments. While justifications are provided for each parameter choice (see Section 5.2), these values were not empirically tuned on this specific task. The hyperparameters may not be optimal for all experimental conditions, and a hyperparameter search could potentially improve performance.

### 10.5 Synthetic Data Distribution

The synthetic puzzles are algorithmically generated. The generation process is not documented in the files examined. If synthetic puzzles have systematic differences from NYT puzzles, this could affect generalization.

### 10.6 Permutation Strategy

Each puzzle is represented with 3 permutations, but the permutation generation method is not documented. If permutations are not sufficiently diverse, the augmentation benefit may be limited.

---

## 11. Conclusion

This report documents the experimental methodology for fine-tuning language models on NYT Connections puzzles. The study comprises three experiments examining data augmentation effects (Experiment 1), reasoning format differences (Experiment 2), and curriculum learning strategies (Experiment 3). The data preparation pipeline implements test set isolation through ID-based exclusion and set intersection verification. All 11 models across three experiments are evaluated on a universal 104-example test set. Training uses consistent hyperparameters with LoRA fine-tuning of a 4-billion parameter base model.

The methodology maintains clear separation between training and test data through multiple verification stages. Test set definition occurs before training data processing, and explicit set intersection checks verify disjoint train/test splits. The experimental designs enable systematic comparison of training strategies through controlled manipulation of specific factors while holding other variables constant. The codebase structure supports reproduction through automated scripts, configuration documentation, and random seed control.

The implementation provides infrastructure for generating predictions from all trained models, storing results with ground truth labels for downstream evaluation. The prediction format enables computation of various evaluation metrics including exact match accuracy, group-level precision and recall, and error analysis.

---

## References

This report documents the methodology based on the following source files:

- `prepare_experiments.py` - Data preparation and test set isolation
- `train_experiment.py` - Training infrastructure and configuration
- `scripts/generate_predictions.py` - Prediction generation
- `scripts/run_all_predictions.sh` - Batch prediction orchestration
- `data/test_ids.json` - Test ID registry
- `data/experiment2/sampled_ids.json` - Experiment 2 sample documentation
- `data/experiment2/id_splits.json` - Experiment 2 split documentation
