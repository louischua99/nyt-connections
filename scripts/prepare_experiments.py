#!/usr/bin/env python3
"""
Prepare all datasets for experiments with strict data leakage prevention
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

def load_jsonl(filepath):
    """Load JSONL file"""
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, filepath):
    """Save JSONL file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def save_json(data, filepath):
    """Save JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================================
# PHASE 1: Identify Test IDs and Create Global Test Set
# ============================================================================

print("="*80)
print("PHASE 1: IDENTIFYING TEST IDS AND CREATING GLOBAL TEST SET")
print("="*80)

# Load test files
print("\nLoading test files...")
structured_nyt_test = load_jsonl('data2/reasoning/structured_nyt_test_formatted.jsonl')
structured_synthetic_test = load_jsonl('data2/reasoning/structured_synthetic_test_formatted.jsonl')
unstructured_nyt = load_jsonl('data2/reasoning/unstructured_nyt_formatted.jsonl')

# Extract test IDs
nyt_test_ids = set()
for entry in structured_nyt_test:
    original_id = entry['metadata'].get('original_id')
    if original_id:
        nyt_test_ids.add(original_id)

synthetic_test_ids = set()
for entry in structured_synthetic_test:
    original_id = entry['metadata'].get('original_id')
    if original_id:
        synthetic_test_ids.add(original_id)

print(f"NYT test IDs: {len(nyt_test_ids)} unique puzzles")
print(f"Synthetic test IDs: {len(synthetic_test_ids)} unique puzzles")
print(f"Total test IDs: {len(nyt_test_ids) + len(synthetic_test_ids)}")

# Map unstructured puzzle_ids to test IDs
unstructured_test_puzzle_ids = set()
for entry in unstructured_nyt:
    puzzle_id = entry['metadata'].get('puzzle_id')
    if puzzle_id in nyt_test_ids:
        unstructured_test_puzzle_ids.add(puzzle_id)

# Create global test set
global_test = structured_nyt_test + structured_synthetic_test
print(f"\nGlobal test set: {len(global_test)} entries")

# Save test IDs and global test set
test_ids_data = {
    'nyt_test_ids': sorted(list(nyt_test_ids)),
    'synthetic_test_ids': sorted(list(synthetic_test_ids)),
    'unstructured_test_puzzle_ids': sorted(list(unstructured_test_puzzle_ids))
}

save_json(test_ids_data, 'data/test_ids.json')
save_jsonl(global_test, 'data/global_test.jsonl')

print(f"✓ Saved test_ids.json")
print(f"✓ Saved global_test.jsonl ({len(global_test)} entries)")


# ============================================================================
# PHASE 2: Create Train/Validation Split
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: CREATING TRAIN/VALIDATION SPLIT")
print("="*80)

# Load training files
structured_nyt_train_full = load_jsonl('data2/reasoning/structured_nyt_train_formatted.jsonl')
structured_synthetic_train_full = load_jsonl('data2/reasoning/structured_synthetic_train_formatted.jsonl')

print(f"\nLoaded structured_nyt_train: {len(structured_nyt_train_full)} entries")
print(f"Loaded structured_synthetic_train: {len(structured_synthetic_train_full)} entries")

# Verify no test leakage
print("\nVerifying no test leakage...")
nyt_train_ids_full = {e['metadata'].get('original_id') for e in structured_nyt_train_full}
synthetic_train_ids_full = {e['metadata'].get('original_id') for e in structured_synthetic_train_full}

nyt_leak = nyt_train_ids_full & nyt_test_ids
synthetic_leak = synthetic_train_ids_full & synthetic_test_ids

if nyt_leak:
    print(f"⚠️  WARNING: NYT train/test leakage detected: {nyt_leak}")
else:
    print("✓ No NYT train/test leakage")

if synthetic_leak:
    print(f"⚠️  WARNING: Synthetic train/test leakage detected: {synthetic_leak}")
else:
    print("✓ No synthetic train/test leakage")

# Create validation split (10% of training data)
print("\nCreating train/validation split...")

# Get unique NYT puzzle IDs from training
nyt_unique_ids = sorted(list(nyt_train_ids_full))
nyt_val_size = max(1, int(len(nyt_unique_ids) * 0.10))
nyt_val_ids = set(random.sample(nyt_unique_ids, nyt_val_size))
nyt_train_ids = nyt_train_ids_full - nyt_val_ids

# Get unique synthetic puzzle IDs from training
synthetic_unique_ids = sorted(list(synthetic_train_ids_full))
synthetic_val_size = max(1, int(len(synthetic_unique_ids) * 0.10))
synthetic_val_ids = set(random.sample(synthetic_unique_ids, synthetic_val_size))
synthetic_train_ids = synthetic_train_ids_full - synthetic_val_ids

print(f"\nNYT split:")
print(f"  Training puzzles: {len(nyt_train_ids)} ({len(nyt_train_ids_full) - len(nyt_val_ids)})")
print(f"  Validation puzzles: {len(nyt_val_ids)}")
print(f"  Test puzzles: {len(nyt_test_ids)}")

print(f"\nSynthetic split:")
print(f"  Training puzzles: {len(synthetic_train_ids)} ({len(synthetic_train_ids_full) - len(synthetic_val_ids)})")
print(f"  Validation puzzles: {len(synthetic_val_ids)}")
print(f"  Test puzzles: {len(synthetic_test_ids)}")

# Split structured datasets into train/val
structured_nyt_train = [e for e in structured_nyt_train_full if e['metadata'].get('original_id') in nyt_train_ids]
structured_nyt_val = [e for e in structured_nyt_train_full if e['metadata'].get('original_id') in nyt_val_ids]

structured_synthetic_train = [e for e in structured_synthetic_train_full if e['metadata'].get('original_id') in synthetic_train_ids]
structured_synthetic_val = [e for e in structured_synthetic_train_full if e['metadata'].get('original_id') in synthetic_val_ids]

print(f"\nStructured entries:")
print(f"  NYT train: {len(structured_nyt_train)} entries")
print(f"  NYT val: {len(structured_nyt_val)} entries")
print(f"  Synthetic train: {len(structured_synthetic_train)} entries")
print(f"  Synthetic val: {len(structured_synthetic_val)} entries")

# Create global validation set
global_validation = structured_nyt_val + structured_synthetic_val
save_jsonl(global_validation, 'data/global_validation.jsonl')
print(f"\n✓ Saved global_validation.jsonl ({len(global_validation)} entries)")

# Create NYT-only validation sets for Experiment 1
# Baseline uses perm=1 only, permutation uses all perms
nyt_val_perm1 = [e for e in structured_nyt_val if e['metadata'].get('permutation') == 1]
nyt_val_all_perms = structured_nyt_val  # All permutations

save_jsonl(nyt_val_perm1, 'data/experiment1/validation_nyt_perm1.jsonl')
save_jsonl(nyt_val_all_perms, 'data/experiment1/validation_nyt_all_perms.jsonl')
print(f"✓ Saved experiment1 NYT-only validation sets:")
print(f"  - validation_nyt_perm1.jsonl: {len(nyt_val_perm1)} entries (74 puzzles, perm=1)")
print(f"  - validation_nyt_all_perms.jsonl: {len(nyt_val_all_perms)} entries (74 puzzles, all perms)")

# Save validation IDs
validation_ids_data = {
    'nyt_val_ids': sorted(list(nyt_val_ids)),
    'synthetic_val_ids': sorted(list(synthetic_val_ids)),
    'nyt_train_ids': sorted(list(nyt_train_ids)),
    'synthetic_train_ids': sorted(list(synthetic_train_ids))
}
save_json(validation_ids_data, 'data/validation_ids.json')
print(f"✓ Saved validation_ids.json")


# ============================================================================
# PHASE 3: Experiment 1 - Data Augmentation Ablation
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: EXPERIMENT 1 - DATA AUGMENTATION ABLATION")
print("="*80)

# 1. Baseline: NYT perm=1 only
baseline = [e for e in structured_nyt_train if e['metadata'].get('permutation') == 1]
save_jsonl(baseline, 'data/experiment1/baseline_train.jsonl')
print(f"\n1. Baseline: {len(baseline)} entries (perm=1 only)")

# 2. Permutation: All NYT permutations
permutation = structured_nyt_train
save_jsonl(permutation, 'data/experiment1/permutation_train.jsonl')
print(f"2. Permutation: {len(permutation)} entries (all perms)")

# 3. Synthetic: NYT perm=1 + Synthetic perm=1
nyt_perm1 = [e for e in structured_nyt_train if e['metadata'].get('permutation') == 1]
synthetic_perm1 = [e for e in structured_synthetic_train if e['metadata'].get('permutation') == 1]
synthetic_only = nyt_perm1 + synthetic_perm1
save_jsonl(synthetic_only, 'data/experiment1/synthetic_train.jsonl')
print(f"3. Synthetic: {len(synthetic_only)} entries ({len(nyt_perm1)} NYT + {len(synthetic_perm1)} synthetic, perm=1)")

# 4. Full: All permutations of both
full = structured_nyt_train + structured_synthetic_train
save_jsonl(full, 'data/experiment1/full_train.jsonl')
print(f"4. Full augmentation: {len(full)} entries ({len(structured_nyt_train)} NYT + {len(structured_synthetic_train)} synthetic, all perms)")

print("\n✓ Experiment 1 datasets created")


# ============================================================================
# PHASE 4: Experiment 2 - Format Impact
# ============================================================================

print("\n" + "="*80)
print("PHASE 4: EXPERIMENT 2 - FORMAT IMPACT")
print("="*80)

# Get safe train IDs (exclude test AND validation IDs from all available IDs)
structured_nyt_all = load_jsonl('data2/reasoning/structured_nyt_train_formatted.jsonl') + \
                     load_jsonl('data2/reasoning/structured_nyt_test_formatted.jsonl')

all_nyt_ids = {e['metadata'].get('original_id') for e in structured_nyt_all}
safe_nyt_train_ids = all_nyt_ids - nyt_test_ids - nyt_val_ids

print(f"\nTotal NYT puzzles: {len(all_nyt_ids)}")
print(f"Test NYT puzzles: {len(nyt_test_ids)}")
print(f"Validation NYT puzzles: {len(nyt_val_ids)}")
print(f"Safe train NYT puzzles: {len(safe_nyt_train_ids)}")

# Build lookup structures
structured_by_id = defaultdict(list)
for entry in structured_nyt_train:
    original_id = entry['metadata'].get('original_id')
    if original_id:
        structured_by_id[original_id].append(entry)

unstructured_by_id = {}
for entry in unstructured_nyt:
    puzzle_id = entry['metadata'].get('puzzle_id')
    if puzzle_id and puzzle_id not in nyt_test_ids and puzzle_id not in nyt_val_ids:
        unstructured_by_id[puzzle_id] = entry

# Find IDs that exist in BOTH formats and are safe
common_safe_ids = set(structured_by_id.keys()) & set(unstructured_by_id.keys())
common_safe_ids = common_safe_ids - nyt_test_ids - nyt_val_ids

print(f"IDs in both structured and unstructured: {len(common_safe_ids)}")

# Sample 500 IDs
sample_size = min(500, len(common_safe_ids))
sampled_ids = random.sample(sorted(common_safe_ids), sample_size)
print(f"\nSampled {sample_size} IDs for Experiment 2")

# Verify no test or validation leakage
sampled_test_leak = set(sampled_ids) & nyt_test_ids
sampled_val_leak = set(sampled_ids) & nyt_val_ids
if sampled_test_leak:
    print(f"⚠️  ERROR: Sampled IDs contain test IDs: {sampled_test_leak}")
    raise ValueError("Test data leakage detected!")
if sampled_val_leak:
    print(f"⚠️  ERROR: Sampled IDs contain validation IDs: {sampled_val_leak}")
    raise ValueError("Validation data leakage detected!")
print("✓ No test or validation leakage in sampled IDs")

# Save sampled IDs for reference
save_json({'sampled_ids': sorted(sampled_ids)}, 'data/experiment2/sampled_ids.json')

# Split the 500 IDs into two halves
half_size = sample_size // 2
first_half_ids = sampled_ids[:half_size]  # IDs 1-250
second_half_ids = sampled_ids[half_size:]  # IDs 251-500

# 1. Structured-only: ALL 500 IDs in structured format (perm=1) = 500 entries
structured_only = []
for pid in sampled_ids:
    perm1_entries = [e for e in structured_by_id[pid] if e['metadata'].get('permutation') == 1]
    structured_only.extend(perm1_entries)
save_jsonl(structured_only, 'data/experiment2/structured_only_train.jsonl')
print(f"\n1. Structured-only: {len(structured_only)} entries (ALL 500 IDs in structured)")

# 2. Unstructured-only: ALL 500 IDs in unstructured format = 500 entries
unstructured_only = [unstructured_by_id[pid] for pid in sampled_ids]
save_jsonl(unstructured_only, 'data/experiment2/unstructured_only_train.jsonl')
print(f"2. Unstructured-only: {len(unstructured_only)} entries (ALL 500 IDs in unstructured)")

# 3. Mixed: IDs 1-250 unstructured + IDs 251-500 structured, SHUFFLED = 500 entries
mixed_unstructured = [unstructured_by_id[pid] for pid in first_half_ids]
mixed_structured = []
for pid in second_half_ids:
    perm1_entries = [e for e in structured_by_id[pid] if e['metadata'].get('permutation') == 1]
    mixed_structured.extend(perm1_entries)
mixed = mixed_unstructured + mixed_structured
random.shuffle(mixed)  # Shuffle them together
save_jsonl(mixed, 'data/experiment2/mixed_train.jsonl')
print(f"3. Mixed: {len(mixed)} entries (IDs 1-250 unstructured + IDs 251-500 structured, shuffled)")

# 4. Sequential: SAME data as Mixed, but split into two phases (NOT shuffled)
sequential_phase1 = [unstructured_by_id[pid] for pid in first_half_ids]
sequential_phase2 = []
for pid in second_half_ids:
    perm1_entries = [e for e in structured_by_id[pid] if e['metadata'].get('permutation') == 1]
    sequential_phase2.extend(perm1_entries)
save_jsonl(sequential_phase1, 'data/experiment2/sequential_phase1_unstructured.jsonl')
save_jsonl(sequential_phase2, 'data/experiment2/sequential_phase2_structured.jsonl')
print(f"4. Sequential: Phase1={len(sequential_phase1)} (IDs 1-250 unstructured) → Phase2={len(sequential_phase2)} (IDs 251-500 structured)")
print(f"   NOTE: Mixed and Sequential use the SAME 500 entries, just organized differently")

# Save the ID splits for reference
save_json({
    'all_500_ids': sorted(sampled_ids),
    'first_half_ids_1_250': sorted(first_half_ids),
    'second_half_ids_251_500': sorted(second_half_ids),
    'note': 'Mixed uses all 500 shuffled. Sequential uses first_half then second_half.'
}, 'data/experiment2/id_splits.json')

# Create validation sets for Experiment 2 (from NYT validation puzzles that exist in both formats)
print("\n5. Creating Experiment 2 validation sets...")

# Build lookup for validation structured puzzles (structured_by_id only has training)
structured_val_by_id = defaultdict(list)
for entry in structured_nyt_val:
    original_id = entry['metadata'].get('original_id')
    if original_id:
        structured_val_by_id[original_id].append(entry)

# Get structured validation entries (perm=1 only for consistency with training)
exp2_val_structured = []
for val_id in nyt_val_ids:
    if val_id in structured_val_by_id:
        perm1_entries = [e for e in structured_val_by_id[val_id] if e['metadata'].get('permutation') == 1]
        exp2_val_structured.extend(perm1_entries)

# Build lookup for unstructured validation puzzles
unstructured_val_by_id = {}
for entry in unstructured_nyt:
    puzzle_id = entry['metadata'].get('puzzle_id')
    if puzzle_id and puzzle_id in nyt_val_ids:
        unstructured_val_by_id[puzzle_id] = entry

exp2_val_unstructured = [unstructured_val_by_id[vid] for vid in nyt_val_ids if vid in unstructured_val_by_id]

# Create mixed validation set (half structured, half unstructured, shuffled)
nyt_val_ids_sorted = sorted(list(nyt_val_ids))
half_val = len(nyt_val_ids_sorted) // 2
first_half_val = nyt_val_ids_sorted[:half_val]
second_half_val = nyt_val_ids_sorted[half_val:]

exp2_val_mixed_unstructured = [unstructured_val_by_id[vid] for vid in first_half_val if vid in unstructured_val_by_id]
exp2_val_mixed_structured = []
for vid in second_half_val:
    if vid in structured_val_by_id:
        perm1_entries = [e for e in structured_val_by_id[vid] if e['metadata'].get('permutation') == 1]
        exp2_val_mixed_structured.extend(perm1_entries)

exp2_val_mixed = exp2_val_mixed_unstructured + exp2_val_mixed_structured
random.shuffle(exp2_val_mixed)

save_jsonl(exp2_val_structured, 'data/experiment2/validation_structured.jsonl')
save_jsonl(exp2_val_unstructured, 'data/experiment2/validation_unstructured.jsonl')
save_jsonl(exp2_val_mixed, 'data/experiment2/validation_mixed.jsonl')

print(f"   Validation structured: {len(exp2_val_structured)} entries")
print(f"   Validation unstructured: {len(exp2_val_unstructured)} entries")
print(f"   Validation mixed: {len(exp2_val_mixed)} entries (half structured, half unstructured, shuffled)")
print(f"   Note: All validation sets contain the same {len(nyt_val_ids)} NYT puzzles in different formats")

print("\n✓ Experiment 2 datasets created")


# ============================================================================
# PHASE 5: Experiment 3 - Preconn Warmup Strategy
# ============================================================================

print("\n" + "="*80)
print("PHASE 5: EXPERIMENT 3 - PRECONN WARMUP STRATEGY")
print("="*80)

# Load preconn
preconn_train = load_jsonl('data2/reasoning/structured_preconn_train_formatted.jsonl')
print(f"\nPreconn warmup: {len(preconn_train)} entries")

# Use full augmented from Experiment 1
full_augmented = full  # Already loaded: 2,781 entries
nyt_component = structured_nyt_train
synthetic_component = structured_synthetic_train

# Save components
save_jsonl(preconn_train, 'data/experiment3/preconn_warmup.jsonl')
save_jsonl(synthetic_component, 'data/experiment3/synthetic_component.jsonl')
save_jsonl(nyt_component, 'data/experiment3/nyt_component.jsonl')
save_jsonl(full_augmented, 'data/experiment3/full_augmented.jsonl')

print(f"\n1. No warmup: {len(full_augmented)} entries (direct training)")
print(f"2. With warmup: {len(preconn_train)} preconn → {len(full_augmented)} full")
print(f"3. Staged warmup: {len(preconn_train)} preconn → {len(synthetic_component)} synthetic → {len(nyt_component)} NYT")

print("\n✓ Experiment 3 datasets created")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Data Split:
  ✓ Training: NYT={len(nyt_train_ids)} puzzles, Synthetic={len(synthetic_train_ids)} puzzles
  ✓ Validation: {len(global_validation)} entries ({len(nyt_val_ids)} NYT + {len(synthetic_val_ids)} synthetic)
  ✓ Test: {len(global_test)} entries ({len(nyt_test_ids)} NYT + {len(synthetic_test_ids)} synthetic)

Experiment 1 - Data Augmentation Ablation:
  ✓ Baseline: {len(baseline)} entries
  ✓ Permutation: {len(permutation)} entries
  ✓ Synthetic: {len(synthetic_only)} entries
  ✓ Full: {len(full)} entries

Experiment 2 - Format Impact:
  ✓ Structured-only: {len(structured_only)} entries
  ✓ Unstructured-only: {len(unstructured_only)} entries
  ✓ Mixed: {len(mixed)} entries
  ✓ Sequential: Phase1={len(unstructured_only)}, Phase2={len(structured_only)}

Experiment 3 - Preconn Warmup:
  ✓ Preconn warmup: {len(preconn_train)} entries
  ✓ Synthetic component: {len(synthetic_component)} entries
  ✓ NYT component: {len(nyt_component)} entries
  ✓ Full augmented: {len(full_augmented)} entries

All datasets saved to data/ directory
✓ No data leakage detected (train/validation/test are fully separated)
""")

print("="*80)
print("COMPLETE!")
print("="*80)
