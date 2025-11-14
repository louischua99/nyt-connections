#!/usr/bin/env python3
"""
Evaluate puzzle solving success using ground truth matching
Outputs results to CSV file
"""

import json
import re
import csv
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Set
from datetime import datetime


## Data Checks

def normalize_word(word: str) -> str:
    """Normalize a word for comparison (lowercase, strip whitespace)"""
    return word.strip().lower()


def normalize_group(group: List[str]) -> Set[str]:
    """Convert a group of words to a normalized set for comparison"""
    return set(normalize_word(w) for w in group)


## Extraction Functions

def extract_final_answer(text):
    """
    Extract final answer based on <think> tags or </think> delimiter

    Models like Qwen3-4B-Thinking-2507 use </think> as a delimiter token to separate
    reasoning from final answer, without an opening <think> tag.

    Cases handled:
    - If </think> exists (with or without <think>): return everything after </think>
    - If <think> exists but no </think>: return empty (incomplete reasoning)
    - If neither exist: return full text (no reasoning tags)
    """
    if not text:
        return ""

    has_think_open = '<think>' in text
    has_think_close = '</think>' in text

    # Case 1: </think> delimiter present - extract everything after it
    if has_think_close:
        parts = text.split('</think>')
        return parts[-1].strip()

    # Case 2: incomplete reasoning - <think> without </think>
    if has_think_open and not has_think_close:
        return ""

    # Case 3: no reasoning tags - use full text
    return text.strip()


def extract_ground_truth_groups(text: str, verbose: bool = False) -> List[List[str]]:
    """
    Extract groups from ground truth text which should be in a standard format
    Returns list of 4 groups, each containing 4 words
    """
    if not text:
        return []

    # Extract final answer
    final_answer = extract_final_answer(text)
    if not final_answer:
        return []

    groups = []

    # Primary pattern for ground truth format: **CATEGORY**: WORD1, WORD2, WORD3, WORD4
    pattern = r'\*\*[^*]+\*\*:\s*([^,\n]+),\s*([^,\n]+),\s*([^,\n]+),\s*([^,\n]+)'
    matches = re.findall(pattern, final_answer)

    for match in matches:
        if len(match) == 4:
            # Clean and store the words
            group = [word.strip() for word in match if word.strip()]
            if len(group) == 4:
                groups.append(group)

    # Fallback: Look for any line with 4 comma-separated words after a colon
    if len(groups) < 4:
        lines = final_answer.split('\n')
        for line in lines:
            if ':' in line:
                # Get everything after the colon
                words_part = line.split(':', 1)[-1].strip()
                # Split by commas
                words = [w.strip() for w in words_part.split(',')]
                # Clean each word - remove extra punctuation but keep the word
                cleaned_words = []
                for word in words:
                    # Remove leading/trailing non-alphanumeric chars except hyphens and apostrophes
                    cleaned = re.sub(r'^[^\w\-\']+|[^\w\-\']+$', '', word, flags=re.UNICODE)
                    if cleaned:
                        cleaned_words.append(cleaned)

                if len(cleaned_words) == 4:
                    # Check if this group is not already added
                    is_duplicate = False
                    for existing_group in groups:
                        if normalize_group(cleaned_words) == normalize_group(existing_group):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        groups.append(cleaned_words)

    if verbose and len(groups) != 4:
        print(f"  [WARNING] Extracted {len(groups)} groups from ground truth instead of 4")

    return groups[:4] if len(groups) >= 4 else groups


def extract_predicted_groups_from_final_answer(text: str, verbose: bool = False) -> List[List[str]]:
    """
    Extract the predicted groups from the final answer section (after </think>).
    Only looks at the end of the text where the final answer typically is.
    """
    if not text:
        return []

    final_answer = extract_final_answer(text)
    if not final_answer:
        return []

    # Take the last 1000 characters - this should contain the final answer
    final_section = final_answer[-1000:]

    groups = []

    # Pattern for lines with 4 comma-separated words (possibly with category labels)
    # Examples:
    # **Group 1**: WORD1, WORD2, WORD3, WORD4
    # | 1 | WORD1, WORD2, WORD3, WORD4 |
    # Group 1: WORD1, WORD2, WORD3, WORD4
    lines = final_section.split('\n')

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Try to find 4 comma-separated words
        # First, extract the part after a colon or pipe if present
        parts = line.split('|')  # For markdown tables
        for part in parts:
            # Remove markdown formatting
            cleaned = part.replace('**', '').replace('*', '')

            # Remove LaTeX formatting like \boxed{...}, \quad, etc.
            cleaned = re.sub(r'\\boxed\{', '', cleaned)
            cleaned = re.sub(r'\$+', '', cleaned)  # Remove $ signs
            cleaned = re.sub(r'\\[a-zA-Z]+', '', cleaned)  # Remove LaTeX commands like \quad
            cleaned = cleaned.replace('}', '').replace('{', '')

            # Split by colon to get the words part
            if ':' in cleaned:
                words_part = cleaned.split(':', 1)[-1]
            else:
                words_part = cleaned

            # Split by commas
            words = [w.strip() for w in words_part.split(',')]

            # Clean each word - remove extra punctuation
            cleaned_words = []
            for word in words:
                # Remove parenthetical explanations like "(WOOD" or "(ALL CAN BE..."
                word = re.sub(r'\([^)]*$', '', word)  # Remove unclosed parentheses
                word = re.sub(r'\([^)]*\)', '', word)  # Remove closed parentheses

                # Remove leading/trailing non-alphanumeric chars except hyphens and apostrophes
                cleaned_word = re.sub(r'^[^\w\-\']+|[^\w\-\']+$', '', word, flags=re.UNICODE)

                # Skip if word contains explanatory text (usually longer than a few words)
                if ' ' in cleaned_word and len(cleaned_word.split()) > 3:
                    continue

                if cleaned_word and len(cleaned_word) > 1:  # At least 2 chars
                    cleaned_words.append(cleaned_word.upper())

            # If we found exactly 4 words, add as a group
            if len(cleaned_words) == 4:
                # Check if this group is not already added
                is_duplicate = False
                for existing_group in groups:
                    if normalize_group(cleaned_words) == normalize_group(existing_group):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    groups.append(cleaned_words)
                    if len(groups) == 4:  # Stop once we have 4 groups
                        break

        if len(groups) == 4:
            break

    return groups


def find_groups_in_prediction(ground_truth_groups: List[List[str]],
                            prediction_text: str,
                            verbose: bool = False) -> List[bool]:
    """
    Check which ground truth groups appear in the prediction.

    Extracts predicted groups from the final answer section and compares
    them to ground truth groups.
    """
    if not prediction_text or not ground_truth_groups:
        return [False] * len(ground_truth_groups)

    # Extract predicted groups from the final answer
    predicted_groups = extract_predicted_groups_from_final_answer(prediction_text, verbose)

    if verbose and predicted_groups:
        print(f"    Extracted {len(predicted_groups)} predicted groups:")
        for i, group in enumerate(predicted_groups):
            print(f"      {i+1}. {group}")

    found_groups = []

    # For each ground truth group, check if it matches any predicted group
    for gt_group in ground_truth_groups:
        normalized_gt = normalize_group(gt_group)
        found = False

        for pred_group in predicted_groups:
            normalized_pred = normalize_group(pred_group)
            if normalized_gt == normalized_pred:
                found = True
                if verbose:
                    print(f"    ✓ Ground truth {gt_group} matches predicted {pred_group}")
                break

        if not found and verbose:
            print(f"    ✗ Ground truth {gt_group} not found in predictions")

        found_groups.append(found)

    return found_groups


## Scoring

def em_scoring_by_matching(ground_truth_groups: List[List[str]],
                          prediction_text: str,
                          verbose: bool = False) -> Tuple[float, int]:
    """
    Calculate EM score by finding ground truth groups in prediction
    Returns (score, correct_groups) where score is 0.0-1.0
    """
    if not ground_truth_groups or not prediction_text:
        return 0.0, 0

    # Find which groups appear in the prediction
    found_groups = find_groups_in_prediction(ground_truth_groups, prediction_text, verbose)

    # Calculate score
    correct_groups = sum(found_groups)
    score = correct_groups * 0.25  # Each group is worth 0.25

    return score, correct_groups


def process_single_prediction(
    pred: Dict,
    verbose: bool
) -> Tuple[str, Dict]:
    """Process a single prediction"""
    puzzle_id = pred.get('puzzle_id', 'unknown')
    prediction_text = pred.get('prediction', '')
    ground_truth_text = pred.get('ground_truth', '')

    if verbose:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] PROCESSING | Puzzle: {puzzle_id}")

    # Extract ground truth groups
    ground_truth_groups = extract_ground_truth_groups(ground_truth_text, verbose)

    if len(ground_truth_groups) != 4:
        if verbose:
            print(f"  [WARNING] Puzzle {puzzle_id}: Could not extract 4 groups from ground truth (got {len(ground_truth_groups)})")

    # Extract predicted groups
    predicted_groups = extract_predicted_groups_from_final_answer(prediction_text, verbose)

    # Score by finding ground truth groups in prediction
    score, correct_groups = em_scoring_by_matching(ground_truth_groups, prediction_text, verbose)

    result = {
        'score': score,
        'correct_groups': correct_groups,
        'total_groups': 4,
        'extraction_success': len(ground_truth_groups) == 4,
        'ground_truth_groups': ground_truth_groups,
        'predicted_groups': predicted_groups
    }

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] SCORED | Puzzle: {puzzle_id} | Score: {score:.2f} | Correct: {correct_groups}/4")

    if verbose and correct_groups < 4:
        print(f"  [DEBUG] Puzzle {puzzle_id}: Missing groups:")
        found_groups = find_groups_in_prediction(ground_truth_groups, prediction_text, False)
        for i, (group, found) in enumerate(zip(ground_truth_groups, found_groups)):
            if not found:
                print(f"    Group {i+1} not found: {group}")

    return puzzle_id, result


def evaluate_predictions_file(
    filepath: Path,
    verbose: bool
) -> Dict:
    """Evaluate a single predictions JSON file"""
    print(f"\nEvaluating: {filepath.name}")

    if not filepath.exists():
        print(f"  File not found, skipping")
        return {}

    with open(filepath, 'r') as f:
        predictions = json.load(f)

    print(f"  Loaded {len(predictions)} predictions")

    # Process all predictions
    scores = {}
    total_score = 0
    total_count = 0
    perfect_puzzles = 0
    failed_extractions = 0
    puzzle_id_counts = {}  # Track occurrences of each puzzle_id

    for pred in predictions:
        puzzle_id, result = process_single_prediction(pred, verbose)

        # Handle duplicate puzzle IDs (e.g., synthetic vs NYT with same ID)
        # by appending a counter to make them unique
        original_puzzle_id = puzzle_id
        if puzzle_id in scores:
            # This puzzle_id already exists, create a unique key
            puzzle_id_counts[original_puzzle_id] = puzzle_id_counts.get(original_puzzle_id, 1) + 1
            puzzle_id = f"{original_puzzle_id}_dup{puzzle_id_counts[original_puzzle_id]}"

        scores[puzzle_id] = result

        total_score += result['score']
        total_count += 1
        if result['score'] == 1.0:
            perfect_puzzles += 1
        if not result['extraction_success']:
            failed_extractions += 1

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {filepath.name}")
    print(f"{'='*80}")
    if total_count > 0:
        avg_score = total_score / total_count
        print(f"  Total puzzles: {total_count}")
        print(f"  Average score: {avg_score:.4f} ({avg_score*100:.2f}%)")
        print(f"  Perfect puzzles: {perfect_puzzles}/{total_count} ({perfect_puzzles/total_count*100:.2f}%)")
        if failed_extractions > 0:
            print(f"  Failed GT extractions: {failed_extractions}/{total_count} ({failed_extractions/total_count*100:.2f}%)")
    print(f"{'='*80}\n")

    return scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate predictions by matching ground truth groups')
    parser.add_argument('--predictions-dir', default='data/predictions_validation',
                       help='Directory with prediction JSON files')
    parser.add_argument('--output', default='data/evaluation_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--pattern', default='*.json',
                       help='File pattern to match (default: *.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose debug output')
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)

    print("="*80)
    print(f"NYT Connections Evaluation (Ground Truth Matching)")
    print("="*80)

    # Find prediction files
    if '*' in args.pattern:
        prediction_files = list(predictions_dir.glob(args.pattern))
    else:
        # If no wildcard, treat as exact filename
        prediction_files = [predictions_dir / args.pattern] if (predictions_dir / args.pattern).exists() else []

    if not prediction_files:
        print(f"\nNo prediction files matching '{args.pattern}' found in {predictions_dir}")
        all_files = list(predictions_dir.glob("*.json"))
        if all_files:
            print(f"Found {len(all_files)} JSON files in directory:")
            for f in sorted(all_files)[:10]:
                print(f"  - {f.name}")
            if len(all_files) > 10:
                print(f"  ... and {len(all_files) - 10} more")
        return

    print(f"\nFound {len(prediction_files)} matching prediction files:")
    for f in sorted(prediction_files):
        print(f"  - {f.name}")

    # Evaluate each file
    all_results = {}

    for filepath in sorted(prediction_files):
        model_name = filepath.stem
        scores = evaluate_predictions_file(filepath, args.verbose)
        all_results[model_name] = scores

    # Write detailed results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Writing results to: {output_path}")

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model_name', 'puzzle_id', 'score', 'correct_groups'
        ])

        for model_name, scores in sorted(all_results.items()):
            # Sort puzzle IDs, handling various formats (numeric, string, with _perm, with _dup)
            def sort_key(item):
                puzzle_id = str(item[0])
                # Extract numeric part for sorting
                import re
                match = re.match(r'(\d+)', puzzle_id)
                if match:
                    return (int(match.group(1)), puzzle_id)  # Sort by number first, then full string
                return (float('inf'), puzzle_id)  # Non-numeric IDs go to the end

            for puzzle_id, result in sorted(scores.items(), key=sort_key):
                writer.writerow([
                    model_name, puzzle_id, result['score'],
                    result['correct_groups']
                ])

    # Write summary - derive filename from output_path to keep test and validation separate
    summary_filename = output_path.stem.replace('evaluation_results', 'evaluation_summary') + '.csv'
    summary_path = output_path.parent / summary_filename
    print(f"Writing summary to: {summary_path}")

    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model_name', 'total_puzzles', 'avg_score_%',
            'perfect_puzzles', 'total_correct_groups'
        ])

        for model_name, scores in sorted(all_results.items()):
            if not scores:
                continue

            total_score = sum(s['score'] for s in scores.values())
            total_puzzles = len(scores)
            perfect_count = sum(1 for s in scores.values() if s['score'] == 1.0)
            total_correct_groups = sum(s['correct_groups'] for s in scores.values())
            avg_score = total_score / total_puzzles if total_puzzles > 0 else 0

            writer.writerow([
                model_name,
                total_puzzles,
                f"{avg_score*100:.2f}%",
                f"{perfect_count}/{total_puzzles} ({perfect_count/total_puzzles*100:.1f}%)" if total_puzzles > 0 else "0/0 (0.0%)",
                total_correct_groups
            ])

    # Write detailed extraction results as JSON - derive filename from output_path
    extraction_filename = output_path.stem.replace('evaluation_results', 'evaluation_extractions') + '.json'
    extraction_path = output_path.parent / extraction_filename
    print(f"Writing extraction details to: {extraction_path}")

    extraction_details = {}
    for model_name, scores in sorted(all_results.items()):
        extraction_details[model_name] = {}

        # Use same sort_key function as above
        def sort_key_ext(item):
            puzzle_id = str(item[0])
            import re
            match = re.match(r'(\d+)', puzzle_id)
            if match:
                return (int(match.group(1)), puzzle_id)
            return (float('inf'), puzzle_id)

        for puzzle_id, result in sorted(scores.items(), key=sort_key_ext):
            extraction_details[model_name][str(puzzle_id)] = {
                'puzzle_id': puzzle_id,
                'score': result['score'],
                'correct_groups': result['correct_groups'],
                'total_groups': result['total_groups'],
                'ground_truth_groups': result.get('ground_truth_groups', []),
                'predicted_groups': result.get('predicted_groups', [])
            }

    with open(extraction_path, 'w') as f:
        json.dump(extraction_details, f, indent=2)

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"Detailed results: {output_path}")
    print(f"Summary results: {summary_path}")
    print(f"Extraction details: {extraction_path}")
    print("="*80)


if __name__ == '__main__':
    main()