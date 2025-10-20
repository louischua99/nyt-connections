#!/usr/bin/env python3
"""
Connections reasoning generation with guided solutions
Generates plausible human-like chain-of-thought reasoning
"""

import json
import requests
import random
import time
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# OpenRouter configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-r1"
OPENROUTER_API_KEY = "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"

# Processing configuration
BATCH_SIZE = 5
MAX_WORKERS = 2
NUM_PERMUTATIONS = 5


def load_puzzles(filename: str) -> List[Dict]:
    """Load puzzles from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def create_permutation(words: List[str], perm_id: int, base_seed: int) -> List[str]:
    """Create a specific permutation of words"""
    random.seed(base_seed + perm_id)
    shuffled = words.copy()
    random.shuffle(shuffled)
    return shuffled


def generate_puzzle_permutations(puzzles: List[Dict], num_permutations: int) -> List[Dict]:
    """Generate multiple permutations for each puzzle"""
    all_permutations = []
    
    for i, puzzle in enumerate(puzzles):
        if i % 100 == 0:
            print(f"Processing puzzle {i+1}/{len(puzzles)}")
            
        # Extract all words
        all_words = []
        for answer in puzzle['answers']:
            all_words.extend(answer['members'])
        
        base_seed = hash(tuple(all_words))
        
        for perm_id in range(1, num_permutations + 1):
            permuted_words = create_permutation(all_words, perm_id, base_seed)
            
            permuted_puzzle = {
                'id': f"{puzzle['id']}_perm{perm_id}",
                'original_id': puzzle['id'],
                'permutation': perm_id,
                'words': permuted_words,
                'answers': puzzle['answers']
            }
            all_permutations.append(permuted_puzzle)
    
    print(f"Generated {len(all_permutations)} total permutations")
    return all_permutations


def create_reasoning_prompt(words: List[str], answers: List[Dict]) -> str:
    """Create prompt that passes answers but asks for human-like reasoning"""
    
    # Sort answers by difficulty (easiest first, as humans would likely find them)
    sorted_answers = sorted(answers, key=lambda x: x.get('level', 0))
    
    # Format the correct groups
    answer_groups = []
    for group in sorted_answers:
        answer_groups.append(f"{group['group']}: {', '.join(sorted(group['members']))}")
    
    prompt = f"""Solve this Connections puzzle by finding 4 groups of 4 related words:
Words: {', '.join(words)}

The correct groups are:
{chr(10).join(answer_groups)}

Your task: Write a natural problem-solving narrative as if you're exploring and discovering these groups yourself. 

START your response with: "Looking at these 16 words: {', '.join(words)}. "

Pretend you're a person vocallizing their thought process through the puzzle:
- Initial cursory scanning and thinking
- Noticing the first pattern (usually the easiest group)
- Testing connections (having some fail but some successful) 
- Counting remaining words after each group found (12 left, 8 left, 4 left)
- Having "aha!" or "wait..." moments when spotting connections
- Sometimes second-guessing before reasoning it through and confirming and/or dismissing
- Natural phrases like "let me check", "that leaves me with", "these could be"

Write the FULL solving process showing how you work through the puzzle step by step.


ONLY AFTER your complete reasoning, conclude with:
"So my four groups are:"

Then list each group as:
**{sorted_answers[0]['group'].upper()}**: {', '.join(sorted(sorted_answers[0]['members']))}
**{sorted_answers[1]['group'].upper()}**: {', '.join(sorted(sorted_answers[1]['members']))}
**{sorted_answers[2]['group'].upper()}**: {', '.join(sorted(sorted_answers[2]['members']))}
**{sorted_answers[3]['group'].upper()}**: {', '.join(sorted(sorted_answers[3]['members']))}
 
**DO NOT MENTION OR ALLUDE TO ANY HINTS/ANSWER BEING SHOWN PRETEND AS IF YOU ARE FIGURING IT OUT YOURSELF**

Here's an example:
Alright, let me look at these 16 words and find the four groups...
First scan - COMBINE, HARROW, PLOW, TRACTOR... These all sound like farming equipment to me. Yeah, COMBINE is a harvesting machine, HARROW breaks up soil, PLOW turns over the earth, and TRACTOR pulls everything. That's definitely farm machinery.
Now, EXCELLENT, OUTSTANDING, SUPERB, TERRIFIC - these are easy! They're all synonyms meaning "great" or "very good." That's a clear group.
Let me see... COLOSSUS, LIGHTHOUSE, MAUSOLEUM, PYRAMIDS. Hmm, what do these have in common? They're all... structures? Wait, I think I know - aren't these all Wonders of the Ancient World? Let me think... Colossus of Rhodes, Lighthouse of Alexandria, Mausoleum at Halicarnassus, and the Great Pyramids of Giza. Yes! That's it!
So that leaves FELLOWSHIP, LORD, RETURN, TWO TOWERS. Oh, this is clever! These are Lord of the Rings movies! "The Fellowship of the Ring," "The Two Towers," and "The Return of the King." But wait... "LORD" by itself? Oh! It must be referring to the title words - "The LORD of the Rings" is in all the movie titles.
Actually, wait. Let me reconsider. FELLOWSHIP, TWO TOWERS, RETURN... these could be the shortened names of the three Lord of the Rings movies. But then what about LORD? Hmm...
Oh! I think I've been overthinking it. These might just be the key words from the Lord of the Rings trilogy titles:

The FELLOWSHIP of the Ring
The TWO TOWERS
The RETURN of the King
And LORD from "Lord of the Rings"

So my four groups are:
FARM EQUIPMENT: COMBINE, HARROW, PLOW, TRACTOR
SYNONYMS FOR EXCELLENT: EXCELLENT, OUTSTANDING, SUPERB, TERRIFIC
ANCIENT WONDERS OF THE WORLD: COLOSSUS, LIGHTHOUSE, MAUSOLEUM, PYRAMIDS
LORD OF THE RINGS REFERENCES: FELLOWSHIP, LORD, RETURN, TWO TOWERS
"""

    return prompt


def call_openrouter(prompt: str, temperature: float = 0.7) -> str:
    """Call OpenRouter API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/nikerlas/connections",
        "X-Title": "Connections Reasoning Generator"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            if 'message' in choice:
                return choice['message'].get('content', '').strip()
            else:
                return choice.get('text', '').strip()
        
        return ""
    except Exception as e:
        print(f"API error: {e}")
        return ""


def process_puzzle(puzzle: Dict, max_retries: int = 2) -> Dict:
    """Process a single puzzle"""
    
    for attempt in range(max_retries):
        prompt = create_reasoning_prompt(puzzle['words'], puzzle['answers'])
        reasoning = call_openrouter(prompt, temperature=0.7)
        
        if reasoning and len(reasoning) > 100:
            # Create training example with just the puzzle (no answer in user message)
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Solve this Connections puzzle by finding 4 groups of 4 related words:\nWords: {', '.join(puzzle['words'])}"
                    },
                    {
                        "role": "assistant",
                        "content": reasoning
                    }
                ],
                "metadata": {
                    "puzzle_id": puzzle['id'],
                    "original_id": puzzle.get('original_id', puzzle['id']),
                    "permutation": puzzle.get('permutation', 0),
                    "reasoning_length": len(reasoning)
                }
            }
    
    return None


def process_batch_sequential(puzzles: List[Dict]) -> List[Dict]:
    """Process puzzles sequentially"""
    results = []
    
    for puzzle in puzzles:
        try:
            print(f"Processing {puzzle['id']}...")
            result = process_puzzle(puzzle)
            if result:
                results.append(result)
                print(f"✓ Processed {puzzle['id']}")
            else:
                print(f"✗ Failed {puzzle['id']}")
        except Exception as e:
            print(f"✗ Error on {puzzle['id']}: {e}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate reasoning data for Connections puzzles')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--test-mode', action='store_true', help='Process only first 10 puzzles for testing')
    args = parser.parse_args()
    
    print("="*60)
    print("CONNECTIONS REASONING GENERATOR")
    print("="*60)
    
    # Create output directories
    Path('data/output').mkdir(parents=True, exist_ok=True)
    
    # Load puzzles
    print("\nLoading puzzles...")
    puzzles = load_puzzles('data/connections.json')
    print(f"Loaded {len(puzzles)} puzzles")
    
    # Test mode
    if args.test_mode:
        puzzles = puzzles[:2]
        print(f"TEST MODE: Using only {len(puzzles)} puzzles")
    
    # Generate permutations
    print("\nGenerating permutations...")
    all_permuted = generate_puzzle_permutations(puzzles, NUM_PERMUTATIONS)
    print(f"Total examples to process: {len(all_permuted)}")
    
    # Process all puzzles
    all_results = []
    start_time = time.time()
    
    # Process in batches
    for i in range(0, len(all_permuted), args.batch_size):
        batch = all_permuted[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (len(all_permuted) + args.batch_size - 1) // args.batch_size
        
        print(f"\nBatch {batch_num}/{total_batches}")
        batch_results = process_batch_sequential(batch)
        all_results.extend(batch_results)
        
        # Progress stats
        print(f"Progress: {len(all_results)}/{len(all_permuted)} ({len(all_results)/len(all_permuted)*100:.1f}%)")
    
    # Save results
    print("\nSaving dataset...")
    
    # Full dataset
    with open('data/output/connections_reasoning.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    with open('data/output/connections_reasoning.jsonl', 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    # Train/test split
    random.seed(42)
    random.shuffle(all_results)
    split_point = int(len(all_results) * 0.9)
    
    train_data = all_results[:split_point]
    test_data = all_results[split_point:]
    
    with open('data/output/connections_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('data/output/connections_test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"Total examples: {len(all_results)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Success rate: {len(all_results)/len(all_permuted)*100:.1f}%")


if __name__ == "__main__":
    main()