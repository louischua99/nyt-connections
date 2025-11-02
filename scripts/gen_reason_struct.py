#!/usr/bin/env python3
"""
enhanced connections reasoning generation using deepseek api
uses systematic category-based approach from nyt analysis
ASYNC VERSION with concurrent API calls for 10x+ speedup
"""

import json
import os
import random
import time
import asyncio
from pathlib import Path
from typing import List, Dict
from openai import AsyncOpenAI

# deepseek api configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"

# processing configuration
NUM_TRAIN_PERMUTATIONS = 3
TRAIN_TEST_SPLIT = 0.9
BATCH_SIZE = 5
CONCURRENT_REQUESTS = 15  # number of concurrent API calls

# nyt connections category types
CATEGORY_TYPES = """
1. Semantic Taxonomy - types of X, parts of Y, members of category
2. Semantic Synonymy - words with similar meanings
3. Semantic Association - items linked by shared scenario/function
4. Named Entities - proper names (people, places, brands, titles)
5. Collocational/Idiomatic - fill slots in phrases (___X, Y___)
6. Lexical Morphology - shared affixes, compounds, word formation
7. Lexical Orthography - letter patterns (palindromes, anagrams, etc)
8. Phonological Pattern - sound patterns (rhymes, homophones)
9. Grammatical/Syntactic - same part of speech or function
10. Wordplay Double Meaning - polysemy, multiple senses
11. Temporal/Sequential - ordered series
12. Numerical/Quantitative - numbers, counts, measurements
13. Lexical Etymology - shared language origin
14. Sociolinguistic Register - slang, dialect, jargon
15. Cross-Linguistic - translations across languages
"""

client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

def load_puzzles(filename: str) -> List[Dict]:
    with open(filename, 'r') as f:
        return json.load(f)

def split_train_test(puzzles: List[Dict], split_ratio: float = 0.9) -> tuple:
    random.seed(42)
    shuffled = puzzles.copy()
    random.shuffle(shuffled)
    split_point = int(len(shuffled) * split_ratio)
    return shuffled[:split_point], shuffled[split_point:]

def create_permutation(words: List[str], perm_id: int, base_seed: int) -> List[str]:
    random.seed(base_seed + perm_id)
    shuffled = words.copy()
    random.shuffle(shuffled)
    return shuffled

def generate_permutations(puzzles: List[Dict], num_perms: int) -> List[Dict]:
    all_permutations = []
    for puzzle in puzzles:
        all_words = []
        for answer in puzzle['answers']:
            all_words.extend(answer['members'])

        base_seed = hash(tuple(all_words))

        for perm_id in range(1, num_perms + 1):
            permuted_words = create_permutation(all_words, perm_id, base_seed)
            all_permutations.append({
                'id': f"{puzzle['id']}_perm{perm_id}",
                'original_id': puzzle['id'],
                'permutation': perm_id,
                'words': permuted_words,
                'answers': puzzle['answers']
            })
    return all_permutations

def create_reasoning_prompt(words: List[str], answers: List[Dict]) -> str:
    sorted_answers = sorted(answers, key=lambda x: x.get('level', 0))

    answer_groups = []
    for group in sorted_answers:
        answer_groups.append(f"{group['group']}: {', '.join(sorted(group['members']))}")

    prompt = f"""Solve this Connections puzzle by finding 4 groups of 4 related words:
Words: {', '.join(words)}

The correct groups are:
{chr(10).join(answer_groups)}

Your task: Write a structured problem-solving narrative using systematic category checking.

START with: "Looking at these 16 words: {', '.join(words)}. I'll systematically check different connection types from this list: {CATEGORY_TYPES}."

SOLVING FRAMEWORK - Work through categories methodically:

PHASE 1: Quick Visual Scan
"First, let me do a quick scan for obvious patterns..."
- Note any immediate standouts (proper names, numbers, obvious sets)
- Identify potential easy groups

PHASE 2: Systematic Category Checking
List and then work through relevant category types

For each promising category, show your thinking:
"Let me check for [Category Type]..."
"I see [WORD1], [WORD2], [WORD3], [WORD4] - these could be [specific connection]"
"Testing: [WORD1] is [explanation], [WORD2] is [explanation]..."
"Yes, these are all [category]" OR "Actually, [WORD] doesn't fit because..."

PHASE 3: Progressive Narrowing
After finding each group:
"Group found: [CATEGORY]. That leaves me with these 12/8/4 words: [list remaining]"
"With those removed, I can now see..."
"The pattern becomes clearer..."
Relist the promising categories and continue until Phase 4

PHASE 4: Final Group by Elimination
"With only 4 words left: [WORDS]"
"These must be connected by..."
"Let me verify: [explanation of connection]"

KEY SOLVING BEHAVIORS:
- State hypothesis before testing: "Could these be types of...?"
- Show verification: "Let me check: X means..., Y is..."
- Express uncertainty: "Hmm, not sure if..." "Wait, maybe..."
- Backtrack when wrong: "Actually, that doesn't work..."
- Use process of elimination: "Since X, Y, Z are gone..."
- Count remaining words after each group

CONCLUDE with:
"So my four groups are:"

**{sorted_answers[0]['group'].upper()}**: {', '.join(sorted(sorted_answers[0]['members']))}
**{sorted_answers[1]['group'].upper()}**: {', '.join(sorted(sorted_answers[1]['members']))}
**{sorted_answers[2]['group'].upper()}**: {', '.join(sorted(sorted_answers[2]['members']))}
**{sorted_answers[3]['group'].upper()}**: {', '.join(sorted(sorted_answers[3]['members']))}

**CRITICAL: Write as if discovering patterns yourself through systematic checking, never mention being given answers. DON'T WRITE THE PHASE NAMES AND INCLUDE THE FULL CATEGORY LIST**

Here is a gold standard example for you to emulate:
Looking at these 16 words: "TRACTOR", "OUTSTANDING", "COLOSSUS", "MAUSOLEUM", "PYRAMIDS", "SUPERB", "FELLOWSHIP", "TWO TOWERS", "RETURN", "LORD", "EXCELLENT", "PLOW", "LIGHTHOUSE", "COMBINE", "TERRIFIC", "HARROW".
I'll systematically check different connection types from this list:
Semantic Taxonomy - types of X, parts of Y, members of category
Semantic Synonymy - words with similar meanings
Semantic Association - items linked by shared scenario/function
Named Entities - proper names (people, places, brands, titles)
Collocational/Idiomatic - fill slots in phrases (X, Y)
Lexical Morphology - shared affixes, compounds, word formation
Lexical Orthography - letter patterns (palindromes, anagrams, etc)
Phonological Pattern - sound patterns (rhymes, homophones)
Grammatical/Syntactic - same part of speech or function
Wordplay Double Meaning - polysemy, multiple senses
Temporal/Sequential - ordered series
Numerical/Quantitative - numbers, counts, measurements
Lexical Etymology - shared language origin
Sociolinguistic Register - slang, dialect, jargon
Cross-Linguistic - translations across languages

Now, looking at the words: "TRACTOR", "OUTSTANDING", "COLOSSUS", "MAUSOLEUM", "PYRAMIDS", "SUPERB", "FELLOWSHIP", "TWO TOWERS", "RETURN", "LORD", "EXCELLENT", "PLOW", "LIGHTHOUSE", "COMBINE", "TERRIFIC", "HARROW"
Let me identify the first group. I see "EXCELLENT", "OUTSTANDING", "SUPERB", "TERRIFIC" - these are clearly Category 2: Semantic Synonymy - words with similar meanings, all expressing high quality or excellence.
Group 1: EXCELLENT, OUTSTANDING, SUPERB, TERRIFIC (Semantic Synonymy)
Remaining categories most likely to be relevant: Semantic Taxonomy, Named Entities, Semantic Association, Collocational/Idiomatic
Looking at what's left, I notice "COLOSSUS", "MAUSOLEUM", "PYRAMIDS", "LIGHTHOUSE" - these are all Category 1: Semantic Taxonomy - specifically, they're all members of the category "Seven Wonders of the Ancient World" (Colossus of Rhodes, Mausoleum at Halicarnassus, Pyramids of Giza, Lighthouse of Alexandria).
Group 2: COLOSSUS, MAUSOLEUM, PYRAMIDS, LIGHTHOUSE (Semantic Taxonomy - Wonders of the Ancient World)
Remaining categories most likely: Named Entities, Semantic Association, Collocational/Idiomatic
Now I see "TRACTOR", "PLOW", "COMBINE", "HARROW" - these fall under Category 3: Semantic Association - items linked by shared function/scenario, specifically farm equipment used in agriculture.
Group 3: TRACTOR, PLOW, COMBINE, HARROW (Semantic Association - farm equipment)
Remaining category most likely: Named Entities
Finally, "FELLOWSHIP", "TWO TOWERS", "RETURN", "LORD" - these are Category 4: Named Entities - they're all parts of titles from the Lord of the Rings series: "The Fellowship (of the Ring)", "The Two Towers", "The Return (of the King)", and "The Lord (of the Rings)".
Group 4: FELLOWSHIP, TWO TOWERS, RETURN, LORD (Named Entities - Lord of the Rings titles)
So my four groups are:

SYNONYMS FOR EXCELLENT: EXCELLENT, OUTSTANDING, SUPERB, TERRIFIC
ANCIENT WONDERS OF THE WORLD: COLOSSUS, LIGHTHOUSE, MAUSOLEUM, PYRAMIDS
FARM EQUIPMENT: COMBINE, HARROW, PLOW, TRACTOR
LORD OF THE RINGS REFERENCES: FELLOWSHIP, LORD, RETURN, TWO TOWERS"""

    return prompt

async def call_deepseek_api(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    return ""

async def process_puzzle(puzzle: Dict, semaphore: asyncio.Semaphore) -> Dict:
    async with semaphore:  # limit concurrent API calls
        prompt = create_reasoning_prompt(puzzle['words'], puzzle['answers'])
        reasoning = await call_deepseek_api(prompt)

        if reasoning and len(reasoning) > 100:
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

async def process_dataset(puzzles: List[Dict], dataset_name: str) -> tuple:
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")

    # split train/test
    train_puzzles, test_puzzles = split_train_test(puzzles, TRAIN_TEST_SPLIT)
    print(f"Train: {len(train_puzzles)} puzzles")
    print(f"Test: {len(test_puzzles)} puzzles")

    # train: 3x permutations
    print(f"\nGenerating {NUM_TRAIN_PERMUTATIONS} permutations for train set...")
    train_permuted = generate_permutations(train_puzzles, NUM_TRAIN_PERMUTATIONS)
    print(f"Total train examples: {len(train_permuted)}")

    # test: no permutation
    test_data = []
    for puzzle in test_puzzles:
        all_words = []
        for answer in puzzle['answers']:
            all_words.extend(answer['members'])
        test_data.append({
            'id': puzzle['id'],
            'original_id': puzzle['id'],
            'permutation': 0,
            'words': all_words,
            'answers': puzzle['answers']
        })

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    # process train
    print(f"\nProcessing train set ({len(train_permuted)} examples)...")
    print(f"Running {CONCURRENT_REQUESTS} concurrent API requests")
    train_tasks = [process_puzzle(puzzle, semaphore) for puzzle in train_permuted]
    train_results = []
    completed = 0
    for coro in asyncio.as_completed(train_tasks):
        result = await coro
        if result:
            train_results.append(result)
        completed += 1
        if completed % 10 == 0 or completed == len(train_tasks):
            print(f"  {completed}/{len(train_tasks)} ({completed/len(train_tasks)*100:.1f}%) - {len(train_results)} successful")

    # process test
    print(f"\nProcessing test set ({len(test_data)} examples)...")
    test_tasks = [process_puzzle(puzzle, semaphore) for puzzle in test_data]
    test_results = []
    completed = 0
    for coro in asyncio.as_completed(test_tasks):
        result = await coro
        if result:
            test_results.append(result)
        completed += 1
        if completed % 5 == 0 or completed == len(test_tasks):
            print(f"  {completed}/{len(test_tasks)} ({completed/len(test_tasks)*100:.1f}%) - {len(test_results)} successful")

    print(f"\nTrain success: {len(train_results)}/{len(train_permuted)} ({len(train_results)/len(train_permuted)*100:.1f}%)")
    print(f"Test success: {len(test_results)}/{len(test_data)} ({len(test_results)/len(test_data)*100:.1f}%)")

    return train_results, test_results

async def main():
    print("="*60)
    print("DEEPSEEK REASONING GENERATOR V2 (ASYNC)")
    print("Using systematic category-based approach")
    print(f"Using {CONCURRENT_REQUESTS} concurrent API requests")
    print("="*60)

    Path('data2/reasoning').mkdir(parents=True, exist_ok=True)

    # load datasets
    print("\nLoading datasets...")
    connections_puzzles = load_puzzles('data2/connections.json')
    categorical_puzzles = load_puzzles('data2/connections_categorical.json')
    print(f"Loaded {len(connections_puzzles)} real NYT puzzles")
    print(f"Loaded {len(categorical_puzzles)} categorical puzzles")

    start_time = time.time()

    # process real nyt puzzles
    conn_train, conn_test = await process_dataset(connections_puzzles, "Real NYT Connections")

    # process categorical puzzles
    cat_train, cat_test = await process_dataset(categorical_puzzles, "Categorical Synthetic")

    # save results
    print(f"\n{'='*60}")
    print("Saving datasets...")
    print(f"{'='*60}")

    # real puzzles
    with open('data2/reasoning/structured_nyt_train.jsonl', 'w') as f:
        for item in conn_train:
            f.write(json.dumps(item) + '\n')
    print(f"Saved structured_nyt_train.jsonl ({len(conn_train)} examples)")

    with open('data2/reasoning/structured_nyt_test.jsonl', 'w') as f:
        for item in conn_test:
            f.write(json.dumps(item) + '\n')
    print(f"Saved structured_nyt_test.jsonl ({len(conn_test)} examples)")

    # categorical puzzles
    with open('data2/reasoning/structured_synthetic_train.jsonl', 'w') as f:
        for item in cat_train:
            f.write(json.dumps(item) + '\n')
    print(f"Saved structured_synthetic_train.jsonl ({len(cat_train)} examples)")

    with open('data2/reasoning/structured_synthetic_test.jsonl', 'w') as f:
        for item in cat_test:
            f.write(json.dumps(item) + '\n')
    print(f"Saved structured_synthetic_test.jsonl ({len(cat_test)} examples)")

    # summary
    total_time = time.time() - start_time
    total_examples = len(conn_train) + len(conn_test) + len(cat_train) + len(cat_test)

    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Real NYT - Train: {len(conn_train)}, Test: {len(conn_test)}")
    print(f"Categorical - Train: {len(cat_train)}, Test: {len(cat_test)}")
    print(f"Total examples: {total_examples}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average per example: {total_time/total_examples:.1f} seconds")
    print(f"Speedup vs sequential: ~{total_examples*4/total_time:.1f}x")

if __name__ == "__main__":
    asyncio.run(main())