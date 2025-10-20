#!/usr/bin/env python3
"""
unstructured reasoning generation using deepseek api
processes ALL ~800 original + ALL 200 synthetic puzzles
no permutation, no train/test split - just generates all ~1000 examples
"""

import json
import time
from pathlib import Path
from typing import List, Dict
from openai import OpenAI

# deepseek api configuration
DEEPSEEK_API_KEY = "YOUR_API_KEY_HERE"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

def load_puzzles(filename: str) -> List[Dict]:
    with open(filename, 'r') as f:
        return json.load(f)

def create_reasoning_prompt(words: List[str], answers: List[Dict]) -> str:
    sorted_answers = sorted(answers, key=lambda x: x.get('level', 0))

    answer_groups = []
    for group in sorted_answers:
        answer_groups.append(f"{group['group']}: {', '.join(sorted(group['members']))}")

    prompt = f"""Solve this Connections puzzle by finding 4 groups of 4 related words:
Words: {', '.join(words)}

The correct groups are:
{chr(10).join(answer_groups)}

Your task: Write a natural problem-solving narrative as if you're exploring and discovering these groups yourself.

START your response with: "Looking at these 16 words: {', '.join(words)}. "

Pretend you're a person vocalizing their thought process through the puzzle:
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

def call_deepseek_api(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return ""

def process_puzzle(puzzle: Dict) -> Dict:
    all_words = []
    for answer in puzzle['answers']:
        all_words.extend(answer['members'])

    prompt = create_reasoning_prompt(all_words, puzzle['answers'])
    reasoning = call_deepseek_api(prompt)

    if reasoning and len(reasoning) > 100:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"Solve this Connections puzzle by finding 4 groups of 4 related words:\nWords: {', '.join(all_words)}"
                },
                {
                    "role": "assistant",
                    "content": reasoning
                }
            ],
            "metadata": {
                "puzzle_id": puzzle['id'],
                "reasoning_length": len(reasoning)
            }
        }
    return None

def process_dataset(puzzles: List[Dict], dataset_name: str) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")
    print(f"Total puzzles: {len(puzzles)}")

    results = []
    for i, puzzle in enumerate(puzzles):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(puzzles)} ({(i+1)/len(puzzles)*100:.1f}%)")

        result = process_puzzle(puzzle)
        if result:
            results.append(result)
        time.sleep(0.5)

    print(f"\nSuccess: {len(results)}/{len(puzzles)} ({len(results)/len(puzzles)*100:.1f}%)")
    return results

def main():
    print("="*60)
    print("UNSTRUCTURED REASONING GENERATOR")
    print("Processes ALL puzzles (no permutation, no split)")
    print("="*60)

    Path('data2/output').mkdir(parents=True, exist_ok=True)

    # load datasets
    print("\nLoading datasets...")
    connections_puzzles = load_puzzles('data2/connections.json')
    categorical_puzzles = load_puzzles('data2/connections_categorical.json')
    print(f"Loaded {len(connections_puzzles)} real NYT puzzles")
    print(f"Loaded {len(categorical_puzzles)} categorical puzzles")

    start_time = time.time()

    # process all real nyt puzzles
    conn_results = process_dataset(connections_puzzles, "Real NYT Connections")

    # process all categorical puzzles
    cat_results = process_dataset(categorical_puzzles, "Categorical Synthetic")

    # save results
    print(f"\n{'='*60}")
    print("Saving datasets...")
    print(f"{'='*60}")

    # real puzzles
    with open('data2/output/unstructured_nyt.jsonl', 'w') as f:
        for item in conn_results:
            f.write(json.dumps(item) + '\n')
    print(f"Saved unstructured_nyt.jsonl ({len(conn_results)} examples)")

    # categorical puzzles
    with open('data2/output/unstructured_synthetic.jsonl', 'w') as f:
        for item in cat_results:
            f.write(json.dumps(item) + '\n')
    print(f"Saved unstructured_synthetic.jsonl ({len(cat_results)} examples)")

    # summary
    total_time = time.time() - start_time
    total_examples = len(conn_results) + len(cat_results)

    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Real NYT: {len(conn_results)} examples")
    print(f"Categorical: {len(cat_results)} examples")
    print(f"Total examples: {total_examples}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average per example: {total_time/total_examples:.1f} seconds")

if __name__ == "__main__":
    main()
