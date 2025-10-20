#!/usr/bin/env python3
"""
structured reasoning generation for preconn categorical puzzles
processes all patterns: 4:1, 5:2, 7:3, 8:2:2, 10:3:3
uses deepseek api with systematic category-based approach
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict
from openai import OpenAI

# deepseek api configuration
DEEPSEEK_API_KEY = "YOUR_API_KEY_HERE"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"

# processing configuration
TRAIN_TEST_SPLIT = 0.9

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

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

def load_preconn_data(filename: str) -> List[Dict]:
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['examples']

def split_train_test(examples: List[Dict], split_ratio: float = 0.9) -> tuple:
    random.seed(42)
    shuffled = examples.copy()
    random.shuffle(shuffled)
    split_point = int(len(shuffled) * split_ratio)
    return shuffled[:split_point], shuffled[split_point:]

def extract_words_from_input(input_text: str) -> List[str]:
    # extract words from various input formats
    if "Pick the odd word out:" in input_text or "Pick the odd words out:" in input_text:
        parts = input_text.split(": ", 1)
        if len(parts) == 2:
            words_str = parts[1]
            return [w.strip() for w in words_str.split(',')]
    elif "There are 3 word groups" in input_text:
        parts = input_text.split(": ", 1)
        if len(parts) == 2:
            words_str = parts[1]
            return [w.strip() for w in words_str.split(',')]
    return []

def get_odd_words(target_scores: Dict[str, int]) -> List[str]:
    return [word for word, score in target_scores.items() if score == 1]

def create_reasoning_prompt_4_1(words: List[str], odd_word: str, explanation: str) -> str:
    return f"""Solve this word puzzle by finding the odd word out:
Words: {', '.join(words)}

The correct answer is: {odd_word}
Pattern explanation: {explanation}

Your task: Write a concise problem-solving narrative using category analysis.

START with: "Looking at these {len(words)} words: {', '.join(words)}. I'll check which category connects 4 of them."

APPROACH:
1. List the category types: Semantic Taxonomy, Semantic Synonymy, Semantic Association, Named Entities, Collocational/Idiomatic, Lexical Morphology, Lexical Orthography, Phonological Pattern, Grammatical/Syntactic, Wordplay Double Meaning, Temporal/Sequential, Numerical/Quantitative, Lexical Etymology, Sociolinguistic Register, Cross-Linguistic
2. Identify which category type connects 4 words and name the specific pattern
3. Note which word doesn't fit

CONCLUDE with: "Therefore, the odd word out is: {odd_word}"

Example:
Looking at these 5 words: GARDEN, STAR, FACE, SALT, STRATUS. I'll check which category connects 4 of them.

Category types: Semantic Taxonomy, Semantic Synonymy, Semantic Association, Named Entities, Collocational/Idiomatic, Lexical Morphology, Lexical Orthography, Phonological Pattern, Grammatical/Syntactic, Wordplay Double Meaning, Temporal/Sequential, Numerical/Quantitative, Lexical Etymology, Sociolinguistic Register, Cross-Linguistic

I notice GARDEN, STAR, FACE, and SALT could follow a pattern. Checking Collocational/Idiomatic - can these complete a phrase? Yes, they all work as "rock___" compounds: rock GARDEN, rock STAR, rock FACE, rock SALT.

STRATUS doesn't fit this pattern - it's a cloud type, not a "rock___" compound.

Therefore, the odd word out is: STRATUS"""

def create_reasoning_prompt_5_2(words: List[str], odd_words: List[str], explanation: str) -> str:
    main_count = len(words) - len(odd_words)

    return f"""Solve this word puzzle by finding the odd words out:
Words: {', '.join(words)}

The correct answer is: {', '.join(sorted(odd_words))}
Pattern explanation: {explanation}

Your task: Write a concise problem-solving narrative using category analysis.

START with: "Looking at these {len(words)} words: {', '.join(words)}. I need to find the main group of {main_count} and identify the {len(odd_words)} that don't fit."

APPROACH:
1. List the category types: Semantic Taxonomy, Semantic Synonymy, Semantic Association, Named Entities, Collocational/Idiomatic, Lexical Morphology, Lexical Orthography, Phonological Pattern, Grammatical/Syntactic, Wordplay Double Meaning, Temporal/Sequential, Numerical/Quantitative, Lexical Etymology, Sociolinguistic Register, Cross-Linguistic
2. Identify which category connects the main group of {main_count} words and name the specific pattern
3. Identify what pattern the {len(odd_words)} outliers share (they form a smaller group)

CONCLUDE with: "Therefore, the odd words out are: {', '.join(sorted(odd_words))}"

Example:
Looking at these 7 words: SUPPLEMENTARY, REFLEX, COMPUTER, ADJACENT, RIGHT, ACUTE, SONIC. I need to find the main group of 5 and identify the 2 that don't fit.

Category types: Semantic Taxonomy, Semantic Synonymy, Semantic Association, Named Entities, Collocational/Idiomatic, Lexical Morphology, Lexical Orthography, Phonological Pattern, Grammatical/Syntactic, Wordplay Double Meaning, Temporal/Sequential, Numerical/Quantitative, Lexical Etymology, Sociolinguistic Register, Cross-Linguistic

I notice SUPPLEMENTARY, REFLEX, ADJACENT, RIGHT, ACUTE - these could be types of angles. Checking Semantic Taxonomy: SUPPLEMENTARY angle (180 degrees), REFLEX angle (greater than 180), ADJACENT angles (next to each other), RIGHT angle (90 degrees), ACUTE angle (less than 90 degrees). Yes, 5 words are angle-related terms.

That leaves COMPUTER and SONIC. What do these share? Checking Collocational/Idiomatic - they both work with the prefix "super___": SUPERCOMPUTER and SUPERSONIC. These 2 words form a "super___" prefix pattern.

Therefore, the odd words out are: COMPUTER, SONIC"""

def create_reasoning_prompt_7_3(words: List[str], odd_words: List[str], explanation: str) -> str:
    main_count = len(words) - len(odd_words)

    return f"""Solve this word puzzle by finding the odd words out:
Words: {', '.join(words)}

The correct answer is: {', '.join(sorted(odd_words))}
Pattern explanation: {explanation}

Your task: Write a concise problem-solving narrative using category analysis.

START with: "Looking at these {len(words)} words: {', '.join(words)}. I need to find the main group of {main_count} and identify the {len(odd_words)} that don't fit."

APPROACH:
1. List the category types: Semantic Taxonomy, Semantic Synonymy, Semantic Association, Named Entities, Collocational/Idiomatic, Lexical Morphology, Lexical Orthography, Phonological Pattern, Grammatical/Syntactic, Wordplay Double Meaning, Temporal/Sequential, Numerical/Quantitative, Lexical Etymology, Sociolinguistic Register, Cross-Linguistic
2. Identify which category connects the main group of {main_count} words and name the specific pattern
3. Identify what pattern the {len(odd_words)} outliers share (they form a smaller group)

CONCLUDE with: "Therefore, the odd words out are: {', '.join(sorted(odd_words))}"

Example:
Looking at these 10 words: ARES, ATHENA, HADES, ZEUS, PILLOW, APHRODITE, BOTTLE, HERA, SUMMER, APOLLO. I need to find the main group of 7 and identify the 3 that don't fit.

Category types: Semantic Taxonomy, Semantic Synonymy, Semantic Association, Named Entities, Collocational/Idiomatic, Lexical Morphology, Lexical Orthography, Phonological Pattern, Grammatical/Syntactic, Wordplay Double Meaning, Temporal/Sequential, Numerical/Quantitative, Lexical Etymology, Sociolinguistic Register, Cross-Linguistic

I notice ARES, ATHENA, HADES, ZEUS, APHRODITE, HERA, APOLLO - these are all Greek gods. Checking Named Entities: ARES (god of war), ATHENA (goddess of wisdom), HADES (god of underworld), ZEUS (king of gods), APHRODITE (goddess of love), HERA (queen of gods), APOLLO (god of sun). Yes, 7 words are Greek deities.

That leaves PILLOW, BOTTLE, SUMMER. What do these share? Checking Lexical Orthography - they all contain double consonants: PILLOW (LL), BOTTLE (TT), SUMMER (MM). These 3 words share a double consonant pattern.

Therefore, the odd words out are: BOTTLE, PILLOW, SUMMER"""

def create_reasoning_prompt_groups(words: List[str], explanation: str, num_groups: int) -> str:
    return f"""Solve this word puzzle by identifying {num_groups} word groups and their themes:
Words: {', '.join(words)}

The correct groups are:
{explanation}

Your task: Write a concise problem-solving narrative using category analysis.

START with: "Looking at these {len(words)} words: {', '.join(words)}. I need to identify {num_groups} distinct groups."

APPROACH:
1. List the category types: Semantic Taxonomy, Semantic Synonymy, Semantic Association, Named Entities, Collocational/Idiomatic, Lexical Morphology, Lexical Orthography, Phonological Pattern, Grammatical/Syntactic, Wordplay Double Meaning, Temporal/Sequential, Numerical/Quantitative, Lexical Etymology, Sociolinguistic Register, Cross-Linguistic
2. Scan words and identify first group - which category type and specific pattern
3. Remove those words, identify second group - category type and pattern
4. Identify third group from remaining words
5. State all {num_groups} groups clearly with their themes

CONCLUDE with all {num_groups} groups and their themes based on the explanation format.

Example:
Looking at these 12 words: OJI, SOFU, TITUS, MARACAS, HAHA, ANI, ITOKO, CHICHI, BONGO, KAZOKU, SOBO, RICHARD. I need to identify 3 distinct groups.

Category types: Semantic Taxonomy, Semantic Synonymy, Semantic Association, Named Entities, Collocational/Idiomatic, Lexical Morphology, Lexical Orthography, Phonological Pattern, Grammatical/Syntactic, Wordplay Double Meaning, Temporal/Sequential, Numerical/Quantitative, Lexical Etymology, Sociolinguistic Register, Cross-Linguistic

I notice OJI, SOFU, HAHA, ANI, ITOKO, CHICHI, KAZOKU, SOBO - these look like Japanese words. Checking Cross-Linguistic: these are family members in Japanese. That's 8 words in group 1.

Remaining: TITUS, MARACAS, BONGO, RICHARD. I see TITUS and RICHARD - these could be Shakespeare plays. Checking Named Entities: "Titus Andronicus" and "Richard III" are Shakespeare plays. That's 2 words in group 2.

That leaves MARACAS and BONGO. Checking Semantic Taxonomy: both are percussion instruments. That's 2 words in group 3.

Group 1: family in japanese (OJI, SOFU, HAHA, ANI, ITOKO, CHICHI, KAZOKU, SOBO)
Group 2: shakespeare plays (TITUS, RICHARD)
Group 3: percussion instruments (MARACAS, BONGO)"""

def create_reasoning_prompt(example: Dict) -> str:
    pattern = example['pattern']
    input_text = example['input']
    words = extract_words_from_input(input_text)
    target_scores = example['target_scores']
    explanation = example['explanation']

    if pattern == "4:1":
        odd_words = get_odd_words(target_scores)
        if odd_words:
            return create_reasoning_prompt_4_1(words, odd_words[0], explanation)
    elif pattern == "5:2":
        odd_words = get_odd_words(target_scores)
        return create_reasoning_prompt_5_2(words, odd_words, explanation)
    elif pattern == "7:3":
        odd_words = get_odd_words(target_scores)
        return create_reasoning_prompt_7_3(words, odd_words, explanation)
    elif pattern in ["8:2:2", "10:3:3"]:
        return create_reasoning_prompt_groups(words, explanation, 3)

    return ""

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

def process_example(example: Dict, index: int, total: int) -> Dict:
    input_text = example['input']
    pattern = example['pattern']

    prompt = create_reasoning_prompt(example)
    if not prompt:
        return None

    reasoning = call_deepseek_api(prompt)

    if reasoning and len(reasoning) > 100:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": input_text
                },
                {
                    "role": "assistant",
                    "content": reasoning
                }
            ],
            "metadata": {
                "pattern": pattern,
                "explanation": example['explanation'],
                "reasoning_length": len(reasoning)
            }
        }
    return None

def process_dataset(examples: List[Dict], dataset_name: str) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")
    print(f"Total examples: {len(examples)}")

    results = []
    for i, example in enumerate(examples):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(examples)} ({(i+1)/len(examples)*100:.1f}%)")

        result = process_example(example, i+1, len(examples))
        if result:
            results.append(result)
        time.sleep(0.5)

    print(f"\nSuccess: {len(results)}/{len(examples)} ({len(results)/len(examples)*100:.1f}%)")
    return results

def main():
    print("="*60)
    print("PRECONN STRUCTURED REASONING GENERATOR")
    print("Patterns: 4:1, 5:2, 7:3, 8:2:2, 10:3:3")
    print("="*60)

    Path('data2/output').mkdir(parents=True, exist_ok=True)

    # load preconn data
    print("\nLoading preconn categorical data...")
    examples = load_preconn_data('data2/preconn_categorical_raw.json')
    print(f"Loaded {len(examples)} preconn examples")

    # split train/test (90/10)
    train_examples, test_examples = split_train_test(examples, TRAIN_TEST_SPLIT)
    print(f"Train: {len(train_examples)} examples")
    print(f"Test: {len(test_examples)} examples")

    start_time = time.time()

    # process train
    train_results = process_dataset(train_examples, "Preconn Train")

    # process test
    test_results = process_dataset(test_examples, "Preconn Test")

    # save results
    print(f"\n{'='*60}")
    print("Saving datasets...")
    print(f"{'='*60}")

    with open('data2/output/structured_preconn_train.jsonl', 'w') as f:
        for item in train_results:
            f.write(json.dumps(item) + '\n')
    print(f"Saved structured_preconn_train.jsonl ({len(train_results)} examples)")

    with open('data2/output/structured_preconn_test.jsonl', 'w') as f:
        for item in test_results:
            f.write(json.dumps(item) + '\n')
    print(f"Saved structured_preconn_test.jsonl ({len(test_results)} examples)")

    # summary
    total_time = time.time() - start_time
    total_examples = len(train_results) + len(test_results)

    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Train: {len(train_results)} examples")
    print(f"Test: {len(test_results)} examples")
    print(f"Total examples: {total_examples}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average per example: {total_time/total_examples:.1f} seconds")

if __name__ == "__main__":
    main()
