#!/usr/bin/env python3
"""
test structured reasoning generation on one preconn puzzle
"""

import json
from openai import OpenAI

DEEPSEEK_API_KEY = "YOUR_API_KEY_HERE"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# nyt category types
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

def extract_words_from_input(input_text: str):
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

def get_odd_words(target_scores):
    return [word for word, score in target_scores.items() if score == 1]

def create_structured_prompt_4_1(words, odd_word, explanation):
    prompt = f"""Solve this word puzzle by finding the odd word out:
Words: {', '.join(words)}

The correct answer is: {odd_word}
Pattern explanation: {explanation}

Your task: Write a concise problem-solving narrative using category analysis.

START with: "Looking at these {len(words)} words: {', '.join(words)}. I'll check which category connects 4 of them."

APPROACH:
1. List the category types from: {CATEGORY_TYPES}
2. Identify which category type connects the main group of 4 words
3. Name the specific pattern (e.g., "european capitals", "rock___ compounds", "silent k words")
4. Note which word doesn't fit

Keep it concise - focus on identifying the relevant category and the pattern, not exhaustively checking every category.

CONCLUDE with:
"Therefore, the odd word out is: {odd_word}"

Example:
Looking at these 5 words: GARDEN, STAR, FACE, SALT, STRATUS. I'll check which category connects 4 of them.

Category types:
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

I notice GARDEN, STAR, FACE, and SALT could follow a pattern. Checking Collocational/Idiomatic - can these complete a phrase? Yes, they all work as "rock___" compounds: rock GARDEN, rock STAR, rock FACE, rock SALT.

STRATUS doesn't fit this pattern - it's a cloud type, not a "rock___" compound.

Therefore, the odd word out is: STRATUS"""

    return prompt

# load one puzzle from each pattern type
with open('data2/preconn_categorical_raw.json', 'r') as f:
    data = json.load(f)

# get one example of 4:1 pattern
example_4_1 = None
for ex in data['examples']:
    if ex['pattern'] == '4:1':
        example_4_1 = ex
        break

if example_4_1:
    words = extract_words_from_input(example_4_1['input'])
    odd_words = get_odd_words(example_4_1['target_scores'])
    odd_word = odd_words[0] if odd_words else ""

    print("="*60)
    print("TESTING PRECONN STRUCTURED REASONING (4:1 PATTERN)")
    print("="*60)
    print(f"\nInput: {example_4_1['input']}")
    print(f"Pattern: {example_4_1['pattern']}")
    print(f"Explanation: {example_4_1['explanation']}")
    print(f"Correct answer: {odd_word}")

    print("\n" + "="*60)
    print("Generating reasoning...")
    print("="*60 + "\n")

    prompt = create_structured_prompt_4_1(words, odd_word, example_4_1['explanation'])
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content

    print(content)

    # save output
    with open('test_output_preconn.txt', 'w', encoding='utf-8') as f:
        f.write(f"Pattern: {example_4_1['pattern']}\n")
        f.write(f"Input: {example_4_1['input']}\n")
        f.write(f"Explanation: {example_4_1['explanation']}\n\n")
        f.write("="*60 + "\n")
        f.write("PRECONN STRUCTURED REASONING OUTPUT\n")
        f.write("="*60 + "\n\n")
        f.write(content)

    print("\n" + "="*60)
    print("Saved to test_output_preconn.txt")
