#!/usr/bin/env python3
"""
test unstructured reasoning generation on one puzzle
"""

import json
from openai import OpenAI

DEEPSEEK_API_KEY = "YOUR_API_KEY_HERE"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

def create_unstructured_prompt(words, answers):
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

# load one puzzle
with open('data2/connections.json', 'r') as f:
    puzzles = json.load(f)

# use first puzzle
puzzle = puzzles[0]
words = []
for answer in puzzle['answers']:
    words.extend(answer['members'])

print("="*60)
print("TESTING UNSTRUCTURED REASONING")
print("="*60)
print(f"\nPuzzle ID: {puzzle['id']}")
print(f"Words: {', '.join(words)}")
print(f"\nActual groups:")
for answer in sorted(puzzle['answers'], key=lambda x: x['level']):
    print(f"  {answer['group']}: {', '.join(answer['members'])}")

print("\n" + "="*60)
print("Generating reasoning...")
print("="*60 + "\n")

prompt = create_unstructured_prompt(words, puzzle['answers'])
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": prompt}]
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print(content)

# save output
with open('test_output_unstructured.txt', 'w', encoding='utf-8') as f:
    f.write(f"Puzzle ID: {puzzle['id']}\n")
    f.write(f"Words: {', '.join(words)}\n\n")
    f.write("="*60 + "\n")
    f.write("UNSTRUCTURED REASONING OUTPUT\n")
    f.write("="*60 + "\n\n")
    f.write(content)

print("\n" + "="*60)
print("Saved to test_output_unstructured.txt")
