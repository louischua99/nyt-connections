#!/usr/bin/env python3
"""
test structured reasoning generation on one puzzle
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

def create_structured_prompt(words, answers):
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
Final Answer:

EXCELLENT, OUTSTANDING, SUPERB, TERRIFIC - Semantic Synonymy (words meaning "very good")
COLOSSUS, MAUSOLEUM, PYRAMIDS, LIGHTHOUSE - Semantic Taxonomy (Ancient Wonders)
TRACTOR, PLOW, COMBINE, HARROW - Semantic Association (farm equipment)
FELLOWSHIP, TWO TOWERS, RETURN, LORD - Named Entities (Lord of the Rings titles)"""

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
print("TESTING STRUCTURED REASONING")
print("="*60)
print(f"\nPuzzle ID: {puzzle['id']}")
print(f"Words: {', '.join(words)}")
print(f"\nActual groups:")
for answer in sorted(puzzle['answers'], key=lambda x: x['level']):
    print(f"  {answer['group']}: {', '.join(answer['members'])}")

print("\n" + "="*60)
print("Generating reasoning...")
print("="*60 + "\n")

prompt = create_structured_prompt(words, puzzle['answers'])
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": prompt}]
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print(content)

# save output
with open('test_output_structured.txt', 'w', encoding='utf-8') as f:
    f.write(f"Puzzle ID: {puzzle['id']}\n")
    f.write(f"Words: {', '.join(words)}\n\n")
    f.write("="*60 + "\n")
    f.write("STRUCTURED REASONING OUTPUT\n")
    f.write("="*60 + "\n\n")
    f.write(content)

print("\n" + "="*60)
print("Saved to test_output_structured.txt")
