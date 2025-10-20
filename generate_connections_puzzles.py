"""
generate full connections puzzles from categorical patterns
creates 4 groups of 4 words per puzzle matching nyt format
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta

# load patterns from categorical generator
from generate_preconn_categorical import CATEGORICAL_PATTERNS

class ConnectionsPuzzleGenerator:
    """generates full 4x4 connections puzzles"""

    def __init__(self, patterns: Dict = None):
        self.patterns = patterns or CATEGORICAL_PATTERNS
        self.puzzle_count = 0
        self.used_combinations = set()

    def get_unique_subgroup(self, pattern_name: str, exclude_indices: set) -> tuple:
        """get subgroup that hasn't been used in current puzzle"""
        pattern = self.patterns[pattern_name]
        available = [i for i in range(len(pattern["examples"])) if i not in exclude_indices]

        if not available:
            return None, None, None

        idx = random.choice(available)
        example = pattern["examples"][idx]
        return example["words"], example["subgroup"], idx

    def generate_puzzle(self, start_date: datetime, puzzle_id: int) -> Dict:
        """generate single 4x4 puzzle with difficulty levels"""

        # select 4 different pattern categories for diversity
        selected_patterns = random.sample(list(self.patterns.keys()), 4)

        groups = []
        used_indices = {pattern: set() for pattern in selected_patterns}
        all_words_set = set()

        # difficulty levels: 0=easiest, 3=hardest
        difficulty_levels = [0, 1, 2, 3]
        random.shuffle(difficulty_levels)

        for i, pattern_type in enumerate(selected_patterns):
            words, subgroup, idx = self.get_unique_subgroup(pattern_type, used_indices[pattern_type])

            if not words or len(words) < 4:
                return None

            # sample 4 unique words not already used
            available_words = [w for w in words if w not in all_words_set]
            # ensure we have at least 4 unique words available
            unique_available = list(set(available_words))
            if len(unique_available) < 4:
                return None

            selected_words = random.sample(unique_available, 4)
            all_words_set.update(selected_words)
            used_indices[pattern_type].add(idx)

            groups.append({
                "level": difficulty_levels[i],
                "group": subgroup.upper(),
                "members": sorted(selected_words)
            })

        # sort by difficulty level
        groups.sort(key=lambda x: x["level"])

        puzzle = {
            "id": puzzle_id,
            "date": start_date.strftime("%Y-%m-%d"),
            "answers": groups
        }

        return puzzle

    def generate_dataset(self, num_puzzles: int = 100, start_date_str: str = "2024-01-01") -> List[Dict]:
        """generate full dataset of puzzles"""
        puzzles = []
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

        attempts = 0
        max_attempts = num_puzzles * 5

        while len(puzzles) < num_puzzles and attempts < max_attempts:
            attempts += 1

            puzzle = self.generate_puzzle(
                start_date + timedelta(days=len(puzzles)),
                len(puzzles) + 1
            )

            if puzzle:
                puzzles.append(puzzle)

                if len(puzzles) % 10 == 0:
                    print(f"generated {len(puzzles)}/{num_puzzles} puzzles")

        return puzzles

def main():
    print("="*60)
    print("connections puzzle generator")
    print("generating 4x4 puzzles from categorical patterns")
    print("="*60)

    Path('data/output').mkdir(parents=True, exist_ok=True)

    generator = ConnectionsPuzzleGenerator()

    # generate puzzles
    num_puzzles = 200
    print(f"\ngenerating {num_puzzles} puzzles...")
    puzzles = generator.generate_dataset(num_puzzles=num_puzzles, start_date_str="2024-01-01")

    print(f"\nsuccessfully generated {len(puzzles)} puzzles")

    # save as json
    output_file = 'data/output/connections_categorical.json'
    with open(output_file, 'w') as f:
        json.dump(puzzles, f, indent=2)

    print(f"\nsaved to {output_file}")

    # show sample
    if puzzles:
        print("\nsample puzzle:")
        sample = puzzles[0]
        print(f"id: {sample['id']}")
        print(f"date: {sample['date']}")
        print("\ngroups:")
        for answer in sample['answers']:
            print(f"  level {answer['level']}: {answer['group']}")
            print(f"    {', '.join(answer['members'])}")

    # stats
    all_categories = {}
    for puzzle in puzzles:
        for answer in puzzle['answers']:
            cat = answer['group']
            all_categories[cat] = all_categories.get(cat, 0) + 1

    print(f"\ntotal unique categories used: {len(all_categories)}")
    print("\ntop 10 categories:")
    for cat, count in sorted(all_categories.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cat}: {count} times")

if __name__ == "__main__":
    main()
