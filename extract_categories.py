#!/usr/bin/env python3
"""
Script to analyze Connections game categories using GPT-5 API.
Processes connections in batches of 3 and maintains a running list of category types.
"""

import json
import os
import sys
from typing import List, Dict, Any
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Unbuffer stdout and stderr for real-time logging
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Load environment variables
load_dotenv()


class CategoryAnalyzer:
    def __init__(self, api_key: str = None, output_dir: str = "logs/categories"):
        """Initialize the analyzer with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.category_types = []
        self.processed_count = 0
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_connections(self, file_path: str) -> List[Dict[str, Any]]:
        """Load connections from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def format_batch_for_analysis(self, batch: List[Dict[str, Any]]) -> str:
        """Format a batch of connections for GPT-5 analysis."""
        formatted = []
        for connection in batch:
            formatted.append(f"Connection #{connection['id']} ({connection['date']}):")
            for answer in connection['answers']:
                formatted.append(f"  - {answer['group']}: {', '.join(answer['members'])}")
        return "\n".join(formatted)

    def build_prompt(self, batch_text: str, is_first_batch: bool = False) -> str:
        """Build the prompt for GPT-5."""
        base_prompt = f"""You are a linguistics expert analyzing NYT Connections puzzles. Your goal is to identify the HIGH-LEVEL linguistic/cognitive patterns used. Aim for a COMPACT taxonomy of 10-20 broad types that can classify ANY Connections puzzle.

{batch_text}

"""

        if is_first_batch:
            base_prompt += """IMPORTANT: Each category should map to exactly ONE broad type. Do NOT create subtypes or separate categories for specific domains.

HIGH-LEVEL PATTERN TYPES TO CONSIDER:

1. **Semantic Taxonomy** - Words sharing a hypernym (types of X, parts of X) - applies to ALL taxonomic categories regardless of domain
2. **Semantic Synonymy** - Words with similar/overlapping meanings
3. **Semantic Association** - Words related by theme, co-occurrence, or context (not strict synonyms/hypernyms)
4. **Named Entities** - Proper nouns grouped by type (teams, people, brands, places, titles) - ONE category for ALL named entity sets
5. **Lexical Morphology** - Word formation patterns (compounds, affixes, truncations, blends)
6. **Lexical Orthography** - Visual/spelling patterns (palindromes, anagrams, letter patterns)
7. **Phonological Pattern** - Sound-based grouping (homophones, rhymes, alliteration)
8. **Grammatical/Syntactic** - Part-of-speech or syntactic function
9. **Collocational/Idiomatic** - Words that form fixed phrases or idioms with a target word
10. **Wordplay - Hidden String** - Words containing/hiding a common substring
11. **Wordplay - Double Meaning** - Words with dual interpretations (puns, polysemy)
12. **Wordplay - Cryptic/Rebus** - Visual or symbolic representation patterns
13. **Functional/Pragmatic** - Words grouped by real-world function or usage context
14. **Temporal/Sequential** - Ordered sequences or time-based relationships
15. **Numerical/Quantitative** - Number-related patterns

KEY RULES:
- "NBA TEAMS", "MLB TEAMS", "MAGAZINES" are ALL just **Named Entities** - don't split by domain
- "Types of shoes", "Types of weather", "Types of fish" are ALL just **Semantic Taxonomy** - don't split by semantic field
- BE ABSTRACT: focus on the LINGUISTIC MECHANISM, not the content domain
- Aim for 10-20 types TOTAL, not 100+

For each puzzle category, identify which ONE high-level type it exemplifies.

Output as JSON:
{
  "category_types": [
    {
      "type": "type_name",
      "description": "abstract linguistic description covering all instances of this pattern",
      "examples": ["EXAMPLE1", "EXAMPLE2"]
    }
  ]
}"""
        else:
            base_prompt += f"""Current taxonomy (target: 10-20 types TOTAL):
{json.dumps(self.category_types, indent=2)}

CRITICAL RULES:
1. For each new category, find the EXISTING type it fits - resist creating new types
2. Only add a NEW type if the pattern is fundamentally different from all existing types
3. AGGRESSIVELY MERGE over-specific types:
   - Different named entity domains → merge into "Named Entities"
   - Different semantic fields → merge into "Semantic Taxonomy" or "Semantic Association"
   - Different wordplay domains → merge into appropriate wordplay type
4. Add examples to existing types rather than creating variants

REMEMBER:
- "Sports teams", "magazines", "celebrities", "brands" are ALL → **Named Entities**
- "Types of X", "Parts of Y", "Kinds of Z" are ALL → **Semantic Taxonomy**
- Different content domains should NOT create new categories
- Focus on MECHANISM, not content

Before adding a new type, ask: "Is this really a different LINGUISTIC PATTERN, or just a different content domain using an existing pattern?"

Output the COMPLETE updated taxonomy as JSON:
{{
  "category_types": [
    {{
      "type": "type_name",
      "description": "abstract description",
      "examples": ["EXAMPLE1", "EXAMPLE2", ...]
    }}
  ]
}}"""

        return base_prompt

    def analyze_batch(self, batch: List[Dict[str, Any]], batch_num: int) -> Dict[str, Any]:
        """Analyze a batch of connections using GPT-5."""
        batch_text = self.format_batch_for_analysis(batch)
        is_first = (batch_num == 1)
        prompt = self.build_prompt(batch_text, is_first)

        print(f"\n{'='*60}")
        print(f"Processing Batch {batch_num} (Connections {batch[0]['id']}-{batch[-1]['id']})")
        print(f"{'='*60}")

        try:
            # Use GPT-5 with Responses API
            response = self.client.responses.create(
                model="gpt-5",
                input=prompt,
                reasoning={"effort": "medium"},  # Medium reasoning for balanced analysis
                text={"verbosity": "medium"}     # Medium verbosity for complete but concise output
            )

            # Parse the response
            result = json.loads(response.output_text)

            # Update category types
            self.category_types = result.get("category_types", [])
            self.processed_count += len(batch)

            print(f"✓ Processed {len(batch)} connections")
            print(f"✓ Total category types: {len(self.category_types)}")

            return result

        except Exception as e:
            print(f"✗ Error processing batch: {e}")
            raise

    def process_all_connections(self, file_path: str, batch_size: int = 3):
        """Process all connections in batches."""
        connections = self.load_connections(file_path)
        total_connections = len(connections)

        print(f"\n{'='*60}")
        print(f"Starting Category Analysis")
        print(f"{'='*60}")
        print(f"Total connections: {total_connections}")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: {(total_connections + batch_size - 1) // batch_size}")

        batch_num = 1
        for i in range(0, total_connections, batch_size):
            batch = connections[i:i + batch_size]
            self.analyze_batch(batch, batch_num)

            # Save progress after each batch
            self.save_progress(self.output_dir / f"category_analysis_progress_batch_{batch_num}.json")

            batch_num += 1

        # Save final results
        final_path = self.output_dir / "category_analysis_final.json"
        self.save_results(final_path)
        print(f"\n{'='*60}")
        print(f"Analysis Complete!")
        print(f"{'='*60}")
        print(f"Total connections processed: {self.processed_count}")
        print(f"Total category types identified: {len(self.category_types)}")
        print(f"Results saved to: {final_path}")

    def save_progress(self, filename: str):
        """Save current progress to a file."""
        output = {
            "processed_count": self.processed_count,
            "total_category_types": len(self.category_types),
            "category_types": self.category_types
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

    def save_results(self, filename: str):
        """Save final results with summary statistics."""
        # Calculate some stats
        total_examples = sum(len(ct.get("examples", [])) for ct in self.category_types)

        output = {
            "summary": {
                "total_connections_analyzed": self.processed_count,
                "total_category_types": len(self.category_types),
                "total_examples": total_examples,
                "avg_examples_per_type": round(total_examples / len(self.category_types), 2) if self.category_types else 0
            },
            "category_types": self.category_types
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        # Also save a human-readable version
        self.save_readable_report(filename.replace('.json', '_report.txt'))

    def save_readable_report(self, filename: str):
        """Save a human-readable report."""
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CONNECTIONS CATEGORY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Total Connections Analyzed: {self.processed_count}\n")
            f.write(f"Total Category Types Identified: {len(self.category_types)}\n\n")

            f.write("="*80 + "\n")
            f.write("CATEGORY TYPES\n")
            f.write("="*80 + "\n\n")

            for i, ct in enumerate(self.category_types, 1):
                f.write(f"{i}. {ct['type'].upper()}\n")
                f.write(f"   Description: {ct['description']}\n")
                f.write(f"   Examples ({len(ct.get('examples', []))}):\n")
                for ex in ct.get('examples', []):
                    f.write(f"     • {ex}\n")
                f.write("\n")

        print(f"Readable report saved to: {filename}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Connections categories using GPT-5")
    parser.add_argument(
        "--input",
        default="data/connections.json",
        help="Path to connections JSON file (default: data/connections.json)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of connections to process per batch (default: 3)"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--output-dir",
        default="logs/categories",
        help="Directory to save output files (default: logs/categories)"
    )

    args = parser.parse_args()

    try:
        analyzer = CategoryAnalyzer(api_key=args.api_key, output_dir=args.output_dir)
        analyzer.process_all_connections(args.input, args.batch_size)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
