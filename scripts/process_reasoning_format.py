#!/usr/bin/env python3
"""
Process JSONL files to:
1. Wrap assistant reasoning in <think></think> tags
2. Append the actual answer from source JSON
"""

import json
from pathlib import Path

def load_source_data(filepath):
    """Load source JSON and create id-to-puzzle mapping"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Create mapping from id to puzzle answers
    id_to_answers = {}
    for puzzle in data:
        puzzle_id = puzzle['id']
        answers = puzzle['answers']
        id_to_answers[puzzle_id] = answers

    return id_to_answers

def format_answer(answers):
    """Format answers in the requested format"""
    answer_lines = []
    for answer_group in answers:
        group_name = answer_group['group']
        members = answer_group['members']
        # Join members with comma-space
        members_str = ', '.join(members)
        answer_lines.append(f"**{group_name}**: {members_str}")

    return '\n'.join(answer_lines)

def process_jsonl_file(input_file, output_file, id_to_answers):
    """Process a single JSONL file"""
    processed_count = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            entry = json.loads(line)

            # Get the original puzzle ID
            original_id = entry['metadata'].get('original_id') or entry['metadata'].get('puzzle_id')
            if isinstance(original_id, str) and '_perm' in original_id:
                # Extract the numeric part before _perm
                original_id = int(original_id.split('_')[0])

            # Get the answers from source
            if original_id in id_to_answers:
                answers = id_to_answers[original_id]
                formatted_answer = format_answer(answers)

                # Process messages
                for message in entry['messages']:
                    if message['role'] == 'assistant':
                        # Wrap in think tags and append answer
                        reasoning = message['content']
                        message['content'] = f"<think>\n{reasoning}\n</think>\n\n{formatted_answer}"

                processed_count += 1

            # Write the modified entry
            outfile.write(json.dumps(entry) + '\n')

    return processed_count

def main():
    # Define file mappings
    files_to_process = [
        {
            'input': 'data2/reasoning/structured_nyt_test.jsonl',
            'output': 'data2/reasoning/structured_nyt_test_formatted.jsonl',
            'source': 'data2/connections.json'
        },
        {
            'input': 'data2/reasoning/structured_nyt_train.jsonl',
            'output': 'data2/reasoning/structured_nyt_train_formatted.jsonl',
            'source': 'data2/connections.json'
        },
        {
            'input': 'data2/reasoning/structured_synthetic_test.jsonl',
            'output': 'data2/reasoning/structured_synthetic_test_formatted.jsonl',
            'source': 'data2/connections_categorical.json'
        },
        {
            'input': 'data2/reasoning/structured_synthetic_train.jsonl',
            'output': 'data2/reasoning/structured_synthetic_train_formatted.jsonl',
            'source': 'data2/connections_categorical.json'
        },
        {
            'input': 'data2/reasoning/unstructured_nyt.jsonl',
            'output': 'data2/reasoning/unstructured_nyt_formatted.jsonl',
            'source': 'data2/connections.json'
        },
        {
            'input': 'data2/reasoning/unstructured_synthetic.jsonl',
            'output': 'data2/reasoning/unstructured_synthetic_formatted.jsonl',
            'source': 'data2/connections_categorical.json'
        }
    ]

    for file_info in files_to_process:
        print(f"\nProcessing {file_info['input']}...")

        # Load source data
        id_to_answers = load_source_data(file_info['source'])
        print(f"  Loaded {len(id_to_answers)} puzzles from {file_info['source']}")

        # Process the file
        count = process_jsonl_file(
            file_info['input'],
            file_info['output'],
            id_to_answers
        )
        print(f"  Processed {count} entries -> {file_info['output']}")

    print("\n✓ All files processed successfully!")

if __name__ == '__main__':
    main()
