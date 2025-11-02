#!/usr/bin/env python3
"""
Process preconn JSONL files to:
1. Wrap assistant reasoning in <think></think> tags
2. Keep the last sentence (after last period) as the answer
3. Capitalize the entire answer
"""

import json
from pathlib import Path


def process_jsonl_file(input_file, output_file):
    """Process a single JSONL file"""
    processed_count = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            entry = json.loads(line)

            # Process messages
            for message in entry['messages']:
                if message['role'] == 'assistant':
                    content = message['content']

                    # Find the last period
                    last_period_idx = content.rfind('.')

                    if last_period_idx != -1:
                        # Split at last period
                        reasoning = content[:last_period_idx + 1]  # Include the period
                        answer = content[last_period_idx + 1:].strip()  # Everything after

                        # Capitalize the entire answer
                        answer = answer.upper()

                        # Format: <think>reasoning</think>\n\nanswer
                        message['content'] = f"<think>\n{reasoning}\n</think>\n\n{answer}"
                    else:
                        # No period found, wrap entire content
                        message['content'] = f"<think>\n{content}\n</think>"

                    processed_count += 1

            # Write the modified entry
            outfile.write(json.dumps(entry) + '\n')

    return processed_count


def main():
    # Define file mappings
    files_to_process = [
        {
            'input': 'data2/reasoning/structured_preconn_test.jsonl',
            'output': 'data2/reasoning/structured_preconn_test_formatted.jsonl'
        },
        {
            'input': 'data2/reasoning/structured_preconn_train.jsonl',
            'output': 'data2/reasoning/structured_preconn_train_formatted.jsonl'
        }
    ]

    print("Processing preconn files...\n")

    for file_info in files_to_process:
        print(f"Processing {file_info['input']}...")

        # Process the file
        count = process_jsonl_file(
            file_info['input'],
            file_info['output']
        )
        print(f"  Processed {count} entries -> {file_info['output']}\n")

    print("✓ All preconn files processed successfully!")


if __name__ == '__main__':
    main()
