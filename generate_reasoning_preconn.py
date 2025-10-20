#!/usr/bin/env python3
"""
Generate reasoning for complex BigBench OOO examples using vLLM server
Uses concurrent processing for faster generation
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm
import argparse
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reasoning_generation_complex.log'),
        logging.StreamHandler()
    ]
)

# vLLM server configuration
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
API_KEY = "YOUR_VLLM_API_KEY_HERE"

def call_vllm(prompt: str, temperature: float = 0.7) -> str:
    """Call vLLM server for reasoning generation"""
    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "stream": False
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            timeout=600
        )
        
        if response.status_code == 200:
            result = response.json()
            choice = result['choices'][0]
            
            # Handle reasoning parser output structure
            if 'message' in choice:
                msg = choice['message']
                reasoning_content = msg.get('reasoning_content', '')
                content = msg.get('content', '')
                
                # Combine reasoning and content if both exist
                if reasoning_content and content:
                    return f"{reasoning_content}\n\n{content}".strip()
                elif reasoning_content:
                    return reasoning_content.strip()
                elif content:
                    return content.strip()
                else:
                    return ""
            else:
                return choice.get('text', '').strip()
        else:
            logging.error(f"vLLM error: {response.status_code} - {response.text}")
            return ""
            
    except requests.exceptions.Timeout:
        logging.error("Request timed out")
        return ""
    except Exception as e:
        logging.error(f"Error calling vLLM: {e}")
        return ""

def generate_reasoning_prompt(example: Dict) -> str:
    """Generate reasoning prompt for complex OOO examples"""
    input_text = example['messages'][0]['content']
    answer = example['messages'][1]['content']
    metadata = example.get('metadata', {})
    
    # Extract pattern info
    pattern = metadata.get('pattern', '')
    explanation = metadata.get('explanation', '')
    
    # Extract words from the puzzle
    import re
    words_match = re.search(r'Pick the odd word out: (.+)', input_text)
    if words_match:
        words_str = words_match.group(1)
        words = [w.strip() for w in words_str.split(',')]
    else:
        # Fallback if format is different
        words = input_text.split(': ')[1].split(', ') if ': ' in input_text else []
    
    # Extract the odd word from answer
    odd_word_match = re.search(r'The odd word\(s\) out: (.+)', answer)
    if odd_word_match:
        odd_word = odd_word_match.group(1)
    else:
        odd_word = answer
    
    prompt = f"""Solve this word puzzle by finding the odd word out:
Words: {', '.join(words)}

The correct answer is: {odd_word}
The pattern is: {pattern} ({explanation})

Your task: You're teaching someone how to approach these puzzles by demonstrating your thought process.

START your response with: "Looking at these words: {', '.join(words)}. "

Write naturally as someone solving this puzzle for the first time, showing genuine exploration and discovery:
- Initial scanning and pattern recognition
- Noticing connections between most words
- Testing different groupings and relationships
- Identifying why one word doesn't fit the pattern
- Having "aha!" or "wait..." moments when spotting connections
- Sometimes second-guessing before reasoning it through
- Natural phrases like "let me check", "I notice", "these could be"

Write the FULL solving process showing how you work through the puzzle step by step. Be thorough and detailed - aim for at least 800 words of reasoning that shows:
- Multiple scanning passes through the words
- Testing different potential patterns (some that work, some that don't)
- Explaining your thought process for each connection you consider
- Moments of uncertainty and how you resolve them
- Clear reasoning for why certain words belong together
- Your process of elimination to find the odd one out
- Walking through your logic step by step

Focus on sophisticated wordplay, linguistic patterns, and hidden connections like:
- Etymology and word origins (Latin, Greek, Sanskrit, etc.)
- Linguistic categories (modal verbs, contractions, suffixes, etc.)
- Technical terminology from specific fields
- Homophones and sound patterns
- Cultural references and loanwords
- Multiple meanings and polysemy
- Compound words and word combinations
- Historical and literary references

ONLY AFTER your complete detailed reasoning, conclude with:
"Therefore, the odd word out is: {odd_word}"

**DO NOT MENTION OR ALLUDE TO ANY HINTS/ANSWER BEING SHOWN. PRETEND AS IF YOU ARE FIGURING IT OUT YOURSELF**

Remember to make your reasoning feel natural and exploratory, as if you're genuinely discovering the pattern through analysis."""
    
    return prompt

def process_example(example: Dict, index: int, total: int) -> Dict:
    """Process a single example to generate reasoning"""
    try:
        # Generate the prompt for reasoning
        prompt = generate_reasoning_prompt(example)
        reasoning = call_vllm(prompt)
        
        if reasoning:
            # Get the original puzzle without any instructions
            original_input = example['messages'][0]['content']
            
            # Extract the answer from the original assistant response
            original_answer = example['messages'][1]['content']
            
            # Create new example with clean prompt and generated reasoning
            new_example = {
                "messages": [
                    {
                        "role": "user",
                        "content": original_input  # Keep the simple puzzle prompt
                    },
                    {
                        "role": "assistant",
                        "content": reasoning  # The generated reasoning
                    }
                ],
                "metadata": {
                    **example.get('metadata', {}),
                    "correct_answer": original_answer,  # Store the correct answer
                    "reasoning_length": len(reasoning)
                }
            }
            
            logging.info(f"Processed example {index}/{total}")
            return new_example
        else:
            logging.warning(f"No reasoning generated for example {index}")
            # Return with the original structure but add a note in metadata
            return {
                "messages": example['messages'],
                "metadata": {
                    **example.get('metadata', {}),
                    "generation_failed": True
                }
            }
            
    except Exception as e:
        logging.error(f"Error processing example {index}: {e}")
        return {
            "messages": example['messages'],
            "metadata": {
                **example.get('metadata', {}),
                "generation_error": str(e)
            }
        }

def generate_reasoning_dataset(input_file: str, output_file: str, max_workers: int = 10, limit: Optional[int] = None):
    """Generate reasoning for complex OOO dataset with concurrent processing"""
    
    # Load dataset
    logging.info(f"Loading dataset from {input_file}")
    examples = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    if limit:
        examples = examples[:limit]
    
    total = len(examples)
    logging.info(f"Processing {total} examples with {max_workers} workers")
    
    # Process examples concurrently
    processed_examples = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_example, example, i+1, total): i 
            for i, example in enumerate(examples)
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=total, desc="Generating reasoning") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    processed_examples.append((index, result))
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"Failed to process example {index}: {e}")
                    processed_examples.append((index, examples[index]))
                    pbar.update(1)
    
    # Sort by original index to maintain order
    processed_examples.sort(key=lambda x: x[0])
    
    # Save results
    logging.info(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        for _, example in processed_examples:
            f.write(json.dumps(example) + '\n')
    
    logging.info(f"Successfully generated reasoning for {len(processed_examples)} examples")
    
    # Print statistics
    print("\n" + "="*60)
    print("REASONING GENERATION COMPLETE")
    print("="*60)
    print(f"Total examples processed: {len(processed_examples)}")
    print(f"Output file: {output_file}")
    
    # Sample reasoning for verification
    if processed_examples:
        sample = processed_examples[0][1]
        print("\nSample reasoning generated:")
        print("-"*40)
        if len(sample['messages'][1]['content']) > 500:
            print(sample['messages'][1]['content'][:500] + "...")
        else:
            print(sample['messages'][1]['content'])

def main():
    parser = argparse.ArgumentParser(description='Generate reasoning for complex BigBench OOO examples')
    parser.add_argument('--input', default='data/output/preconn_raw.jsonl',
                       help='Input JSONL file with examples')
    parser.add_argument('--output', default='data/output/preconn_reasoning.jsonl',
                       help='Output JSONL file with reasoning')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of concurrent workers (default: 10)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of examples to process (for testing)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for generation (default: 0.7)')
    
    args = parser.parse_args()
    
    # Check if vLLM server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("Warning: vLLM server may not be running properly")
    except:
        print("Error: Cannot connect to vLLM server at http://localhost:8000")
        print("Please ensure the vLLM server is running with:")
        print("  vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --reasoning-parser deepseek_r1")
        return
    
    # Generate reasoning
    generate_reasoning_dataset(
        args.input,
        args.output,
        max_workers=args.workers,
        limit=args.limit
    )

if __name__ == "__main__":
    main()