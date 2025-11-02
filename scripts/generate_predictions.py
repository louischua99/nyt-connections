#!/usr/bin/env python3
"""
Generate predictions for test set using trained models
Saves raw model outputs without evaluation
"""

import argparse
import json
from pathlib import Path
from unsloth import FastLanguageModel
from transformers import TextStreamer
from tqdm import tqdm
import torch

# Model configuration
MAX_SEQ_LENGTH = 10000
LOAD_IN_4BIT = False


def load_test_data(filepath):
    """Load test JSONL"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def generate_prediction(model, tokenizer, user_message, max_new_tokens=2048):
    """Generate model prediction for a single puzzle"""
    messages = [{"role": "user", "content": user_message}]

    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response (after the prompt)
    # Find where the assistant's response starts
    if "<|im_start|>assistant" in full_output:
        response = full_output.split("<|im_start|>assistant")[-1]
        response = response.split("<|im_end|>")[0].strip()
    else:
        # Fallback: try to extract after the user message
        response = full_output[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):]
        response = response.strip()

    return response


def main():
    parser = argparse.ArgumentParser(description='Generate predictions using trained model')
    parser.add_argument('--model', required=True, help='Path to trained model directory')
    parser.add_argument('--test', default='data/global_test.jsonl', help='Path to test data')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--max-new-tokens', type=int, default=2048, help='Max tokens to generate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
    args = parser.parse_args()

    # Set GPU device
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print(f"Loading model from: {args.model}")
    print(f"Test data: {args.test}")
    print(f"Output: {args.output}")
    print("="*80)

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Set to inference mode
    FastLanguageModel.for_inference(model)

    # Load test data
    test_data = load_test_data(args.test)
    print(f"Loaded {len(test_data)} test examples")

    # Generate predictions
    predictions = []

    for i, example in enumerate(tqdm(test_data, desc="Generating predictions")):
        # Extract user message (the puzzle)
        user_message = None
        for msg in example['messages']:
            if msg['role'] == 'user':
                user_message = msg['content']
                break

        if not user_message:
            print(f"Warning: No user message found in example {i}")
            continue

        # Generate prediction
        try:
            prediction = generate_prediction(
                model,
                tokenizer,
                user_message,
                max_new_tokens=args.max_new_tokens
            )

            # Store result
            result = {
                'puzzle_id': example.get('metadata', {}).get('puzzle_id', i),
                'user_message': user_message,
                'prediction': prediction,
                'ground_truth': example['messages'][1]['content'] if len(example['messages']) > 1 else None,
                'metadata': example.get('metadata', {})
            }
            predictions.append(result)

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append({
                'puzzle_id': example.get('metadata', {}).get('puzzle_id', i),
                'user_message': user_message,
                'prediction': None,
                'error': str(e),
                'metadata': example.get('metadata', {})
            })

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"\nSaved {len(predictions)} predictions to {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()
