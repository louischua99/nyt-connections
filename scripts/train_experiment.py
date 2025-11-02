#!/usr/bin/env python3
"""
Unsloth training script for NYT Connections experiments
Supports all experiment types with configurable parameters
"""

import argparse
import json
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer

# Model configuration
MODEL_NAME = "unsloth/Qwen3-4B-Thinking-2507"
MAX_SEQ_LENGTH = 10000
LOAD_IN_4BIT = False

# LoRA configuration
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training configuration defaults
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_WARMUP = 5
DEFAULT_LR = 2e-4


def load_jsonl(filepath):
    """Load JSONL dataset"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_dataset(data):
    """Convert JSONL to HuggingFace Dataset with conversations"""
    conversations = []
    for entry in data:
        # Extract messages
        messages = entry.get('messages', [])
        if messages:
            conversations.append(messages)

    return Dataset.from_dict({"conversations": conversations})


def apply_chat_template(examples, tokenizer):
    """Apply Qwen3 thinking chat template"""
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}


def train_single_phase(
    train_data_path,
    test_data_path,  # Actually validation data - kept as test_data_path for API compatibility
    output_dir,
    num_train_epochs=1,
    max_steps=None,
    learning_rate=DEFAULT_LR,
    batch_size=DEFAULT_BATCH_SIZE,
    grad_accum=DEFAULT_GRAD_ACCUM,
    warmup_steps=DEFAULT_WARMUP,
    save_steps=100,
    model_to_load=MODEL_NAME,
):
    """Train a single phase (test_data_path is actually validation data for monitoring)"""

    print("="*80)
    print(f"TRAINING: {output_dir}")
    print(f"Loading from: {model_to_load}")
    print("="*80)

    # Load model and tokenizer
    print("\n1. Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_to_load,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Add LoRA adapters if loading base model
    if model_to_load == MODEL_NAME:
        print("2. Adding LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    # Apply chat template
    print("3. Applying Qwen3-thinking chat template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen3-thinking",
    )

    # Load and prepare dataset
    print(f"4. Loading training data from {train_data_path}...")
    train_data = load_jsonl(train_data_path)
    train_dataset = prepare_dataset(train_data)
    print(f"   Loaded {len(train_dataset)} training examples")

    # Apply chat template to dataset
    train_dataset = train_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        batched=True
    )

    # Load validation data if provided (for monitoring during training)
    eval_dataset = None
    if test_data_path:
        print(f"5. Loading validation data from {test_data_path}...")
        test_data = load_jsonl(test_data_path)
        eval_dataset = prepare_dataset(test_data)
        eval_dataset = eval_dataset.map(
            lambda x: apply_chat_template(x, tokenizer),
            batched=True
        )
        print(f"   Loaded {len(eval_dataset)} validation examples")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup trainer
    print("6. Setting up trainer...")

    # Handle epochs vs max_steps
    if max_steps is not None:
        actual_epochs = 1  # Set to 1 when using max_steps
        actual_max_steps = max_steps
    else:
        actual_epochs = num_train_epochs
        actual_max_steps = -1  # -1 means use epochs instead

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=warmup_steps,
            num_train_epochs=actual_epochs,
            max_steps=actual_max_steps,
            learning_rate=learning_rate,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            save_steps=save_steps,
            save_total_limit=3,
            report_to="none",
        ),
    )

    # Train on responses only (mask out user prompts)
    print("7. Setting up response-only training...")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\n8. GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"   {start_gpu_memory} GB of memory reserved.")

    # Train
    print("\n9. Starting training...")
    print("="*80)
    trainer_stats = trainer.train()

    # Show final stats
    print("\n" + "="*80)
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)

    print(f"Training complete!")
    print(f"Time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
    print(f"Peak memory: {used_memory} GB ({used_percentage}%)")
    print(f"Memory for training: {used_memory_for_lora} GB")

    # Save model
    print(f"\n10. Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("="*80)
    print("TRAINING PHASE COMPLETE!")
    print("="*80)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train NYT Connections model")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["exp1_baseline", "exp1_permutation", "exp1_synthetic", "exp1_full",
                               "exp2_structured", "exp2_unstructured", "exp2_mixed", "exp2_sequential",
                               "exp3_no_warmup", "exp3_warmup", "exp3_staged"],
                       help="Experiment to run")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps (overrides epochs)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    # Auto-generate output directory
    if args.output_dir is None:
        args.output_dir = f"models/{args.experiment}"

    # Get validation data path for training monitoring
    # NOTE: Test set is kept separate and only used for final evaluation after all training

    # Route to appropriate experiment
    if args.experiment == "exp1_baseline":
        # Use NYT-only validation with perm=1 (matches training distribution)
        train_single_phase("data/experiment1/baseline_train.jsonl",
                          "data/experiment1/validation_nyt_perm1.jsonl",
                          args.output_dir,
                          num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
                          batch_size=args.batch_size)

    elif args.experiment == "exp1_permutation":
        # Use NYT-only validation with all permutations (matches training distribution)
        train_single_phase("data/experiment1/permutation_train.jsonl",
                          "data/experiment1/validation_nyt_all_perms.jsonl",
                          args.output_dir,
                          num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
                          batch_size=args.batch_size)

    elif args.experiment == "exp1_synthetic":
        # Use global validation (includes both NYT and synthetic)
        train_single_phase("data/experiment1/synthetic_train.jsonl",
                          "data/global_validation.jsonl",
                          args.output_dir,
                          num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
                          batch_size=args.batch_size)

    elif args.experiment == "exp1_full":
        # Use global validation (includes both NYT and synthetic)
        train_single_phase("data/experiment1/full_train.jsonl",
                          "data/global_validation.jsonl",
                          args.output_dir,
                          num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
                          batch_size=args.batch_size)

    elif args.experiment == "exp2_structured":
        # Use structured validation for structured training
        train_single_phase("data/experiment2/structured_only_train.jsonl",
                          "data/experiment2/validation_structured.jsonl",
                          args.output_dir,
                          num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
                          batch_size=args.batch_size)

    elif args.experiment == "exp2_unstructured":
        # Use unstructured validation for unstructured training
        train_single_phase("data/experiment2/unstructured_only_train.jsonl",
                          "data/experiment2/validation_unstructured.jsonl",
                          args.output_dir,
                          num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
                          batch_size=args.batch_size)

    elif args.experiment == "exp2_mixed":
        # Mixed training uses mixed validation (half structured, half unstructured)
        train_single_phase("data/experiment2/mixed_train.jsonl",
                          "data/experiment2/validation_mixed.jsonl",
                          args.output_dir,
                          num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
                          batch_size=args.batch_size)

    elif args.experiment == "exp2_sequential":
        # Two-phase training
        print("\n" + "="*80)
        print("SEQUENTIAL TRAINING (2 PHASES)")
        print("="*80)

        phase1_dir = f"{args.output_dir}/phase1_unstructured"
        train_single_phase(
            "data/experiment2/sequential_phase1_unstructured.jsonl",
            "data/experiment2/validation_unstructured.jsonl",
            phase1_dir,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size
        )

        phase2_dir = f"{args.output_dir}/phase2_structured_final"
        train_single_phase(
            "data/experiment2/sequential_phase2_structured.jsonl",
            "data/experiment2/validation_structured.jsonl",
            phase2_dir,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            model_to_load=phase1_dir
        )

    elif args.experiment == "exp3_no_warmup":
        # Direct training on full dataset with validation
        train_single_phase("data/experiment3/full_augmented.jsonl",
                          "data/global_validation.jsonl",
                          args.output_dir,
                          num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
                          batch_size=args.batch_size)

    elif args.experiment == "exp3_warmup":
        # Two-phase training
        print("\n" + "="*80)
        print("WARMUP TRAINING (2 PHASES)")
        print("="*80)

        # Phase 1: Preconn warmup with LOWER LR and NO validation (different task type)
        warmup_lr = args.lr / 2  # Use half the standard LR for warmup (e.g., 1e-4 instead of 2e-4)
        phase1_dir = f"{args.output_dir}/phase1_preconn"
        print(f"\nPhase 1: Warmup on Pre-Connections (LR={warmup_lr}, no validation)")
        train_single_phase(
            "data/experiment3/preconn_warmup.jsonl",
            None,  # Disable validation - different task type
            phase1_dir,
            num_train_epochs=args.epochs,
            learning_rate=warmup_lr,
            batch_size=args.batch_size
        )

        # Phase 2: Full training with standard LR and validation
        phase2_dir = f"{args.output_dir}/phase2_full_final"
        print(f"\nPhase 2: Full training on Connections (LR={args.lr}, with validation)")
        train_single_phase(
            "data/experiment3/full_augmented.jsonl",
            "data/global_validation.jsonl",
            phase2_dir,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            model_to_load=phase1_dir
        )

    elif args.experiment == "exp3_staged":
        # Three-stage training: preconn → synthetic → nyt
        print("\n" + "="*80)
        print("STAGED TRAINING (3 PHASES)")
        print("="*80)

        # Phase 1: Preconn warmup with LOWER LR and NO validation (different task type)
        warmup_lr = args.lr / 2  # Use half the standard LR for warmup
        phase1_dir = f"{args.output_dir}/phase1_preconn"
        print(f"\nPhase 1: Warmup on Pre-Connections (LR={warmup_lr}, no validation)")
        train_single_phase(
            "data/experiment3/preconn_warmup.jsonl",
            None,  # Disable validation - different task type
            phase1_dir,
            num_train_epochs=args.epochs,
            learning_rate=warmup_lr,
            batch_size=args.batch_size
        )

        # Phase 2: Synthetic with LOWER LR and NO validation (warming up to NYT difficulty)
        phase2_dir = f"{args.output_dir}/phase2_synthetic"
        print(f"\nPhase 2: Training on Synthetic Connections (LR={warmup_lr}, no validation)")
        train_single_phase(
            "data/experiment3/synthetic_component.jsonl",
            None,  # Disable validation - not representative of final task (NYT only)
            phase2_dir,
            num_train_epochs=args.epochs,
            learning_rate=warmup_lr,
            batch_size=args.batch_size,
            model_to_load=phase1_dir
        )

        # Phase 3: NYT with standard LR and validation
        phase3_dir = f"{args.output_dir}/phase3_nyt_final"
        print(f"\nPhase 3: Training on NYT Connections (LR={args.lr}, with validation)")
        train_single_phase(
            "data/experiment3/nyt_component.jsonl",
            "data/global_validation.jsonl",
            phase3_dir,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            model_to_load=phase2_dir
        )


if __name__ == "__main__":
    main()
