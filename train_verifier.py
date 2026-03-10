#!/usr/bin/env python3
"""train_verifier.py -- Generate synthetic data and fine-tune a small verifier model.

Usage:
    python train_verifier.py --principles principles.txt --output-dir ./my-verifier
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from openai import OpenAI
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SYNTH_MODEL = "gpt-4o-mini"
EXAMPLES_PER_PRINCIPLE = 30  # half compliant, half violating
EVAL_SPLIT = 0.15
MAX_SEQ_LEN = 512


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def generate_synthetic_examples(principles: list[str], client: OpenAI) -> list[dict]:
    """Use GPT-4o-mini to create compliant and violating examples for each principle."""
    examples = []
    for principle in principles:
        for label, directive in [("pass", "follows"), ("fail", "violates")]:
            n = EXAMPLES_PER_PRINCIPLE // 2
            prompt = (
                f"Generate {n} short, realistic AI assistant responses that each "
                f"{directive} the following principle:\n\n"
                f"Principle: \"{principle}\"\n\n"
                f"Return a JSON array of strings, nothing else."
            )
            resp = client.chat.completions.create(
                model=SYNTH_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                response_format={"type": "json_object"},
            )
            try:
                content = json.loads(resp.choices[0].message.content)
                # Handle {"responses": [...]} or [...]
                items = content if isinstance(content, list) else list(content.values())[0]
            except (json.JSONDecodeError, IndexError):
                print(f"  Warning: failed to parse response for '{principle}' ({label}), skipping batch")
                continue

            for text in items:
                violations = [principle] if label == "fail" else []
                confidence = round(random.uniform(0.80, 0.99), 2)
                verdict_json = json.dumps({"violations": violations, "confidence": confidence})
                examples.append({
                    "principle": principle,
                    "response": text,
                    "label": label,
                    "verdict": verdict_json,
                })
        print(f"  Generated {EXAMPLES_PER_PRINCIPLE} examples for: {principle[:60]}")
    random.shuffle(examples)
    return examples


def format_for_training(example: dict) -> str:
    """Format a single example as a chat-style prompt + completion."""
    return (
        f"<|user|>\n"
        f"Verify the following response against the principle: \"{example['principle']}\"\n\n"
        f"Response: \"{example['response']}\"\n"
        f"<|assistant|>\n"
        f"{example['verdict']}"
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(examples: list[dict], output_dir: Path, base_model: str):
    print(f"\nLoading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    # LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Split train / eval
    split_idx = int(len(examples) * (1 - EVAL_SPLIT))
    train_texts = [format_for_training(e) for e in examples[:split_idx]]
    eval_texts = [format_for_training(e) for e in examples[split_idx:]]

    def tokenize(texts):
        encodings = tokenizer(
            texts, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length"
        )
        encodings["labels"] = encodings["input_ids"].copy()
        return Dataset.from_dict(encodings)

    train_ds = tokenize(train_texts)
    eval_ds = tokenize(eval_texts)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print(f"\nTraining on {len(train_ds)} examples, evaluating on {len(eval_ds)}...")
    trainer.train()

    # Save merged model
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nModel saved to {output_dir}")
    return merged, tokenizer, examples[split_idx:]


# ---------------------------------------------------------------------------
# Functional evaluation
# ---------------------------------------------------------------------------
def evaluate(model, tokenizer, eval_examples: list[dict], output_dir: Path):
    """Generate verdicts on held-out set and compute metrics."""
    print(f"\nRunning functional evaluation on {len(eval_examples)} examples...")
    y_true, y_pred = [], []

    device = next(model.parameters()).device
    for ex in eval_examples:
        prompt = (
            f"<|user|>\n"
            f"Verify the following response against the principle: \"{ex['principle']}\"\n\n"
            f"Response: \"{ex['response']}\"\n"
            f"<|assistant|>\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Parse prediction
        try:
            verdict = json.loads(generated)
            pred_label = "fail" if verdict.get("violations") else "pass"
        except json.JSONDecodeError:
            pred_label = "fail"  # conservative default

        y_true.append(1 if ex["label"] == "fail" else 0)
        y_pred.append(1 if pred_label == "fail" else 0)

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "eval_count": len(eval_examples),
    }

    metrics_path = output_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nEvaluation metrics saved to {metrics_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a VerifAI verifier from principles.")
    parser.add_argument("--principles", required=True, help="Path to principles text file (one per line).")
    parser.add_argument("--output-dir", required=True, help="Directory to save trained model and metrics.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="HuggingFace model ID to fine-tune.")
    args = parser.parse_args()

    # Validate
    principles_path = Path(args.principles)
    if not principles_path.exists():
        sys.exit(f"Error: principles file not found: {principles_path}")

    principles = [line.strip() for line in principles_path.read_text().splitlines() if line.strip()]
    if not principles:
        sys.exit("Error: principles file is empty.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY environment variable is required.")

    client = OpenAI(api_key=api_key)

    # Step 1: Generate synthetic data
    print(f"Generating synthetic data for {len(principles)} principles...")
    examples = generate_synthetic_examples(principles, client)
    print(f"Total examples: {len(examples)}")

    # Save synthetic data
    data_path = output_dir / "synthetic_data.jsonl"
    with open(data_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Synthetic data saved to {data_path}")

    # Step 2: Fine-tune
    model, tokenizer, eval_examples = train(examples, output_dir, args.base_model)

    # Step 3: Evaluate
    evaluate(model, tokenizer, eval_examples, output_dir)

    print(f"\nDone! Your verifier is ready at: {output_dir}")


if __name__ == "__main__":
    main()
