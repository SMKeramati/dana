#!/usr/bin/env python3
"""Train a LoRA/QLoRA adapter on a dataset.

Usage:
    python scripts/finetune/train_lora.py \
        --dataset datasets/persian-instruct/samples.jsonl \
        --base-model Qwen/Qwen3-0.6B \
        --output /tmp/dana-finetune/persian-v1 \
        --epochs 3 --lr 2e-4 --rank 16
"""
from __future__ import annotations

import argparse
import sys
import uuid

sys.path.insert(0, "services/finetuning-service/src")

from training.data_loader import DatasetLoader
from training.data_validator import DataValidator
from training.trainer import LoRAConfig, LoRATrainer, QuantizationMode, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Dana LoRA Fine-tuning Script")
    parser.add_argument("--dataset", required=True, help="Path to JSONL/JSON dataset")
    parser.add_argument("--format", default="jsonl", choices=["jsonl", "alpaca", "sharegpt"])
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="/tmp/dana-finetune")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--quant", default="int4", choices=["none", "int8", "int4"])
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--validate-data", action="store_true", default=True)
    args = parser.parse_args()

    job_id = f"ft-{uuid.uuid4().hex[:8]}"
    print("=== Dana LoRA Fine-tuning ===")
    print(f"Job ID: {job_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Base model: {args.base_model}")
    print(f"LoRA rank: {args.rank}, alpha: {args.alpha}")
    print(f"Quantization: {args.quant}")
    print()

    # 1. Load dataset
    loader = DatasetLoader(normalize=True, max_length=args.max_seq_length)
    if args.format == "jsonl":
        samples = loader.load_jsonl(args.dataset)
    elif args.format == "alpaca":
        samples = loader.load_alpaca(args.dataset)
    else:
        samples = loader.load_sharegpt(args.dataset)

    stats = loader.compute_stats(samples)
    print(f"Loaded {stats.total_samples} samples")
    print(f"  Avg prompt length: {stats.avg_prompt_length:.0f} chars")
    print(f"  Avg completion length: {stats.avg_completion_length:.0f} chars")
    print(f"  Languages: {stats.languages_detected}")
    print()

    # 2. Validate data
    if args.validate_data:
        validator = DataValidator(expected_language="fa")
        report = validator.validate(samples)
        print(f"Validation: {report.valid_samples}/{report.total_samples} valid ({report.pass_rate:.1%})")
        if report.duplicates_found:
            print(f"  Duplicates: {report.duplicates_found}")
        if report.toxicity_flagged:
            print(f"  Toxicity flagged: {report.toxicity_flagged}")

        samples = validator.filter_valid(samples, report)
        print(f"  After filtering: {len(samples)} samples")
        print()

    # 3. Split train/val
    train_samples, val_samples = loader.split(samples, val_ratio=args.val_ratio)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    print()

    # 4. Configure training
    config = TrainingConfig(
        base_model=args.base_model,
        quantization=QuantizationMode(args.quant),
        lora=LoRAConfig(r=args.rank, alpha=args.alpha),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        output_dir=args.output,
    )

    # 5. Train
    trainer = LoRATrainer(config)
    print("Preparing model...")
    model_info = trainer.prepare_model()
    print(f"  Trainable params: {model_info['trainable_params']:,} ({model_info['trainable_percent']}%)")
    print(f"  Total params: {model_info['total_params']:,}")
    print()

    print("Starting training...")
    result = trainer.train(train_samples, val_samples, job_id=job_id)

    print()
    print(f"=== Training {'Complete' if result.status == 'completed' else 'Failed'} ===")
    print(f"  Status: {result.status}")
    print(f"  Steps: {result.total_steps}")
    print(f"  Final loss: {result.final_train_loss:.4f}")
    print(f"  Time: {result.training_time_s:.1f}s")
    if result.adapter_path:
        print(f"  Adapter saved: {result.adapter_path}")
    if result.error:
        print(f"  Error: {result.error}")


if __name__ == "__main__":
    main()
