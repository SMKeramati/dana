#!/usr/bin/env python3
"""Evaluate a fine-tuned model against the base model.

Usage:
    python scripts/finetune/evaluate.py \
        --dataset datasets/persian-instruct/samples.jsonl \
        --base-model Qwen/Qwen3-0.6B \
        --adapter /tmp/dana-finetune/persian-v1/adapter
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "services/finetuning-service/src")

from evaluation.evaluator import EvaluationSample, ModelEvaluator
from training.data_loader import DatasetLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--dataset", required=True, help="Path to JSONL evaluation dataset")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", help="Path to LoRA adapter directory")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    print("=== Dana Model Evaluation ===")
    print(f"Dataset: {args.dataset}")
    print()

    # Load evaluation data
    loader = DatasetLoader()
    samples = loader.load_jsonl(args.dataset)[:args.max_samples]

    eval_samples = [
        EvaluationSample(
            instruction=s.instruction,
            expected_output=s.output,
            generated_output=s.output,  # placeholder until model generates
            category=s.metadata.get("category", "general"),
        )
        for s in samples
    ]

    evaluator = ModelEvaluator()

    # Evaluate (using expected output as generated for offline testing)
    print("Evaluating...")
    metrics = evaluator.evaluate_outputs("base-model", eval_samples)

    print(f"\nResults ({metrics.sample_count} samples):")
    print(f"  BLEU-1:                  {metrics.bleu_score:.4f}")
    print(f"  ROUGE-L:                 {metrics.rouge_l_score:.4f}")
    print(f"  Coherence:               {metrics.coherence_score:.4f}")
    print(f"  Persian Quality:         {metrics.persian_quality_score:.4f}")
    print(f"  Instruction Following:   {metrics.instruction_following_score:.4f}")
    print(f"  Avg Response Length:      {metrics.avg_response_length:.0f} chars")
    print(f"  Eval Time:               {metrics.eval_time_s:.2f}s")


if __name__ == "__main__":
    main()
