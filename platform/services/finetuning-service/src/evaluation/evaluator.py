"""Post-training model evaluation.

Daneshbonyan: Internal R&D - Custom AI Evaluation Pipeline

Evaluates fine-tuned models against base models on:
- Perplexity (lower = better language modeling)
- Task accuracy (instruction following quality)
- Persian language quality (specific to Dana's use case)
- Response coherence and relevance scoring
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

from dana_common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationSample:
    """A sample for evaluation with expected output."""
    instruction: str
    expected_output: str
    generated_output: str = ""
    category: str = "general"


@dataclass
class ModelEvalMetrics:
    """Evaluation metrics for a single model."""
    model_name: str
    perplexity: float = 0.0
    avg_response_length: float = 0.0
    coherence_score: float = 0.0
    persian_quality_score: float = 0.0
    instruction_following_score: float = 0.0
    bleu_score: float = 0.0
    rouge_l_score: float = 0.0
    eval_time_s: float = 0.0
    sample_count: int = 0


@dataclass
class ComparisonResult:
    """Comparison between base model and fine-tuned model."""
    base_metrics: ModelEvalMetrics
    finetuned_metrics: ModelEvalMetrics
    improvement: dict[str, float] = field(default_factory=dict)
    verdict: str = ""  # improved, degraded, similar


def compute_perplexity(losses: list[float]) -> float:
    """Compute perplexity from a list of per-token losses."""
    if not losses:
        return float("inf")
    avg_loss = sum(losses) / len(losses)
    return math.exp(min(avg_loss, 100))  # cap to avoid overflow


def compute_bleu_unigram(reference: str, hypothesis: str) -> float:
    """Simplified unigram BLEU score (no external dependencies).

    Daneshbonyan: Internal R&D - Custom Evaluation Metric
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    matches = 0
    hyp_counts: dict[str, int] = {}
    for t in hyp_tokens:
        hyp_counts[t] = hyp_counts.get(t, 0) + 1

    for token, count in hyp_counts.items():
        matches += min(count, ref_counts.get(token, 0))

    precision = matches / len(hyp_tokens)

    # Brevity penalty
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(hyp_tokens))

    return bp * precision


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score using longest common subsequence.

    Daneshbonyan: Internal R&D - Custom Evaluation Metric
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # LCS via DP
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    precision = lcs_len / n
    recall = lcs_len / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_persian_quality(text: str) -> float:
    """Heuristic scorer for Persian text quality.

    Daneshbonyan: Internal R&D - Persian Language Quality Assessment

    Checks:
    - Persian character ratio (should be high for Persian responses)
    - Sentence structure (periods, question marks)
    - No excessive Latin text mixing
    - Proper use of half-space (ZWNJ)
    """
    if not text:
        return 0.0

    scores: list[float] = []

    # Persian character ratio
    persian_chars = len(re.findall(r"[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]", text))
    total_alpha = len(re.findall(r"[a-zA-Z\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]", text))
    if total_alpha > 0:
        ratio = persian_chars / total_alpha
        scores.append(min(ratio * 1.2, 1.0))  # slight boost

    # Sentence structure - has proper endings
    sentence_endings = len(re.findall(r"[.؟!،;\n]", text))
    sentences_expected = max(len(text) / 200, 1)
    structure_score = min(sentence_endings / sentences_expected, 1.0)
    scores.append(structure_score)

    # Length adequacy (not too short, not too long for a response)
    len_score = 1.0
    if len(text) < 20:
        len_score = len(text) / 20
    elif len(text) > 5000:
        len_score = max(0.5, 5000 / len(text))
    scores.append(len_score)

    # ZWNJ usage (indicates proper Persian typography)
    has_zwnj = "\u200c" in text
    if persian_chars > 50:
        scores.append(0.8 if has_zwnj else 0.5)

    return sum(scores) / len(scores) if scores else 0.0


def score_instruction_following(instruction: str, output: str) -> float:
    """Score how well the output follows the instruction.

    Daneshbonyan: Internal R&D - Custom Instruction Following Metric
    """
    if not output:
        return 0.0

    scores: list[float] = []

    # Non-empty, reasonable length response
    if len(output) > 10:
        scores.append(0.8)
    else:
        scores.append(0.2)

    # If instruction asks a question, output should not be a question
    is_question = "?" in instruction or "\u061f" in instruction
    output_is_question = output.strip().endswith(("?", "\u061f"))
    if is_question and not output_is_question:
        scores.append(0.9)
    elif is_question and output_is_question:
        scores.append(0.3)
    else:
        scores.append(0.7)

    # Key term overlap between instruction and output
    inst_words = set(instruction.lower().split())
    out_words = set(output.lower().split())
    if inst_words:
        overlap = len(inst_words & out_words) / len(inst_words)
        scores.append(min(overlap * 2, 1.0))

    # Not just echoing the instruction
    if output.strip() == instruction.strip():
        return 0.1

    return sum(scores) / len(scores) if scores else 0.0


def score_coherence(text: str) -> float:
    """Score text coherence using heuristics.

    Daneshbonyan: Internal R&D - Custom Coherence Assessment
    """
    if not text or len(text) < 10:
        return 0.0

    scores: list[float] = []

    # Repetition check (bigram repetition rate)
    words = text.lower().split()
    if len(words) >= 4:
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        unique_bigrams = len(set(bigrams))
        repetition_rate = 1 - (unique_bigrams / len(bigrams))
        scores.append(max(0, 1 - repetition_rate * 2))

    # Sentence variety (not all same length)
    sentences = re.split(r"[.!?\u061f\n]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        # Some variance is good (not all sentences same length)
        scores.append(min(math.sqrt(variance) / 5, 1.0))

    # Not truncated (ends with proper punctuation)
    if text.strip()[-1:] in ".!?\u061f\u06d4":
        scores.append(1.0)
    else:
        scores.append(0.5)

    return sum(scores) / len(scores) if scores else 0.0


class ModelEvaluator:
    """Evaluate and compare models before/after fine-tuning.

    Daneshbonyan: Internal R&D - Custom Model Evaluation Pipeline
    """

    def evaluate_outputs(
        self,
        model_name: str,
        samples: list[EvaluationSample],
    ) -> ModelEvalMetrics:
        """Evaluate generated outputs against expected outputs."""
        start = time.monotonic()

        bleu_scores = []
        rouge_scores = []
        coherence_scores = []
        persian_scores = []
        following_scores = []
        response_lengths = []

        for sample in samples:
            if not sample.generated_output:
                continue

            bleu_scores.append(compute_bleu_unigram(sample.expected_output, sample.generated_output))
            rouge_scores.append(compute_rouge_l(sample.expected_output, sample.generated_output))
            coherence_scores.append(score_coherence(sample.generated_output))
            persian_scores.append(score_persian_quality(sample.generated_output))
            following_scores.append(score_instruction_following(sample.instruction, sample.generated_output))
            response_lengths.append(len(sample.generated_output))

        n = max(len(bleu_scores), 1)

        return ModelEvalMetrics(
            model_name=model_name,
            avg_response_length=sum(response_lengths) / n,
            coherence_score=sum(coherence_scores) / n,
            persian_quality_score=sum(persian_scores) / n,
            instruction_following_score=sum(following_scores) / n,
            bleu_score=sum(bleu_scores) / n,
            rouge_l_score=sum(rouge_scores) / n,
            eval_time_s=time.monotonic() - start,
            sample_count=len(samples),
        )

    def compare(
        self,
        base_metrics: ModelEvalMetrics,
        finetuned_metrics: ModelEvalMetrics,
    ) -> ComparisonResult:
        """Compare base model vs fine-tuned model metrics."""
        improvement: dict[str, float] = {}
        improved_count = 0
        degraded_count = 0

        for metric_name in [
            "coherence_score", "persian_quality_score",
            "instruction_following_score", "bleu_score", "rouge_l_score",
        ]:
            base_val = getattr(base_metrics, metric_name, 0)
            ft_val = getattr(finetuned_metrics, metric_name, 0)
            if base_val > 0:
                delta = (ft_val - base_val) / base_val
            else:
                delta = ft_val
            improvement[metric_name] = round(delta * 100, 2)

            if delta > 0.02:
                improved_count += 1
            elif delta < -0.02:
                degraded_count += 1

        if improved_count > degraded_count:
            verdict = "improved"
        elif degraded_count > improved_count:
            verdict = "degraded"
        else:
            verdict = "similar"

        logger.info(
            "model_comparison",
            base=base_metrics.model_name,
            finetuned=finetuned_metrics.model_name,
            verdict=verdict,
            improvements=improvement,
        )

        return ComparisonResult(
            base_metrics=base_metrics,
            finetuned_metrics=finetuned_metrics,
            improvement=improvement,
            verdict=verdict,
        )
