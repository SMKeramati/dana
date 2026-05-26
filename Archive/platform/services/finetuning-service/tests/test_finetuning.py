"""Tests for fine-tuning service modules (offline - no GPU needed)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.evaluation.evaluator import (
    EvaluationSample,
    ModelEvaluator,
    compute_bleu_unigram,
    compute_rouge_l,
    score_coherence,
    score_instruction_following,
    score_persian_quality,
)
from src.training.data_loader import (
    DatasetLoader,
    TrainingSample,
    detect_language,
    normalize_persian,
)
from src.training.data_validator import DataValidator
from src.training.trainer import (
    LoRAConfig,
    TrainingConfig,
    TrainingResult,
    cosine_lr_schedule,
)

# =========================================================================
# Data Loader Tests
# =========================================================================


class TestNormalizePersian:
    def test_arabic_kaf_to_persian(self) -> None:
        assert normalize_persian("\u0643\u062a\u0627\u0628") == "\u06a9\u062a\u0627\u0628"

    def test_arabic_ya_to_persian(self) -> None:
        assert normalize_persian("\u0639\u0644\u064a") == "\u0639\u0644\u06cc"

    def test_arabic_digits_to_latin(self) -> None:
        assert normalize_persian("\u0661\u0662\u0663") == "123"

    def test_persian_digits_to_latin(self) -> None:
        assert normalize_persian("\u06f4\u06f5\u06f6") == "456"

    def test_whitespace_normalized(self) -> None:
        assert normalize_persian("hello   world") == "hello world"

    def test_empty_string(self) -> None:
        assert normalize_persian("") == ""


class TestDetectLanguage:
    def test_persian_text(self) -> None:
        assert detect_language("سلام دنیا. این یک متن فارسی است.") == "fa"

    def test_english_text(self) -> None:
        assert detect_language("Hello world. This is English.") == "en"

    def test_mixed_mostly_persian(self) -> None:
        assert detect_language("سلام دنیا Python برنامه فارسی است") == "fa"

    def test_empty(self) -> None:
        assert detect_language("") == "en"


class TestTrainingSample:
    def test_prompt_with_system(self) -> None:
        s = TrainingSample(instruction="سلام", output="درود", system="تو دستیار هستی")
        assert "<|system|>" in s.prompt
        assert "<|user|>" in s.prompt

    def test_prompt_without_system(self) -> None:
        s = TrainingSample(instruction="سلام", output="درود")
        assert "<|system|>" not in s.prompt
        assert "<|user|>" in s.prompt

    def test_completion(self) -> None:
        s = TrainingSample(instruction="x", output="y")
        assert "<|assistant|>" in s.completion

    def test_full_text(self) -> None:
        s = TrainingSample(instruction="x", output="y")
        assert s.full_text == f"{s.prompt}\n{s.completion}"


class TestDatasetLoader:
    def _make_jsonl(self, samples: list[dict], tmp: Path) -> Path:
        path = tmp / "data.jsonl"
        with path.open("w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return path

    def test_load_jsonl_basic(self, tmp_path: Path) -> None:
        path = self._make_jsonl([
            {"instruction": "What is 2+2?", "output": "4. Two plus two equals four."},
        ], tmp_path)
        loader = DatasetLoader(normalize=False, min_output_length=1)
        samples = loader.load_jsonl(path)
        assert len(samples) == 1
        assert samples[0].instruction == "What is 2+2?"

    def test_load_jsonl_skips_invalid(self, tmp_path: Path) -> None:
        path = self._make_jsonl([
            {"instruction": "Valid question?", "output": "Valid answer here."},
            {"instruction": "", "output": "no instruction"},
            {"instruction": "no output", "output": ""},
        ], tmp_path)
        loader = DatasetLoader(normalize=False, min_output_length=1)
        samples = loader.load_jsonl(path)
        assert len(samples) == 1

    def test_load_jsonl_persian(self, tmp_path: Path) -> None:
        path = self._make_jsonl([
            {"instruction": "پایتخت ایران کجاست؟", "output": "پایتخت ایران تهران است."},
        ], tmp_path)
        loader = DatasetLoader(normalize=True, min_output_length=5)
        samples = loader.load_jsonl(path)
        assert len(samples) == 1

    def test_load_alpaca(self, tmp_path: Path) -> None:
        path = tmp_path / "alpaca.json"
        path.write_text(json.dumps([
            {"instruction": "Translate to Persian", "input": "Hello", "output": "This is the translation result."},
        ]))
        loader = DatasetLoader(normalize=False, min_output_length=5)
        samples = loader.load_alpaca(path)
        assert len(samples) == 1
        assert samples[0].input == "Hello"

    def test_compute_stats(self) -> None:
        samples = [
            TrainingSample(instruction="سلام", output="درود بر شما"),
            TrainingSample(instruction="Hello world", output="Hi there friend"),
        ]
        loader = DatasetLoader()
        stats = loader.compute_stats(samples)
        assert stats.total_samples == 2
        assert "fa" in stats.languages_detected

    def test_split(self) -> None:
        samples = [TrainingSample(instruction=f"q{i}", output=f"answer number {i}") for i in range(20)]
        loader = DatasetLoader()
        train, val = loader.split(samples, val_ratio=0.2)
        assert len(train) == 16
        assert len(val) == 4

    def test_load_messages_format(self, tmp_path: Path) -> None:
        path = self._make_jsonl([{
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is AI and tell me more?"},
                {"role": "assistant", "content": "AI is artificial intelligence technology."},
            ]
        }], tmp_path)
        loader = DatasetLoader(normalize=False, min_output_length=5)
        samples = loader.load_jsonl(path)
        assert len(samples) == 1
        assert "AI" in samples[0].output


# =========================================================================
# Data Validator Tests
# =========================================================================


class TestDataValidator:
    def _make_sample(self, **kwargs: str) -> TrainingSample:
        defaults = {
            "instruction": "این یک سوال تست است؟",
            "output": "این یک پاسخ تست است که باید به اندازه کافی طولانی باشد.",
        }
        defaults.update(kwargs)
        return TrainingSample(**defaults)

    def test_valid_sample_passes(self) -> None:
        v = DataValidator()
        report = v.validate([self._make_sample()])
        assert report.valid_samples == 1
        assert report.is_acceptable

    def test_empty_instruction_fails(self) -> None:
        v = DataValidator()
        report = v.validate([self._make_sample(instruction="")])
        assert report.valid_samples == 0
        assert report.format_errors == 1

    def test_empty_output_fails(self) -> None:
        v = DataValidator()
        report = v.validate([self._make_sample(output="")])
        assert report.format_errors == 1

    def test_short_output_fails(self) -> None:
        v = DataValidator(min_output_len=20)
        report = v.validate([self._make_sample(output="کوتاه")])
        assert report.too_short == 1

    def test_duplicate_detection(self) -> None:
        s = self._make_sample()
        v = DataValidator()
        report = v.validate([s, s])
        assert report.duplicates_found == 1

    def test_toxicity_detection(self) -> None:
        v = DataValidator()
        report = v.validate([self._make_sample(output="این متن شامل تروریسم و خشونت است.")])
        assert report.toxicity_flagged >= 1

    def test_placeholder_output_rejected(self) -> None:
        v = DataValidator()
        report = v.validate([self._make_sample(output="[TODO: fill this in]")])
        assert report.valid_samples == 0

    def test_filter_valid(self) -> None:
        v = DataValidator()
        samples = [
            self._make_sample(),
            self._make_sample(instruction=""),
        ]
        filtered = v.filter_valid(samples)
        assert len(filtered) == 1

    def test_language_mismatch_warning(self) -> None:
        v = DataValidator(expected_language="fa")
        report = v.validate([self._make_sample(
            instruction="What is Python programming language?",
            output="Python is a versatile programming language used worldwide.",
        )])
        lang_issues = [i for i in report.issues if i.issue_type == "language_mismatch"]
        assert len(lang_issues) >= 1

    def test_identical_instruction_output_rejected(self) -> None:
        text = "این یک متن تکراری است"
        v = DataValidator()
        report = v.validate([self._make_sample(instruction=text, output=text)])
        assert report.valid_samples == 0


# =========================================================================
# Trainer Config Tests
# =========================================================================


class TestTrainerConfig:
    def test_lora_scaling(self) -> None:
        cfg = LoRAConfig(r=16, alpha=32)
        assert cfg.scaling == 2.0

    def test_effective_batch_size(self) -> None:
        cfg = TrainingConfig(batch_size=4, gradient_accumulation_steps=8)
        assert cfg.effective_batch_size == 32

    def test_cosine_lr_warmup(self) -> None:
        lr = cosine_lr_schedule(step=5, total_steps=100, base_lr=1e-3, warmup_steps=10)
        assert lr == pytest.approx(5e-4, rel=0.01)

    def test_cosine_lr_at_warmup_end(self) -> None:
        lr = cosine_lr_schedule(step=10, total_steps=100, base_lr=1e-3, warmup_steps=10)
        assert lr == pytest.approx(1e-3, rel=0.01)

    def test_cosine_lr_at_end(self) -> None:
        lr = cosine_lr_schedule(step=100, total_steps=100, base_lr=1e-3, warmup_steps=10)
        assert lr == pytest.approx(1e-4, rel=0.05)  # min_lr_ratio=0.1

    def test_cosine_lr_mid_training(self) -> None:
        lr = cosine_lr_schedule(step=55, total_steps=100, base_lr=1e-3, warmup_steps=10)
        assert 1e-4 < lr < 1e-3

    def test_training_result_default(self) -> None:
        r = TrainingResult(job_id="test", status="completed")
        assert r.best_val_loss == float("inf")
        assert r.metrics_history == []


# =========================================================================
# Evaluator Tests
# =========================================================================


class TestBLEU:
    def test_identical_strings(self) -> None:
        assert compute_bleu_unigram("hello world", "hello world") == pytest.approx(1.0, rel=0.01)

    def test_no_overlap(self) -> None:
        assert compute_bleu_unigram("hello world", "foo bar") == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self) -> None:
        score = compute_bleu_unigram("the cat sat on the mat", "the cat on the mat")
        assert 0.5 < score <= 1.0

    def test_empty_hypothesis(self) -> None:
        assert compute_bleu_unigram("hello", "") == 0.0

    def test_brevity_penalty(self) -> None:
        long_ref = "this is a much longer reference sentence with many words"
        short_hyp = "this is"
        score = compute_bleu_unigram(long_ref, short_hyp)
        assert score < 0.5  # brevity penalty should reduce score


class TestROUGEL:
    def test_identical(self) -> None:
        assert compute_rouge_l("a b c d", "a b c d") == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert compute_rouge_l("a b c", "x y z") == pytest.approx(0.0)

    def test_partial_match(self) -> None:
        score = compute_rouge_l("the cat sat on the mat", "the cat on the rug")
        assert 0.3 < score < 1.0

    def test_empty(self) -> None:
        assert compute_rouge_l("hello", "") == 0.0


class TestPersianQuality:
    def test_pure_persian(self) -> None:
        score = score_persian_quality("این یک متن فارسی است. کیفیت آن باید بالا باشد.")
        assert score > 0.5

    def test_english_text(self) -> None:
        score = score_persian_quality("This is English text with no Persian.")
        assert score < 0.7

    def test_empty(self) -> None:
        assert score_persian_quality("") == 0.0


class TestInstructionFollowing:
    def test_good_answer(self) -> None:
        score = score_instruction_following(
            "پایتخت ایران کجاست؟",
            "پایتخت ایران تهران است. تهران بزرگترین شهر ایران است.",
        )
        assert score > 0.5

    def test_empty_output(self) -> None:
        assert score_instruction_following("سوال", "") == 0.0

    def test_echo_penalty(self) -> None:
        score = score_instruction_following("سلام", "سلام")
        assert score < 0.2


class TestCoherence:
    def test_coherent_text(self) -> None:
        text = "این یک متن منسجم است. چندین جمله دارد. هر جمله معنی‌دار است."
        score = score_coherence(text)
        assert score > 0.3

    def test_empty(self) -> None:
        assert score_coherence("") == 0.0

    def test_short_text(self) -> None:
        assert score_coherence("short") == 0.0


class TestModelEvaluator:
    def test_evaluate_basic(self) -> None:
        samples = [
            EvaluationSample(
                instruction="What is AI?",
                expected_output="AI is artificial intelligence.",
                generated_output="AI is artificial intelligence technology.",
            ),
        ]
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_outputs("test-model", samples)
        assert metrics.sample_count == 1
        assert metrics.bleu_score > 0
        assert metrics.rouge_l_score > 0

    def test_compare_improved(self) -> None:
        from src.evaluation.evaluator import ModelEvalMetrics
        base = ModelEvalMetrics(
            model_name="base",
            coherence_score=0.5,
            persian_quality_score=0.5,
            instruction_following_score=0.5,
            bleu_score=0.3,
            rouge_l_score=0.3,
        )
        finetuned = ModelEvalMetrics(
            model_name="finetuned",
            coherence_score=0.7,
            persian_quality_score=0.8,
            instruction_following_score=0.7,
            bleu_score=0.5,
            rouge_l_score=0.5,
        )
        evaluator = ModelEvaluator()
        result = evaluator.compare(base, finetuned)
        assert result.verdict == "improved"
        assert all(v > 0 for v in result.improvement.values())

    def test_compare_degraded(self) -> None:
        from src.evaluation.evaluator import ModelEvalMetrics
        base = ModelEvalMetrics(model_name="base", bleu_score=0.8, rouge_l_score=0.8,
                                coherence_score=0.8, persian_quality_score=0.8,
                                instruction_following_score=0.8)
        finetuned = ModelEvalMetrics(model_name="ft", bleu_score=0.3, rouge_l_score=0.3,
                                     coherence_score=0.3, persian_quality_score=0.3,
                                     instruction_following_score=0.3)
        evaluator = ModelEvaluator()
        result = evaluator.compare(base, finetuned)
        assert result.verdict == "degraded"
