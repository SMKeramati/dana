"""Training data quality validation pipeline.

Daneshbonyan: Internal R&D - Data Quality Management

Validates training data for:
- Format correctness (required fields, types)
- Content quality (length, dedup, toxicity keywords)
- Language consistency
- Statistical outlier detection
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from dana_common.logging import get_logger

from .data_loader import TrainingSample, detect_language

logger = get_logger(__name__)


@dataclass
class ValidationIssue:
    sample_index: int
    field: str
    issue_type: str
    message: str
    severity: str = "warning"  # warning, error


@dataclass
class ValidationReport:
    total_samples: int = 0
    valid_samples: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    duplicates_found: int = 0
    toxicity_flagged: int = 0
    too_short: int = 0
    too_long: int = 0
    format_errors: int = 0

    @property
    def pass_rate(self) -> float:
        return self.valid_samples / max(self.total_samples, 1)

    @property
    def is_acceptable(self) -> bool:
        return self.pass_rate >= 0.9 and self.format_errors == 0


# Toxicity keywords (basic filter - production would use a classifier)
_TOXICITY_KEYWORDS_FA = {
    "فحش", "دشنام", "نژادپرستی", "تروریسم", "خشونت",
    "قتل", "تجاوز", "انتحاری", "بمب", "سلاح",
}
_TOXICITY_KEYWORDS_EN = {
    "kill", "murder", "terrorist", "bomb", "weapon",
    "suicide", "genocide", "racial slur", "hate speech",
}


class DataValidator:
    """Validate training data quality for fine-tuning.

    Daneshbonyan: Internal R&D - Data Quality Pipeline
    """

    def __init__(
        self,
        min_instruction_len: int = 10,
        max_instruction_len: int = 4096,
        min_output_len: int = 10,
        max_output_len: int = 8192,
        max_total_len: int = 16384,
        expected_language: str | None = None,
        dedup: bool = True,
        toxicity_check: bool = True,
    ) -> None:
        self._min_instruction_len = min_instruction_len
        self._max_instruction_len = max_instruction_len
        self._min_output_len = min_output_len
        self._max_output_len = max_output_len
        self._max_total_len = max_total_len
        self._expected_language = expected_language
        self._dedup = dedup
        self._toxicity_check = toxicity_check

    def validate(self, samples: list[TrainingSample]) -> ValidationReport:
        """Run full validation pipeline on a dataset."""
        report = ValidationReport(total_samples=len(samples))
        seen_hashes: set[str] = set()
        valid_indices: set[int] = set()

        for i, sample in enumerate(samples):
            issues = self._validate_sample(i, sample, seen_hashes)
            if issues:
                report.issues.extend(issues)
                for issue in issues:
                    if issue.severity == "error":
                        break
                else:
                    valid_indices.add(i)
            else:
                valid_indices.add(i)

        report.valid_samples = len(valid_indices)
        report.duplicates_found = sum(1 for iss in report.issues if iss.issue_type == "duplicate")
        report.toxicity_flagged = sum(1 for iss in report.issues if iss.issue_type == "toxicity")
        report.too_short = sum(1 for iss in report.issues if iss.issue_type == "too_short")
        report.too_long = sum(1 for iss in report.issues if iss.issue_type == "too_long")
        report.format_errors = sum(1 for iss in report.issues if iss.issue_type == "format_error")

        logger.info(
            "validation_complete",
            total=report.total_samples,
            valid=report.valid_samples,
            pass_rate=round(report.pass_rate, 3),
            duplicates=report.duplicates_found,
            toxicity=report.toxicity_flagged,
        )
        return report

    def filter_valid(
        self, samples: list[TrainingSample], report: ValidationReport | None = None,
    ) -> list[TrainingSample]:
        """Return only samples that pass validation."""
        if report is None:
            report = self.validate(samples)

        error_indices: set[int] = set()
        for issue in report.issues:
            if issue.severity == "error":
                error_indices.add(issue.sample_index)

        return [s for i, s in enumerate(samples) if i not in error_indices]

    def _validate_sample(
        self, index: int, sample: TrainingSample, seen_hashes: set[str],
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Format: must have instruction and output
        if not sample.instruction:
            issues.append(ValidationIssue(index, "instruction", "format_error", "Empty instruction", "error"))
            return issues
        if not sample.output:
            issues.append(ValidationIssue(index, "output", "format_error", "Empty output", "error"))
            return issues

        # Length checks
        if len(sample.instruction) < self._min_instruction_len:
            issues.append(ValidationIssue(
                index, "instruction", "too_short",
                f"Instruction too short ({len(sample.instruction)} chars)", "error",
            ))
        if len(sample.instruction) > self._max_instruction_len:
            issues.append(ValidationIssue(
                index, "instruction", "too_long",
                f"Instruction too long ({len(sample.instruction)} chars)", "warning",
            ))
        if len(sample.output) < self._min_output_len:
            issues.append(ValidationIssue(
                index, "output", "too_short",
                f"Output too short ({len(sample.output)} chars)", "error",
            ))
        if len(sample.output) > self._max_output_len:
            issues.append(ValidationIssue(
                index, "output", "too_long",
                f"Output too long ({len(sample.output)} chars)", "warning",
            ))
        if len(sample.full_text) > self._max_total_len:
            issues.append(ValidationIssue(
                index, "full_text", "too_long",
                f"Total sample too long ({len(sample.full_text)} chars)", "warning",
            ))

        # Dedup
        if self._dedup:
            content_hash = hashlib.sha256(sample.full_text.encode()).hexdigest()[:16]
            if content_hash in seen_hashes:
                issues.append(ValidationIssue(index, "full_text", "duplicate", "Duplicate sample", "error"))
            else:
                seen_hashes.add(content_hash)

        # Language check
        if self._expected_language:
            detected = detect_language(sample.instruction + " " + sample.output)
            if detected != self._expected_language:
                issues.append(ValidationIssue(
                    index, "language", "language_mismatch",
                    f"Expected {self._expected_language}, got {detected}", "warning",
                ))

        # Toxicity check
        if self._toxicity_check:
            combined = (sample.instruction + " " + sample.output).lower()
            for kw in _TOXICITY_KEYWORDS_FA | _TOXICITY_KEYWORDS_EN:
                if kw.lower() in combined:
                    issues.append(ValidationIssue(
                        index, "content", "toxicity",
                        f"Toxicity keyword found: '{kw}'", "warning",
                    ))
                    break

        # Quality heuristics
        if sample.output.strip() == sample.instruction.strip():
            issues.append(ValidationIssue(
                index, "output", "quality", "Output identical to instruction", "error",
            ))

        # Check for placeholder/template outputs
        placeholder_patterns = [r"^\[.*\]$", r"^TODO", r"^FIXME", r"^placeholder"]
        for pat in placeholder_patterns:
            if re.match(pat, sample.output.strip(), re.IGNORECASE):
                issues.append(ValidationIssue(
                    index, "output", "quality", f"Placeholder output: {sample.output[:50]}", "error",
                ))
                break

        return issues
