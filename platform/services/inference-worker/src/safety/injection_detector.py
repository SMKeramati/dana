"""Custom prompt injection detection.

Daneshbonyan: Internal Design & Development

Pattern-based + heuristic detection of prompt injection attempts.
Combines regex pattern matching for known injection templates with
statistical heuristics (entropy analysis, role-boundary violations,
instruction-override detection) to flag suspicious prompts before
they reach the model.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionResult:
    """Result of running injection detection on a prompt."""

    threat_level: ThreatLevel
    score: float  # 0.0 (safe) to 1.0 (certain injection)
    matched_patterns: list[str] = field(default_factory=list)
    heuristic_flags: list[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return self.threat_level in (ThreatLevel.SAFE, ThreatLevel.LOW)


# ------------------------------------------------------------------
# Known injection patterns (compiled once at module load)
# ------------------------------------------------------------------

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    (
        "role_override",
        re.compile(
            r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|rules?|prompts?)",
            re.IGNORECASE,
        ),
        0.9,
    ),
    (
        "system_prompt_leak",
        re.compile(
            r"(?:reveal|show|print|output|repeat)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        "role_impersonation",
        re.compile(
            r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|evil|unrestricted|jailbroken)",
            re.IGNORECASE,
        ),
        0.9,
    ),
    (
        "delimiter_injection",
        re.compile(
            r"(?:---|\*\*\*|===|```)\s*(?:system|assistant|admin|root)\s*(?:---|\*\*\*|===|```)",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "encoding_evasion",
        re.compile(
            r"(?:base64|rot13|hex|unicode)\s*(?:decode|encode|convert)",
            re.IGNORECASE,
        ),
        0.6,
    ),
    (
        "instruction_boundary",
        re.compile(
            r"\[/?(?:INST|SYS|SYSTEM)\]",
            re.IGNORECASE,
        ),
        0.75,
    ),
    (
        "dan_jailbreak",
        re.compile(
            r"(?:DAN|Do\s+Anything\s+Now|developer\s+mode|god\s+mode)",
            re.IGNORECASE,
        ),
        0.95,
    ),
]


class PromptInjectionDetector:
    """Detects prompt injection attempts using patterns and heuristics.

    Daneshbonyan: Internal Design & Development
    """

    def __init__(
        self,
        pattern_weight: float = 0.6,
        heuristic_weight: float = 0.4,
        threshold_low: float = 0.25,
        threshold_medium: float = 0.5,
        threshold_high: float = 0.75,
        threshold_critical: float = 0.9,
    ) -> None:
        self._pattern_weight = pattern_weight
        self._heuristic_weight = heuristic_weight
        self._thresholds = {
            ThreatLevel.LOW: threshold_low,
            ThreatLevel.MEDIUM: threshold_medium,
            ThreatLevel.HIGH: threshold_high,
            ThreatLevel.CRITICAL: threshold_critical,
        }
        self._scan_count = 0
        self._threat_count = 0

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    def _scan_patterns(self, text: str) -> tuple[float, list[str]]:
        """Scan text against known injection patterns.

        Returns (max_score, list_of_matched_pattern_names).
        """
        matched: list[str] = []
        max_score = 0.0

        for name, pattern, score in _INJECTION_PATTERNS:
            if pattern.search(text):
                matched.append(name)
                max_score = max(max_score, score)

        return max_score, matched

    # ------------------------------------------------------------------
    # Heuristic analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _char_entropy(text: str) -> float:
        """Shannon entropy of the character distribution in *text*."""
        if not text:
            return 0.0
        freq: dict[str, int] = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        n = len(text)
        entropy = 0.0
        for count in freq.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _count_role_markers(text: str) -> int:
        """Count tokens that look like role boundary markers."""
        markers = re.findall(
            r"\b(?:system|user|assistant|human|ai|admin|root)\s*:",
            text,
            re.IGNORECASE,
        )
        return len(markers)

    @staticmethod
    def _excessive_special_chars(text: str, threshold: float = 0.3) -> bool:
        """Check if the ratio of non-alphanumeric characters is unusually high."""
        if not text:
            return False
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return (special / len(text)) > threshold

    @staticmethod
    def _instruction_density(text: str) -> float:
        """Ratio of imperative / instruction-like words in the text."""
        imperative_words = {
            "ignore", "forget", "disregard", "override", "bypass",
            "pretend", "act", "simulate", "behave", "reveal",
            "output", "print", "repeat", "translate", "decode",
        }
        words = re.findall(r"[a-zA-Z]+", text.lower())
        if not words:
            return 0.0
        hits = sum(1 for w in words if w in imperative_words)
        return hits / len(words)

    def _heuristic_score(self, text: str) -> tuple[float, list[str]]:
        """Compute a heuristic injection score in [0, 1]."""
        flags: list[str] = []
        scores: list[float] = []

        # Entropy check -- very low or very high can be suspicious
        entropy = self._char_entropy(text)
        if entropy < 2.0 and len(text) > 50:
            flags.append("low_entropy")
            scores.append(0.3)
        if entropy > 5.5:
            flags.append("high_entropy")
            scores.append(0.4)

        # Role markers
        role_count = self._count_role_markers(text)
        if role_count >= 2:
            flags.append(f"role_markers:{role_count}")
            scores.append(min(role_count * 0.25, 0.8))

        # Special character ratio
        if self._excessive_special_chars(text):
            flags.append("excessive_special_chars")
            scores.append(0.35)

        # Instruction density
        density = self._instruction_density(text)
        if density > 0.1:
            flags.append(f"instruction_density:{density:.2f}")
            scores.append(min(density * 3.0, 0.9))

        # Length anomaly -- extremely long user turns may be padding attacks
        if len(text) > 10_000:
            flags.append("excessive_length")
            scores.append(0.25)

        if not scores:
            return 0.0, flags

        return float(np.max(scores)), flags

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, text: str) -> DetectionResult:
        """Analyse *text* for prompt injection indicators."""
        self._scan_count += 1

        pattern_score, matched_patterns = self._scan_patterns(text)
        heuristic_score, heuristic_flags = self._heuristic_score(text)

        combined = (
            self._pattern_weight * pattern_score
            + self._heuristic_weight * heuristic_score
        )
        combined = min(combined, 1.0)

        # Determine threat level
        if combined >= self._thresholds[ThreatLevel.CRITICAL]:
            level = ThreatLevel.CRITICAL
        elif combined >= self._thresholds[ThreatLevel.HIGH]:
            level = ThreatLevel.HIGH
        elif combined >= self._thresholds[ThreatLevel.MEDIUM]:
            level = ThreatLevel.MEDIUM
        elif combined >= self._thresholds[ThreatLevel.LOW]:
            level = ThreatLevel.LOW
        else:
            level = ThreatLevel.SAFE

        if level not in (ThreatLevel.SAFE, ThreatLevel.LOW):
            self._threat_count += 1
            logger.warning(
                "Injection detected (%s, score=%.3f): patterns=%s flags=%s",
                level.value,
                combined,
                matched_patterns,
                heuristic_flags,
            )

        return DetectionResult(
            threat_level=level,
            score=combined,
            matched_patterns=matched_patterns,
            heuristic_flags=heuristic_flags,
        )

    def scan_multi_turn(self, messages: list[dict[str, str]]) -> DetectionResult:
        """Scan a multi-turn conversation for injection across messages.

        Each message should have ``role`` and ``content`` keys.
        """
        # Concatenate with role markers to detect cross-message attacks
        parts: list[str] = []
        for msg in messages:
            parts.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
        full_text = "\n".join(parts)
        return self.scan(full_text)

    @property
    def scan_count(self) -> int:
        return self._scan_count

    @property
    def threat_count(self) -> int:
        return self._threat_count

    @property
    def threat_rate(self) -> float:
        return self._threat_count / self._scan_count if self._scan_count > 0 else 0.0
