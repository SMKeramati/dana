"""Custom dataset loader and formatter for fine-tuning.

Daneshbonyan: Internal R&D - Custom Data Pipeline

Supports multiple input formats (Alpaca, ShareGPT, JSONL) and converts
them to a unified training format. Handles Persian text normalization,
tokenization alignment, and train/validation splitting.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dana_common.logging import get_logger

logger = get_logger(__name__)

# Persian text normalization
_PERSIAN_NORMALIZE_MAP = {
    "\u0643": "\u06a9",  # Arabic kaf -> Persian kaf
    "\u064a": "\u06cc",  # Arabic ya -> Persian ya
    "\u0649": "\u06cc",  # Alef maqsura -> Persian ya
    "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3",
    "\u0664": "4", "\u0665": "5", "\u0666": "6", "\u0667": "7",
    "\u0668": "8", "\u0669": "9",
    "\u06f0": "0", "\u06f1": "1", "\u06f2": "2", "\u06f3": "3",
    "\u06f4": "4", "\u06f5": "5", "\u06f6": "6", "\u06f7": "7",
    "\u06f8": "8", "\u06f9": "9",
}


@dataclass
class TrainingSample:
    """A single training example in unified format."""
    instruction: str
    input: str = ""
    output: str = ""
    system: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        parts = []
        if self.system:
            parts.append(f"<|system|>\n{self.system}")
        parts.append(f"<|user|>\n{self.instruction}")
        if self.input:
            parts.append(self.input)
        return "\n".join(parts)

    @property
    def completion(self) -> str:
        return f"<|assistant|>\n{self.output}"

    @property
    def full_text(self) -> str:
        return f"{self.prompt}\n{self.completion}"


@dataclass
class DatasetStats:
    total_samples: int = 0
    avg_prompt_length: float = 0.0
    avg_completion_length: float = 0.0
    max_prompt_length: int = 0
    max_completion_length: int = 0
    languages_detected: dict[str, int] = field(default_factory=dict)
    format_source: str = ""


def normalize_persian(text: str) -> str:
    """Normalize Persian/Arabic text for consistent training."""
    for src, dst in _PERSIAN_NORMALIZE_MAP.items():
        text = text.replace(src, dst)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]", "", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Simple heuristic language detection."""
    persian_chars = len(re.findall(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]", text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    total = max(persian_chars + latin_chars, 1)
    if persian_chars / total > 0.3:
        return "fa"
    return "en"


class DatasetLoader:
    """Load and convert training datasets from multiple formats.

    Daneshbonyan: Internal R&D - Custom Data Pipeline
    """

    def __init__(
        self,
        normalize: bool = True,
        max_length: int = 4096,
        min_output_length: int = 10,
    ) -> None:
        self._normalize = normalize
        self._max_length = max_length
        self._min_output_length = min_output_length

    def load_jsonl(self, path: str | Path) -> list[TrainingSample]:
        """Load from JSONL file (one JSON object per line)."""
        path = Path(path)
        samples: list[TrainingSample] = []
        skipped = 0

        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("invalid_json", line=line_no, path=str(path))
                    skipped += 1
                    continue

                sample = self._parse_sample(obj)
                if sample and self._validate(sample):
                    samples.append(sample)
                else:
                    skipped += 1

        logger.info(
            "dataset_loaded",
            path=str(path),
            loaded=len(samples),
            skipped=skipped,
            format="jsonl",
        )
        return samples

    def load_alpaca(self, path: str | Path) -> list[TrainingSample]:
        """Load Alpaca-format JSON (list of {instruction, input, output})."""
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        samples: list[TrainingSample] = []
        skipped = 0

        for item in data:
            sample = TrainingSample(
                instruction=self._clean(item.get("instruction", "")),
                input=self._clean(item.get("input", "")),
                output=self._clean(item.get("output", "")),
                system=self._clean(item.get("system", "")),
            )
            if self._validate(sample):
                samples.append(sample)
            else:
                skipped += 1

        logger.info(
            "dataset_loaded",
            path=str(path),
            loaded=len(samples),
            skipped=skipped,
            format="alpaca",
        )
        return samples

    def load_sharegpt(self, path: str | Path) -> list[TrainingSample]:
        """Load ShareGPT-format JSON (conversations with turns)."""
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        samples: list[TrainingSample] = []
        skipped = 0

        for conversation in data:
            turns = conversation.get("conversations", [])
            system_msg = ""
            for i, turn in enumerate(turns):
                role = turn.get("from", turn.get("role", ""))
                value = turn.get("value", turn.get("content", ""))
                if role in ("system",):
                    system_msg = self._clean(value)
                elif role in ("human", "user") and i + 1 < len(turns):
                    next_turn = turns[i + 1]
                    next_role = next_turn.get("from", next_turn.get("role", ""))
                    if next_role in ("gpt", "assistant"):
                        sample = TrainingSample(
                            instruction=self._clean(value),
                            output=self._clean(next_turn.get("value", next_turn.get("content", ""))),
                            system=system_msg,
                        )
                        if self._validate(sample):
                            samples.append(sample)
                        else:
                            skipped += 1

        logger.info(
            "dataset_loaded",
            path=str(path),
            loaded=len(samples),
            skipped=skipped,
            format="sharegpt",
        )
        return samples

    def compute_stats(self, samples: list[TrainingSample]) -> DatasetStats:
        """Compute statistics over a dataset."""
        if not samples:
            return DatasetStats()

        prompt_lengths = [len(s.prompt) for s in samples]
        completion_lengths = [len(s.completion) for s in samples]
        lang_counts: dict[str, int] = {}
        for s in samples:
            lang = detect_language(s.instruction + " " + s.output)
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        return DatasetStats(
            total_samples=len(samples),
            avg_prompt_length=sum(prompt_lengths) / len(prompt_lengths),
            avg_completion_length=sum(completion_lengths) / len(completion_lengths),
            max_prompt_length=max(prompt_lengths),
            max_completion_length=max(completion_lengths),
            languages_detected=lang_counts,
        )

    def split(
        self,
        samples: list[TrainingSample],
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[TrainingSample], list[TrainingSample]]:
        """Split dataset into training and validation sets."""
        import random
        rng = random.Random(seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
        return shuffled[:split_idx], shuffled[split_idx:]

    def _parse_sample(self, obj: dict[str, Any]) -> TrainingSample | None:
        """Parse a generic JSON object into a TrainingSample."""
        if "instruction" in obj:
            return TrainingSample(
                instruction=self._clean(obj["instruction"]),
                input=self._clean(obj.get("input", "")),
                output=self._clean(obj.get("output", "")),
                system=self._clean(obj.get("system", "")),
                metadata={k: v for k, v in obj.items() if k not in {"instruction", "input", "output", "system"}},
            )
        if "prompt" in obj and "completion" in obj:
            return TrainingSample(
                instruction=self._clean(obj["prompt"]),
                output=self._clean(obj["completion"]),
            )
        if "messages" in obj:
            return self._parse_messages(obj["messages"])
        return None

    def _parse_messages(self, messages: list[dict[str, str]]) -> TrainingSample | None:
        """Parse OpenAI chat format messages into a single sample."""
        system = ""
        instruction = ""
        output = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system = self._clean(content)
            elif role == "user":
                instruction = self._clean(content)
            elif role == "assistant":
                output = self._clean(content)
        if instruction and output:
            return TrainingSample(instruction=instruction, output=output, system=system)
        return None

    def _clean(self, text: str) -> str:
        """Clean and optionally normalize text."""
        text = text.strip()
        if self._normalize:
            text = normalize_persian(text)
        return text

    def _validate(self, sample: TrainingSample) -> bool:
        """Validate a training sample."""
        if not sample.instruction:
            return False
        if not sample.output or len(sample.output) < self._min_output_length:
            return False
        if len(sample.full_text) > self._max_length * 4:  # rough char estimate
            return False
        return True
