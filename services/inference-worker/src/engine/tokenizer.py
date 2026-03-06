"""Tokenizer wrapper with prefix caching for shared context detection.

Provides a lightweight tokenizer abstraction that tracks common prefixes
across requests so the inference engine can reuse KV cache entries for
shared system prompts and few-shot examples.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PrefixCacheEntry:
    """A cached tokenised prefix and its hash."""

    text: str
    token_ids: np.ndarray
    hash_key: str
    hit_count: int = 0


class SimpleTokenizer:
    """Byte-level tokenizer with prefix caching.

    This is a *simplified* tokenizer for environments where the real
    model tokenizer is not available (testing, benchmarking).  It maps
    each byte to a token id in [0, 255] with a few special tokens above
    that range.
    """

    PAD_ID: int = 256
    BOS_ID: int = 257
    EOS_ID: int = 258
    UNK_ID: int = 259
    VOCAB_SIZE: int = 260

    def __init__(self, max_prefix_cache: int = 256) -> None:
        self._prefix_cache: dict[str, PrefixCacheEntry] = {}
        self._max_prefix_cache = max_prefix_cache

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_bos: bool = True) -> np.ndarray:
        """Encode text to a 1-D int32 numpy array of token ids."""
        ids = [int(b) for b in text.encode("utf-8", errors="replace")]
        if add_bos:
            ids = [self.BOS_ID] + ids
        return np.array(ids, dtype=np.int32)

    def decode(self, token_ids: np.ndarray | list[int]) -> str:
        """Decode token ids back to text (best-effort)."""
        raw_bytes: list[int] = []
        for tid in token_ids:
            tid = int(tid)
            if 0 <= tid <= 255:
                raw_bytes.append(tid)
            # Skip special tokens
        return bytes(raw_bytes).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Prefix caching
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def cache_prefix(self, text: str) -> PrefixCacheEntry:
        """Tokenise *text* and store in the prefix cache."""
        h = self._hash_text(text)
        if h in self._prefix_cache:
            entry = self._prefix_cache[h]
            entry.hit_count += 1
            return entry

        # Evict oldest if at capacity
        if len(self._prefix_cache) >= self._max_prefix_cache:
            oldest_key = min(
                self._prefix_cache, key=lambda k: self._prefix_cache[k].hit_count
            )
            del self._prefix_cache[oldest_key]

        token_ids = self.encode(text, add_bos=True)
        entry = PrefixCacheEntry(text=text, token_ids=token_ids, hash_key=h)
        self._prefix_cache[h] = entry
        return entry

    def find_shared_prefix(self, text: str) -> tuple[PrefixCacheEntry | None, int]:
        """Find the longest cached prefix that matches the start of *text*.

        Returns ``(entry, match_length)`` where *match_length* is the number
        of matching tokens.  Returns ``(None, 0)`` on miss.
        """
        best_entry: PrefixCacheEntry | None = None
        best_length = 0

        encoded = self.encode(text, add_bos=True)

        for entry in self._prefix_cache.values():
            prefix_len = len(entry.token_ids)
            if prefix_len > len(encoded):
                continue
            if np.array_equal(encoded[:prefix_len], entry.token_ids):
                if prefix_len > best_length:
                    best_entry = entry
                    best_length = prefix_len

        if best_entry is not None:
            best_entry.hit_count += 1

        return best_entry, best_length

    def encode_with_prefix_cache(
        self, text: str
    ) -> tuple[np.ndarray, int]:
        """Encode text, returning (token_ids, num_cached_prefix_tokens).

        If a cached prefix matches, the caller can skip KV computation
        for the first *num_cached_prefix_tokens* positions.
        """
        entry, match_len = self.find_shared_prefix(text)
        token_ids = self.encode(text, add_bos=True)
        return token_ids, match_len

    @property
    def prefix_cache_size(self) -> int:
        return len(self._prefix_cache)

    def clear_prefix_cache(self) -> None:
        self._prefix_cache.clear()
