# coding=utf-8
"""Auxiliary repetition-severity metrics (continuous / non-loop signals).

These describe severity on a continuum, so a gradient is still visible when
the binary ``loop`` flag saturates at 0 or 1:

* ``hit_max_tokens`` -- ``finish_reason == "length"`` (generation ran to the cap),
* ``rep4``           -- ``1 - distinct-4-gram ratio`` (token-level),
* ``gzip_ratio``     -- compressed/raw byte ratio (lower = more repetitive).

Token-level metrics operate on the SAME token ids the loop flag uses (the text
is tokenised once by the caller), so all four unified metrics are consistent.
"""

import gzip
from typing import Sequence


def rep_n(token_ids: Sequence[int], n: int = 4) -> float:
    """Repetition rate ``1 - distinct-n-gram / total-n-gram`` (0.0 if too short)."""
    total = len(token_ids) - n + 1
    if total <= 0:
        return 0.0
    grams = {tuple(token_ids[i:i + n]) for i in range(total)}
    return 1.0 - len(grams) / total


def gzip_ratio(text: str) -> float:
    """Compressed/raw byte ratio; lower means more repetitive. 1.0 for empty text."""
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    return len(gzip.compress(raw, compresslevel=6)) / len(raw)


def hit_max_tokens(finish_reason: str) -> int:
    """1 iff generation stopped because it ran into the length cap."""
    return 1 if finish_reason == "length" else 0
