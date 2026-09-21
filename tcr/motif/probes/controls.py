"""Deterministic nonrecent length-matched motif controls."""

from __future__ import annotations

from collections.abc import Sequence

from ..schemas import ParsedBlock
from .serialization import motif_surface


def select_nonrecent_motif(
    blocks: Sequence[ParsedBlock], *, k_0based: int, m: int, tokenizer=None
) -> tuple[int, str] | None:
    recent = tuple(block.canonical_signature for block in blocks[k_0based - m + 1 : k_0based + 1])
    recent_surface = motif_surface(blocks, start=k_0based - m + 1, length=m)
    recent_len = len(tokenizer(recent_surface, add_special_tokens=False)["input_ids"]) if tokenizer else len(recent_surface)
    candidates: list[tuple[int, int, str]] = []
    latest_start = k_0based - 2 * m + 1
    for start in range(0, latest_start + 1):
        window = blocks[start : start + m]
        signature = tuple(block.canonical_signature for block in window)
        if signature == recent:
            continue
        surface = motif_surface(blocks, start=start, length=m)
        length = len(tokenizer(surface, add_special_tokens=False)["input_ids"]) if tokenizer else len(surface)
        candidates.append((abs(length - recent_len), start, surface))
    if not candidates:
        return None
    _, start, surface = min(candidates)
    return start, surface
