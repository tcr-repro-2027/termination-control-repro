"""Adjacent complete-block motif repetition and capture detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

from .block_parser import ParsedBlock


@dataclass(frozen=True)
class MotifRun:
    signature_kind: str
    sequence_segment: int
    block_onset: int
    block_period: int
    block_num_repeats: int
    strict_block_end: int
    coverage_blocks: int
    motif_signatures: tuple[Hashable, ...]

    @property
    def second_copy_start(self) -> int:
        return self.block_onset + self.block_period

    def confirmation_block_exclusive(self, repeats: int) -> int:
        return self.block_onset + repeats * self.block_period


def _values(blocks: Sequence[ParsedBlock], kind: str) -> list[Hashable]:
    if kind == "triple":
        return [block.triple_signature for block in blocks]
    if kind == "quad":
        return [block.canonical_signature for block in blocks]
    raise ValueError(f"signature kind must be triple or quad, got {kind!r}")


def _segment_slices(blocks: Sequence[ParsedBlock]) -> list[tuple[int, int, int]]:
    if not blocks:
        return []
    slices: list[tuple[int, int, int]] = []
    start = 0
    segment = blocks[0].sequence_segment
    for index in range(1, len(blocks)):
        if blocks[index].sequence_segment != segment:
            slices.append((start, index, segment))
            start = index
            segment = blocks[index].sequence_segment
    slices.append((start, len(blocks), segment))
    return slices


def find_motif_runs(
    blocks: Sequence[ParsedBlock],
    *,
    signature_kind: str,
    min_repeats: int,
    max_period_blocks: int | None = None,
) -> list[MotifRun]:
    """Find every strict consecutive motif run inside contiguous valid segments."""
    if min_repeats < 2:
        raise ValueError("min_repeats must be at least 2")
    signatures = _values(blocks, signature_kind)
    runs: list[MotifRun] = []
    for seg_start, seg_end, segment in _segment_slices(blocks):
        local = signatures[seg_start:seg_end]
        n = len(local)
        upper = n // min_repeats
        if max_period_blocks is not None and max_period_blocks > 0:
            upper = min(upper, max_period_blocks)
        for period in range(1, upper + 1):
            run_start: int | None = None
            # local j denotes the second sequence element participating in the
            # p-shift equality.  A run of L equalities covers L+p blocks.
            for j in range(period, n + 1):
                equal = j < n and local[j] == local[j - period]
                if equal and run_start is None:
                    run_start = j
                if (not equal or j == n) and run_start is not None:
                    eq_end = j
                    eq_len = eq_end - run_start
                    coverage = eq_len + period
                    if coverage >= min_repeats * period:
                        onset_local = run_start - period
                        repeats = coverage // period
                        strict_end_local = onset_local + repeats * period
                        runs.append(
                            MotifRun(
                                signature_kind=signature_kind,
                                sequence_segment=segment,
                                block_onset=seg_start + onset_local,
                                block_period=period,
                                block_num_repeats=repeats,
                                strict_block_end=seg_start + strict_end_local,
                                coverage_blocks=coverage,
                                motif_signatures=tuple(local[onset_local:onset_local + period]),
                            )
                        )
                    run_start = None
    return sorted(
        runs,
        key=lambda run: (
            run.confirmation_block_exclusive(min_repeats),
            run.block_period,
            run.block_onset,
            -run.block_num_repeats,
        ),
    )


def primary_motif_event(
    runs: Sequence[MotifRun], *, confirmation_repeats: int
) -> MotifRun | None:
    if not runs:
        return None
    eligible = [run for run in runs if run.block_num_repeats >= confirmation_repeats]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda run: (
            run.confirmation_block_exclusive(confirmation_repeats),
            run.block_period,
            run.block_onset,
            -run.block_num_repeats,
        ),
    )
