# coding=utf-8
"""Periodicity-based loop detection (the decision kernel).

A loop is, structurally, a contiguous run of tokens that equals itself shifted
by a fixed period ``p`` -- i.e. ``line[j] == line[j - p]`` holds over a long,
*unbroken* stretch.  For each period ``p`` in ``[p_min, p_max]`` this module
finds the first contiguous run of positions satisfying ``line[j] == line[j-p]``
that is long enough to contain ``k`` back-to-back copies of the ``p``-token
unit (run length ``>= (k - 1) * p``), and reports the earliest such run.

Requiring an *unbroken* periodic run makes the decision robust: any gap resets
the run, so a short motif scattered across the response can never accumulate
into a false positive.  Token ids are compared directly (no string matching).

Compared to the previous implementation, the run is reported as a single
:class:`PeriodicRun` carrying ``onset`` (loop start token index), ``period``
(repeating-unit length in tokens) and ``num_repeats`` (consecutive copies) --
the three quantities the downstream analyses depend on.
"""

import dataclasses
from typing import List, Optional, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class PeriodicRun:
    """The earliest consecutive periodic run found in a token sequence."""
    unit: List[int]     # the repeating token-id unit (length == period)
    onset: int          # token index where the periodic run starts
    period: int         # repeating-unit length, in tokens
    num_repeats: int    # consecutive copies of the unit in the run


def find_periodic_run(
    line: Sequence[int], p_min: int, p_max: int, k: int
) -> Optional[PeriodicRun]:
    """Find the earliest contiguous loop in ``line``, or ``None``.

    Scan periods ``p`` in ``[p_min, p_max]``; for each, locate the first run of
    consecutive positions with ``line[j] == line[j - p]`` whose length is at
    least ``(k - 1) * p`` (``k`` copies of the unit back to back).  Across all
    periods, return the run that starts earliest; ties are broken by the
    smaller period (the minimal repeating unit).
    """
    n = len(line)
    if n == 0 or k < 2:
        return None

    arr = np.asarray(line)
    # k copies of a p-token unit span k*p tokens, so periods above n // k are
    # impossible -- this bounds the work and skips short responses outright.
    upper = min(p_max, n // k)

    best_onset = -1
    best_p = -1
    best_run_len = -1
    for p in range(max(1, p_min), upper + 1):
        # eq[i] is True iff line[i + p] == line[i]; an unbroken run of True of
        # length L means [i, i + p + L) is periodic with period p.
        eq = (arr[p:] == arr[:-p]).astype(np.int8)
        if not eq.any():
            continue
        # A padded diff marks run starts (+1) and ends (-1); flatnonzero then
        # yields alternating start, end, start, end, ... indices into ``eq``.
        edges = np.flatnonzero(np.diff(np.concatenate(([np.int8(0)], eq, [np.int8(0)]))))
        starts = edges[0::2]
        lengths = edges[1::2] - starts
        qualifying = np.flatnonzero(lengths >= (k - 1) * p)   # >= k copies
        if qualifying.size == 0:
            continue
        onset = int(starts[qualifying[0]])                    # earliest run for this p
        # Prefer the earliest onset; scanning p ascending means a later (larger)
        # period with the same onset never overrides the smaller minimal unit.
        if best_onset == -1 or onset < best_onset:
            best_onset, best_p = onset, p
            best_run_len = int(lengths[qualifying[0]])

    if best_onset == -1:
        return None
    return PeriodicRun(
        unit=[int(x) for x in arr[best_onset:best_onset + best_p]],
        onset=best_onset,
        period=best_p,
        # A run of length L in eq-space covers L + p tokens = L // p + 1 copies.
        num_repeats=best_run_len // best_p + 1,
    )
