"""Align task-native block events with legacy raw-token stable orbits."""

from __future__ import annotations

from typing import Any, Sequence

from .block_parser import ParseDiagnostic, ParsedBlock
from .legacy import LegacyOrbit
from .motif_capture import MotifRun


def _block_boundary_map(blocks: Sequence[ParsedBlock]) -> dict[int, int]:
    return {
        int(block.token_start): block.block_index
        for block in blocks
        if block.token_start is not None
    }


def _raw_overlap_blocks(
    blocks: Sequence[ParsedBlock], raw_start: int, raw_end: int
) -> list[int]:
    return [
        block.block_index
        for block in blocks
        if block.token_start is not None
        and block.token_end is not None
        and block.token_start < raw_end
        and block.token_end > raw_start
    ]


def raw_onset_location(blocks: Sequence[ParsedBlock], token_onset: int) -> dict[str, Any]:
    if not blocks:
        return {"location": "no_complete_blocks", "block_index": None, "offset_in_block": None}
    for block in blocks:
        if block.token_start is None or block.token_end is None:
            continue
        if token_onset == block.token_start:
            return {"location": "block_start", "block_index": block.block_index, "offset_in_block": 0}
        if block.token_start < token_onset < block.token_end:
            return {
                "location": "inside_block",
                "block_index": block.block_index,
                "offset_in_block": token_onset - block.token_start,
            }
    starts = [block.token_start for block in blocks if block.token_start is not None]
    ends = [block.token_end for block in blocks if block.token_end is not None]
    if starts and token_onset < min(starts):
        location = "before_first_block"
    elif ends and token_onset >= max(ends):
        location = "after_last_block"
    else:
        location = "separator_between_blocks"
    return {"location": location, "block_index": None, "offset_in_block": None}


def classify_legacy_alignment(
    *,
    blocks: Sequence[ParsedBlock],
    quad_capture_runs: Sequence[MotifRun],
    triple_capture_runs: Sequence[MotifRun],
    diagnostics: Sequence[ParseDiagnostic],
    legacy: LegacyOrbit,
) -> tuple[str, dict[str, Any]]:
    evidence: dict[str, Any] = {
        "raw_onset_moved": False,
        "matching_quad_run": None,
        "boundary_pair": None,
        "phase_offset_tokens": None,
        "overlap_blocks": [],
    }
    if not legacy.exists:
        return "no_legacy_orbit", evidence
    if legacy.raw_onset_token is None or legacy.raw_period_tokens is None or legacy.raw_span_end_token_exclusive is None:
        evidence["reason"] = "legacy_coordinates_missing"
        return "unresolved", evidence
    if any(block.token_start is None or block.token_end is None for block in blocks):
        evidence["reason"] = "block_token_offsets_missing"
        return "unresolved", evidence

    raw_start = legacy.raw_onset_token
    raw_end = legacy.raw_span_end_token_exclusive
    period = legacy.raw_period_tokens
    boundaries = _block_boundary_map(blocks)
    evidence["raw_onset_location"] = raw_onset_location(blocks, raw_start)

    # Strict semantic alignment: raw onset is a complete block start and one raw
    # token period lands on the start of the same motif phase in the next copy.
    for run in quad_capture_runs:
        if run.block_onset >= len(blocks) or run.block_onset + run.block_period >= len(blocks):
            continue
        start = blocks[run.block_onset].token_start
        peer = blocks[run.block_onset + run.block_period].token_start
        if start is None or peer is None:
            continue
        if raw_start == start and peer - start == period:
            evidence["matching_quad_run"] = {
                "block_onset": run.block_onset,
                "block_period": run.block_period,
                "block_num_repeats": run.block_num_repeats,
                "token_period": peer - start,
            }
            return (
                "single_block_aligned" if run.block_period == 1 else "multi_block_aligned",
                evidence,
            )

    # Phase rotation: preserve the detector's raw onset.  A complete block
    # boundary inside the raw period must map to another block boundary exactly
    # one raw period later, and that macro-cycle must be supported by a strict
    # full-quad capture run.
    if raw_start not in boundaries:
        for boundary_token, block_index in sorted(boundaries.items()):
            if not (raw_start < boundary_token < min(raw_start + period, raw_end)):
                continue
            peer_token = boundary_token + period
            peer_index = boundaries.get(peer_token)
            if peer_index is None or peer_index <= block_index:
                continue
            block_period = peer_index - block_index
            matching = next(
                (
                    run
                    for run in quad_capture_runs
                    if run.block_period == block_period
                    and run.block_onset <= block_index
                    and run.strict_block_end >= peer_index + block_period
                ),
                None,
            )
            if matching is not None:
                evidence["matching_quad_run"] = {
                    "block_onset": matching.block_onset,
                    "block_period": matching.block_period,
                    "block_num_repeats": matching.block_num_repeats,
                }
                evidence["boundary_pair"] = [boundary_token, peer_token]
                evidence["phase_offset_tokens"] = boundary_token - raw_start
                return "phase_rotated_structured", evidence

    overlaps = _raw_overlap_blocks(blocks, raw_start, raw_end)
    evidence["overlap_blocks"] = overlaps
    incomplete_or_invalid = any(
        diagnostic.parse_status
        in {"incomplete_tail", "invalid_json", "invalid_schema", "field_span_failure"}
        for diagnostic in diagnostics
    )
    evidence["incomplete_or_invalid"] = incomplete_or_invalid
    if overlaps and (quad_capture_runs or triple_capture_runs or incomplete_or_invalid):
        return "partial_structured", evidence
    return "non_structured", evidence
