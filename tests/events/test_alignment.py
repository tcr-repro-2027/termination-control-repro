from __future__ import annotations

from conftest import char_offsets, relation, surface
from tcr.events.alignment import classify_legacy_alignment
from tcr.events.block_parser import parse_relation_blocks
from tcr.events.legacy import LegacyOrbit
from tcr.events.motif_capture import find_motif_runs


def _legacy(onset: int, period: int, repeats: int, n: int):
    return LegacyOrbit(
        True,
        onset,
        period,
        repeats,
        onset + period,
        onset + 3 * period,
        onset + repeats * period,
        True,
        None,
        None,
    )


def test_exact_single_block_alignment():
    text = surface([relation(0)] * 5)
    parsed = parse_relation_blocks(text, char_offsets(text))
    starts = [b.token_start for b in parsed.blocks]
    period = starts[1] - starts[0]
    legacy = _legacy(starts[0], period, 3, len(text))
    runs = find_motif_runs(parsed.blocks, signature_kind="quad", min_repeats=3)
    alignment, evidence = classify_legacy_alignment(
        blocks=parsed.blocks,
        quad_capture_runs=runs,
        triple_capture_runs=runs,
        diagnostics=parsed.diagnostics,
        legacy=legacy,
    )
    assert alignment == "single_block_aligned"
    assert evidence["raw_onset_moved"] is False


def test_phase_rotated_multi_block_alignment_preserves_raw_onset():
    motif = [relation(0), relation(1)]
    text = surface(motif * 5)
    parsed = parse_relation_blocks(text, char_offsets(text))
    starts = [b.token_start for b in parsed.blocks]
    period = starts[2] - starts[0]
    raw_onset = starts[0] + 3
    legacy = _legacy(raw_onset, period, 3, len(text))
    runs = find_motif_runs(parsed.blocks, signature_kind="quad", min_repeats=3)
    alignment, evidence = classify_legacy_alignment(
        blocks=parsed.blocks,
        quad_capture_runs=runs,
        triple_capture_runs=runs,
        diagnostics=parsed.diagnostics,
        legacy=legacy,
    )
    assert alignment == "phase_rotated_structured"
    assert evidence["phase_offset_tokens"] > 0
    assert legacy.raw_onset_token == raw_onset
