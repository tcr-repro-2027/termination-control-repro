from __future__ import annotations

from conftest import relation, surface
from tcr.events.block_parser import parse_relation_blocks
from tcr.events.motif_capture import find_motif_runs, primary_motif_event


def test_two_block_motif_capture_has_online_confirmation_after_third_copy():
    motif = [relation(0), relation(1)]
    blocks = parse_relation_blocks(surface(motif * 4)).blocks
    runs = find_motif_runs(
        blocks, signature_kind="quad", min_repeats=3, max_period_blocks=None
    )
    run = primary_motif_event(runs, confirmation_repeats=3)
    assert run is not None
    assert run.block_onset == 0
    assert run.block_period == 2
    assert run.block_num_repeats == 4
    assert run.second_copy_start == 2
    assert run.confirmation_block_exclusive(3) == 6


def test_primary_event_uses_earliest_confirmation_then_minimal_period():
    rows = [relation(0)] * 6
    blocks = parse_relation_blocks(surface(rows)).blocks
    runs = find_motif_runs(blocks, signature_kind="quad", min_repeats=3)
    run = primary_motif_event(runs, confirmation_repeats=3)
    assert run is not None
    assert run.block_period == 1
    assert run.confirmation_block_exclusive(3) == 3


def test_two_copies_are_adjacent_repeat_but_not_capture():
    motif = [relation(0), relation(1)]
    blocks = parse_relation_blocks(surface(motif * 2)).blocks
    adjacent = find_motif_runs(blocks, signature_kind="triple", min_repeats=2)
    capture = find_motif_runs(blocks, signature_kind="triple", min_repeats=3)
    assert primary_motif_event(adjacent, confirmation_repeats=2) is not None
    assert not capture
