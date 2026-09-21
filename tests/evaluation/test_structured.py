# coding=utf-8
"""The structured/legacy event row: the thing every E1 number is built on."""

from __future__ import annotations

from conftest import block_text, fake_encode, response_of
from tcr.evaluation import protocol
from tcr.evaluation.structured import build_event_row, legacy_detector, response_text


def row_for(text: str, finish_reason: str = "length", **kwargs):
    ids, offsets = fake_encode(text)
    return build_event_row(model_tag="unit", key=1, seed=0, text=text,
                           token_ids=ids, offsets=offsets,
                           finish_reason=finish_reason,
                           config=legacy_detector(), **kwargs)


def test_looping_response_is_captured_and_orbits(looping_response):
    row, _ = row_for(looping_response)

    assert row["n_blocks"] == 60
    assert row["first_triple_reuse"]["exists"]
    # The very first repeat is the seed: block 2 (1-based) reuses block 1.
    assert row["first_triple_reuse"]["block_index_1based"] == 2

    capture = row["motif_capture_triple"]
    assert capture["exists"]
    assert capture["block_period"] == 1
    assert capture["block_onset_0based"] == 0
    # capture is confirmed after CAPTURE_MIN_REPEATS copies, not at the end
    assert capture["confirmed_at_block_exclusive_0based"] == protocol.CAPTURE_MIN_REPEATS

    assert row["legacy_orbit"]["exists"], "60 identical blocks must trip the " \
                                          "token-period detector as well"
    assert row["stage_flags"] == {
        "seed_reuse": True, "capture": True, "stable_orbit": True,
        "hit_max": True, "runaway_legacy": True, "runaway_block_hitmax": True,
        "runaway_any": True,
    }
    assert row["stage_chain"] == "seed_to_capture_to_orbit_to_hitmax"


def test_capture_is_confirmed_before_the_legacy_orbit(looping_response):
    """The whole reason 01b exists: the block event is an ONLINE signal that
    fires long before the raw token orbit is confirmed."""
    row, _ = row_for(looping_response)
    temporal = row["temporal_alignment"]

    assert temporal["complete_chain_available"]
    assert temporal["online_confirmation_order_valid"] is True
    assert (temporal["seed_confirmed_at_token_exclusive"]
            < temporal["capture_confirmed_at_token_exclusive"]
            < temporal["orbit_confirmed_at_token_exclusive"])


def test_legacy_orbit_maps_onto_whole_blocks(looping_response):
    """A raw onset that lands mid-dict is exactly the defect E1 avoids; on a
    pure block loop the two coordinate systems must agree."""
    row, _ = row_for(looping_response)
    assert row["alignment_type"] == "single_block_aligned"
    assert row["alignment_evidence"]["matching_quad_run"]["block_period"] == 1


def test_clean_response_has_no_events(clean_response):
    row, _ = row_for(clean_response, finish_reason="stop")

    assert row["n_blocks"] == 5
    assert not row["first_triple_reuse"]["exists"]
    assert not row["motif_capture_triple"]["exists"]
    assert not row["legacy_orbit"]["exists"]
    assert row["stage_chain"] == "clean_progress"
    assert row["top_level_list_complete"]
    assert row["severity"]["hit_max_tokens"] == 0


def test_a_truncated_tail_is_not_a_block(clean_response):
    """A response cut off inside the last dict must not contribute a block --
    that partial dict is the single most common false reuse."""
    truncated = clean_response.rstrip("]") + ', {"source": "甲", "target": "乙"'
    row, _ = row_for(truncated)

    assert row["n_blocks"] == 5
    assert row["has_incomplete_tail"]
    assert not row["top_level_list_complete"]


def test_description_only_change_still_reuses_the_triple():
    """Reuse is defined on (source,target,relation); a reworded description
    is the same relation and must count."""
    text = response_of([
        block_text("甲", "乙", description="第一次说明"),
        block_text("丙", "丁"),
        block_text("甲", "乙", description="换个说法的说明"),
    ])
    row, _ = row_for(text, finish_reason="stop")

    reuse = row["first_triple_reuse"]
    assert reuse["exists"] and reuse["block_index_1based"] == 3
    assert reuse["description_changed"] is True
    assert not row["first_exact_quad_reuse"]["exists"]


def test_short_loops_are_below_the_length_gate():
    """The legacy detector has a length gate; a genuinely short answer must
    never be called a loop no matter how repetitive it looks."""
    text = response_of([block_text("甲", "乙")] * 3)
    ids, _ = fake_encode(text)
    assert len(ids) <= protocol.LEGACY_THRESHOLD

    row, _ = row_for(text, finish_reason="stop")
    assert not row["legacy_orbit"]["exists"]
    assert row["motif_capture_triple"]["exists"], "the block event does not " \
                                                  "need a length gate"


def test_diagnostics_and_snippets_are_off_the_row_by_default(clean_response):
    """Bulk stays off the 8848 rows, but the caller still gets the snippets so
    it can write a bounded manual-audit sample."""
    row, snippets = row_for(clean_response, finish_reason="stop")
    assert "parse_diagnostics" not in row
    assert "audit_snippets" not in row
    assert snippets["parser_blocks"]

    row, _ = row_for(clean_response, finish_reason="stop",
                     keep_diagnostics=True, keep_snippets=True)
    assert "parse_diagnostics" in row
    assert row["audit_snippets"]["parser_blocks"]


def test_engine_token_count_mismatch_is_recorded(clean_response):
    ids, offsets = fake_encode(clean_response)
    row, _ = build_event_row(model_tag="unit", key=1, seed=0,
                             text=clean_response, token_ids=ids, offsets=offsets,
                             finish_reason="stop", gen_tokens_engine=len(ids))
    assert row["gen_tokens_mismatch"] is False

    row, _ = build_event_row(model_tag="unit", key=1, seed=0,
                             text=clean_response, token_ids=ids, offsets=offsets,
                             finish_reason="stop", gen_tokens_engine=len(ids) + 3)
    assert row["gen_tokens_mismatch"] is True


def test_response_text_is_the_answer_only():
    sample = {"response": "ANSWER", "reasoning": "THOUGHTS"}
    assert protocol.LEGACY_TARGET == "answer"
    assert response_text(sample) == "ANSWER"
