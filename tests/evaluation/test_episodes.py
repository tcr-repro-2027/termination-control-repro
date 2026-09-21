# coding=utf-8
"""The marginal P0d episode summary."""

from __future__ import annotations

import pytest

from conftest import block_text, fake_encode, response_of
from tcr.evaluation.episodes import episode_summary, response_outcomes
from tcr.evaluation.structured import build_event_row, legacy_detector


def event_row(text: str, *, key: int, seed: int = 0,
              finish_reason: str = "stop"):
    ids, offsets = fake_encode(text)
    row, _ = build_event_row(model_tag="unit", key=key, seed=seed, text=text,
                             token_ids=ids, offsets=offsets,
                             finish_reason=finish_reason,
                             config=legacy_detector())
    return row


def test_a_clean_response_has_no_episode(clean_response):
    summary = episode_summary([event_row(clean_response, key=1)])
    assert summary["n_errors"] == 0
    assert summary["episodes_per_response"] == 0.0
    assert summary["per_episode_capture_hazard"] is None
    assert summary["gap_stop_hazard"] == 1.0     # the one gap slot ended in a stop


def test_a_captured_response_is_a_terminal_capture(looping_response):
    summary = episode_summary([event_row(looping_response, key=1,
                                         finish_reason="length")])
    assert summary["n_errors"] == 0
    assert summary["episodes_per_response"] == 1.0
    assert summary["per_episode_capture_hazard"] == 1.0
    assert summary["capture_terminal_rate"] == 1.0
    assert summary["gap_stop_hazard"] == 0.0
    assert summary["p_orbit_given_terminal_capture"] == 1.0


def test_a_recovered_episode_is_exposure_without_capture():
    """Two reuse blocks then novel content again: the response experienced an
    episode and survived it.  That is the exposure term -- it must be counted
    even though nothing was captured."""
    text = response_of([
        block_text("甲", "乙"),
        block_text("丙", "丁"),
        block_text("甲", "乙"),      # reuse -> episode opens
        block_text("戊", "己"),      # novel  -> episode closes, recovered
        block_text("庚", "辛"),
    ])
    rows = [event_row(text, key=1)]
    outcomes, errors = response_outcomes(rows)

    assert errors == []
    assert outcomes[0].n_episodes_experienced == 1
    assert outcomes[0].recovered_episodes == 1
    assert outcomes[0].terminal_event == "stop"
    assert outcomes[0].terminal_stage == "gap"

    summary = episode_summary(rows)
    assert summary["episodes_per_response"] == 1.0
    assert summary["per_episode_capture_hazard"] == 0.0
    assert summary["responses_with_any_episode_rate"] == 1.0


def test_exposure_and_propensity_are_separable(clean_response,
                                               looping_response):
    """The fork P0d exists for: a model that merely stops less accumulates
    episodes without any change in the per-episode hazard."""
    recovered = response_of(
        [block_text("甲", "乙"), block_text("丙", "丁"), block_text("甲", "乙"),
         block_text("戊", "己"), block_text("庚", "辛"), block_text("戊", "己"),
         block_text("壬", "癸")])

    low_exposure = episode_summary([event_row(clean_response, key=1)])
    high_exposure = episode_summary([event_row(recovered, key=2)])

    assert low_exposure["episodes_per_response"] == 0.0
    assert high_exposure["episodes_per_response"] == 2.0
    assert high_exposure["per_episode_capture_hazard"] == 0.0


def test_a_hit_max_response_without_capture_is_censored(clean_response):
    summary = episode_summary([event_row(clean_response, key=1,
                                         finish_reason="length")])
    assert summary["censor_rate"] == 1.0
    assert summary["gap_stop_hazard"] == 0.0     # censoring is not a stop


def test_bad_rows_are_counted_not_raised():
    broken = {"sample_id": "x", "model_tag": "unit", "stable_prompt_id": "1",
              "seed": 0, "n_blocks": 2, "block_index": [{}, {}],
              "n_identity_complete_blocks": 99,       # deliberately wrong
              "first_nonempty_triple_reuse": {"exists": False},
              "legacy_orbit": {"exists": False}, "alignment_evidence": {}}
    summary = episode_summary([broken])
    assert summary["n_errors"] == 1
    assert summary["errors_sample"]
    assert summary["n_scored_responses"] == 0


def test_responses_with_no_complete_block_stay_in_the_denominator(clean_response):
    """The bug this guards: skipping unparseable responses shrank the
    denominator, so the models that produce the most of them looked like the
    ones with the fewest episodes per response."""
    blank = event_row("", key=1, finish_reason="stop")
    good = event_row(clean_response, key=2, finish_reason="stop")
    reusing = event_row(response_of([
        block_text("甲", "乙"), block_text("丙", "丁"),
        block_text("甲", "乙"), block_text("戊", "己")]), key=3)

    summary = episode_summary([blank, good, reusing])
    assert summary["n_errors"] == 0
    assert summary["n_scored_responses"] == 3, "no response may be dropped"
    assert summary["denominator_is_complete"] is True
    # one episode over three responses, not over the two that had blocks
    assert summary["episodes_per_response"] == pytest.approx(1 / 3)
    assert summary["responses_with_any_episode_rate"] == pytest.approx(1 / 3)
    # Four gap slots at risk: one each for the two responses that never opened
    # an episode, and two for the third (it SURVIVED its first gap slot into an
    # episode, then stopped in the second).  Three of them ended in a stop.
    assert summary["gap_stop_hazard"] == pytest.approx(0.75)


def test_a_blank_response_terminates_in_the_first_gap_slot():
    outcomes, errors = response_outcomes([event_row("", key=1, finish_reason="stop")])
    assert errors == []
    assert outcomes[0].n_episodes_experienced == 0
    assert outcomes[0].terminal_stage == "gap"
    assert outcomes[0].terminal_event == "stop"


def test_a_hit_max_blank_response_is_censored_not_stopped():
    outcomes, _ = response_outcomes([event_row("", key=1, finish_reason="length")])
    assert outcomes[0].terminal_event == "censor"
