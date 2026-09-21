# coding=utf-8
"""Model-level aggregation: rates, trigger rates, CIs and the CSV view."""

from __future__ import annotations

import json

import pytest

from conftest import fake_encode
from tcr.evaluation import aggregate
from tcr.evaluation.aggregate import (
    CSV_FIELDS, bootstrap_rates, distribution, project_row, summarize_events,
    summary_to_row,
)
from tcr.evaluation.structured import build_event_row, legacy_detector


def make_row(text: str, *, key: int, seed: int, finish_reason: str):
    ids, offsets = fake_encode(text)
    row, _ = build_event_row(model_tag="unit", key=key, seed=seed, text=text,
                             token_ids=ids, offsets=offsets,
                             finish_reason=finish_reason,
                             config=legacy_detector())
    return project_row(row)


@pytest.fixture
def mixed_rows(looping_response, clean_response):
    """Four records x two seeds; record 0 loops on both seeds, record 1 loops
    on one, records 2 and 3 are clean."""
    rows = []
    for key in range(4):
        for seed in range(2):
            loops = key == 0 or (key == 1 and seed == 0)
            rows.append(make_row(
                looping_response if loops else clean_response,
                key=key, seed=seed,
                finish_reason="length" if loops else "stop"))
    return rows


def test_projection_keeps_everything_the_summary_reads(looping_response):
    row = make_row(looping_response, key=1, seed=0, finish_reason="length")
    for name, predicate in aggregate.EVENT_FLAGS.items():
        assert isinstance(predicate(row), bool), name
    assert summarize_events([row])["n_responses"] == 1


def test_rate_and_trigger_rate_answer_different_questions(mixed_rows):
    summary = summarize_events(mixed_rows)

    assert summary["n_records"] == 4
    assert summary["n_responses"] == 8
    # 3 of 8 responses capture; 2 of 4 records have at least one that does
    assert summary["semantic_capture_rate"] == pytest.approx(3 / 8)
    assert summary["semantic_capture_trigger_rate"] == pytest.approx(2 / 4)
    assert summary["stable_orbit_rate"] == pytest.approx(3 / 8)
    assert summary["hit_max_rate"] == pytest.approx(3 / 8)


def test_conditional_cascade_probabilities(mixed_rows):
    summary = summarize_events(mixed_rows)
    assert summary["p_orbit_given_capture"] == pytest.approx(1.0)
    assert summary["p_hitmax_given_orbit"] == pytest.approx(1.0)
    assert 0.0 < summary["p_capture_given_reuse"] <= 1.0


def test_onset_distributions_only_count_the_events_that_happened(mixed_rows):
    summary = summarize_events(mixed_rows)
    assert summary["capture_block_onset_n"] == 3
    assert summary["capture_block_onset_median"] == 1     # 1-based, first block
    assert summary["capture_block_period_median"] == 1
    assert summary["legacy_onset_token_n"] == 3
    assert summary["n_blocks_n"] == 8


def test_alignment_is_reported_for_orbits(mixed_rows):
    summary = summarize_events(mixed_rows)
    assert summary["alignment_counts"] == {"single_block_aligned": 3}
    assert summary["legacy_structured_alignment_rate"] == pytest.approx(1.0)
    assert summary["temporal_order_valid_rate"] == pytest.approx(1.0)


def test_distribution_of_nothing_is_none_not_zero():
    empty = distribution([], "x")
    assert empty["x_n"] == 0
    assert empty["x_mean"] is None and empty["x_median"] is None


def test_bootstrap_brackets_the_point_estimate(mixed_rows):
    summary = summarize_events(mixed_rows)
    ci = bootstrap_rates(mixed_rows, n_boot=500)

    for name in ("semantic_capture", "stable_orbit", "hit_max"):
        low = ci[f"{name}_rate_ci_low"]
        high = ci[f"{name}_rate_ci_high"]
        assert low <= summary[f"{name}_rate"] <= high
        assert 0.0 <= low <= high <= 1.0
    assert ci["bootstrap_unit"] == "record"


def test_bootstrap_resamples_records_not_responses(clean_response,
                                                   looping_response):
    """Eight seeds of ONE record carry one record's worth of information; a
    response-level bootstrap would report a spuriously tight interval."""
    rows = [make_row(looping_response, key=0, seed=seed, finish_reason="length")
            for seed in range(8)]
    rows += [make_row(clean_response, key=1, seed=seed, finish_reason="stop")
             for seed in range(8)]
    ci = bootstrap_rates(rows, n_boot=1000)
    width = ci["stable_orbit_rate_ci_high"] - ci["stable_orbit_rate_ci_low"]
    # With only two records the interval must span essentially the whole range.
    assert width > 0.9


def test_strict_f1_ci_needs_aligned_quality_samples(mixed_rows):
    quality = [{"strict": {"tp": 1, "pred_n": 1, "gold_n": 2}}
               for _ in mixed_rows]
    ci = bootstrap_rates(mixed_rows, quality, n_boot=200)
    assert 0.0 <= ci["strict_f1_ci_low"] <= ci["strict_f1_ci_high"] <= 1.0

    with pytest.raises(ValueError, match="aligned"):
        bootstrap_rates(mixed_rows, quality[:-1], n_boot=10)


def test_csv_row_is_flat_and_complete(mixed_rows):
    summary = {
        "task": {"tag": "qwen3-4b-clean", "run_name": "qwen3-4b-clean",
                 "kind": "trained", "size": "4B", "data_variant": "clean",
                 "arm_family": "stage", "seed": 42, "ckpt_step": 417,
                 "is_final": True, "save_mode": "final_only",
                 "terminal_mode": "", "priority": 3, "tier": 3,
                 "model_path": "/x", "tokenizer_path": "/x",
                 "tokenizer_is_fallback": False},
        "mode": "nothink", "protocol_version": "e1-v1.0",
        "events": summarize_events(mixed_rows),
        "quality": {"strict_f1": 0.5, "json_valid_rate": 0.9,
                    "n_responses": 8},
        "episodes": {"episodes_per_response": 1.2, "n_errors": 0},
        "bootstrap_ci": bootstrap_rates(mixed_rows, n_boot=100),
        "generation": {"eval_data_sha256": "abc", "chat_prefix_ok": True,
                       "n_skipped_too_long": 0, "generation_minutes": 12.5},
        "analysis_minutes": 3.0, "analyzed_at": "2026-08-31 12:00:00",
        "responses_path": "/r", "event_rows_path": "/e",
    }
    row = summary_to_row(summary)

    assert set(row) <= set(CSV_FIELDS), set(row) - set(CSV_FIELDS)
    assert row["tag"] == "qwen3-4b-clean"
    assert row["n_records"] == 4 and row["n_responses"] == 8
    assert row["strict_f1"] == 0.5
    assert row["episodes_per_response"] == 1.2
    assert row["semantic_capture_rate"] == pytest.approx(3 / 8)
    assert row["chat_prefix_ok"] is True
    assert json.dumps(row, default=str)          # must be CSV/JSON-serialisable


def test_no_duplicate_csv_columns():
    assert len(CSV_FIELDS) == len(set(CSV_FIELDS))
