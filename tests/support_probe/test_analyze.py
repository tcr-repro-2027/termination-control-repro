# coding=utf-8
"""The measures and the gates, against readouts whose truth is known.

Every number E2 reports comes out of this arithmetic, and none of it is checked
by anything downstream: a slope computed with the wrong sign, or a positive
control that quietly passes when it should fail, would look entirely normal in
the output.  So the whole chain is driven here from synthetic readouts built
with a planted effect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tcr.support_probe import protocol
from tcr.support_probe.analyze import (
    instrument_gates, load_views, slope_through_origin, summarize,
    summary_to_row,
)

TRUE_SLOPE = {"admissibility": 0.9, "evidence": 0.6}
PC_STOP_EFFECT = 1.2
PC_CONTINUE_EFFECT = 0.8


def cell(anchor_id: str, axis: str, arm: str, level: float, margin: float,
         *, close_prob: float = 0.2, close_rank: int = 3,
         survives: bool = True, anchor_type: str | None = None) -> Dict[str, Any]:
    return {
        "cell_id": f"{anchor_id}|{arm}|{level}",
        "anchor_id": anchor_id,
        "anchor_type": anchor_type or axis,
        "key": anchor_id, "source": "doc", "n_remainder": 2,
        "variant_id": f"{anchor_id}:{arm}:{level}",
        "arm": arm, "axis": axis if arm in ("base", "manip", "neutral") else "none",
        "level": level, "n_edits": 0, "n_remaining_supported": 2,
        "n_context_tokens": 100,
        "close_token": 1, "continue_token": 2,
        "close_token_str": "]", "continue_token_str": ",",
        "sm_primary": margin, "sm_raw": margin * 0.7,
        "close_logprob": -1.0, "continue_logprob": -0.5,
        "close_rank": close_rank, "continue_rank": 1,
        "close_in_top_k": True, "close_survives_top_p": survives,
        "close_sampler_prob": close_prob, "continue_sampler_prob": 0.5,
        "close_prob_pre_filter": close_prob, "n_kept_by_sampler": 5,
        # the state the short continuations are sampled in (the prefix carries
        # no presence penalty there), which gate 7 is read against
        "sm_unpenalised": margin * 0.9,
        "close_rank_unpenalised": close_rank,
        "close_in_top_k_unpenalised": True,
        "close_survives_top_p_unpenalised": survives,
        "close_sampler_prob_unpenalised": close_prob,
        "continue_sampler_prob_unpenalised": 0.5,
        "close_prob_pre_filter_unpenalised": close_prob,
        "n_kept_by_sampler_unpenalised": 5,
    }


def synth_rows(n: int = 24, *, neutral_drift: float = 0.0,
               pc_stop: float = PC_STOP_EFFECT,
               pc_continue: float = PC_CONTINUE_EFFECT) -> List[Dict[str, Any]]:
    """One anchor per axis per record, with a planted slope."""
    rows: List[Dict[str, Any]] = []
    for index in range(n):
        base_margin = -2.0 + 0.1 * index          # a spread of baselines
        for axis in ("admissibility", "evidence"):
            anchor = f"{axis}:{index}"
            rows.append(cell(anchor, axis, "base", 0.0, base_margin))
            for level in (0.5, 1.0):
                rows.append(cell(anchor, axis, "neutral", level,
                                 base_margin + neutral_drift))
                # closing gets easier as support is removed, and the close token
                # climbs the ranking with it
                rows.append(cell(
                    anchor, axis, "manip", level,
                    base_margin + neutral_drift + TRUE_SLOPE[axis] * level,
                    close_rank=3 - int(2 * level), close_prob=0.2 + 0.3 * level))
            rows.append(cell(anchor, axis, "pc_stop", 0.0, base_margin + pc_stop))
            rows.append(cell(anchor, axis, "pc_continue", 0.0,
                             base_margin - pc_continue))
            rows.append(cell(anchor, axis, "pc_neutral", 0.0, base_margin))
        zero = f"zero_answer:{index}"
        for arm, margin in (("base", 1.5), ("pc_stop", 2.5),
                            ("pc_continue", 0.5), ("pc_neutral", 1.5)):
            rows.append(cell(zero, "none", arm, 0.0, margin,
                             anchor_type="zero_answer", close_rank=1,
                             close_prob=0.7))
    return rows


def write_rows(tmp_path: Path, rows: List[Dict[str, Any]]) -> Path:
    path = tmp_path / "readouts.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                              for row in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def summary(tmp_path: Path):
    return summarize(load_views(write_rows(tmp_path, synth_rows())), tag="unit")


# ------------------------------------------------------------------- slopes

def test_slope_through_origin_recovers_a_planted_line():
    assert slope_through_origin([0.0, 0.5, 1.0], [0.0, 0.45, 0.9]) == pytest.approx(0.9)


def test_slope_is_the_predicted_delta_at_full_manipulation():
    """§9.5 reads b as "the StopMargin change from no manipulation to full
    manipulation", so a perfectly linear anchor must return exactly delta(1)."""
    assert slope_through_origin([0.0, 0.5, 1.0], [0.0, 0.3, 0.6]) == pytest.approx(0.6)


def test_aci_and_eci_recover_their_planted_slopes(summary):
    for axis in ("admissibility", "evidence"):
        block = summary["sm_primary"][axis]
        assert block["n_anchors"] == 24
        assert block["slope"]["mean"] == pytest.approx(TRUE_SLOPE[axis])
        assert block["slope"]["ci_low"] > 0
        assert block[f"delta_r1.0"]["mean"] == pytest.approx(TRUE_SLOPE[axis])


def test_the_two_axes_are_reported_separately(summary):
    """H2 needs them dissociable; a single blended index would hide exactly the
    contrast the hypothesis is about."""
    aci = summary["sm_primary"]["admissibility"]["slope"]["mean"]
    eci = summary["sm_primary"]["evidence"]["slope"]["mean"]
    assert aci != pytest.approx(eci)
    assert aci == pytest.approx(0.9) and eci == pytest.approx(0.6)


def test_neutral_drift_is_zero_when_the_neutral_edit_is_inert(summary):
    for axis in ("admissibility", "evidence"):
        drift = summary["sm_primary"][axis]["neutral_drift_r1.0"]
        assert drift["mean"] == pytest.approx(0.0)


def test_delta_subtracts_the_neutral_not_the_base(tmp_path: Path):
    """With a drifting neutral, delta must stay clean: the drift is exactly the
    'the input was edited at all' component the neutral exists to remove."""
    views = load_views(write_rows(tmp_path, synth_rows(neutral_drift=0.4)))
    block = summarize(views, tag="unit")["sm_primary"]["admissibility"]
    assert block["slope"]["mean"] == pytest.approx(TRUE_SLOPE["admissibility"])
    assert block["neutral_drift_r1.0"]["mean"] == pytest.approx(0.4)


# ---------------------------------------------------------- positive control

def test_pc_is_sign_unified_and_decomposed(summary):
    block = summary["sm_primary"]["admissibility"]
    assert block["pc"]["mean"] == pytest.approx(PC_STOP_EFFECT + PC_CONTINUE_EFFECT)
    assert block["pc_stop_effect"]["mean"] == pytest.approx(PC_STOP_EFFECT)
    assert block["pc_continue_effect"]["mean"] == pytest.approx(PC_CONTINUE_EFFECT)


def test_nsci_normalises_the_slope_by_the_control(summary):
    block = summary["sm_primary"]["admissibility"]
    expected = TRUE_SLOPE["admissibility"] / (PC_STOP_EFFECT + PC_CONTINUE_EFFECT)
    assert block["nsci"]["mean"] == pytest.approx(expected)
    assert block["nsci"]["n_qualified"] == 24
    assert block["nsci"]["tau"] == protocol.PC_TAU


def test_nsci_drops_anchors_below_tau(tmp_path: Path):
    """§9.5 keeps only anchors with |PC| >= tau: dividing by a near-zero control
    manufactures enormous ratios out of noise."""
    views = load_views(write_rows(tmp_path, synth_rows(pc_stop=0.05,
                                                       pc_continue=0.05)))
    block = summarize(views, tag="unit")["sm_primary"]["admissibility"]
    assert block["nsci"]["n_qualified"] == 0
    assert block["nsci"]["mean"] is None


# ---------------------------------------------------------------- floor flags

def test_floor_and_ceiling_anchors_are_excluded_and_counted(tmp_path: Path):
    rows = synth_rows(n=4)
    for row in rows:
        if row["anchor_id"] == "admissibility:0":
            row["close_prob_pre_filter"] = 0.9995   # already certain to close
        if row["anchor_id"] == "admissibility:1":
            row["close_prob_pre_filter"] = 0.0      # unreachable even at r=1
    block = summarize(load_views(write_rows(tmp_path, rows)),
                      tag="unit")["sm_primary"]["admissibility"]
    assert block["n_anchors"] == 4
    assert block["n_informative"] == 2
    assert block["uninformative"] == {"ceiling": 1, "floor": 1}


# --------------------------------------------------------------- zero answer

def test_zero_answer_reports_the_empty_margin(summary):
    block = summary["sm_primary"]["zero_answer"]
    assert block["n_anchors"] == 24
    assert block["empty_margin"]["mean"] == pytest.approx(1.5)
    assert block["close_is_top1_rate"]["mean"] == pytest.approx(1.0)


# --------------------------------------------------------------------- gates

def test_a_healthy_instrument_passes_every_gate(summary):
    gates = instrument_gates(summary, anchor_report={"n_rejected": 0},
                             score_manifest={"chat_prefix_ok": True})
    failed = [gate["gate"] for gate in gates if not gate["pass"]]
    assert failed == [], failed
    assert len(gates) == 6


def test_a_drifting_neutral_edit_fails_gate_1(tmp_path: Path):
    views = load_views(write_rows(tmp_path, synth_rows(neutral_drift=1.5)))
    gates = instrument_gates(summarize(views, tag="unit"),
                             anchor_report={"n_rejected": 0},
                             score_manifest={"chat_prefix_ok": True})
    gate = next(g for g in gates if g["gate"] == "1_neutral_effect_near_zero")
    assert gate["pass"] is False


def test_an_unresponsive_model_fails_gate_2(tmp_path: Path):
    """H3's negation condition: with no response to a direct instruction, a
    flat support slope cannot be read as a SUPPORT failure."""
    views = load_views(write_rows(tmp_path, synth_rows(pc_stop=0.0,
                                                       pc_continue=0.0)))
    gates = instrument_gates(summarize(views, tag="unit"))
    gate = next(g for g in gates if g["gate"] == "2_positive_control_direction")
    assert gate["pass"] is False


def test_a_failed_anchor_build_fails_gate_4(summary):
    gates = instrument_gates(summary, anchor_report={"n_rejected": 3},
                             score_manifest={"chat_prefix_ok": True})
    gate = next(g for g in gates if g["gate"] == "4_parser_prefix_token_alignment")
    assert gate["pass"] is False

    gates = instrument_gates(summary, anchor_report={"n_rejected": 0},
                             score_manifest={"chat_prefix_ok": False})
    gate = next(g for g in gates if g["gate"] == "4_parser_prefix_token_alignment")
    assert gate["pass"] is False


def test_gate_3_is_per_model_and_flags_a_flat_response(tmp_path: Path):
    """A trained arm showing no support response is a RESULT; the gate records
    it without pretending the instrument broke."""
    rows = [row for row in synth_rows()]
    for row in rows:
        if row["arm"] == "manip":
            row["sm_primary"] = -2.0 + 0.1 * int(row["anchor_id"].split(":")[1])
    gates = instrument_gates(summarize(load_views(write_rows(tmp_path, rows)),
                                       tag="flat"))
    gate = next(g for g in gates if g["gate"] == "3_support_manipulation_responds")
    assert gate["pass"] is False
    assert "RESULT" in gate["note"]


# ----------------------------------------------------------------- csv view

def test_csv_row_is_flat_and_complete(summary):
    from tcr.support_probe.analyze import CSV_FIELDS
    gates = instrument_gates(summary, anchor_report={"n_rejected": 0},
                             score_manifest={"chat_prefix_ok": True})
    row = summary_to_row(summary, gates)

    assert set(row) <= set(CSV_FIELDS), set(row) - set(CSV_FIELDS)
    assert row["aci_slope"] == pytest.approx(0.9)
    assert row["eci_slope"] == pytest.approx(0.6)
    assert row["gates_failed"] == 0
    assert row["empty_margin"] == pytest.approx(1.5)
    assert json.dumps(row, default=str)


# ------------------------------------------------------------- stop hazard

#: Chosen so every planted rate lands on a whole number of draws out of 16:
#: the writer derives `stop_hazard` from `n_closed`, and the merge now checks
#: that they agree, so a fixture that plants an unreachable rate is testing a
#: file the scorer could never produce.
HAZARD_SLOPE = {"admissibility": 0.25, "evidence": 0.125}


def hazard_rows(rows: List[Dict[str, Any]], *, other_per_cell: int = 0,
                sign: float = 1.0) -> List[Dict[str, Any]]:
    """A hazard file keyed to the same cells, with its own planted slope."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row["arm"] not in ("base", "manip", "neutral"):
            continue                        # default --arms skips the controls
        axis = row["anchor_type"]
        lift = (HAZARD_SLOPE.get(axis, 0.0) * row["level"] * sign
                if row["arm"] == "manip" else 0.0)
        decided = 16 - other_per_cell
        closed = round((0.25 + lift) * decided)
        out.append({
            "cell_id": row["cell_id"], "anchor_id": row["anchor_id"],
            "anchor_type": axis, "variant_id": row["variant_id"],
            "arm": row["arm"], "axis": row["axis"], "level": row["level"],
            "n_samples": 16, "n_closed": closed,
            "n_continued": decided - closed,
            "n_other": other_per_cell, "n_noncanonical": 0,
            # over ALL draws, exactly as e2.hazard writes it
            "stop_hazard": closed / 16,
            "stop_hazard_decided": (closed / decided) if decided else None,
            "presence_state": protocol.HAZARD_PRESENCE_STATE,
        })
    return out


def write_hazard(tmp_path: Path, rows: List[Dict[str, Any]]) -> Path:
    path = tmp_path / "hazard.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                              for row in rows) + "\n", encoding="utf-8")
    return path


def summary_with_hazard(tmp_path: Path, **kwargs):
    rows = synth_rows()
    views = load_views(write_rows(tmp_path, rows),
                       write_hazard(tmp_path, hazard_rows(rows, **kwargs)))
    return summarize(views, tag="unit")


def test_without_a_hazard_file_the_summary_says_so(summary):
    """NO_HAZARD=1 must leave an empty column, not a zero."""
    assert summary["has_hazard"] is False
    assert summary["stop_hazard"]["evidence"]["n_with_hazard"] == 0
    assert summary_to_row(summary, [])["aci_hazard_slope"] is None


def test_the_hazard_reaches_the_summary_and_the_csv(tmp_path: Path):
    """The bug this pins: the hazard stage used to write a file that nothing
    downstream ever read, so the GPU time bought nothing."""
    summary = summary_with_hazard(tmp_path)
    assert summary["has_hazard"] is True
    for axis, prefix in (("admissibility", "aci"), ("evidence", "eci")):
        block = summary["stop_hazard"][axis]
        assert block["n_with_hazard"] == 24
        assert block["slope"]["mean"] == pytest.approx(HAZARD_SLOPE[axis])
        row = summary_to_row(summary, [])
        assert row[f"{prefix}_hazard_slope"] == pytest.approx(HAZARD_SLOPE[axis])
        assert row[f"{prefix}_hazard_base_rate"] == pytest.approx(0.25)


def test_the_hazard_is_measured_on_the_same_anchors_as_the_margin(tmp_path: Path):
    """Merging into the shared cells is what makes 'they moved together'
    checkable: the two summaries cover an identical anchor set."""
    summary = summary_with_hazard(tmp_path)
    for axis in ("admissibility", "evidence"):
        assert (summary["stop_hazard"][axis]["n_informative"]
                == summary["sm_primary"][axis]["n_informative"])


def test_the_neutral_arm_shows_no_hazard_drift(tmp_path: Path):
    summary = summary_with_hazard(tmp_path)
    drift = summary["stop_hazard"]["evidence"]["neutral_drift_r1.0"]
    assert drift["mean"] == pytest.approx(0.0)


def test_gate_7_passes_when_score_and_behaviour_agree(tmp_path: Path):
    summary = summary_with_hazard(tmp_path)
    gate = next(g for g in instrument_gates(summary)
                if g["gate"] == "7_hazard_is_measurable_and_agrees")
    assert gate["pass"] is True
    assert summary_to_row(summary, [])["hazard_agrees"] is True


def test_gate_7_fails_when_the_model_does_the_opposite_of_what_it_scores(tmp_path):
    """§9.5 forbids the claim when the margin and the behaviour disagree."""
    summary = summary_with_hazard(tmp_path, sign=-1.0)
    gate = next(g for g in instrument_gates(summary)
                if g["gate"] == "7_hazard_is_measurable_and_agrees")
    assert gate["pass"] is False


def test_gate_7_fails_when_the_continuations_stop_being_decisions(tmp_path):
    """A high `other` rate is not a small effect -- it is no measurement, and
    it is exactly what a broken classifier would produce."""
    summary = summary_with_hazard(tmp_path, other_per_cell=15)
    gate = next(g for g in instrument_gates(summary)
                if g["gate"] == "7_hazard_is_measurable_and_agrees")
    assert gate["pass"] is False
    assert gate["detail"]["other_rate"] > 0.5


def test_gate_7_is_absent_rather_than_failed_without_the_hazard(summary):
    names = [gate["gate"] for gate in instrument_gates(summary)]
    assert "7_hazard_is_measurable_and_agrees" not in names


def test_the_csv_records_which_anchor_types_were_actually_built(summary):
    """`evidence_add` is specified by §E2 but not built, so no reader can call
    this a four-type run."""
    row = summary_to_row(summary, [])
    assert row["built_anchor_types"] == "evidence;admissibility;zero_answer"
    assert row["unbuilt_anchor_types"] == "evidence_add"


# ------------------------------------------------- the hazard must really join

def test_a_hazard_file_for_another_model_does_not_pass_as_no_hazard(tmp_path: Path):
    """Cell ids are stable across models, so a hazard file that joins onto
    NOTHING means it was keyed to another anchor set.  Silently reporting
    `has_hazard: false` would drop gate 7 while every other gate reads green."""
    rows = synth_rows()
    foreign = hazard_rows(rows)
    for row in foreign:
        row["cell_id"] = "OTHER_ANCHORS|" + row["cell_id"]
    with pytest.raises(SystemExit) as excinfo:
        load_views(write_rows(tmp_path, rows),
                   write_hazard(tmp_path, foreign),
                   hazard_arms="base,manip,neutral", strict_hazard=True)
    assert "no hazard row joined" in str(excinfo.value)


def test_a_half_finished_hazard_is_a_failure_not_a_thinner_sample(tmp_path: Path):
    rows = synth_rows()
    partial = hazard_rows(rows)[:20]
    with pytest.raises(SystemExit) as excinfo:
        load_views(write_rows(tmp_path, rows),
                   write_hazard(tmp_path, partial),
                   hazard_arms="base,manip,neutral", strict_hazard=True)
    assert "of" in str(excinfo.value) and "joined" in str(excinfo.value)


def test_a_requested_hazard_file_that_is_absent_is_an_error(tmp_path: Path):
    with pytest.raises(SystemExit) as excinfo:
        load_views(write_rows(tmp_path, synth_rows()),
                   tmp_path / "not_here.jsonl", strict_hazard=True)
    assert "missing" in str(excinfo.value)


def test_the_expected_cell_count_comes_from_the_arms_that_were_run(tmp_path: Path):
    """`--arms all` covers the controls too, so the same file is complete under
    one arm set and short under the other."""
    from tcr.support_probe.analyze import expected_hazard_cells
    rows = synth_rows(n=2)
    by_cell = {row["cell_id"]: row for row in rows}
    assert expected_hazard_cells(by_cell, "all") == len(by_cell)
    three = expected_hazard_cells(by_cell, "base,manip,neutral")
    assert 0 < three < len(by_cell)
    assert expected_hazard_cells(by_cell, None) is None


def test_a_complete_hazard_join_is_silent(tmp_path: Path):
    rows = synth_rows()
    views = load_views(write_rows(tmp_path, rows),
                       write_hazard(tmp_path, hazard_rows(rows)),
                       hazard_arms="base,manip,neutral", strict_hazard=True)
    assert summarize(views, tag="unit")["has_hazard"] is True


# ------------------------------------------------ the CI's resampling unit

def test_the_hazard_rates_are_bootstrapped_over_anchors_not_cells(tmp_path: Path):
    """An anchor contributes five hazard cells and they are anything but
    independent; pooling cells reports an interval several times too narrow and
    weights anchors by how many cells they happen to have."""
    rows = synth_rows()
    summary = summary_with_hazard(tmp_path, other_per_cell=2)
    block = summary["stop_hazard"]["admissibility"]
    assert block["other_rate"]["n"] == block["n_informative"] == 24
    assert block["n_hazard_cells"] > block["other_rate"]["n"]
    assert block["other_rate"]["mean"] == pytest.approx(2 / 16)


def test_every_reported_interval_resamples_the_same_number_of_units(tmp_path: Path):
    """`n` is the anchor count everywhere, so no block is silently a per-cell
    interval hiding among per-anchor ones."""
    summary = summary_with_hazard(tmp_path)
    for axis in ("admissibility", "evidence"):
        margin = summary["sm_primary"][axis]
        hazard = summary["stop_hazard"][axis]
        assert margin["slope"]["n"] == hazard["slope"]["n"]
        assert hazard["other_rate"]["n"] == hazard["slope"]["n"]


# ------------------------------------------------------------ --bootstrap

def test_the_bootstrap_argument_actually_changes_the_resampling(tmp_path: Path):
    """It used to be parsed and dropped: `--bootstrap 100` produced the
    protocol's 2000-resample CI and said nothing about it."""
    views = load_views(write_rows(tmp_path, synth_rows()))
    few = summarize(views, tag="unit", n_boot=25)
    many = summarize(views, tag="unit", n_boot=protocol.BOOTSTRAP)
    assert few["bootstrap"] == 25 and many["bootstrap"] == protocol.BOOTSTRAP
    # same point estimate, different interval: only the resampling changed.
    # Read on the baseline margin, which genuinely varies across anchors -- the
    # planted slope is identical on every one, so its CI is degenerate whatever
    # the resample count.
    for axis in ("admissibility", "evidence"):
        assert (few["sm_primary"][axis]["slope"]["mean"]
                == pytest.approx(many["sm_primary"][axis]["slope"]["mean"]))
    spread = "baseline_margin"
    assert (few["sm_primary"]["admissibility"][spread]["mean"]
            == pytest.approx(many["sm_primary"]["admissibility"][spread]["mean"]))
    assert (few["sm_primary"]["admissibility"][spread]["ci_low"]
            != many["sm_primary"]["admissibility"][spread]["ci_low"])


# ------------------------------- the hazard merge must count REAL cells

def test_duplicate_hazard_rows_cannot_pad_the_completeness_check(tmp_path: Path):
    """Counting rows lets a resume that rewrote a chunk cover for a cell that
    never landed: 2560 rows, 2559 cells, and the shortfall is invisible."""
    rows = synth_rows()
    hazard = hazard_rows(rows)
    missing = hazard.pop()                      # one cell never measured
    hazard += hazard[:5]                        # five duplicated by a resume
    assert len(hazard) > len(hazard_rows(rows))  # padded past the target
    with pytest.raises(SystemExit) as excinfo:
        load_views(write_rows(tmp_path, rows), write_hazard(tmp_path, hazard),
                   hazard_arms="base,manip,neutral", strict_hazard=True)
    assert "joined" in str(excinfo.value)
    assert missing["cell_id"]


def test_a_row_without_a_hazard_is_not_a_measured_cell(tmp_path: Path):
    """A row that joins but carries no `stop_hazard` contributes to nothing;
    counting it as merged would let a half-written file pass by volume."""
    rows = synth_rows()
    hazard = hazard_rows(rows)
    for row in hazard[:10]:
        row["stop_hazard"] = None
        row["n_samples"] = None
    with pytest.raises(SystemExit) as excinfo:
        load_views(write_rows(tmp_path, rows), write_hazard(tmp_path, hazard),
                   hazard_arms="base,manip,neutral", strict_hazard=True)
    assert "not a usable measurement" in str(excinfo.value)


def test_merge_stats_separate_the_three_failure_modes(tmp_path: Path):
    from tcr.support_probe.analyze import merge_hazard
    rows = synth_rows(n=2)
    by_cell = {row["cell_id"]: row for row in rows}
    hazard = hazard_rows(rows)
    hazard = hazard + hazard[:2]                      # duplicates
    hazard[3] = {**hazard[3], "stop_hazard": None}    # incomplete
    hazard.append({**hazard[0], "cell_id": "elsewhere"})   # unmatched
    stats = merge_hazard(by_cell, write_hazard(tmp_path, hazard))
    assert stats["duplicates"] == 2
    assert stats["malformed"] == 1
    assert stats["unmatched"] == 1


# ------------------------- the evidence axis's edit-magnitude sensitivity

def test_the_evidence_slope_is_repeated_on_the_magnitude_matched_subset(tmp_path):
    """§9.5 sensitivity.  The evidence axis substitutes over the whole
    document, so "the edit itself cancels" holds exactly only where the two
    arms rewrote the same amount of text.  Reported beside the pooled slope,
    never instead of it."""
    rows = synth_rows()
    for row in rows:
        if row["anchor_type"] == "evidence":
            index = int(row["anchor_id"].split(":")[1])
            row["occurrence_gap"] = 0 if index < 10 else 3
    block = summarize(load_views(write_rows(tmp_path, rows)),
                      tag="unit")["sm_primary"]["evidence"]
    assert block["n_occurrence_matched"] == 10
    assert block["slope_occurrence_matched"]["n"] == 10
    assert block["slope_occurrence_matched"]["mean"] == pytest.approx(
        TRUE_SLOPE["evidence"])
    assert block["occurrence_gap"]["mean"] == pytest.approx(
        (10 * 0 + 14 * 3) / 24)
    # the pooled slope is unchanged: a subset is a smaller sample, not a better
    # estimator, so it never replaces the headline number
    assert block["slope"]["n"] == 24


def test_the_sensitivity_is_absent_when_nothing_recorded_the_gap(summary):
    """Readouts written before the gap was recorded must not silently become
    an empty 'matched subset'."""
    block = summary["sm_primary"]["evidence"]
    assert "slope_occurrence_matched" not in block
    assert summary_to_row(summary, [])["eci_slope_matched"] is None


def test_the_admissibility_axis_has_no_edit_magnitude_to_match(tmp_path: Path):
    """It rewrites the candidate list, not the document, so the whole question
    does not arise -- and the CSV must not imply that it does."""
    rows = synth_rows()
    for row in rows:
        if row["anchor_type"] == "evidence":
            row["occurrence_gap"] = 0
    summary = summarize(load_views(write_rows(tmp_path, rows)), tag="unit")
    assert "slope_occurrence_matched" not in summary["sm_primary"]["admissibility"]
    row = summary_to_row(summary, [])
    assert "aci_slope_matched" not in row
    assert row["eci_slope_matched"] is not None


def test_a_duplicated_cell_is_a_failure_not_a_resume(tmp_path: Path):
    """The writer skips cells already in the file, so a repeat is not a resume:
    it is two runs' output concatenated, and the later row silently wins."""
    rows = synth_rows()
    hazard = hazard_rows(rows)
    hazard.append(dict(hazard[0], n_closed=1, n_continued=15,
                       stop_hazard=1 / 16, stop_hazard_decided=1 / 16))
    with pytest.raises(SystemExit) as excinfo:
        load_views(write_rows(tmp_path, rows), write_hazard(tmp_path, hazard),
                   hazard_arms="base,manip,neutral", strict_hazard=True)
    assert "more than once" in str(excinfo.value)


@pytest.mark.parametrize("broken, expect", [
    ({"n_samples": 8}, "not the protocol"),
    ({"stop_hazard": 1.7}, "not a rate"),
    ({"stop_hazard": "x"}, "not a number"),
    ({"stop_hazard_decided": 2.0}, "not a rate"),
    ({"n_closed": -1, "n_continued": 17}, "not a count"),
    ({"n_samples": True}, "not the protocol"),
    ({"n_other": 3}, "do not add up"),
    # the sum still holds; only the derived rate disagrees
    ({"n_closed": 9, "n_continued": 7}, "disagrees with n_closed"),
])
def test_a_corrupt_hazard_row_is_not_a_measurement(broken, expect):
    """Every one of these would flow into the rates and the CI as if it were a
    measurement.  The writer produces none of them, so each means the file was
    corrupted, concatenated, or written under another protocol."""
    from tcr.support_probe.analyze import hazard_row_problem
    row = {"cell_id": "a", "n_samples": 16, "n_closed": 4, "n_continued": 12,
           "n_other": 0, "n_noncanonical": 0, "stop_hazard": 0.25,
           "stop_hazard_decided": 0.25}
    assert hazard_row_problem(row) is None
    problem = hazard_row_problem({**row, **broken})
    assert problem and expect in problem


def test_the_summary_records_which_analysis_produced_it(tmp_path: Path):
    """`protocol_version` pins the measurement; this pins the code that turned
    it into numbers, and they move independently."""
    summary = summarize(load_views(write_rows(tmp_path, synth_rows())), tag="u")
    assert summary["summary_schema"] == protocol.SUMMARY_SCHEMA_VERSION


# ---------------- gate 7 must compare the hazard to its OWN sampling state

def test_gate_7_is_read_against_the_state_the_hazard_was_sampled_in(tmp_path):
    """The hazard's prefix carries no presence penalty; `sm_primary`'s does.
    Comparing them would be comparing two distributions."""
    summary = summary_with_hazard(tmp_path)
    assert summary["hazard_margin_family"] == "sm_unpenalised"
    gate = next(g for g in instrument_gates(summary)
                if g["gate"] == "7_hazard_is_measurable_and_agrees")
    assert gate["detail"]["compared_against"] == "sm_unpenalised"
    assert gate["detail"]["presence_state"] == protocol.HAZARD_PRESENCE_STATE
    for axis in ("admissibility", "evidence"):
        assert summary["stop_hazard"][axis]["presence_state"] == \
            protocol.HAZARD_PRESENCE_STATE


def test_a_hazard_out_of_reach_in_its_own_state_is_not_a_disagreement(tmp_path):
    """Without the prefix penalty the close token sits ~2.1 logits lower, so it
    can be outside the sampler on most anchors.  A flat hazard is then about the
    state, not the model, and gate 7 must say `not measurable` rather than
    reading it as the model contradicting its own margin."""
    rows = synth_rows()
    for row in rows:
        row["close_prob_pre_filter_unpenalised"] = 0.0  # unreachable there
    summary = summarize(load_views(write_rows(tmp_path, rows),
                                   write_hazard(tmp_path, hazard_rows(rows))),
                        tag="unit")
    for axis in ("admissibility", "evidence"):
        assert summary["stop_hazard"][axis]["reachable_share"] == 0.0
    gate = next(g for g in instrument_gates(summary)
                if g["gate"] == "7_hazard_is_measurable_and_agrees")
    assert gate["pass"] is False
    assert gate["detail"]["measurable"] is False


def test_hazard_rows_from_another_sampling_state_are_refused(tmp_path):
    rows = synth_rows()
    hazard = [dict(row, presence_state="prefix_penalised")
              for row in hazard_rows(rows)]
    with pytest.raises(SystemExit) as excinfo:
        load_views(write_rows(tmp_path, rows), write_hazard(tmp_path, hazard),
                   hazard_arms="base,manip,neutral", strict_hazard=True,
                   expect_presence_state=protocol.HAZARD_PRESENCE_STATE)
    assert "sampled in" in str(excinfo.value)


# ------------------------------- the score readouts' own integrity

def test_a_duplicated_score_cell_is_a_failure(tmp_path: Path):
    """The scorer skips cells the file already holds, so a repeat is two runs'
    output in one file -- and `load_views` silently kept the later row."""
    rows = synth_rows()
    rows.append(dict(rows[0], sm_primary=99.0))
    with pytest.raises(SystemExit) as excinfo:
        load_views(write_rows(tmp_path, rows))
    assert "duplicate score cell" in str(excinfo.value)


def test_a_short_readout_file_is_a_failure_not_a_thinner_sample(tmp_path: Path):
    """Fewer anchors only widen a CI, so nothing downstream would ever reveal
    that a third of the cells never landed."""
    rows = synth_rows()
    with pytest.raises(SystemExit) as excinfo:
        load_views(write_rows(tmp_path, rows), expect_cells=len(rows) + 40)
    assert "records" in str(excinfo.value)
    # the honest count passes
    assert load_views(write_rows(tmp_path, rows), expect_cells=len(rows))


# ------------------- the manifest must prove the FILE, not just its shape

def test_a_readout_file_that_does_not_match_its_manifest_is_refused(tmp_path):
    """Same cell ids, same count, same manifest beside it -- a swapped-in file
    passes every other check.  The digest is what binds the two."""
    path = write_rows(tmp_path, synth_rows())
    with pytest.raises(SystemExit) as excinfo:
        load_views(path, readouts_sha256="0" * 64)
    assert "sha256" in str(excinfo.value)
    from tcr.support_probe.io_utils import sha256_file
    assert load_views(path, readouts_sha256=sha256_file(path))


def test_a_hazard_file_that_does_not_match_its_manifest_is_refused(tmp_path):
    rows = synth_rows()
    hazard = write_hazard(tmp_path, hazard_rows(rows))
    with pytest.raises(SystemExit) as excinfo:
        load_views(write_rows(tmp_path, rows), hazard,
                   hazard_arms="base,manip,neutral", strict_hazard=True,
                   hazard_sha256="0" * 64)
    assert "sha256" in str(excinfo.value)


def test_rows_tagged_for_another_model_are_refused(tmp_path):
    """Cell ids are identical across models, so a whole file from the wrong
    model joins perfectly.  Every row carries its tag; check it."""
    rows = [dict(row, tag="qwen3-4b-OTHER") for row in synth_rows()]
    with pytest.raises(SystemExit) as excinfo:
        load_views(write_rows(tmp_path, rows), expect_tag="qwen3-4b-cleanv2-s42")
    assert "tagged" in str(excinfo.value)


def test_hazard_rows_tagged_for_another_model_are_refused(tmp_path):
    rows = synth_rows()
    hazard = [dict(row, tag="qwen3-4b-OTHER") for row in hazard_rows(rows)]
    with pytest.raises(SystemExit) as excinfo:
        load_views(write_rows(tmp_path, rows), write_hazard(tmp_path, hazard),
                   hazard_arms="base,manip,neutral", strict_hazard=True,
                   expect_tag="qwen3-4b-cleanv2-s42")
    assert "another model" in str(excinfo.value)


def test_a_missing_count_field_is_not_a_zero():
    """`row.get(field, 0)` let an incomplete row through as a legitimate zero;
    the writer always emits these, so an absent one means a corrupted row."""
    from tcr.support_probe.analyze import hazard_row_problem
    row = {"cell_id": "a", "n_samples": 16, "n_closed": 4, "n_continued": 12,
           "n_other": 0, "n_noncanonical": 0, "stop_hazard": 0.25,
           "stop_hazard_decided": 0.25}
    assert hazard_row_problem(row) is None
    for field in ("n_closed", "n_continued", "n_other", "n_noncanonical"):
        short = {name: value for name, value in row.items() if name != field}
        problem = hazard_row_problem(short)
        assert problem and field in problem, field


def test_a_sampler_excluded_anchor_is_still_informative(tmp_path: Path):
    """The floor rule asks a GRADED question, not a step-function one.

    `close_sampler_prob` is exactly 0.0 whenever the close token falls outside
    the top-p nucleus, and at a structured JSON boundary the model is peaked
    enough that top_p=0.8 keeps little more than the argmax.  On the real
    calibration models that rejected 78-81% of anchors whose margins sat at
    about -1.6 nat and moved by +1.0 to +2.3 nat under manipulation.  Sampler
    reachability is reported instead of excluding them."""
    rows = synth_rows(n=8)
    for row in rows:
        row["close_sampler_prob"] = 0.0             # nothing survives top-p
        row["close_survives_top_p"] = False
        row["close_prob_pre_filter"] = 0.17         # but perfectly graded
    block = summarize(load_views(write_rows(tmp_path, rows)),
                      tag="unit")["sm_primary"]["admissibility"]
    assert block["n_informative"] == 8              # all kept
    assert block["uninformative"] == {}
    assert block["sampler_reachable_share"] == 0.0  # and the caveat is reported
    assert block["slope"]["mean"] == pytest.approx(TRUE_SLOPE["admissibility"])


def test_gate_6_does_not_fire_on_readouts_that_are_all_noise(tmp_path: Path):
    """A sign disagreement between two quantities that both span zero is not a
    systematic reversal -- it is noise, and it used to fail the gate."""
    rows = synth_rows()
    for row in rows:                     # kill the effect: every arm identical
        row["sm_primary"] = row["sm_raw"] = row["sm_unpenalised"] = -1.0
        row["close_rank"] = 3
    summary = summarize(load_views(write_rows(tmp_path, rows)), tag="unit")
    gate = next(g for g in instrument_gates(summary)
                if g["gate"] == "6_no_systematic_sign_flip")
    assert gate["pass"] is True
    assert all(v["n_decided"] == 0 for v in gate["detail"].values())
