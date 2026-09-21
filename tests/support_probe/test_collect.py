# coding=utf-8
"""The report a human actually reads, and the two ways it can mislead.

`e2_metrics.csv` carries everything, but nobody reads a 50-column CSV before
deciding whether the instrument passed.  They read `e2_instrument_report.md`.
So the failures that matter here are the ones where the CSV is right and the
report is reassuring:

* a gate that fails but is not in the table at all (gate 7 used to be);
* a gate that was never MEASURED rendering the same as one that passed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcr.support_probe import protocol                                        # noqa: E402
from tcr.support_probe.io_utils import write_json                             # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "e2_collect", ROOT / "experiments" / "5_support_probe" / "e2_collect.py")
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)


def summary(tag: str, *, hazard: bool, gate7_pass: bool = True):
    gates = [{"gate": name, "pass": True, "detail": {}, "note": ""}
             for name in ("1_neutral_effect_near_zero",
                          "2_positive_control_direction",
                          "3_support_manipulation_responds",
                          "4_parser_prefix_token_alignment",
                          "5_floor_anchors_flagged",
                          "6_no_systematic_sign_flip")]
    if hazard:
        gates.append({"gate": "7_hazard_is_measurable_and_agrees",
                      "pass": gate7_pass, "detail": {"other_rate": 0.9},
                      "note": ""})
    block = {"slope": {"mean": 0.5, "ci_low": 0.2, "ci_high": 0.8},
             "pc": {"mean": 2.0, "ci_low": 1.5, "ci_high": 2.5}}
    hazard_block = {
        "slope": {"mean": 0.2}, "base_rate": {"mean": 0.3},
        "other_rate": {"mean": 0.9 if not gate7_pass else 0.02},
        "noncanonical_rate": {"mean": 0.01},
        "n_with_hazard": 24,
    } if hazard else {"n_anchors": 24, "n_with_hazard": 0}
    return {
        "tag": tag, "protocol": protocol.describe(),
        "summary_schema": protocol.SUMMARY_SCHEMA_VERSION,
        "anchors_sha256": "deadbeef", "n_anchors": 384, "n_cells": 2560,
        "sm_primary": {"admissibility": block, "evidence": block,
                       "zero_answer": {"empty_margin": {"mean": 1.5},
                                       "close_is_top1_rate": {"mean": 0.9}}},
        "sm_raw": {"admissibility": block, "evidence": block},
        "stop_hazard": {"admissibility": hazard_block, "evidence": hazard_block},
        "has_hazard": hazard,
        "hazard_agrees_with_margin": {"admissibility": gate7_pass,
                                      "evidence": gate7_pass},
        "gates": gates, "task": {"data_variant": "cleanv2"},
    }


@pytest.fixture
def report(tmp_path):
    def build(*summaries) -> str:
        analysis = tmp_path / "analysis"
        analysis.mkdir(exist_ok=True)
        for body in summaries:
            write_json(analysis / f"{body['tag']}_e2_summary.json", body)
        out = tmp_path / "report.md"
        # the real comparability pass, not a stub: half of what this file
        # checks is whether an incomparable table says so
        problems, notes = collect.comparability(analysis)
        collect.write_report(out, analysis, problems, notes)
        return out.read_text(encoding="utf-8")
    return build


def test_a_failing_hazard_gate_appears_in_the_main_table(report):
    """The bug this pins: gate 7 reached the CSV but not the report, so a run
    whose behaviour column disagreed with its margins read as all-green."""
    text = report(summary("qwen3-4b-cleanv2-s42", hazard=True, gate7_pass=False))
    assert "7 短续写" in text
    header, row = [line for line in text.splitlines()
                   if "7 短续写" in line or "qwen3-4b-cleanv2-s42 |" in line][:2]
    assert row.count("✗") == 1                      # exactly gate 7


def test_a_passing_hazard_gate_is_a_tick(report):
    text = report(summary("qwen3-4b-cleanv2-s42", hazard=True))
    row = next(line for line in text.splitlines()
               if line.startswith("| qwen3-4b-cleanv2-s42 |"))
    assert "✗" not in row and row.count("✓") == 6


def test_an_unmeasured_hazard_is_a_dash_not_a_tick(report):
    """NO_HAZARD=1 SKIPS gate 7.  Rendering that as a pass would claim the
    behaviour agreed with the margin when it was never generated."""
    text = report(summary("qwen3-4b-notrain", hazard=False))
    row = next(line for line in text.splitlines()
               if line.startswith("| qwen3-4b-notrain |"))
    assert row.count("–") == 1 and "✗" not in row
    assert "未测" in text


def test_the_hazard_numbers_get_their_own_section_when_they_exist(report):
    text = report(summary("a", hazard=True))
    assert "## 3. 短续写" in text
    assert "other 率" in text


def test_the_hazard_section_is_absent_when_nothing_measured_it(report):
    text = report(summary("a", hazard=False))
    assert "## 3. 短续写" not in text
    assert "## 4. 零答案参照" in text and "## 5. 可比性" in text


# ------------------------------------------------------- comparability

def test_a_different_prompt_makes_the_numbers_incomparable(report, tmp_path):
    """A margin measured under a different instruction is a different margin;
    the report used to call that "one instrument"."""
    a = summary("a", hazard=False)
    b = summary("b", hazard=False)
    a["prompt_rendered_sha256"] = "aaaa"
    b["prompt_rendered_sha256"] = "bbbb"
    text = report(a, b)
    assert "prompt_rendered_sha256 differs" in text


def test_a_different_resample_count_makes_the_intervals_incomparable(report):
    a, b = summary("a", hazard=False), summary("b", hazard=False)
    a["bootstrap"], b["bootstrap"] = 2000, 100
    assert "bootstrap differs" in report(a, b)


def test_different_hazard_arms_are_a_comparability_problem(report):
    a, b = summary("a", hazard=True), summary("b", hazard=True)
    a["hazard_arms"], b["hazard_arms"] = "base,manip,neutral", "all"
    assert "hazard_arms differs" in report(a, b)


def test_partial_hazard_coverage_is_reported_not_hidden(report):
    """Running the short continuations on only the key arms is a documented
    choice, so it is a NOTE -- but a silent one would leave gate 7 absent on
    half the table with nothing saying why."""
    text = report(summary("a", hazard=True), summary("b", hazard=False))
    assert "hazard coverage is partial" in text
    assert "1 model(s) have a behaviour column" in text


def test_one_instrument_still_reads_as_one_instrument(report):
    text = report(summary("a", hazard=True), summary("b", hazard=True))
    assert "所有模型使用同一套锚点" in text


def test_a_summary_from_older_analysis_code_is_not_comparable(report):
    """`protocol_version` can match while the numbers came from different
    analysis code."""
    a, b = summary("a", hazard=False), summary("b", hazard=False)
    b["summary_schema"] = "e2-summary-v1"
    assert "summary_schema differs" in report(a, b)


def test_partial_hazard_coverage_is_a_note_not_a_problem(tmp_path):
    """`--strict` has to be usable on the recommended workflow (§5 says to run
    the short continuations on the key arms only), so this must not fail it."""
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    for body in (summary("a", hazard=True), summary("b", hazard=False)):
        write_json(analysis / f"{body['tag']}_e2_summary.json", body)
    problems, notes = collect.comparability(analysis)
    assert problems == []
    assert any("hazard coverage is partial" in note for note in notes)


def test_a_real_incomparability_is_a_problem(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    a, b = summary("a", hazard=False), summary("b", hazard=False)
    b["anchors_sha256"] = "OTHER"
    for body in (a, b):
        write_json(analysis / f"{body['tag']}_e2_summary.json", body)
    problems, _notes = collect.comparability(analysis)
    assert any("anchors_sha256 differs" in problem for problem in problems)


# --------------------------------------------------- the CSV's own hygiene

def orphan_csv(tmp_path):
    """A CSV row whose summary is gone -- what an `ONLY=` run on an existing
    RESULT_ROOT leaves behind."""
    analysis = tmp_path / "analysis"
    analysis.mkdir(exist_ok=True)
    write_json(analysis / "a_e2_summary.json", summary("a", hazard=False))
    csv_path = tmp_path / "e2_metrics.csv"
    csv_path.write_text("tag,protocol_version\nGONE,e2-v1.0\n", encoding="utf-8")
    return csv_path


def run_collect(tmp_path, *extra) -> int:
    import subprocess
    return subprocess.run(
        [sys.executable,
         str(ROOT / "experiments" / "5_support_probe" / "e2_collect.py"),
         "--result_root", str(tmp_path), *extra],
        capture_output=True, text=True).returncode


def test_a_row_with_no_summary_behind_it_is_dropped(tmp_path):
    """A number nobody can reproduce is worse than a missing one."""
    csv_path = orphan_csv(tmp_path)
    assert run_collect(tmp_path) == 0
    assert "GONE" not in csv_path.read_text(encoding="utf-8")


def test_keeping_an_orphan_row_is_possible_but_is_then_a_problem(tmp_path):
    csv_path = orphan_csv(tmp_path)
    assert run_collect(tmp_path, "--keep_orphan_rows") == 0
    assert "GONE" in csv_path.read_text(encoding="utf-8")
    # ...and strict refuses it, because nothing can reproduce that row
    assert run_collect(tmp_path, "--keep_orphan_rows", "--strict") == 1


def test_the_report_shows_eci_next_to_its_matched_subset(report):
    """§9.5 sensitivity, in the document a human actually reads: the evidence
    axis substitutes over the whole document, so pooled ECI alone cannot be
    called a pure evidence-axis effect."""
    body = summary("a", hazard=False)
    body["sm_primary"]["evidence"] = {
        **body["sm_primary"]["evidence"],
        "slope_occurrence_matched": {"mean": 0.42, "ci_low": 0.1, "ci_high": 0.7},
        "n_occurrence_matched": 80,
    }
    text = report(body)
    assert "ECI 等量子集" in text
    # §1 is the gate table, §2 the support-response one
    row = [line for line in text.splitlines() if line.startswith("| a |")][1]
    assert "+0.420" in row and "(n=80)" in row
    assert "纯证据轴效应是不成立的" in text


def test_a_summary_without_the_subset_says_so_rather_than_implying_zero(report):
    text = report(summary("a", hazard=False))
    row = [line for line in text.splitlines() if line.startswith("| a |")][1]
    assert "未记录" in row


def test_an_unverified_hazard_summary_is_refused_by_the_collector(tmp_path):
    """`--allow_unverified_hazard` is a hand-inspection door.  It used to print
    a warning and then write the summary and the CSV row anyway."""
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    body = summary("a", hazard=True)
    body["hazard_verified"] = False
    write_json(analysis / "a_e2_summary.json", body)
    problems, _notes = collect.comparability(analysis)
    assert any("WITHOUT a manifest" in problem for problem in problems)


def test_hazard_measured_in_two_different_states_is_incomparable(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    a, b = summary("a", hazard=True), summary("b", hazard=True)
    for axis in ("admissibility", "evidence"):
        a["stop_hazard"][axis] = {**a["stop_hazard"][axis],
                                  "presence_state": "prefill_unpenalised"}
        b["stop_hazard"][axis] = {**b["stop_hazard"][axis],
                                  "presence_state": "prefix_penalised"}
    for body in (a, b):
        write_json(analysis / f"{body['tag']}_e2_summary.json", body)
    problems, _notes = collect.comparability(analysis)
    assert any("hazard_presence_state differs" in problem for problem in problems)


def test_strict_does_not_leave_a_csv_it_calls_invalid(tmp_path):
    """The exit code is transient; the file is not.  A table written and then
    declared invalid is what a reader finds three weeks later."""
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    a, b = summary("a", hazard=False), summary("b", hazard=False)
    b["anchors_sha256"] = "OTHER"                    # genuinely incomparable
    for body in (a, b):
        write_json(analysis / f"{body['tag']}_e2_summary.json", body)

    csv_path = tmp_path / "e2_metrics.csv"
    assert run_collect(tmp_path, "--strict") == 1
    assert not csv_path.exists()
    # the report IS written: that is where the reason lives
    assert "anchors_sha256 differs" in (
        tmp_path / "e2_instrument_report.md").read_text(encoding="utf-8")
    # without --strict the table is still produced, with the problem reported
    assert run_collect(tmp_path) == 0
    assert csv_path.exists()


def test_a_previously_good_csv_is_not_overwritten_by_a_bad_run(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    write_json(analysis / "a_e2_summary.json", summary("a", hazard=False))
    csv_path = tmp_path / "e2_metrics.csv"
    assert run_collect(tmp_path, "--strict") == 0
    good = csv_path.read_text(encoding="utf-8")

    b = summary("b", hazard=False)
    b["summary_schema"] = "e2-summary-v1"
    write_json(analysis / "b_e2_summary.json", b)
    assert run_collect(tmp_path, "--strict") == 1
    assert csv_path.read_text(encoding="utf-8") == good
