# coding=utf-8
"""The helper scripts: preflight validation, CSV rebuild, M0/M1 pairing.

These are the places where a silent wrong answer is cheapest to produce and
most expensive to notice, so each test below pins one specific way of being
silently wrong.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tcr.evaluation.io_utils import append_csv_row, read_csv_rows, write_json

ROOT = Path(__file__).resolve().parents[2]


def _script(name: str):
    """Import a `scripts/*.py` module (they are not a package)."""
    path = ROOT / "experiments" / "3_evaluate" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_e1_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preflight = _script("e1_preflight")
collect = _script("e1_collect")
make_pair = _script("e1_make_pair")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                              for row in rows) + "\n", encoding="utf-8")
    return path


def good_record(key: int) -> Dict[str, Any]:
    return {"key": key, "source": "s", "text": "甲与乙相关。",
            "entities_str": "['甲', '乙']",
            "output": [{"source": "甲", "target": "乙", "relation": "r",
                        "description": "d"}]}


# ------------------------------------------------------- preflight: eval set

def test_eval_validation_reads_every_record_not_just_the_first(tmp_path: Path):
    """The bug: only `records[0]` was checked, so a broken record anywhere else
    was scored against an empty gold set and looked perfectly valid."""
    rows = [good_record(1), good_record(2), good_record(3)]
    del rows[2]["output"]
    problems = preflight.validate_eval_set(write_jsonl(tmp_path / "e.jsonl", rows))
    assert len(problems) == 1
    assert "key=3" in problems[0] and "output" in problems[0]


def test_eval_validation_catches_every_broken_shape(tmp_path: Path):
    rows = [good_record(1), good_record(2), good_record(3), good_record(4)]
    rows[0]["text"] = "   "
    rows[1]["entities_str"] = ""
    rows[2]["output"] = {"not": "a list"}
    rows[3]["output"] = "{not json"
    problems = preflight.validate_eval_set(write_jsonl(tmp_path / "e.jsonl", rows))
    assert len(problems) == 4
    assert any("'text'" in p for p in problems)
    assert any("'entities_str'" in p for p in problems)
    assert any("expected a list" in p for p in problems)
    assert any("unparseable" in p for p in problems)


def test_eval_validation_rejects_duplicates_and_empty_files(tmp_path: Path):
    duplicated = write_jsonl(tmp_path / "dup.jsonl",
                             [good_record(1), good_record(1)])
    assert any("duplicate key" in p
               for p in preflight.validate_eval_set(duplicated))

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert preflight.validate_eval_set(empty) == ["the file is empty"]


def test_a_correct_eval_set_has_no_problems(tmp_path: Path):
    path = write_jsonl(tmp_path / "e.jsonl", [good_record(i) for i in range(5)])
    assert preflight.validate_eval_set(path) == []


# ------------------------------------------------- collect: the stage curve

def _row(tag: str, size: str, variant: str, seed: Any, capture: float, *,
         family: str = "stage", terminal: str = "") -> Dict[str, Any]:
    return {"tag": tag, "run_name": tag, "size": size, "data_variant": variant,
            "arm_family": family, "seed": seed, "is_final": "True",
            "terminal_mode": terminal, "semantic_capture_rate": capture,
            "strict_f1": 0.1}


def test_stage_curve_uses_the_canonical_seed_only():
    """`4B-cleanv2-s42` and `4B-cleanv2-s123` are both in the matrix; the curve
    must not depend on which one happens to be read last."""
    rows = [_row("qwen3-4b-clean", "4B", "clean", 42, 0.3),
            _row("qwen3-4b-cleanv2-s42", "4B", "cleanv2", 42, 0.1),
            _row("qwen3-4b-cleanv2-s123", "4B", "cleanv2", 123, 0.9)]
    problems: List[str] = []
    curve = collect.stage_curve(rows, problems)

    assert problems == []
    entry = next(e for e in curve if e["comparison"] == "clean->cleanv2")
    assert entry["after_tag"] == "qwen3-4b-cleanv2-s42"
    assert entry["semantic_capture_rate_after"] == pytest.approx(0.1)


def test_stage_curve_refuses_an_ambiguous_cell():
    rows = [_row("a", "4B", "clean", 42, 0.3), _row("b", "4B", "clean", 42, 0.4),
            _row("c", "4B", "cleanv2", 42, 0.1)]
    problems: List[str] = []
    curve = collect.stage_curve(rows, problems)

    assert any("ambiguous" in p for p in problems)
    assert not [e for e in curve if e["comparison"] == "clean->cleanv2"]


def test_stage_curve_pairs_terminal_aware_with_its_own_data_base():
    rows = [_row("qwen3-4b-clean", "4B", "clean", 42, 0.3),
            _row("qwen3-4b-clean-ta", "4B", "clean", 42, 0.05,
                 terminal="close_eos")]
    entry = collect.stage_curve(rows, [])[0]
    assert entry["before_tag"] == "qwen3-4b-clean"
    assert entry["after_tag"] == "qwen3-4b-clean-ta"
    assert entry["semantic_capture_rate_delta"] == pytest.approx(-0.25)


def test_rebuild_keeps_a_row_appended_concurrently(tmp_path: Path, monkeypatch):
    """collect rewrites the CSV the orchestrator is appending to; a row that
    landed between reading the summaries and writing the file must survive."""
    root = tmp_path / "results"
    (root / "summary").mkdir(parents=True)
    write_json(root / "summary" / "a_summary.json", {
        "task": {"tag": "a", "size": "4B", "data_variant": "clean",
                 "arm_family": "stage", "seed": 42, "is_final": True,
                 "tier": 3, "priority": 3},
        "events": {"n_records": 1, "n_responses": 8,
                   "semantic_capture_rate": 0.5},
        "quality": {"strict_f1": 0.2}, "episodes": {}, "bootstrap_ci": {},
        "generation": {"chat_prefix_ok": True}})
    append_csv_row(root / "e1_metrics.csv",
                   {"tag": "late-arrival", "semantic_capture_rate": 0.9},
                   collect.CSV_FIELDS)

    monkeypatch.setattr(sys, "argv",
                        ["e1_collect.py", "--result_root", str(root)])
    assert collect.main() == 0

    tags = {row["tag"] for row in read_csv_rows(root / "e1_metrics.csv")}
    assert tags == {"a", "late-arrival"}


# ------------------------------------------------------- make_pair: guards

def _event_row(prompt: str, seed: int) -> Dict[str, Any]:
    return {"stable_prompt_id": prompt, "seed": seed, "key": int(prompt),
            "model_tag": "m", "sample_id": f"m:{prompt}:{seed}",
            "selection_roles": ["prevalence"]}


def test_pairing_refuses_a_file_that_mixes_two_runs(tmp_path: Path):
    path = write_jsonl(tmp_path / "events.jsonl",
                       [_event_row("1", 0), _event_row("1", 0)])
    with pytest.raises(SystemExit, match="two rows for prompt"):
        make_pair._index(path)


def test_pairing_refuses_an_empty_file(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="empty"):
        make_pair._index(path)


def _arm(root: Path, tag: str, *, eval_sha: str) -> Path:
    events = root / "events" / f"{tag}_event_rows.jsonl"
    write_jsonl(events, [_event_row("1", 0), _event_row("2", 0)])
    write_json(root / "summary" / f"{tag}_summary.json", {
        "protocol_version": "e1-v1.0",
        "generation": {"eval_data_sha256": eval_sha,
                       "prompt_rendered_sha256": "PROMPT"}})
    return events


def test_pairing_refuses_two_arms_scored_on_different_eval_sets(tmp_path: Path):
    m0 = _arm(tmp_path, "arm0", eval_sha="AAA")
    m1 = _arm(tmp_path, "arm1", eval_sha="BBB")
    with pytest.raises(SystemExit, match="evaluation set"):
        make_pair.check_comparable(m0, m1)


def test_pairing_accepts_two_comparable_arms(tmp_path: Path, capsys):
    m0 = _arm(tmp_path, "arm0", eval_sha="AAA")
    m1 = _arm(tmp_path, "arm1", eval_sha="AAA")
    make_pair.check_comparable(m0, m1)
    assert "share one protocol" in capsys.readouterr().out


def test_three_duplicates_do_not_crash_the_curve():
    """The ambiguity path used to poison the map with None; a THIRD match then
    dereferenced it."""
    rows = [_row(tag, "4B", "clean", 42, 0.3) for tag in ("a", "b", "c")]
    rows.append(_row("v2", "4B", "cleanv2", 42, 0.1))
    problems: List[str] = []
    curve = collect.stage_curve(rows, problems)

    assert len(problems) == 1
    assert "3 rows (a, b, c)" in problems[0]
    assert not [e for e in curve if e["comparison"] == "clean->cleanv2"]


def _summary_doc(tag: str, *, eval_sha: str = "EVALSHA",
                 protocol_version: str = "e1-v1.0") -> Dict[str, Any]:
    return {"task": {"tag": tag, "size": "4B", "data_variant": "clean",
                     "arm_family": "stage", "seed": 42, "is_final": True,
                     "tier": 3, "priority": 3},
            "protocol_version": protocol_version,
            "events": {"n_records": 1, "n_responses": 8,
                       "semantic_capture_rate": 0.5},
            "quality": {"strict_f1": 0.2}, "episodes": {"n_errors": 0},
            "bootstrap_ci": {},
            "generation": {"chat_prefix_ok": True, "eval_data_sha256": eval_sha,
                           "prompt_rendered_sha256": "PROMPT"}}


def test_a_stale_carried_row_is_audited_not_waved_through(tmp_path: Path,
                                                          monkeypatch, capsys):
    """The bug: carried rows went into the CSV but the audit only looked at the
    summary rows, so a leftover from an incompatible run got reported as
    'comparability checks passed'."""
    root = tmp_path / "results"
    (root / "summary").mkdir(parents=True)
    write_json(root / "summary" / "current_summary.json", _summary_doc("current"))
    append_csv_row(root / "e1_metrics.csv",
                   {"tag": "leftover", "eval_data_sha256": "AN_OLD_EVAL_SET",
                    "prompt_rendered_sha256": "PROMPT",
                    "protocol_version": "e1-v0.9", "chat_prefix_ok": "True",
                    "n_records": "1", "n_responses": "8"},
                   collect.CSV_FIELDS)

    monkeypatch.setattr(sys, "argv", ["e1_collect.py", "--result_root",
                                      str(root), "--strict"])
    assert collect.main() == 1                    # --strict must refuse

    out = capsys.readouterr().out
    assert "evaluation set differs across tasks" in out
    assert "protocol version differs across tasks" in out
    assert "no summary/*.json backs it" in out
    assert "comparability checks passed" not in out


def test_a_concurrently_appended_row_is_rebuilt_from_its_summary(
        tmp_path: Path, monkeypatch, capsys):
    """The row we are protecting is backed by a summary (the orchestrator
    appends only after analysis wrote one), so the re-read inside the lock
    turns it into a first-class, auditable row rather than a copy."""
    root = tmp_path / "results"
    (root / "summary").mkdir(parents=True)
    write_json(root / "summary" / "a_summary.json", _summary_doc("a"))
    write_json(root / "summary" / "late_summary.json", _summary_doc("late"))
    append_csv_row(root / "e1_metrics.csv", {"tag": "late"}, collect.CSV_FIELDS)

    monkeypatch.setattr(sys, "argv", ["e1_collect.py", "--result_root", str(root)])
    assert collect.main() == 0

    rows = read_csv_rows(root / "e1_metrics.csv")
    assert {row["tag"] for row in rows} == {"a", "late"}
    late = next(row for row in rows if row["tag"] == "late")
    assert late["semantic_capture_rate"] == "0.5"     # from the summary, not ""
    assert "no summary/*.json backs it" not in capsys.readouterr().out


def test_pairing_fails_closed_without_a_summary(tmp_path: Path):
    """Missing provenance is not the same as compatible provenance: the prompt
    keys line up either way, so the pairing would look perfectly healthy."""
    events = write_jsonl(tmp_path / "events" / "arm_event_rows.jsonl",
                         [_event_row("1", 0)])
    other = write_jsonl(tmp_path / "events" / "other_event_rows.jsonl",
                        [_event_row("1", 0)])
    with pytest.raises(SystemExit, match="no summary"):
        make_pair.check_comparable(events, other)


def test_pairing_without_a_summary_needs_an_explicit_override(tmp_path: Path,
                                                              capsys):
    events = write_jsonl(tmp_path / "events" / "arm_event_rows.jsonl",
                         [_event_row("1", 0)])
    other = write_jsonl(tmp_path / "events" / "other_event_rows.jsonl",
                        [_event_row("1", 0)])
    make_pair.check_comparable(events, other, allow_unverified=True)
    assert "--allow_unverified" in capsys.readouterr().out
