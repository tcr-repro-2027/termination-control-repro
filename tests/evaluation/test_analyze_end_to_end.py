# coding=utf-8
"""The whole CPU stage, end to end, on a synthetic responses file.

No model files and no vLLM: the tokenizer is injected (one token per
character), which is enough because everything downstream of it only compares
token ids and maps token indices to char offsets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from conftest import block_text, response_of
from tcr.evaluation.aggregate import CSV_FIELDS, summary_to_row
from tcr.evaluation.analyze import analyze
from tcr.evaluation.io_utils import append_csv_row, read_csv_rows, read_jsonl


class CharTokenizer:
    """Minimal stand-in for a HF fast tokenizer: one token per character."""

    is_fast = True

    def __call__(self, texts, add_special_tokens=False,
                 return_offsets_mapping=False):
        if isinstance(texts, str):
            texts = [texts]
        ids = [[ord(char) for char in text] for text in texts]
        result: Dict[str, Any] = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                [(index, index + 1) for index in range(len(text))]
                for text in texts]
        return result


LOOP = response_of([block_text("甲", "乙")] * 60, closed=False)
CLEAN = response_of([block_text("甲", "乙"), block_text("实体0", "目标0")])


def eval_rows() -> List[Dict[str, Any]]:
    return [
        {"key": index, "source": "unit",
         "text": "甲和乙相关，实体0 与 目标0 也相关。",
         "entities_str": "['甲', '乙', '实体0', '目标0']",
         "output": [{"source": "甲", "target": "乙", "relation": "关系",
                     "description": "d"}]}
        for index in range(3)
    ]


def response_rows() -> List[Dict[str, Any]]:
    rows = []
    for index, record in enumerate(eval_rows()):
        samples = []
        for seed in range(4):
            loops = index == 0 or (index == 1 and seed < 2)
            text = LOOP if loops else CLEAN
            samples.append({"seed": seed, "response": text, "reasoning": "",
                            "finish_reason": "length" if loops else "stop",
                            "gen_tokens_engine": len(text)})
        rows.append({"key": record["key"], "source": record["source"],
                     "text": record["text"],
                     "entities_str": record["entities_str"],
                     "prompt_tokens": 1234, "skipped_reason": "",
                     "responses": samples})
    # a record whose prompt did not fit: no responses, must not be scored
    rows.append({"key": 99, "source": "unit", "text": "x", "entities_str": "[]",
                 "prompt_tokens": 40000, "skipped_reason": "prompt_exceeds_context",
                 "responses": []})
    return rows


@pytest.fixture
def workspace(tmp_path: Path) -> Dict[str, Path]:
    eval_path = tmp_path / "eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as handle:
        for row in eval_rows():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        # E-Natural has no empty-gold record, and 01b's gold loader rejects
        # one, so the over-long record carries a normal gold row too.
        handle.write(json.dumps(
            {"key": 99, "source": "unit", "text": "甲与乙", "entities_str": "['甲', '乙']",
             "output": [{"source": "甲", "target": "乙", "relation": "关系",
                         "description": "d"}]},
            ensure_ascii=False) + "\n")

    responses = tmp_path / "results" / "responses" / "unit_nothink_n8.jsonl"
    responses.parent.mkdir(parents=True, exist_ok=True)
    with responses.open("w", encoding="utf-8") as handle:
        for row in response_rows():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"root": tmp_path / "results", "eval": eval_path,
            "responses": responses}


@pytest.fixture
def summary(workspace):
    return analyze(responses_path=str(workspace["responses"]),
                   eval_data=str(workspace["eval"]),
                   tokenizer_path="unused",
                   output_root=str(workspace["root"]), tag="unit-model",
                   task={"tag": "unit-model", "size": "4B",
                         "data_variant": "clean", "is_final": True},
                   audit_samples=3, n_boot=100, tokenizer=CharTokenizer())


def test_denominators_exclude_the_skipped_record(summary):
    assert summary["events"]["n_records"] == 3
    assert summary["events"]["n_responses"] == 12
    assert summary["n_skipped_records"] == 1


def test_repetition_rates_match_the_construction(summary):
    events = summary["events"]
    # 6 of 12 responses loop: all 4 of record 0, 2 of record 1
    assert events["semantic_capture_rate"] == pytest.approx(0.5)
    assert events["stable_orbit_rate"] == pytest.approx(0.5)
    assert events["hit_max_rate"] == pytest.approx(0.5)
    assert events["semantic_capture_trigger_rate"] == pytest.approx(2 / 3)


def test_quality_reflects_the_loops(summary):
    quality = summary["quality"]
    assert quality["n_responses"] == 12
    # the looping answers are truncated mid-array -> unparseable
    assert quality["json_valid_rate"] == pytest.approx(0.5)
    assert 0.0 < quality["strict_f1"] <= 1.0
    assert quality["strict_gold_n"] == 12          # every response owes gold


def test_event_rows_are_written_in_the_01b_schema(summary, workspace):
    rows = list(read_jsonl(summary["event_rows_path"]))
    assert len(rows) == 12
    for name in ("sample_id", "stable_prompt_id", "model_tag", "seed",
                 "n_blocks", "n_identity_complete_blocks", "block_index",
                 "first_triple_reuse", "first_nonempty_triple_reuse",
                 "first_exact_quad_reuse", "motif_capture_triple",
                 "motif_capture_quad", "legacy_orbit", "alignment_type",
                 "alignment_evidence", "stage_flags", "stage_chain",
                 "selection_roles", "parse_diagnostic_counts"):
        assert name in rows[0], name
    assert rows[0]["selection_roles"] == ["prevalence"]
    assert len({row["sample_id"] for row in rows}) == 12
    assert "quality" in rows[0], "per-response utility travels with the row"
    assert not list(Path(summary["event_rows_path"]).parent.glob("*.partial"))


def test_the_01b_p0_analysers_accept_the_rows(summary):
    """The reason the schema matters: P0b/P0c/P0d must run on E1 output."""
    from tcr.events.p0b_competing_risk import (
        load_gold_mapping, validate_prevalence_rows,
    )
    rows = [dict(row, model_tag="M0" if index % 2 else "M1",
                 sample_id=f"{'M0' if index % 2 else 'M1'}:"
                           f"{row['key']}:{row['seed']}")
            for index, row in enumerate(list(read_jsonl(summary["event_rows_path"])))]
    # rebuild a clean pairing: every prompt/seed present for both arms
    paired = []
    for row in rows:
        for tag in ("M0", "M1"):
            paired.append(dict(row, model_tag=tag,
                               sample_id=f"{tag}:{row['key']}:{row['seed']}"))
    prevalence, prompt_ids = validate_prevalence_rows(paired)
    assert len(prompt_ids) == 3
    assert len(prevalence) == 24

    gold = load_gold_mapping(summary["eval_data"],
                             [row["key"] for row in rows])
    assert gold.n_rows == 4


def test_audit_sample_is_bounded(summary, workspace):
    audit = workspace["root"] / "audit" / "unit-model_audit_samples.jsonl"
    rows = list(read_jsonl(audit))
    assert len(rows) == 3
    assert rows[0]["snippets"]["capture"]


def test_summary_becomes_one_csv_row(summary, tmp_path: Path):
    row = summary_to_row(summary)
    path = tmp_path / "e1_metrics.csv"
    append_csv_row(path, row, CSV_FIELDS)

    written = read_csv_rows(path)[0]
    assert written["tag"] == "unit-model"
    assert written["n_responses"] == "12"
    assert float(written["semantic_capture_rate"]) == pytest.approx(0.5)
    assert float(written["stable_orbit_rate_ci_low"]) <= 0.5
    assert float(written["stable_orbit_rate_ci_high"]) >= 0.5


def test_episode_columns_are_populated(summary):
    episodes = summary["episodes"]
    assert episodes["n_errors"] == 0
    assert episodes["n_scored_responses"] == 12
    assert episodes["episodes_per_response"] == pytest.approx(0.5)
    # This is the exposure/propensity split, and the two numbers differ here on
    # purpose: half the responses ever open an episode, but every episode that
    # opens is captured.  A response-level capture rate of 0.5 with a
    # per-episode hazard of 1.0 is a pure EXPOSURE story.
    assert episodes["per_episode_capture_hazard"] == pytest.approx(1.0)
    assert episodes["responses_with_any_episode_rate"] == pytest.approx(0.5)


def test_a_responses_file_from_another_eval_set_is_rejected(workspace):
    """Scoring against the wrong gold silently produces plausible-looking F1,
    so it must be an error, not a warning."""
    other = workspace["root"] / "other_eval.jsonl"
    other.write_text(json.dumps({"key": 12345, "text": "x",
                                 "entities_str": "[]", "output": []},
                                ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no gold row"):
        analyze(responses_path=str(workspace["responses"]),
                eval_data=str(other), tokenizer_path="unused",
                output_root=str(workspace["root"]), tag="mismatch",
                n_boot=10, tokenizer=CharTokenizer())
