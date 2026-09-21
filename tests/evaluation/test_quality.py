# coding=utf-8
"""Task utility and the two support axes."""

from __future__ import annotations

import json

import pytest

from tcr.evaluation.quality import (
    candidate_entities, empty_prediction, gold_record, pooled, score_sample,
)


def relation(source: str, target: str, name: str = "关系") -> dict:
    return {"source": source, "target": target, "relation": name,
            "description": "描述"}


@pytest.fixture
def gold(eval_record):
    return gold_record(eval_record)


def test_candidate_entities_reads_the_prompt_form():
    assert candidate_entities("['甲', '乙', \"丙\"]") == ["甲", "乙", "丙"]
    assert candidate_entities('["a", "b"]') == ["a", "b"]
    assert candidate_entities(["a", "b"]) == ["a", "b"]
    assert candidate_entities("") == []
    # a malformed list must still yield its entities rather than abort a task
    assert candidate_entities("['甲', '乙'") == ["甲", "乙"]


def test_a_perfect_answer_scores_one(gold):
    text = json.dumps([relation("甲", "乙")], ensure_ascii=False)
    scored = score_sample(text, gold)

    assert scored["json_valid"] == 1
    assert scored["strict"] == {"tp": 1, "pred_n": 1, "gold_n": 1}
    assert scored["strict_f1"] == 1.0
    assert scored["n_supported"] == 1
    assert scored["n_dual_conflict"] == 0


def test_a_looping_answer_is_unparseable_and_costs_recall(gold):
    """A response truncated mid-array scores an empty prediction set and the
    full gold count -- the response-level counterpart of a loop."""
    text = '[{"source": "甲", "target": "乙", "relation": "关系", "descri'
    scored = score_sample(text, gold)

    assert scored["json_valid"] == 0
    assert scored["strict"] == {"tp": 0, "pred_n": 0, "gold_n": 1}
    assert scored["n_pred_blocks"] == 0


def test_the_two_support_axes_are_separable(gold):
    """A block can violate admissibility, evidence, both, or neither, and each
    case has to land in its own counter."""
    text = json.dumps([
        relation("甲", "乙"),          # supported on both axes
        relation("甲", "外来实体"),     # target not a candidate, not in text
        relation("实体0", "目标0"),     # candidate and in text
    ], ensure_ascii=False)
    scored = score_sample(text, gold)

    assert scored["n_pred_blocks"] == 3
    assert scored["n_in_candidate"] == 2
    assert scored["n_in_text"] == 2
    assert scored["n_supported"] == 2
    assert scored["n_dual_conflict"] == 1


def test_normalisation_is_shared_with_the_f1_metrics():
    """Case and internal whitespace must not read as a support violation --
    E1 uses one normalisation everywhere, and `*_exact` carries the stricter
    reading.  (Leading/trailing space is already stripped upstream by the
    frozen v1.0 parser, so it never reaches either counter.)"""
    record = {"key": 7, "text": "Tesla ships the Model 3 with an FSD chip.",
              "entities_str": "['Tesla', 'Model 3', 'FSD chip']",
              "output": [{"source": "Tesla", "target": "Model 3",
                          "relation": "ships", "description": "d"}]}
    gold = gold_record(record)
    text = json.dumps([{"source": "tesla", "target": "Model  3",
                        "relation": "ships", "description": "d"}],
                      ensure_ascii=False)
    scored = score_sample(text, gold)

    assert scored["n_in_candidate"] == 1
    assert scored["n_in_candidate_exact"] == 0
    assert scored["strict"]["tp"] == 1


def test_repeated_blocks_are_deduplicated_for_f1_but_not_for_support(gold):
    """A captured response emits one relation hundreds of times: F1 must not
    reward the repetition, and the support denominator must still see it."""
    text = json.dumps([relation("甲", "乙")] * 50, ensure_ascii=False)
    scored = score_sample(text, gold)

    assert scored["strict"] == {"tp": 1, "pred_n": 1, "gold_n": 1}
    assert scored["n_pred_blocks"] == 50
    assert scored["n_unique_supported"] == 1


def test_empty_prediction_detection():
    assert empty_prediction("[]")
    assert empty_prediction("```json\n[]\n```")
    assert not empty_prediction('[{"source": "a"}]')
    assert not empty_prediction("")


def test_pooling_weights_every_response_equally(gold):
    perfect = score_sample(json.dumps([relation("甲", "乙")],
                                      ensure_ascii=False), gold)
    broken = score_sample("[{", gold)
    summary = pooled([perfect, broken], total_gen_tokens=1000)

    assert summary["n_responses"] == 2
    assert summary["strict_tp"] == 1
    assert summary["strict_gold_n"] == 2       # the broken one still owes gold
    assert summary["strict_recall"] == 0.5
    assert summary["json_valid_rate"] == 0.5
    assert summary["unique_supported_per_1k_tokens"] == 1.0


def test_undefined_rates_are_none_not_zero(gold):
    """A model that produced no block at all has an UNDEFINED out-of-candidate
    rate.  Reporting 0.0 would put it top of the compliance table."""
    summary = pooled([score_sample("", gold)])
    assert summary["n_pred_blocks_total"] == 0
    assert summary["out_of_candidate_block_rate"] is None
    assert summary["support_valid_block_precision"] is None
    # no empty-gold record in E-Natural, so this stays undefined too
    assert summary["empty_gold_accuracy"] is None


def test_empty_gold_accuracy_when_the_gold_is_empty():
    record = {"key": 9, "text": "无关文本", "entities_str": "['甲']",
              "output": []}
    gold = gold_record(record)
    summary = pooled([score_sample("[]", gold),
                      score_sample('[{"source": "甲", "target": "甲", '
                                   '"relation": "r", "description": "d"}]', gold)])
    assert summary["n_empty_gold_responses"] == 2
    assert summary["empty_gold_accuracy"] == 0.5
