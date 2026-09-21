# coding=utf-8
"""Tests for the frozen relation-extraction metrics (micro-F1 pooling).

Run from the project dir:   python tests/test_metrics.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tcr.extraction.evaluation.metrics import (
    evaluate_record,
    gold_relations,
    pair_set,
    summarize,
    triple_set,
)


def _rel(s, t, r, d="d"):
    return {"source": s, "target": t, "relation": r, "description": d}


GOLD = [_rel("特斯拉", "Model 3", "产品研发方"), _rel("Model 3", "FSD 芯片", "核心硬件")]


def _record(responses):
    return {"key": "k1", "source": "src", "text": "t", "entities_str": "e",
            "responses": [{"seed": i, "response": r, "finish_reason": "stop",
                           "reasoning": ""} for i, r in enumerate(responses)]}


def test_normalisation_and_dedup():
    # Case/whitespace-insensitive matching; duplicated predictions collapse.
    pred = [_rel("  特斯拉 ", "model 3", "产品研发方"),
            _rel("特斯拉", "Model 3", "产品研发方")]
    assert triple_set(pred) == {("特斯拉", "model 3", "产品研发方")}
    assert pair_set(pred) == {("特斯拉", "model 3")}
    print("PASS test_normalisation_and_dedup")


def test_evaluate_record_counts():
    # Response 0: one exact triple + one pair-only match.  Response 1: unparseable.
    resp0 = json.dumps([_rel("特斯拉", "Model 3", "产品研发方"),
                        _rel("Model 3", "FSD 芯片", "别的关系")], ensure_ascii=False)
    line = evaluate_record(_record([resp0, "not json"]), GOLD)
    r0, r1 = line["per_response"]
    assert r0["strict"] == {"tp": 1, "pred_n": 2, "gold_n": 2,
                            "precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert r0["relaxed"]["tp"] == 2 and r0["relaxed"]["f1"] == 1.0
    assert r0["json_valid"] == 1
    assert r1["strict"] == {"tp": 0, "pred_n": 0, "gold_n": 2,
                            "precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert r1["json_valid"] == 0
    assert line["json_valid_rate"] == 0.5
    assert line["mean_strict_f1"] == 0.25
    print("PASS test_evaluate_record_counts")


def test_micro_pooling():
    resp_half = json.dumps([_rel("特斯拉", "Model 3", "产品研发方")], ensure_ascii=False)
    lines = [evaluate_record(_record([resp_half, "not json"]), GOLD)]
    summary = summarize(lines, "m", skipped_no_gold=0)
    # Pooled: tp=1, pred_n=1, gold_n=4 -> P=1.0, R=0.25, F1=0.4.
    assert summary["strict"] == {"precision": 1.0, "recall": 0.25, "f1": 0.4,
                                 "tp": 1, "pred_n": 1, "gold_n": 4}
    assert summary["num_responses"] == 2
    assert summary["json_valid_rate"] == 0.5
    assert summary["protocol_version"] == "v1.0"
    print("PASS test_micro_pooling")


def test_gold_as_json_string():
    gold = gold_relations(json.dumps(GOLD, ensure_ascii=False), key="k")
    assert len(gold) == 2 and gold[0]["source"] == "特斯拉"
    try:
        gold_relations("not json", key="k")
        raise AssertionError("expected ValueError for unparseable gold")
    except ValueError:
        pass
    try:
        gold_relations({"not": "a list"}, key="k")
        raise AssertionError("expected ValueError for non-list gold")
    except ValueError:
        pass
    print("PASS test_gold_as_json_string")


if __name__ == "__main__":
    test_normalisation_and_dedup()
    test_evaluate_record_counts()
    test_micro_pooling()
    test_gold_as_json_string()
    print("\nAll metrics tests passed.")
