# coding=utf-8
"""The frozen relation-extraction metrics (protocol v1.0).

Exactly three metrics, no more:

* **strict**  -- micro-F1 over normalised ``(source, target, relation)``
  triples ("蒸馏一致性 F1", the paper's headline quality metric);
* **relaxed** -- micro-F1 over normalised ``(source, target)`` entity pairs;
* **json_valid_rate** -- share of responses that parse to a ``list[dict]``
  with all four fields present.

Aggregation over the K=8 samples: every response is one evaluation unit; its
prediction set is DEDUPLICATED after normalisation and matched against the
gold set, and tp / pred_n / gold_n are pooled over ALL (record x sample) units
before computing P/R/F1 (micro).  This weights every generated answer equally
-- a looping/truncated answer contributes an empty prediction set and its full
gold count, so quality degradation from repetition is reflected directly --
and it is the response-level counterpart of LoopRate.  Per-record macro means
are kept in the detail lines for diagnostics (e.g. correlating with p_i).

Gold is the eval set's ``output`` field (a relation list, or a JSON string
encoding one).
"""

import json
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from tcr.extraction import protocol
from tcr.extraction.evaluation.parsing import parse_response

Triple = Tuple[str, str, str]
Pair = Tuple[str, str]


def _norm(value: Any) -> str:
    """Normalise a field for matching: strip, lowercase, collapse whitespace."""
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def gold_relations(output: Any, key: Any) -> List[Dict]:
    """Coerce a record's ``output`` field into a list of relation dicts."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError as exc:
            raise ValueError(f"record key={key!r}: gold 'output' is an "
                             f"unparseable JSON string") from exc
    if not isinstance(output, list):
        raise ValueError(f"record key={key!r}: gold 'output' must be a list, "
                         f"got {type(output).__name__}")
    return [item for item in output if isinstance(item, dict)]


def triple_set(relations: Sequence[Dict]) -> Set[Triple]:
    """Deduplicated normalised (source, target, relation) triples."""
    return {(_norm(r.get("source")), _norm(r.get("target")), _norm(r.get("relation")))
            for r in relations}


def pair_set(relations: Sequence[Dict]) -> Set[Pair]:
    """Deduplicated normalised (source, target) pairs."""
    return {(_norm(r.get("source")), _norm(r.get("target"))) for r in relations}


def match_counts(gold: Set, pred: Set) -> Dict[str, int]:
    return {"tp": len(gold & pred), "pred_n": len(pred), "gold_n": len(gold)}


def prf(tp: int, pred_n: int, gold_n: int) -> Dict[str, float]:
    p = tp / pred_n if pred_n else 0.0
    r = tp / gold_n if gold_n else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def _pooled(counts: Iterable[Dict[str, int]]) -> Dict[str, Any]:
    """Micro P/R/F1 from pooled tp / pred_n / gold_n."""
    tp = pred_n = gold_n = 0
    for c in counts:
        tp += c["tp"]
        pred_n += c["pred_n"]
        gold_n += c["gold_n"]
    return {**prf(tp, pred_n, gold_n), "tp": tp, "pred_n": pred_n, "gold_n": gold_n}


def evaluate_record(record: Dict, output: Any) -> Dict:
    """Score one response record against its gold ``output``; returns the
    detail line.  ``output`` comes from the eval set, joined by ``key`` (the
    responses file does not carry gold)."""
    gold = gold_relations(output, record.get("key"))
    gold_triples, gold_pairs = triple_set(gold), pair_set(gold)

    per_response: List[Dict] = []
    for sample in record["responses"]:
        parsed = parse_response(sample.get("response", ""))
        strict = match_counts(gold_triples, triple_set(parsed.relations))
        relaxed = match_counts(gold_pairs, pair_set(parsed.relations))
        per_response.append({
            "seed": sample.get("seed"),
            "json_valid": int(parsed.json_valid),
            "strict": {**strict, **prf(**strict)},
            "relaxed": {**relaxed, **prf(**relaxed)},
        })

    n = len(per_response)
    return {
        "key": record.get("key"),
        "source": record.get("source"),
        "gold_n": len(gold_triples),
        "n": n,
        "mean_strict_f1": round(sum(r["strict"]["f1"] for r in per_response) / n, 4) if n else 0.0,
        "mean_relaxed_f1": round(sum(r["relaxed"]["f1"] for r in per_response) / n, 4) if n else 0.0,
        "json_valid_rate": round(sum(r["json_valid"] for r in per_response) / n, 4) if n else 0.0,
        "per_response": per_response,
    }


def summarize(item_lines: List[Dict], model_name: str, skipped_no_gold: int) -> Dict:
    """Pool every (record x sample) unit into the protocol summary line."""
    responses = [r for line in item_lines for r in line["per_response"]]
    num_responses = len(responses)
    return {
        "summary": True,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "model_name": model_name,
        "num_records": len(item_lines),
        "num_responses": num_responses,
        "skipped_no_gold": skipped_no_gold,
        "strict": _pooled(r["strict"] for r in responses),
        "relaxed": _pooled(r["relaxed"] for r in responses),
        "json_valid_rate": round(
            sum(r["json_valid"] for r in responses) / num_responses, 4
        ) if num_responses else 0.0,
    }
