# coding=utf-8
"""Task utility (§9.1) and support compliance (§9.2) for one response.

Utility reuses the FROZEN v1.0 evaluator
(`tcr/extraction/evaluation`) verbatim -- same
lenient JSON extraction, same strict/relaxed normalisation, same pooled micro
counts -- so an E1 F1 sits on the same scale as `loop_results.csv`.  Only the
aggregation lives here.

Support compliance is new in E1 and is the reason this module exists at all.
The repetition story of this project is a *support* story: a block whose
endpoints are not in the candidate list, or not in the document, is a block the
model produced without licence, and its rate is what separates "the model
degraded" from "the model kept emitting unlicensed material until it ran out of
context".  Two axes, reported separately (§9.2):

* **admissibility** -- both endpoints appear in the candidate entity list;
* **evidence**      -- both endpoints appear in the source document.

Normalisation is the SAME `_norm` the F1 metrics use (strip, lowercase,
collapse whitespace).  Using one normalisation across every E1 number is the
point of E1; the stricter reading is reported alongside as `*_exact` rather
than instead.  Note that the frozen v1.0 parser already strips leading and
trailing whitespace off every field, so `*_exact` differs from the primary
reading by case folding and internal whitespace only.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Iterable, List, Set, Tuple

from tcr.extraction.evaluation.metrics import (           # noqa: E402
    _norm as norm_field, gold_relations, match_counts, pair_set, prf, triple_set,
)
from tcr.extraction.evaluation.parsing import parse_response  # noqa: E402

__all__ = [
    "GoldRecord", "candidate_entities", "empty_prediction", "gold_record",
    "score_sample", "norm_field", "pooled",
]

_QUOTED = re.compile(r"""['"]([^'"]*)['"]""")


class GoldRecord:
    """Everything the scorer needs about one eval record, computed once."""

    __slots__ = ("key", "source", "gold", "gold_triples", "gold_pairs",
                 "n_gold_blocks", "candidates", "candidates_exact",
                 "norm_text", "raw_text")

    def __init__(self, record: Dict[str, Any]) -> None:
        self.key = record.get("key")
        self.source = record.get("source", "")
        # Not `record.get("output", [])`: a record with no gold would silently
        # score gold_n = 0, which makes every prediction a false positive and
        # every recall denominator smaller -- wrong numbers that look valid.
        if "output" not in record:
            raise ValueError(f"eval record key={self.key!r} has no 'output' "
                             "field; it cannot be scored against")
        self.gold = gold_relations(record["output"], self.key)
        self.gold_triples = triple_set(self.gold)
        self.gold_pairs = pair_set(self.gold)
        self.n_gold_blocks = len(self.gold)
        entities = candidate_entities(record.get("entities_str", ""))
        self.candidates = {norm_field(name) for name in entities}
        self.candidates_exact = {name.strip() for name in entities}
        self.raw_text = record.get("text", "") or ""
        self.norm_text = norm_field(self.raw_text)


def gold_record(record: Dict[str, Any]) -> GoldRecord:
    return GoldRecord(record)


def candidate_entities(entities_str: Any) -> List[str]:
    """The candidate list as the prompt shows it.

    `entities_str` is a Python-literal list rendered into the prompt, e.g.
    ``"['TR5', 'TR6', ...]"``.  `literal_eval` handles it; the quoted-string
    fallback keeps a malformed row scorable instead of aborting a whole task
    over one record.
    """
    if isinstance(entities_str, (list, tuple)):
        return [str(value) for value in entities_str]
    text = str(entities_str or "").strip()
    if not text:
        return []
    for loader in (ast.literal_eval, json.loads):
        try:
            value = loader(text)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
    return [match.group(1) for match in _QUOTED.finditer(text)]


def empty_prediction(response: str) -> bool:
    """True iff the response is exactly the empty relation list `[]`."""
    from tcr.extraction.evaluation.parsing import extract_json_value

    value = extract_json_value(response)
    return isinstance(value, list) and not value


def score_sample(response: str, gold: GoldRecord) -> Dict[str, Any]:
    """Utility + support counts for ONE generated response.

    Counts, not rates: rates are pooled over (record x sample) units by
    :func:`pooled`, so a looping response contributes its full gold count to
    recall and its hundreds of unlicensed blocks to the support denominators,
    exactly as a response-level LoopRate would.
    """
    parsed = parse_response(response)
    relations = parsed.relations

    pred_triples = triple_set(relations)
    pred_pairs = pair_set(relations)
    strict = match_counts(gold.gold_triples, pred_triples)
    relaxed = match_counts(gold.gold_pairs, pred_pairs)

    in_candidate = in_text = supported = dual_conflict = 0
    in_candidate_exact = 0
    supported_triples: Set[Tuple[str, str, str]] = set()
    for relation in relations:
        source = relation.get("source", "")
        target = relation.get("target", "")
        norm_source, norm_target = norm_field(source), norm_field(target)
        admissible = (norm_source in gold.candidates
                      and norm_target in gold.candidates)
        evidenced = (bool(norm_source) and bool(norm_target)
                     and norm_source in gold.norm_text
                     and norm_target in gold.norm_text)
        in_candidate += int(admissible)
        in_text += int(evidenced)
        in_candidate_exact += int(source.strip() in gold.candidates_exact
                                  and target.strip() in gold.candidates_exact)
        if admissible and evidenced:
            supported += 1
            supported_triples.add(
                (norm_source, norm_target, norm_field(relation.get("relation", ""))))
        elif not admissible and not evidenced:
            dual_conflict += 1

    n_blocks = len(relations)
    return {
        "json_valid": int(parsed.json_valid),
        "empty_list": int(empty_prediction(response)),
        "strict": strict,
        "relaxed": relaxed,
        "strict_f1": prf(**strict)["f1"],
        "relaxed_f1": prf(**relaxed)["f1"],
        "n_pred_blocks": n_blocks,
        "n_unique_pred_triples": len(pred_triples),
        "n_in_candidate": in_candidate,
        "n_in_candidate_exact": in_candidate_exact,
        "n_in_text": in_text,
        "n_supported": supported,
        "n_dual_conflict": dual_conflict,
        "n_unique_supported": len(supported_triples),
        "gold_n_blocks": gold.n_gold_blocks,
    }


def pooled(samples: Iterable[Dict[str, Any]], *,
           total_gen_tokens: int = 0) -> Dict[str, Any]:
    """Pool per-response counts into the §9.1/§9.2 model-level metrics.

    Every response is one unit with equal weight (the response-level
    counterpart of LoopRate).  Rates whose denominator is empty come back as
    ``None`` rather than 0.0 -- a model that produced no block at all has an
    undefined out-of-candidate rate, not a perfect one.
    """
    samples = list(samples)
    n = len(samples)
    if n == 0:
        return {"n_responses": 0}

    def total(field: str) -> int:
        return sum(int(sample[field]) for sample in samples)

    strict_tp = sum(sample["strict"]["tp"] for sample in samples)
    strict_pred = sum(sample["strict"]["pred_n"] for sample in samples)
    strict_gold = sum(sample["strict"]["gold_n"] for sample in samples)
    relaxed_tp = sum(sample["relaxed"]["tp"] for sample in samples)
    relaxed_pred = sum(sample["relaxed"]["pred_n"] for sample in samples)
    relaxed_gold = sum(sample["relaxed"]["gold_n"] for sample in samples)

    blocks = total("n_pred_blocks")
    supported = total("n_supported")
    empty_gold = [sample for sample in samples if sample["gold_n_blocks"] == 0]

    def rate(numerator: int, denominator: int) -> float | None:
        return (numerator / denominator) if denominator else None

    strict = prf(strict_tp, strict_pred, strict_gold)
    relaxed = prf(relaxed_tp, relaxed_pred, relaxed_gold)
    return {
        "n_responses": n,
        # ---- §9.1 task utility
        "strict_precision": strict["precision"],
        "strict_recall": strict["recall"],
        "strict_f1": strict["f1"],
        "relaxed_precision": relaxed["precision"],
        "relaxed_recall": relaxed["recall"],
        "relaxed_f1": relaxed["f1"],
        "strict_tp": strict_tp, "strict_pred_n": strict_pred,
        "strict_gold_n": strict_gold,
        "json_valid_rate": total("json_valid") / n,
        "mean_response_strict_f1": sum(s["strict_f1"] for s in samples) / n,
        "mean_response_relaxed_f1": sum(s["relaxed_f1"] for s in samples) / n,
        "mean_pred_blocks": blocks / n,
        "mean_unique_pred_triples": total("n_unique_pred_triples") / n,
        "unique_supported_per_1k_tokens": rate(1000 * total("n_unique_supported"),
                                               total_gen_tokens),
        # ---- §9.2 support compliance
        "n_pred_blocks_total": blocks,
        "out_of_candidate_block_rate": rate(blocks - total("n_in_candidate"), blocks),
        "out_of_candidate_block_rate_exact": rate(
            blocks - total("n_in_candidate_exact"), blocks),
        "out_of_text_endpoint_rate": rate(blocks - total("n_in_text"), blocks),
        "dual_conflict_rate": rate(total("n_dual_conflict"), blocks),
        "support_valid_block_precision": rate(supported, blocks),
        "unsupported_nonempty_rate": sum(
            1 for sample in samples
            if sample["n_pred_blocks"] > 0 and sample["n_supported"] == 0) / n,
        "empty_prediction_rate": total("empty_list") / n,
        "empty_gold_accuracy": rate(
            sum(sample["empty_list"] for sample in empty_gold), len(empty_gold)),
        "n_empty_gold_responses": len(empty_gold),
    }
