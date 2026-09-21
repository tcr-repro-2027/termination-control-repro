# coding=utf-8
"""Model-level summary: rates, distributions, record-clustered bootstrap CIs.

Two aggregation levels are reported for every binary event, and they answer
different questions -- both are in §9.3, so both are columns:

* **rate**    -- share of the 8848 (record x seed) RESPONSES that show it.
                 This is the deployment-facing number and the one the F1
                 pooling is aligned with.
* **trigger** -- share of the 1106 RECORDS where at least one of the 8 seeds
                 shows it.  This is the "can this input go wrong at all"
                 number, and it saturates much earlier than the rate.

Confidence intervals resample RECORDS, not responses: the eight samples of one
prompt are strongly correlated, so a response-level bootstrap would report an
interval several times too narrow.  One index matrix is drawn and shared by
every metric, so the CIs are mutually consistent and a difference computed
between two columns of the same run uses the same resampling.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np

from . import protocol

__all__ = [
    "CSV_FIELDS", "bootstrap_rates", "distribution", "project_row",
    "summarize_events", "summary_to_row",
]


def project_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The ~1 KB slice of an event row that this module actually reads.

    A full event row carries one record per parsed block, so a task's 8848 rows
    can reach a gigabyte in memory.  `analyze.py` streams the full rows to disk
    and keeps only this projection for summarising, which is why several
    analysis workers can run beside eight busy GPUs without a memory scare.
    Every key below is read by :data:`EVENT_FLAGS`, :func:`summarize_events` or
    :func:`bootstrap_rates`; keep the three in sync.
    """
    def sub(name: str, *fields: str) -> Dict[str, Any]:
        value = row.get(name) or {}
        return {field: value.get(field) for field in fields}

    return {
        "stable_prompt_id": row.get("stable_prompt_id"),
        "seed": row.get("seed"),
        "gen_tokens": row.get("gen_tokens"),
        "n_blocks": row.get("n_blocks"),
        "top_level_list_complete": row.get("top_level_list_complete"),
        "has_incomplete_tail": row.get("has_incomplete_tail"),
        "alignment_type": row.get("alignment_type"),
        "stage_chain": row.get("stage_chain"),
        "gen_tokens_mismatch": row.get("gen_tokens_mismatch"),
        "severity": sub("severity", "rep4", "gzip_ratio", "gen_len",
                        "hit_max_tokens"),
        "stage_flags": sub("stage_flags", "seed_reuse", "capture",
                           "stable_orbit", "hit_max", "runaway_block_hitmax",
                           "runaway_any"),
        "first_triple_reuse": sub("first_triple_reuse", "exists",
                                  "block_index_1based"),
        "first_nonempty_triple_reuse": sub("first_nonempty_triple_reuse", "exists"),
        "first_exact_quad_reuse": sub("first_exact_quad_reuse", "exists"),
        "first_adjacent_motif_repeat_triple": sub(
            "first_adjacent_motif_repeat_triple", "exists"),
        "motif_capture_triple": sub("motif_capture_triple", "exists",
                                    "block_onset_1based", "block_period",
                                    "confirmed_at_token_exclusive"),
        "motif_capture_quad": sub("motif_capture_quad", "exists"),
        "legacy_orbit": sub("legacy_orbit", "exists", "raw_onset_token",
                            "raw_period_tokens", "observed_num_repeats",
                            "hit_max_tokens"),
        "temporal_alignment": sub("temporal_alignment",
                                  "complete_chain_available",
                                  "online_confirmation_order_valid"),
    }


# ------------------------------------------------------------- distributions

def distribution(values: Sequence[float], prefix: str) -> Dict[str, Any]:
    """mean / median / p25 / p75 / max of a value list, `None` when empty."""
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {f"{prefix}_n": 0, f"{prefix}_mean": None, f"{prefix}_median": None,
                f"{prefix}_p25": None, f"{prefix}_p75": None, f"{prefix}_max": None}
    ordered = sorted(numeric)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] * (high - position) + ordered[high] * (position - low)

    return {
        f"{prefix}_n": len(numeric),
        f"{prefix}_mean": mean(numeric),
        f"{prefix}_median": median(numeric),
        f"{prefix}_p25": percentile(0.25),
        f"{prefix}_p75": percentile(0.75),
        f"{prefix}_max": max(numeric),
    }


# ------------------------------------------------------------ event features

def _flag(row: Mapping[str, Any], *path: str) -> bool:
    value: Any = row
    for step in path:
        if not isinstance(value, Mapping):
            return False
        value = value.get(step)
    return bool(value)


#: Binary per-response events reported as both a rate and a trigger rate.
#: The order is the story the paper tells: a triple is reused, the reuse
#: becomes an adjacent repeat, the repeat is captured into a motif, the motif
#: becomes a stable orbit, the orbit eats the context window.
EVENT_FLAGS: Dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "first_triple_reuse": lambda row: _flag(row, "first_triple_reuse", "exists"),
    "first_nonempty_triple_reuse":
        lambda row: _flag(row, "first_nonempty_triple_reuse", "exists"),
    "first_quad_reuse": lambda row: _flag(row, "first_exact_quad_reuse", "exists"),
    "adjacent_motif_repeat":
        lambda row: _flag(row, "first_adjacent_motif_repeat_triple", "exists"),
    "semantic_capture": lambda row: _flag(row, "stage_flags", "capture"),
    "quad_capture": lambda row: _flag(row, "motif_capture_quad", "exists"),
    "stable_orbit": lambda row: _flag(row, "stage_flags", "stable_orbit"),
    "runaway_block_hitmax": lambda row: _flag(row, "stage_flags", "runaway_block_hitmax"),
    "runaway_any": lambda row: _flag(row, "stage_flags", "runaway_any"),
    "legacy_loop": lambda row: _flag(row, "legacy_orbit", "exists"),
    "hit_max": lambda row: _flag(row, "severity", "hit_max_tokens"),
    "json_list_complete": lambda row: bool(row.get("top_level_list_complete")),
    "incomplete_tail": lambda row: bool(row.get("has_incomplete_tail")),
}

#: Metrics that get a bootstrap CI.  Kept short on purpose: every extra one
#: costs a full resampling pass, and these are the six numbers E1's verdict
#: (§E1 判定) is actually read off.
CI_METRICS = ("semantic_capture", "stable_orbit", "legacy_loop", "hit_max",
              "first_triple_reuse", "runaway_any")


def summarize_events(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Rates, trigger rates and distributions over one model's event rows."""
    by_record: Dict[Any, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_record[row.get("stable_prompt_id")].append(row)

    n_responses = len(rows)
    n_records = len(by_record)
    summary: Dict[str, Any] = {
        "n_records": n_records,
        "n_responses": n_responses,
        "responses_per_record_min": min((len(items) for items in by_record.values()),
                                        default=0),
        "responses_per_record_max": max((len(items) for items in by_record.values()),
                                        default=0),
    }
    if not rows:
        return summary

    for name, predicate in EVENT_FLAGS.items():
        flags = [bool(predicate(row)) for row in rows]
        summary[f"{name}_rate"] = sum(flags) / n_responses
        summary[f"{name}_trigger_rate"] = sum(
            1 for items in by_record.values() if any(predicate(row) for row in items)
        ) / n_records

    # ---- conditional structure of the cascade
    captured = [row for row in rows if EVENT_FLAGS["semantic_capture"](row)]
    orbits = [row for row in rows if EVENT_FLAGS["stable_orbit"](row)]
    reused = [row for row in rows if EVENT_FLAGS["first_triple_reuse"](row)]
    summary["p_orbit_given_capture"] = (
        sum(1 for row in captured if EVENT_FLAGS["stable_orbit"](row)) / len(captured)
        if captured else None)
    summary["p_capture_given_reuse"] = (
        sum(1 for row in reused if EVENT_FLAGS["semantic_capture"](row)) / len(reused)
        if reused else None)
    summary["p_hitmax_given_orbit"] = (
        sum(1 for row in orbits if EVENT_FLAGS["hit_max"](row)) / len(orbits)
        if orbits else None)

    # ---- where the events happen (the onset figures)
    summary.update(distribution([row["gen_tokens"] for row in rows], "gen_tokens"))
    summary.update(distribution([row["n_blocks"] for row in rows], "n_blocks"))
    summary.update(distribution(
        [row["severity"]["rep4"] for row in rows if "severity" in row], "rep4"))
    summary.update(distribution(
        [row["severity"]["gzip_ratio"] for row in rows if "severity" in row],
        "gzip_ratio"))
    summary.update(distribution(
        [row["first_triple_reuse"]["block_index_1based"] for row in reused],
        "first_reuse_block"))
    summary.update(distribution(
        [row["motif_capture_triple"]["block_onset_1based"] for row in captured],
        "capture_block_onset"))
    summary.update(distribution(
        [row["motif_capture_triple"]["block_period"] for row in captured],
        "capture_block_period"))
    summary.update(distribution(
        [row["motif_capture_triple"]["confirmed_at_token_exclusive"]
         for row in captured], "capture_token"))
    summary.update(distribution(
        [row["legacy_orbit"]["raw_onset_token"] for row in orbits],
        "legacy_onset_token"))
    summary.update(distribution(
        [row["legacy_orbit"]["raw_period_tokens"] for row in orbits],
        "legacy_period_tokens"))

    # ---- how well the raw token orbit maps onto complete blocks
    alignment: Dict[str, int] = defaultdict(int)
    for row in orbits:
        alignment[str(row.get("alignment_type"))] += 1
    summary["alignment_counts"] = dict(alignment)
    structured_aligned = sum(
        count for name, count in alignment.items()
        if name in ("single_block_aligned", "multi_block_aligned",
                    "phase_rotated_structured"))
    summary["legacy_structured_alignment_rate"] = (
        structured_aligned / len(orbits) if orbits else None)

    ordered = [row["temporal_alignment"]["online_confirmation_order_valid"]
               for row in rows
               if row.get("temporal_alignment", {}).get("complete_chain_available")]
    summary["temporal_order_valid_rate"] = (
        sum(1 for value in ordered if value) / len(ordered) if ordered else None)
    summary["temporal_chain_available_n"] = len(ordered)

    chains: Dict[str, int] = defaultdict(int)
    for row in rows:
        chains[str(row.get("stage_chain"))] += 1
    summary["stage_chain_counts"] = dict(chains)

    mismatches = [row for row in rows if row.get("gen_tokens_mismatch")]
    summary["gen_tokens_mismatch_rate"] = len(mismatches) / n_responses
    return summary


# -------------------------------------------------------------- bootstrap CI

def bootstrap_rates(rows: Sequence[Mapping[str, Any]],
                    quality_by_sample: Sequence[Mapping[str, Any]] | None = None,
                    *, n_boot: int = protocol.BOOTSTRAP,
                    seed: int = protocol.BOOTSTRAP_SEED) -> Dict[str, Any]:
    """Record-clustered 95% CIs for the headline rates and strict micro-F1.

    ``rows`` and ``quality_by_sample`` must be response-aligned (same order):
    they are two views of the same 8848 units, and the CI for F1 has to be
    drawn from the same record resampling as the CI for the capture rate.
    """
    if not rows:
        return {}
    record_ids = sorted({row.get("stable_prompt_id") for row in rows},
                        key=repr)
    index_of = {value: index for index, value in enumerate(record_ids)}
    groups = np.array([index_of[row.get("stable_prompt_id")] for row in rows],
                      dtype=np.int64)
    n_records = len(record_ids)

    counts = np.bincount(groups, minlength=n_records).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_records, size=(n_boot, n_records))

    result: Dict[str, Any] = {"bootstrap": n_boot, "bootstrap_seed": seed,
                              "bootstrap_unit": "record"}

    def interval(per_record_numerator: np.ndarray,
                 per_record_denominator: np.ndarray) -> tuple[float, float]:
        numerator = per_record_numerator[draws].sum(axis=1)
        denominator = per_record_denominator[draws].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(denominator > 0, numerator / denominator, np.nan)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (float("nan"), float("nan"))
        return (float(np.percentile(finite, 2.5)),
                float(np.percentile(finite, 97.5)))

    for name in CI_METRICS:
        predicate = EVENT_FLAGS[name]
        flags = np.array([1.0 if predicate(row) else 0.0 for row in rows])
        per_record = np.bincount(groups, weights=flags, minlength=n_records)
        low, high = interval(per_record, counts)
        result[f"{name}_rate_ci_low"] = low
        result[f"{name}_rate_ci_high"] = high

        triggered = (per_record > 0).astype(np.float64)
        ones = np.ones(n_records)
        low, high = interval(triggered, ones)
        result[f"{name}_trigger_rate_ci_low"] = low
        result[f"{name}_trigger_rate_ci_high"] = high

    if quality_by_sample is not None:
        if len(quality_by_sample) != len(rows):
            raise ValueError("quality samples and event rows must be aligned")
        tp = np.bincount(groups, weights=np.array(
            [float(item["strict"]["tp"]) for item in quality_by_sample]),
            minlength=n_records)
        pred = np.bincount(groups, weights=np.array(
            [float(item["strict"]["pred_n"]) for item in quality_by_sample]),
            minlength=n_records)
        gold = np.bincount(groups, weights=np.array(
            [float(item["strict"]["gold_n"]) for item in quality_by_sample]),
            minlength=n_records)
        tp_b = tp[draws].sum(axis=1)
        pred_b = pred[draws].sum(axis=1)
        gold_b = gold[draws].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.where(pred_b > 0, tp_b / pred_b, 0.0)
            recall = np.where(gold_b > 0, tp_b / gold_b, 0.0)
            f1 = np.where(precision + recall > 0,
                          2 * precision * recall / (precision + recall), 0.0)
        result["strict_f1_ci_low"] = float(np.percentile(f1, 2.5))
        result["strict_f1_ci_high"] = float(np.percentile(f1, 97.5))
    return result


# ------------------------------------------------------------------ csv view

#: The unified CSV.  Grouped, and ordered so a reader scanning left to right
#: goes identity -> primary structural repetition -> legacy repetition ->
#: utility -> support compliance -> episode dynamics -> provenance.
CSV_FIELDS: List[str] = [
    # ---- identity
    "tag", "run_name", "kind", "size", "data_variant", "arm_family", "seed",
    "ckpt_step", "is_final", "save_mode", "terminal_mode", "priority", "tier",
    "mode", "protocol_version",
    # ---- denominators
    "n_records", "n_responses", "n_skipped_too_long",
    # ---- primary: structured four-field block repetition (§9.3)
    "semantic_capture_rate", "semantic_capture_rate_ci_low",
    "semantic_capture_rate_ci_high", "semantic_capture_trigger_rate",
    "stable_orbit_rate", "stable_orbit_rate_ci_low", "stable_orbit_rate_ci_high",
    "stable_orbit_trigger_rate",
    "runaway_any_rate", "runaway_any_rate_ci_low", "runaway_any_rate_ci_high",
    "runaway_block_hitmax_rate",
    "first_triple_reuse_rate", "first_triple_reuse_rate_ci_low",
    "first_triple_reuse_rate_ci_high", "first_triple_reuse_trigger_rate",
    "first_quad_reuse_rate", "adjacent_motif_repeat_rate", "quad_capture_rate",
    "p_capture_given_reuse", "p_orbit_given_capture", "p_hitmax_given_orbit",
    "capture_block_onset_median", "capture_block_onset_mean",
    "capture_block_period_median", "capture_token_median",
    "first_reuse_block_median", "n_blocks_mean", "n_blocks_median",
    # ---- legacy token-period detector (continuity with loop_results.csv)
    "legacy_loop_rate", "legacy_loop_rate_ci_low", "legacy_loop_rate_ci_high",
    "legacy_loop_trigger_rate", "hit_max_rate", "hit_max_rate_ci_low",
    "hit_max_rate_ci_high", "rep4_mean", "gzip_ratio_mean",
    "legacy_onset_token_median", "legacy_period_tokens_median",
    "legacy_structured_alignment_rate", "temporal_order_valid_rate",
    "gen_tokens_mean", "gen_tokens_median", "gen_tokens_p75", "gen_tokens_max",
    # ---- task utility (§9.1)
    "strict_precision", "strict_recall", "strict_f1", "strict_f1_ci_low",
    "strict_f1_ci_high", "relaxed_precision", "relaxed_recall", "relaxed_f1",
    "json_valid_rate", "json_list_complete_rate", "incomplete_tail_rate",
    "mean_response_strict_f1", "mean_pred_blocks", "mean_unique_pred_triples",
    "unique_supported_per_1k_tokens",
    # ---- support compliance (§9.2)
    "out_of_candidate_block_rate", "out_of_candidate_block_rate_exact",
    "out_of_text_endpoint_rate", "dual_conflict_rate",
    "support_valid_block_precision", "unsupported_nonempty_rate",
    "empty_prediction_rate", "empty_gold_accuracy",
    # ---- episode dynamics (§9.3, P0d marginal)
    "episodes_per_response", "reuse_blocks_per_response",
    "per_episode_capture_hazard", "gap_stop_hazard", "in_episode_stop_hazard",
    "responses_with_any_episode_rate", "captured_episode_ordinal_mean",
    "distance_last_new_triple_blocks_mean",
    "capture_is_first_episode_after_last_new_rate", "episode_errors",
    # ---- provenance
    "model_path", "tokenizer_path", "tokenizer_is_fallback",
    "eval_data_sha256", "prompt_rendered_sha256", "chat_prefix_ok",
    "gen_tokens_mismatch_rate", "generation_minutes", "analysis_minutes",
    "vllm_version", "analyzed_at", "responses_path", "event_rows_path",
]


def summary_to_row(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten a per-task summary json into one `e1_metrics.csv` row."""
    task = summary.get("task", {})
    events = summary.get("events", {})
    quality = summary.get("quality", {})
    episodes = summary.get("episodes", {})
    boot = summary.get("bootstrap_ci", {})
    generation = summary.get("generation", {})

    row: Dict[str, Any] = {}
    row.update({
        "tag": task.get("tag"), "run_name": task.get("run_name"),
        "kind": task.get("kind"), "size": task.get("size"),
        "data_variant": task.get("data_variant"),
        "arm_family": task.get("arm_family"), "seed": task.get("seed"),
        "ckpt_step": task.get("ckpt_step"), "is_final": task.get("is_final"),
        "save_mode": task.get("save_mode"),
        "terminal_mode": task.get("terminal_mode"),
        "priority": task.get("priority"), "tier": task.get("tier"),
        "mode": summary.get("mode"),
        "protocol_version": summary.get("protocol_version"),
        "n_skipped_too_long": generation.get("n_skipped_too_long"),
        "model_path": task.get("model_path"),
        "tokenizer_path": task.get("tokenizer_path"),
        "tokenizer_is_fallback": task.get("tokenizer_is_fallback"),
        "eval_data_sha256": generation.get("eval_data_sha256"),
        "prompt_rendered_sha256": generation.get("prompt_rendered_sha256"),
        "chat_prefix_ok": generation.get("chat_prefix_ok"),
        "generation_minutes": generation.get("generation_minutes"),
        "analysis_minutes": summary.get("analysis_minutes"),
        "vllm_version": generation.get("vllm_version"),
        "analyzed_at": summary.get("analyzed_at"),
        "responses_path": summary.get("responses_path"),
        "event_rows_path": summary.get("event_rows_path"),
        "episode_errors": episodes.get("n_errors"),
    })
    for source in (events, quality, episodes, boot):
        for name, value in source.items():
            if name in CSV_FIELDS and name not in row:
                row[name] = value
    # `n_records` / `n_responses` live in both events and quality; events wins
    # because it is the denominator every repetition rate is divided by.
    row["n_records"] = events.get("n_records")
    row["n_responses"] = events.get("n_responses")
    return row
