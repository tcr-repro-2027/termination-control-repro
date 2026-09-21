"""P0b event-exposure, competing-risk, and first-seed lineage analysis.

This module consumes frozen ``event_rows.jsonl`` records produced by the
01b relation-event detector.  It does not reparse responses, move the legacy
raw onset, load an LLM, or change any detector definition.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .io_utils import canonical_key, iter_jsonl

MODEL_TAGS = ("M0", "M1")
MODEL_INDEX = {name: index for index, name in enumerate(MODEL_TAGS)}


@dataclass(frozen=True)
class TerminalRecord:
    prompt_id: str
    model_tag: str
    seed: int
    gold_blocks: int
    n_blocks: int
    hit_max: bool
    event_kind: str  # seed | normal_stop | admin_censor
    event_block: int
    event_progress: float
    output_gold_ratio: float
    seed_block: int | None
    seed_progress: float | None


@dataclass(frozen=True)
class LineageRecord:
    sample_id: str
    prompt_id: str
    model_tag: str
    seed: int
    gold_blocks: int
    n_blocks: int
    hit_max: bool
    eligible: bool
    ineligible_reason: str
    previous_block_0based: int | None
    seed_block_0based: int | None
    motif_period_blocks: int | None
    seed_progress: float | None
    motif_all_nonempty: bool | None
    same_sequence_segment: bool | None
    second_status: str
    second_first_mismatch_offset_1based: int | None
    second_confirmation_block_1based: int | None
    third_status: str
    third_first_mismatch_offset_1based: int | None
    third_confirmation_block_1based: int | None
    lineage_capture: bool
    generic_semantic_capture: bool
    stable_orbit: bool


@dataclass(frozen=True)
class GoldMapping:
    key_field: str
    n_rows: int
    values: dict[str, int]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _gold_count(row: Mapping[str, Any]) -> int:
    value = row.get("n_output_dicts")
    if value is not None:
        count = int(value)
        if count > 0:
            return count
    output = row.get("output")
    if isinstance(output, list) and output:
        return len(output)
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list) and parsed:
            return len(parsed)
    raise ValueError("gold row lacks a positive n_output_dicts and parseable non-empty output")


def load_gold_mapping(
    gold_path: str | Path,
    event_keys: Sequence[Any],
    *,
    key_field: str = "auto",
) -> GoldMapping:
    """Map event keys to gold cardinalities with strict full-coverage checks."""
    rows = _read_jsonl(gold_path)
    if not rows:
        raise ValueError(f"gold dataset is empty: {gold_path}")
    event_key_ids = {canonical_key(value) for value in event_keys}

    candidates: list[tuple[str, dict[str, int]]] = []
    field_order = ("key", "index", "id")
    for field in field_order:
        mapping: dict[str, int] = {}
        valid = True
        for row in rows:
            if field not in row:
                valid = False
                break
            key_id = canonical_key(row[field])
            count = _gold_count(row)
            if key_id in mapping and mapping[key_id] != count:
                raise ValueError(f"duplicate gold {field}={row[field]!r} has conflicting counts")
            mapping[key_id] = count
        if valid:
            candidates.append((field, mapping))

    line0 = {canonical_key(index): _gold_count(row) for index, row in enumerate(rows)}
    line1 = {canonical_key(index + 1): _gold_count(row) for index, row in enumerate(rows)}
    candidates.extend((("line_0based", line0), ("line_1based", line1)))

    if key_field != "auto":
        matches = [value for value in candidates if value[0] == key_field]
        if not matches:
            raise ValueError(
                f"unsupported or unavailable gold key field {key_field!r}; "
                f"available={[name for name, _ in candidates]}"
            )
        chosen_name, chosen = matches[0]
        missing = sorted(event_key_ids - set(chosen))
        if missing:
            raise ValueError(
                f"gold key field {chosen_name!r} misses {len(missing)} event keys; "
                f"first={missing[:5]}"
            )
        return GoldMapping(chosen_name, len(rows), {key: chosen[key] for key in event_key_ids})

    full = [(name, mapping) for name, mapping in candidates if event_key_ids <= set(mapping)]
    if not full:
        coverage = sorted(
            ((name, len(event_key_ids & set(mapping))) for name, mapping in candidates),
            key=lambda item: (-item[1], item[0]),
        )
        raise ValueError(
            "no automatic gold-key mapping covers all event keys; "
            f"coverage={coverage}; pass --gold-key-field explicitly after auditing"
        )

    # Prefer semantic identifier fields over line-number fallbacks.
    priority = {"key": 0, "index": 1, "id": 2, "line_0based": 3, "line_1based": 4}
    full.sort(key=lambda item: priority[item[0]])
    chosen_name, chosen = full[0]

    # If several candidates fully cover the keys but disagree on cardinality,
    # automatic selection is unsafe and must stop.
    chosen_values = {key: chosen[key] for key in event_key_ids}
    for other_name, other in full[1:]:
        other_values = {key: other[key] for key in event_key_ids}
        if other_values != chosen_values and priority[other_name] <= 2:
            raise ValueError(
                f"ambiguous automatic gold mapping: {chosen_name!r} and {other_name!r} "
                "both cover all keys but assign different n_output_dicts"
            )
    return GoldMapping(chosen_name, len(rows), chosen_values)


def is_semantic_capture(row: Mapping[str, Any]) -> bool:
    event = row.get("motif_capture_triple", {})
    if not isinstance(event, Mapping) or not event.get("exists"):
        return False
    triples = event.get("motif_triples", [])
    return bool(triples) and all(
        isinstance(triple, list)
        and len(triple) == 3
        and all(isinstance(value, str) and value.strip() for value in triple)
        for triple in triples
    )


def validate_prevalence_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    prevalence = [row for row in rows if "prevalence" in row.get("selection_roles", [])]
    if not prevalence:
        raise ValueError("event_rows.jsonl has no prevalence rows")
    sample_ids = [str(row.get("sample_id")) for row in prevalence]
    duplicates = [sample_id for sample_id, count in Counter(sample_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate prevalence sample_id values: {duplicates[:5]}")

    grouped: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in prevalence:
        model = str(row.get("model_tag"))
        if model not in MODEL_TAGS:
            raise ValueError(f"unexpected model_tag={model!r}")
        prompt_id = str(row.get("stable_prompt_id"))
        grouped[prompt_id][model].add(int(row.get("seed")))
    prompt_ids = sorted(grouped)
    for prompt_id in prompt_ids:
        models = grouped[prompt_id]
        if set(models) != set(MODEL_TAGS):
            raise ValueError(f"prompt {prompt_id} lacks an M0/M1 pair")
        if models["M0"] != models["M1"]:
            raise ValueError(
                f"prompt {prompt_id} has M0/M1 seed mismatch: "
                f"{sorted(models['M0'])} vs {sorted(models['M1'])}"
            )
    return prevalence, prompt_ids


def build_terminal_records(
    rows: Sequence[dict[str, Any]], gold: GoldMapping
) -> list[TerminalRecord]:
    records: list[TerminalRecord] = []
    for row in rows:
        key_id = canonical_key(row.get("key"))
        if key_id not in gold.values:
            raise KeyError(f"event key is missing from gold mapping: {row.get('key')!r}")
        gold_blocks = int(gold.values[key_id])
        n_blocks = int(row.get("n_blocks", 0))
        if n_blocks < 0:
            raise ValueError("n_blocks cannot be negative")
        seed_event = row.get("first_nonempty_triple_reuse", {})
        hit_max = bool(row.get("legacy_orbit", {}).get("hit_max_tokens"))
        if seed_event.get("exists"):
            seed_block = int(seed_event["block_index_1based"])
            if not (1 <= seed_block <= n_blocks):
                raise ValueError(
                    f"invalid seed block {seed_block} for n_blocks={n_blocks}, sample={row.get('sample_id')}"
                )
            event_kind = "seed"
            # Seed is confirmed when the repeated complete block has been emitted.
            event_block = seed_block
            event_progress = seed_block / gold_blocks
            seed_progress: float | None = event_progress
        else:
            seed_block = None
            event_kind = "admin_censor" if hit_max else "normal_stop"
            # The competing stop/censor occurs at the next relation-block opportunity.
            event_block = n_blocks + 1
            event_progress = event_block / gold_blocks
            seed_progress = None
        records.append(
            TerminalRecord(
                prompt_id=str(row["stable_prompt_id"]),
                model_tag=str(row["model_tag"]),
                seed=int(row["seed"]),
                gold_blocks=gold_blocks,
                n_blocks=n_blocks,
                hit_max=hit_max,
                event_kind=event_kind,
                event_block=event_block,
                event_progress=event_progress,
                output_gold_ratio=n_blocks / gold_blocks,
                seed_block=seed_block,
                seed_progress=seed_progress,
            )
        )
    return records


def _copy_status(
    blocks: Sequence[Mapping[str, Any]],
    *,
    expected_hashes: Sequence[str],
    start: int,
    expected_segment: int,
    hit_max: bool,
) -> tuple[str, int | None, int | None]:
    """Return status, first mismatch offset (1-based), confirmation block (1-based)."""
    for offset, expected in enumerate(expected_hashes):
        index = start + offset
        if index >= len(blocks):
            status = "admin_censor" if hit_max else "normal_stop"
            return status, None, None
        block = blocks[index]
        if int(block.get("sequence_segment", -1)) != expected_segment:
            return "continuity_break", offset + 1, None
        if str(block.get("triple_hash")) != str(expected):
            return "mismatch", offset + 1, None
    return "completed", None, start + len(expected_hashes)


def trace_first_seed_lineage(row: Mapping[str, Any], gold_blocks: int) -> LineageRecord:
    event = row.get("first_nonempty_triple_reuse", {})
    sample_id = str(row.get("sample_id"))
    common = dict(
        sample_id=sample_id,
        prompt_id=str(row.get("stable_prompt_id")),
        model_tag=str(row.get("model_tag")),
        seed=int(row.get("seed")),
        gold_blocks=gold_blocks,
        n_blocks=int(row.get("n_blocks", 0)),
        hit_max=bool(row.get("legacy_orbit", {}).get("hit_max_tokens")),
    )
    if not event.get("exists"):
        return LineageRecord(
            **common,
            eligible=False,
            ineligible_reason="no_nonempty_seed",
            previous_block_0based=None,
            seed_block_0based=None,
            motif_period_blocks=None,
            seed_progress=None,
            motif_all_nonempty=None,
            same_sequence_segment=None,
            second_status="not_at_risk",
            second_first_mismatch_offset_1based=None,
            second_confirmation_block_1based=None,
            third_status="not_at_risk",
            third_first_mismatch_offset_1based=None,
            third_confirmation_block_1based=None,
            lineage_capture=False,
            generic_semantic_capture=is_semantic_capture(row),
            stable_orbit=bool(row.get("legacy_orbit", {}).get("exists")),
        )

    blocks = row.get("block_index", [])
    if not isinstance(blocks, list):
        raise TypeError(f"block_index must be a list: {sample_id}")
    previous = int(event["previous_block_index_0based"])
    current = int(event["block_index_0based"])
    period = current - previous
    seed_progress = (current + 1) / gold_blocks
    if period <= 0 or not (0 <= previous < current < len(blocks)):
        return LineageRecord(
            **common,
            eligible=False,
            ineligible_reason="invalid_seed_indices",
            previous_block_0based=previous,
            seed_block_0based=current,
            motif_period_blocks=period,
            seed_progress=seed_progress,
            motif_all_nonempty=None,
            same_sequence_segment=None,
            second_status="not_at_risk",
            second_first_mismatch_offset_1based=None,
            second_confirmation_block_1based=None,
            third_status="not_at_risk",
            third_first_mismatch_offset_1based=None,
            third_confirmation_block_1based=None,
            lineage_capture=False,
            generic_semantic_capture=is_semantic_capture(row),
            stable_orbit=bool(row.get("legacy_orbit", {}).get("exists")),
        )

    motif = blocks[previous:current]
    motif_hashes = [str(block.get("triple_hash")) for block in motif]
    motif_all_nonempty = all(bool(block.get("identity_complete")) for block in motif)
    segment = int(motif[0].get("sequence_segment", -1))
    same_segment = all(int(block.get("sequence_segment", -2)) == segment for block in motif)
    if not motif_all_nonempty:
        reason = "motif_contains_empty_identity"
    elif not same_segment or int(blocks[current].get("sequence_segment", -3)) != segment:
        reason = "cross_segment_seed"
    else:
        reason = ""

    if reason:
        return LineageRecord(
            **common,
            eligible=False,
            ineligible_reason=reason,
            previous_block_0based=previous,
            seed_block_0based=current,
            motif_period_blocks=period,
            seed_progress=seed_progress,
            motif_all_nonempty=motif_all_nonempty,
            same_sequence_segment=same_segment,
            second_status="not_at_risk",
            second_first_mismatch_offset_1based=None,
            second_confirmation_block_1based=None,
            third_status="not_at_risk",
            third_first_mismatch_offset_1based=None,
            third_confirmation_block_1based=None,
            lineage_capture=False,
            generic_semantic_capture=is_semantic_capture(row),
            stable_orbit=bool(row.get("legacy_orbit", {}).get("exists")),
        )

    second_status, second_mismatch, second_confirmation = _copy_status(
        blocks,
        expected_hashes=motif_hashes,
        start=current,
        expected_segment=segment,
        hit_max=common["hit_max"],
    )
    if second_status == "completed":
        third_status, third_mismatch, third_confirmation = _copy_status(
            blocks,
            expected_hashes=motif_hashes,
            start=current + period,
            expected_segment=segment,
            hit_max=common["hit_max"],
        )
    else:
        third_status, third_mismatch, third_confirmation = "not_at_risk", None, None

    return LineageRecord(
        **common,
        eligible=True,
        ineligible_reason="",
        previous_block_0based=previous,
        seed_block_0based=current,
        motif_period_blocks=period,
        seed_progress=seed_progress,
        motif_all_nonempty=True,
        same_sequence_segment=True,
        second_status=second_status,
        second_first_mismatch_offset_1based=second_mismatch,
        second_confirmation_block_1based=second_confirmation,
        third_status=third_status,
        third_first_mismatch_offset_1based=third_mismatch,
        third_confirmation_block_1based=third_confirmation,
        lineage_capture=third_status == "completed",
        generic_semantic_capture=is_semantic_capture(row),
        stable_orbit=bool(row.get("legacy_orbit", {}).get("exists")),
    )


def build_lineage_records(
    rows: Sequence[dict[str, Any]], gold: GoldMapping
) -> list[LineageRecord]:
    return [
        trace_first_seed_lineage(row, gold.values[canonical_key(row.get("key"))])
        for row in rows
    ]


def make_progress_grid(horizon: float, step: float) -> np.ndarray:
    if horizon <= 0 or step <= 0:
        raise ValueError("horizon and step must be positive")
    n_steps = int(round(horizon / step))
    if not math.isclose(n_steps * step, horizon, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("normalized horizon must be an integer multiple of grid step")
    return np.linspace(0.0, horizon, n_steps + 1, dtype=np.float64)


def prompt_competing_arrays(
    records: Sequence[TerminalRecord],
    prompt_ids: Sequence[str],
    edges: np.ndarray,
    *,
    time_scale: str = "gold_progress",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-prompt/model risk, seed-event, and stop-event arrays.

    ``time_scale='gold_progress'`` uses the next-block opportunity divided by
    the prompt's gold relation count.  ``time_scale='block'`` uses the absolute
    complete-block opportunity and is provided to reconcile the earlier B70
    analysis with normal EOS treated as a competing event.
    """
    if time_scale not in {"gold_progress", "block"}:
        raise ValueError(f"unknown time_scale={time_scale!r}")
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    n_bins = len(edges) - 1
    risk = np.zeros((len(prompt_ids), len(MODEL_TAGS), n_bins), dtype=np.int32)
    seed = np.zeros_like(risk)
    stop = np.zeros_like(risk)
    by_prompt_model: dict[tuple[str, str], list[TerminalRecord]] = defaultdict(list)
    for record in records:
        by_prompt_model[(record.prompt_id, record.model_tag)].append(record)

    for prompt_id in prompt_ids:
        pidx = prompt_index[prompt_id]
        for model in MODEL_TAGS:
            midx = MODEL_INDEX[model]
            model_records = by_prompt_model[(prompt_id, model)]
            for record in model_records:
                time = (
                    float(record.event_progress)
                    if time_scale == "gold_progress"
                    else float(record.event_block)
                )
                for k in range(n_bins):
                    left = float(edges[k])
                    right = float(edges[k + 1])
                    if time > left + 1e-12:
                        risk[pidx, midx, k] += 1
                    if left < time <= right + 1e-12:
                        if record.event_kind == "seed":
                            seed[pidx, midx, k] += 1
                        elif record.event_kind == "normal_stop":
                            stop[pidx, midx, k] += 1
    if np.any(seed + stop > risk):
        raise AssertionError("event count exceeds risk set")
    return risk, seed, stop


def _hazards(risk: np.ndarray, seed: np.ndarray, stop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h_seed = np.divide(seed, risk, out=np.zeros_like(seed, dtype=np.float64), where=risk > 0)
    h_stop = np.divide(stop, risk, out=np.zeros_like(stop, dtype=np.float64), where=risk > 0)
    if np.any(h_seed + h_stop > 1.0 + 1e-10):
        raise AssertionError("observed competing hazards sum to more than one")
    return h_seed, h_stop


def _cause_intensities(h_seed: np.ndarray, h_stop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = h_seed + h_stop
    clipped = np.minimum(total, 1.0 - 1e-12)
    lambda_total = -np.log1p(-clipped)
    share_seed = np.divide(h_seed, total, out=np.zeros_like(h_seed), where=total > 0)
    return lambda_total * share_seed, lambda_total * (1.0 - share_seed)


def _hazards_from_intensities(
    lambda_seed: np.ndarray, lambda_stop: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total = lambda_seed + lambda_stop
    event_probability = -np.expm1(-total)
    h_seed = np.divide(
        event_probability * lambda_seed,
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    h_stop = np.divide(
        event_probability * lambda_stop,
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    return h_seed, h_stop


def cif_metrics(h_seed: np.ndarray, h_stop: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized Aalen-Johansen metrics over the final array axis."""
    total = h_seed + h_stop
    one_minus = np.clip(1.0 - total, 0.0, 1.0)
    survival_before = np.concatenate(
        [np.ones((*total.shape[:-1], 1), dtype=np.float64), np.cumprod(one_minus, axis=-1)[..., :-1]],
        axis=-1,
    )
    cif_seed_curve = np.cumsum(survival_before * h_seed, axis=-1)
    cif_stop_curve = np.cumsum(survival_before * h_stop, axis=-1)
    lambda_seed, lambda_stop = _cause_intensities(h_seed, h_stop)
    return {
        "cif_seed_curve": cif_seed_curve,
        "cif_stop_curve": cif_stop_curve,
        "survival_curve": np.cumprod(one_minus, axis=-1),
        "cif_seed": cif_seed_curve[..., -1],
        "cif_stop": cif_stop_curve[..., -1],
        "survival": np.cumprod(one_minus, axis=-1)[..., -1],
        "cumhaz_seed": np.sum(lambda_seed, axis=-1),
        "cumhaz_stop": np.sum(lambda_stop, axis=-1),
    }


def competing_decomposition(
    h_seed_m0: np.ndarray,
    h_stop_m0: np.ndarray,
    h_seed_m1: np.ndarray,
    h_stop_m1: np.ndarray,
) -> dict[str, np.ndarray]:
    """Order-independent Shapley decomposition of seed-CIF change.

    Cause-specific discrete hazards are converted to additive interval
    intensities before cross-model mixing.  For each observed model this
    conversion exactly reconstructs its original discrete hazards.
    """
    ls0, lt0 = _cause_intensities(h_seed_m0, h_stop_m0)
    ls1, lt1 = _cause_intensities(h_seed_m1, h_stop_m1)

    def value(ls: np.ndarray, lt: np.ndarray) -> np.ndarray:
        hs, ht = _hazards_from_intensities(ls, lt)
        return cif_metrics(hs, ht)["cif_seed"]

    f00 = value(ls0, lt0)
    f10 = value(ls1, lt0)
    f01 = value(ls0, lt1)
    f11 = value(ls1, lt1)
    seed_component = 0.5 * ((f10 - f00) + (f11 - f01))
    stop_component = 0.5 * ((f01 - f00) + (f11 - f10))
    return {
        "f00": f00,
        "f10": f10,
        "f01": f01,
        "f11": f11,
        "total_diff": f11 - f00,
        "seed_hazard_component": seed_component,
        "stop_exposure_component": stop_component,
        "identity_error": (seed_component + stop_component) - (f11 - f00),
    }


def aggregate_competing(
    risk: np.ndarray, seed: np.ndarray, stop: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    if weights is None:
        risk_agg = np.sum(risk, axis=0, dtype=np.float64)
        seed_agg = np.sum(seed, axis=0, dtype=np.float64)
        stop_agg = np.sum(stop, axis=0, dtype=np.float64)
    else:
        risk_agg = np.tensordot(weights, risk, axes=(0, 0))
        seed_agg = np.tensordot(weights, seed, axes=(0, 0))
        stop_agg = np.tensordot(weights, stop, axes=(0, 0))
    h_seed, h_stop = _hazards(risk_agg, seed_agg, stop_agg)
    model_metrics = cif_metrics(h_seed, h_stop)
    decomposition = competing_decomposition(
        h_seed[0], h_stop[0], h_seed[1], h_stop[1]
    )
    return {
        "risk": risk_agg,
        "seed_events": seed_agg,
        "stop_events": stop_agg,
        "h_seed": h_seed,
        "h_stop": h_stop,
        **model_metrics,
        **decomposition,
    }


def lineage_prompt_counts(
    lineage: Sequence[LineageRecord], prompt_ids: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    names = [
        "seed",
        "eligible",
        "ineligible",
        "second_completed",
        "second_admin_censor",
        "second_observed_failure",
        "third_at_risk",
        "third_completed",
        "third_admin_censor",
        "third_observed_failure",
        "lineage_capture",
        "generic_capture",
        "stable_orbit",
        "orbit_after_lineage_capture",
    ]
    name_index = {name: index for index, name in enumerate(names)}
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    counts = np.zeros((len(prompt_ids), len(MODEL_TAGS), len(names)), dtype=np.int32)
    for record in lineage:
        if record.prompt_id not in prompt_index:
            continue
        arr = counts[prompt_index[record.prompt_id], MODEL_INDEX[record.model_tag]]
        if record.ineligible_reason == "no_nonempty_seed":
            continue
        arr[name_index["seed"]] += 1
        if record.generic_semantic_capture:
            arr[name_index["generic_capture"]] += 1
        if record.stable_orbit:
            arr[name_index["stable_orbit"]] += 1
        if not record.eligible:
            arr[name_index["ineligible"]] += 1
            continue
        arr[name_index["eligible"]] += 1
        if record.second_status == "completed":
            arr[name_index["second_completed"]] += 1
            arr[name_index["third_at_risk"]] += 1
        elif record.second_status == "admin_censor":
            arr[name_index["second_admin_censor"]] += 1
        else:
            arr[name_index["second_observed_failure"]] += 1
        if record.third_status == "completed":
            arr[name_index["third_completed"]] += 1
        elif record.third_status == "admin_censor":
            arr[name_index["third_admin_censor"]] += 1
        elif record.third_status != "not_at_risk":
            arr[name_index["third_observed_failure"]] += 1
        if record.lineage_capture:
            arr[name_index["lineage_capture"]] += 1
        if record.stable_orbit and record.lineage_capture:
            arr[name_index["orbit_after_lineage_capture"]] += 1
    return counts, names


def safe_ratio(num: np.ndarray | float, den: np.ndarray | float) -> np.ndarray:
    num_arr = np.asarray(num, dtype=np.float64)
    den_arr = np.asarray(den, dtype=np.float64)
    return np.divide(num_arr, den_arr, out=np.full_like(num_arr, np.nan), where=den_arr > 0)


def lineage_metrics_from_counts(counts: np.ndarray, names: Sequence[str]) -> dict[str, np.ndarray]:
    idx = {name: index for index, name in enumerate(names)}
    second_observed_den = (
        counts[..., idx["second_completed"]] + counts[..., idx["second_observed_failure"]]
    )
    third_observed_den = (
        counts[..., idx["third_completed"]] + counts[..., idx["third_observed_failure"]]
    )
    return {
        "eligibility_rate": safe_ratio(counts[..., idx["eligible"]], counts[..., idx["seed"]]),
        "second_completion_observed": safe_ratio(
            counts[..., idx["second_completed"]], second_observed_den
        ),
        "second_admin_censor_rate": safe_ratio(
            counts[..., idx["second_admin_censor"]], counts[..., idx["eligible"]]
        ),
        "third_completion_observed_given_second": safe_ratio(
            counts[..., idx["third_completed"]], third_observed_den
        ),
        "third_admin_censor_rate_given_second": safe_ratio(
            counts[..., idx["third_admin_censor"]], counts[..., idx["third_at_risk"]]
        ),
        "lineage_capture_per_seed": safe_ratio(
            counts[..., idx["lineage_capture"]], counts[..., idx["seed"]]
        ),
        "generic_capture_per_seed": safe_ratio(
            counts[..., idx["generic_capture"]], counts[..., idx["seed"]]
        ),
        "orbit_given_lineage_capture": safe_ratio(
            counts[..., idx["orbit_after_lineage_capture"]], counts[..., idx["lineage_capture"]]
        ),
    }


def response_prompt_summaries(
    terminals: Sequence[TerminalRecord], prompt_ids: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    names = [
        "n",
        "output_gold_ratio_sum",
        "normal_stop",
        "admin_censor",
        "seed",
        "seed_before_0_50",
        "seed_0_50_0_75",
        "seed_0_75_1_00",
        "seed_1_00_1_25",
        "seed_after_1_25",
        "output_under_0_75",
        "output_0_75_1_25",
        "output_over_1_25",
        "output_over_1_50",
    ]
    idx = {name: index for index, name in enumerate(names)}
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    values = np.zeros((len(prompt_ids), len(MODEL_TAGS), len(names)), dtype=np.float64)
    for record in terminals:
        arr = values[prompt_index[record.prompt_id], MODEL_INDEX[record.model_tag]]
        arr[idx["n"]] += 1
        arr[idx["output_gold_ratio_sum"]] += record.output_gold_ratio
        arr[idx[record.event_kind]] += 1
        ratio = record.output_gold_ratio
        if ratio < 0.75:
            arr[idx["output_under_0_75"]] += 1
        elif ratio <= 1.25:
            arr[idx["output_0_75_1_25"]] += 1
        else:
            arr[idx["output_over_1_25"]] += 1
        if ratio > 1.50:
            arr[idx["output_over_1_50"]] += 1
        if record.seed_progress is not None:
            x = record.seed_progress
            if x < 0.50:
                name = "seed_before_0_50"
            elif x < 0.75:
                name = "seed_0_50_0_75"
            elif x < 1.00:
                name = "seed_0_75_1_00"
            elif x < 1.25:
                name = "seed_1_00_1_25"
            else:
                name = "seed_after_1_25"
            arr[idx[name]] += 1
    return values, names


def summary_metrics_from_counts(values: np.ndarray, names: Sequence[str]) -> dict[str, np.ndarray]:
    idx = {name: index for index, name in enumerate(names)}
    n = values[..., idx["n"]]
    seed = values[..., idx["seed"]]
    return {
        "mean_output_gold_ratio": safe_ratio(values[..., idx["output_gold_ratio_sum"]], n),
        "normal_stop_rate": safe_ratio(values[..., idx["normal_stop"]], n),
        "admin_censor_rate": safe_ratio(values[..., idx["admin_censor"]], n),
        "seed_rate": safe_ratio(seed, n),
        "output_under_0_75_rate": safe_ratio(values[..., idx["output_under_0_75"]], n),
        "output_0_75_1_25_rate": safe_ratio(values[..., idx["output_0_75_1_25"]], n),
        "output_over_1_25_rate": safe_ratio(values[..., idx["output_over_1_25"]], n),
        "output_over_1_50_rate": safe_ratio(values[..., idx["output_over_1_50"]], n),
        "seed_before_0_50_share": safe_ratio(values[..., idx["seed_before_0_50"]], seed),
        "seed_0_50_0_75_share": safe_ratio(values[..., idx["seed_0_50_0_75"]], seed),
        "seed_0_75_1_00_share": safe_ratio(values[..., idx["seed_0_75_1_00"]], seed),
        "seed_1_00_1_25_share": safe_ratio(values[..., idx["seed_1_00_1_25"]], seed),
        "seed_after_1_25_share": safe_ratio(values[..., idx["seed_after_1_25"]], seed),
    }


def quantile_ci(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if finite.size else math.nan


def paired_result(m0: float, m1: float, bootstrap_diff: np.ndarray) -> dict[str, float]:
    return {
        "m0": float(m0),
        "m1": float(m1),
        "diff": float(m1 - m0),
        "ci_low": quantile_ci(bootstrap_diff, 0.025),
        "ci_high": quantile_ci(bootstrap_diff, 0.975),
    }


def scalar_result(value: float, bootstrap_values: np.ndarray) -> dict[str, float]:
    return {
        "value": float(value),
        "ci_low": quantile_ci(bootstrap_values, 0.025),
        "ci_high": quantile_ci(bootstrap_values, 0.975),
    }


def bootstrap_competing_only(
    *,
    risk: np.ndarray,
    seed_events: np.ndarray,
    stop_events: np.ndarray,
    n_bootstrap: int,
    random_seed: int,
    batch_size: int = 200,
) -> dict[str, Any]:
    """Prompt-cluster bootstrap for a competing-risk grid only."""
    n_prompts = risk.shape[0]
    observed = aggregate_competing(risk, seed_events, stop_events)
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    rng = np.random.default_rng(random_seed)
    completed = 0
    while completed < n_bootstrap:
        batch = min(batch_size, n_bootstrap - completed)
        weights = rng.multinomial(n_prompts, np.full(n_prompts, 1.0 / n_prompts), size=batch)
        risk_b = np.einsum("bn,nmk->bmk", weights, risk, optimize=True)
        seed_b = np.einsum("bn,nmk->bmk", weights, seed_events, optimize=True)
        stop_b = np.einsum("bn,nmk->bmk", weights, stop_events, optimize=True)
        h_seed_b, h_stop_b = _hazards(risk_b, seed_b, stop_b)
        metrics_b = cif_metrics(h_seed_b, h_stop_b)
        decomp_b = competing_decomposition(
            h_seed_b[:, 0, :], h_stop_b[:, 0, :], h_seed_b[:, 1, :], h_stop_b[:, 1, :]
        )
        for name in ("cif_seed", "cif_stop", "cumhaz_seed", "cumhaz_stop"):
            collected[f"metric_{name}"].append(metrics_b[name][:, 1] - metrics_b[name][:, 0])
        for name in ("total_diff", "seed_hazard_component", "stop_exposure_component"):
            collected[f"decomp_{name}"].append(decomp_b[name])
        completed += batch
    draws = {name: np.concatenate(parts) for name, parts in collected.items()}
    return {
        "observed_competing": observed,
        "competing": {
            name: paired_result(
                observed[name][0], observed[name][1], draws[f"metric_{name}"]
            )
            for name in ("cif_seed", "cif_stop", "cumhaz_seed", "cumhaz_stop")
        },
        "decomposition": {
            name: scalar_result(observed[name], draws[f"decomp_{name}"])
            for name in ("total_diff", "seed_hazard_component", "stop_exposure_component")
        },
    }


def bootstrap_analysis(
    *,
    risk: np.ndarray,
    seed_events: np.ndarray,
    stop_events: np.ndarray,
    lineage_counts: np.ndarray,
    lineage_names: Sequence[str],
    summary_counts: np.ndarray,
    summary_names: Sequence[str],
    n_bootstrap: int,
    random_seed: int,
    batch_size: int = 200,
) -> dict[str, Any]:
    n_prompts = risk.shape[0]
    observed_competing = aggregate_competing(risk, seed_events, stop_events)
    observed_lineage_counts = np.sum(lineage_counts, axis=0, dtype=np.float64)
    observed_lineage = lineage_metrics_from_counts(observed_lineage_counts, lineage_names)
    observed_summary_counts = np.sum(summary_counts, axis=0, dtype=np.float64)
    observed_summary = summary_metrics_from_counts(observed_summary_counts, summary_names)

    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    rng = np.random.default_rng(random_seed)
    completed = 0
    while completed < n_bootstrap:
        batch = min(batch_size, n_bootstrap - completed)
        # Multinomial cluster weights are exactly equivalent to resampling
        # prompt IDs with replacement and retaining every seed within a prompt.
        weights = rng.multinomial(n_prompts, np.full(n_prompts, 1.0 / n_prompts), size=batch)
        risk_b = np.einsum("bn,nmk->bmk", weights, risk, optimize=True)
        seed_b = np.einsum("bn,nmk->bmk", weights, seed_events, optimize=True)
        stop_b = np.einsum("bn,nmk->bmk", weights, stop_events, optimize=True)
        h_seed_b, h_stop_b = _hazards(risk_b, seed_b, stop_b)
        metrics_b = cif_metrics(h_seed_b, h_stop_b)
        decomp_b = competing_decomposition(
            h_seed_b[:, 0, :], h_stop_b[:, 0, :], h_seed_b[:, 1, :], h_stop_b[:, 1, :]
        )
        for name in ("cif_seed", "cif_stop", "cumhaz_seed", "cumhaz_stop"):
            collected[f"competing_{name}_diff"].append(
                metrics_b[name][:, 1] - metrics_b[name][:, 0]
            )
        for name in ("cif_seed_curve", "cif_stop_curve"):
            collected[f"competing_{name}_diff"].append(
                metrics_b[name][:, 1, :] - metrics_b[name][:, 0, :]
            )
        for name in ("total_diff", "seed_hazard_component", "stop_exposure_component"):
            collected[f"decomp_{name}"].append(decomp_b[name])

        lineage_b_counts = np.einsum("bn,nmk->bmk", weights, lineage_counts, optimize=True)
        lineage_b = lineage_metrics_from_counts(lineage_b_counts, lineage_names)
        for name, values in lineage_b.items():
            collected[f"lineage_{name}_diff"].append(values[:, 1] - values[:, 0])

        summary_b_counts = np.einsum("bn,nmk->bmk", weights, summary_counts, optimize=True)
        summary_b = summary_metrics_from_counts(summary_b_counts, summary_names)
        for name, values in summary_b.items():
            collected[f"summary_{name}_diff"].append(values[:, 1] - values[:, 0])
        completed += batch

    draws = {name: np.concatenate(parts) for name, parts in collected.items()}
    competing_results: dict[str, dict[str, float]] = {}
    for name in ("cif_seed", "cif_stop", "cumhaz_seed", "cumhaz_stop"):
        competing_results[name] = paired_result(
            observed_competing[name][0],
            observed_competing[name][1],
            draws[f"competing_{name}_diff"],
        )
    decomposition_results = {
        name: scalar_result(observed_competing[name], draws[f"decomp_{name}"])
        for name in ("total_diff", "seed_hazard_component", "stop_exposure_component")
    }
    lineage_results = {
        name: paired_result(values[0], values[1], draws[f"lineage_{name}_diff"])
        for name, values in observed_lineage.items()
    }
    summary_results = {
        name: paired_result(values[0], values[1], draws[f"summary_{name}_diff"])
        for name, values in observed_summary.items()
    }
    curve_ci = {
        name: {
            "ci_low": np.nanquantile(draws[f"competing_{name}_diff"], 0.025, axis=0),
            "ci_high": np.nanquantile(draws[f"competing_{name}_diff"], 0.975, axis=0),
        }
        for name in ("cif_seed_curve", "cif_stop_curve")
    }
    return {
        "observed_competing": observed_competing,
        "competing": competing_results,
        "decomposition": decomposition_results,
        "lineage": lineage_results,
        "summary": summary_results,
        "curve_ci": curve_ci,
    }


def descriptive_quantiles(terminals: Sequence[TerminalRecord]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for model in MODEL_TAGS:
        model_records = [record for record in terminals if record.model_tag == model]
        ratios = np.asarray([record.output_gold_ratio for record in model_records], dtype=np.float64)
        seed_progress = np.asarray(
            [record.seed_progress for record in model_records if record.seed_progress is not None],
            dtype=np.float64,
        )
        blocks = np.asarray([record.n_blocks for record in model_records], dtype=np.float64)
        output[model] = {
            "n": float(len(model_records)),
            "n_blocks_p25": float(np.quantile(blocks, 0.25)),
            "n_blocks_median": float(np.quantile(blocks, 0.50)),
            "n_blocks_p75": float(np.quantile(blocks, 0.75)),
            "output_gold_ratio_p25": float(np.quantile(ratios, 0.25)),
            "output_gold_ratio_median": float(np.quantile(ratios, 0.50)),
            "output_gold_ratio_p75": float(np.quantile(ratios, 0.75)),
            "seed_progress_p25": float(np.quantile(seed_progress, 0.25)) if seed_progress.size else math.nan,
            "seed_progress_median": float(np.quantile(seed_progress, 0.50)) if seed_progress.size else math.nan,
            "seed_progress_p75": float(np.quantile(seed_progress, 0.75)) if seed_progress.size else math.nan,
        }
    return output


def competing_curve_rows(
    observed: Mapping[str, np.ndarray], edges: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_TAGS:
        midx = MODEL_INDEX[model]
        for k in range(len(edges) - 1):
            rows.append(
                {
                    "model": model,
                    "progress_left": float(edges[k]),
                    "progress_right": float(edges[k + 1]),
                    "at_risk": int(observed["risk"][midx, k]),
                    "seed_events": int(observed["seed_events"][midx, k]),
                    "normal_stop_events": int(observed["stop_events"][midx, k]),
                    "seed_hazard": float(observed["h_seed"][midx, k]),
                    "stop_hazard": float(observed["h_stop"][midx, k]),
                    "seed_cif": float(observed["cif_seed_curve"][midx, k]),
                    "stop_cif": float(observed["cif_stop_curve"][midx, k]),
                    "event_free_survival": float(observed["survival_curve"][midx, k]),
                }
            )
    return rows


def lineage_record_dict(record: LineageRecord) -> dict[str, Any]:
    return record.__dict__.copy()


def terminal_record_dict(record: TerminalRecord) -> dict[str, Any]:
    return record.__dict__.copy()
