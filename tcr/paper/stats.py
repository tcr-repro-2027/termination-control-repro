# coding: utf-8
"""Prompt-clustered paired statistics.

Every headline comparison in this paper is between two models scored on the
SAME 1,106 evaluation records with the same eight generation seeds.  The eight
responses of one record are strongly correlated, so the resampling unit is the
record, not the response, and the two arms must be resampled TOGETHER -- an
interval built by subtracting two independently bootstrapped intervals is not
an interval for the difference.

The same multinomial weights therefore drive both arms in every draw, and each
draw recomputes the statistic from counts rather than averaging per-prompt
rates: a pooled F1 is a ratio of summed TP / predicted / gold counts, and
averaging per-record F1 values would silently answer a different question.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 20260908


@dataclass(frozen=True)
class Effect:
    """One paired comparison: both arms, their difference, and its interval."""

    metric: str
    m0: float
    m1: float
    diff: float
    ci_low: float
    ci_high: float
    n_prompts: int
    n_m0: int
    n_m1: int
    direction: str

    def as_dict(self, **extra: Any) -> dict[str, Any]:
        """The effect as a row, plus caller context.

        Context may not reuse a field name the effect already owns.  Passing
        ``m0=<tag>`` reads naturally and silently replaces the MEASURED level
        of arm 0 with a model name, after which the number is gone and every
        downstream formatter prints NA.  Names like ``m0_tag`` cost nothing and
        cannot do that.
        """
        row = asdict(self)
        clash = sorted(set(extra) & set(row))
        if clash:
            raise ValueError(
                f"as_dict() would overwrite the effect's own field(s) {clash} "
                "with caller context; those names hold the measured values. "
                "Use a distinct key, e.g. 'm0_tag' rather than 'm0'.")
        row.update(extra)
        return row


def direction_of(low: float, high: float) -> str:
    if not (math.isfinite(low) and math.isfinite(high)):
        return "undetermined"
    if low > 0:
        return "m1_higher"
    if high < 0:
        return "m1_lower"
    return "interval_covers_zero"


def _weights(rng: np.random.Generator, n_prompts: int, n_boot: int) -> np.ndarray:
    """(n_boot, n_prompts) multinomial resampling weights over records."""
    return rng.multinomial(n_prompts, np.full(n_prompts, 1.0 / n_prompts),
                           size=n_boot).astype(np.float64)


def _percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    finite = samples[np.isfinite(samples)]
    if finite.size < 2:
        return (math.nan, math.nan)
    return (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)))


def group_by_prompt(rows: Sequence[Mapping[str, Any]], key: str = "stable_prompt_id"
                    ) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key)), []).append(row)
    return grouped


def _aligned_prompt_arrays(
    m0_rows: Sequence[Mapping[str, Any]],
    m1_rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    key: str = "stable_prompt_id",
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-prompt sums of `fields` for each arm, over the shared prompt set.

    Prompts present in only one arm are dropped: a paired difference over a
    prompt set that differs between the arms is not a paired difference.
    """
    left = group_by_prompt(m0_rows, key)
    right = group_by_prompt(m1_rows, key)
    prompts = sorted(set(left) & set(right))
    if not prompts:
        raise ValueError("the two arms share no prompt")
    n_fields = len(fields)
    a = np.zeros((len(prompts), n_fields), dtype=np.float64)
    b = np.zeros((len(prompts), n_fields), dtype=np.float64)
    for index, prompt in enumerate(prompts):
        for position, field in enumerate(fields):
            a[index, position] = sum(float(row.get(field, 0.0) or 0.0)
                                     for row in left[prompt])
            b[index, position] = sum(float(row.get(field, 0.0) or 0.0)
                                     for row in right[prompt])
    return prompts, a, b


def paired_rate(
    m0_rows: Sequence[Mapping[str, Any]],
    m1_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    value_field: str,
    count_field: str = "_unit",
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    key: str = "stable_prompt_id",
) -> Effect:
    """A ratio-of-sums rate, paired over prompts.

    `value_field` is the numerator per response (0/1 for an event rate, a count
    for a block-level rate); `count_field` is the denominator per response,
    defaulting to one unit per response.
    """
    rows0 = [{**row, "_unit": 1.0} for row in m0_rows]
    rows1 = [{**row, "_unit": 1.0} for row in m1_rows]
    prompts, a, b = _aligned_prompt_arrays(rows0, rows1,
                                           [value_field, count_field], key)
    rng = np.random.default_rng(seed)
    weights = _weights(rng, len(prompts), n_boot)

    def rate(matrix: np.ndarray, w: np.ndarray) -> np.ndarray:
        numerator = w @ matrix[:, 0]
        denominator = w @ matrix[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denominator > 0, numerator / denominator, np.nan)

    point0 = float(a[:, 0].sum() / a[:, 1].sum()) if a[:, 1].sum() else math.nan
    point1 = float(b[:, 0].sum() / b[:, 1].sum()) if b[:, 1].sum() else math.nan
    draws = rate(b, weights) - rate(a, weights)
    low, high = _percentile_ci(draws)
    return Effect(metric=metric, m0=point0, m1=point1, diff=point1 - point0,
                  ci_low=low, ci_high=high, n_prompts=len(prompts),
                  n_m0=len(rows0), n_m1=len(rows1),
                  direction=direction_of(low, high))


def paired_mean(
    m0_rows: Sequence[Mapping[str, Any]],
    m1_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    value_field: str,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    key: str = "stable_prompt_id",
) -> Effect:
    """A per-response mean (generation length, block count, ...), paired."""
    return paired_rate(m0_rows, m1_rows, metric=metric, value_field=value_field,
                       n_boot=n_boot, seed=seed, key=key)


def paired_f1(
    m0_rows: Sequence[Mapping[str, Any]],
    m1_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    tp_field: str,
    pred_field: str,
    gold_field: str,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    key: str = "stable_prompt_id",
) -> Effect:
    """Pooled micro-F1, recomputed from resampled TP / predicted / gold counts."""
    prompts, a, b = _aligned_prompt_arrays(
        m0_rows, m1_rows, [tp_field, pred_field, gold_field], key)
    rng = np.random.default_rng(seed)
    weights = _weights(rng, len(prompts), n_boot)

    def f1(matrix: np.ndarray, w: np.ndarray) -> np.ndarray:
        tp = w @ matrix[:, 0]
        pred = w @ matrix[:, 1]
        gold = w @ matrix[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(pred > 0, tp / pred, 0.0)
            recall = np.where(gold > 0, tp / gold, 0.0)
            denominator = precision + recall
            return np.where(denominator > 0,
                            2.0 * precision * recall / denominator, 0.0)

    ones = np.ones((1, len(prompts)))
    point0 = float(f1(a, ones)[0])
    point1 = float(f1(b, ones)[0])
    draws = f1(b, weights) - f1(a, weights)
    low, high = _percentile_ci(draws)
    return Effect(metric=metric, m0=point0, m1=point1, diff=point1 - point0,
                  ci_low=low, ci_high=high, n_prompts=len(prompts),
                  n_m0=len(m0_rows), n_m1=len(m1_rows),
                  direction=direction_of(low, high))


def paired_anchor_effect(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    value_field: str,
    anchor_key: str = "anchor_id",
    prompt_key: str = "prompt_id",
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> Effect:
    """Difference between two models measured on the SAME anchors.

    The pairing is per anchor (identical prefix, identical decision tokens),
    and the resampling unit is still the prompt, because one prompt can
    contribute more than one anchor and its anchors are not independent.

    Exactly one row per anchor is required on each side.  Silently keeping the
    last of several would quietly throw away every other draw and report an
    interval built from a fraction of the data, so repeated anchors are an
    error here and the caller has to say how it wants them aggregated.
    """
    for label, rows in (("m0", rows_a), ("m1", rows_b)):
        counts = Counter(str(row[anchor_key]) for row in rows)
        repeated = [anchor for anchor, n in counts.items() if n > 1]
        if repeated:
            raise ValueError(
                f"{metric}: {label} has {len(repeated)} anchor(s) appearing more "
                f"than once (e.g. {sorted(repeated)[:3]}).  Aggregate the repeats "
                "into one value per anchor before calling this.")
    left = {str(row[anchor_key]): row for row in rows_a}
    right = {str(row[anchor_key]): row for row in rows_b}
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError(f"{metric}: the two models share no anchor")
    prompts = sorted({str(left[a][prompt_key]) for a in shared})
    index = {prompt: position for position, prompt in enumerate(prompts)}
    sums_a = np.zeros(len(prompts))
    sums_b = np.zeros(len(prompts))
    counts = np.zeros(len(prompts))
    for anchor in shared:
        value_a = left[anchor].get(value_field)
        value_b = right[anchor].get(value_field)
        if value_a is None or value_b is None:
            continue
        position = index[str(left[anchor][prompt_key])]
        sums_a[position] += float(value_a)
        sums_b[position] += float(value_b)
        counts[position] += 1.0
    if counts.sum() == 0:
        raise ValueError(f"{metric}: no anchor carries {value_field!r} in both models")
    rng = np.random.default_rng(seed)
    weights = _weights(rng, len(prompts), n_boot)

    def mean(sums: np.ndarray, w: np.ndarray) -> np.ndarray:
        denominator = w @ counts
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denominator > 0, (w @ sums) / denominator, np.nan)

    point_a = float(sums_a.sum() / counts.sum())
    point_b = float(sums_b.sum() / counts.sum())
    draws = mean(sums_b, weights) - mean(sums_a, weights)
    low, high = _percentile_ci(draws)
    return Effect(metric=metric, m0=point_a, m1=point_b, diff=point_b - point_a,
                  ci_low=low, ci_high=high, n_prompts=len(prompts),
                  n_m0=int(counts.sum()), n_m1=int(counts.sum()),
                  direction=direction_of(low, high))


def paired_condition_effect(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    value_field: str,
    condition_field: str,
    baseline: str,
    treatment: str,
    anchor_key: str = "anchor_id",
    prompt_key: str = "prompt_id",
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> Effect:
    """Two conditions of ONE model on the same anchors (intervention effects)."""
    left = [row for row in rows if str(row.get(condition_field)) == baseline]
    right = [row for row in rows if str(row.get(condition_field)) == treatment]
    return paired_anchor_effect(left, right, metric=metric, value_field=value_field,
                                anchor_key=anchor_key, prompt_key=prompt_key,
                                n_boot=n_boot, seed=seed)


def specificity_effect(
    main_rows: Sequence[Mapping[str, Any]],
    random_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    value_field: str,
    anchor_key: str = "anchor_id",
    prompt_key: str = "prompt_id",
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> Effect:
    """Main-direction effect minus the mean of the equal-norm random controls.

    Random directions are averaged WITHIN an anchor before the comparison, so
    the eight controls of one anchor count once rather than eight times.
    """
    pooled: dict[str, dict[str, Any]] = {}
    for row in random_rows:
        anchor = str(row[anchor_key])
        bucket = pooled.setdefault(anchor, {anchor_key: anchor,
                                            prompt_key: row[prompt_key],
                                            "_sum": 0.0, "_n": 0})
        value = row.get(value_field)
        if value is None:
            continue
        bucket["_sum"] += float(value)
        bucket["_n"] += 1
    averaged = [{anchor_key: bucket[anchor_key], prompt_key: bucket[prompt_key],
                 value_field: bucket["_sum"] / bucket["_n"]}
                for bucket in pooled.values() if bucket["_n"]]
    return paired_anchor_effect(averaged, main_rows, metric=metric,
                                value_field=value_field, anchor_key=anchor_key,
                                prompt_key=prompt_key, n_boot=n_boot, seed=seed)


def mean_per_anchor(rows: Sequence[Mapping[str, Any]], *, value_field: str,
                    anchor_key: str = "anchor_id",
                    prompt_key: str = "prompt_id") -> list[dict[str, Any]]:
    """Collapse several measurements of one anchor into that anchor's mean.

    The full-continuation experiment draws a response several times per
    (anchor, condition); the paired test compares CONDITIONS on an anchor, so
    the draws are the anchor's within-cell replicates and belong inside its
    single value, not as extra rows the pairing would collide on.
    """
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(value_field)
        if value is None:
            continue
        anchor = str(row[anchor_key])
        bucket = buckets.setdefault(anchor, {anchor_key: anchor,
                                             prompt_key: row.get(prompt_key),
                                             "_sum": 0.0, "_n": 0})
        bucket["_sum"] += float(value)
        bucket["_n"] += 1
    return [{anchor_key: bucket[anchor_key], prompt_key: bucket[prompt_key],
             value_field: bucket["_sum"] / bucket["_n"], "n_draws": bucket["_n"]}
            for bucket in buckets.values() if bucket["_n"]]


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation with average ranks for ties (numpy only)."""
    def ranks(values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        order = np.argsort(array, kind="stable")
        rank = np.empty_like(array)
        rank[order] = np.arange(1, array.size + 1, dtype=np.float64)
        for value in np.unique(array):
            mask = array == value
            if mask.sum() > 1:
                rank[mask] = float(rank[mask].mean())
        return rank

    if len(x) < 3:
        return math.nan
    rx, ry = ranks(x), ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denominator) if denominator else math.nan


def effects_to_rows(effects: Sequence[Effect], **shared: Any) -> list[dict[str, Any]]:
    return [effect.as_dict(**shared) for effect in effects]
