"""d_stop identification: dev/holdout split, layer scan, separability (numpy).

Direction recipe (plan v2 R2, frozen): on the SAME anchor prefix, the
candidate termination-damage direction at layer l is the dev-anchor mean of
``h_M1(l) - h_M0(l)`` at the boundary position.  The layer is chosen by
paired holdout separability (AUC of the projection difference), and the
frozen direction is then behaviorally cross-checked against the S1 StopMargin
per anchor (diagnostic, not gating — the causal gate is S2c steering).
"""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import (
    DEV_FRACTION,
    HOLDOUT_AUC_MIN,
    N_LAYERS,
    PRIMARY_ANCHOR_TYPE,
    SPLIT_SEED,
)


def split_prompts(anchors: Sequence[Mapping[str, Any]], *, anchor_type: str) -> tuple[set[str], set[str]]:
    """Deterministic dev/holdout split by prompt within one anchor type."""
    prompts = sorted({str(a["prompt_id"]) for a in anchors if a["anchor_type"] == anchor_type})
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(prompts)
    n_dev = max(1, int(round(len(prompts) * DEV_FRACTION)))
    return set(prompts[:n_dev]), set(prompts[n_dev:])


def stack_boundary(
    activations: Mapping[str, np.ndarray],
    anchors: Sequence[Mapping[str, Any]],
    *,
    anchor_type: str,
    prompt_filter: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """(n_anchors, n_layers, d_model) float32 stack + anchor id order."""
    rows, ids = [], []
    for anchor in anchors:
        if anchor["anchor_type"] != anchor_type:
            continue
        if prompt_filter is not None and str(anchor["prompt_id"]) not in prompt_filter:
            continue
        anchor_id = str(anchor["anchor_id"])
        h = activations.get(anchor_id)
        if h is None:
            raise KeyError(f"boundary activations missing for anchor {anchor_id}")
        if h.shape != (N_LAYERS, h.shape[-1]):
            raise ValueError(f"unexpected activation shape {h.shape} for {anchor_id}")
        rows.append(np.asarray(h, dtype=np.float32))
        ids.append(anchor_id)
    if not rows:
        raise ValueError(f"no anchors of type {anchor_type} matched the filter")
    return np.stack(rows), ids


def layer_directions(dev_m0: np.ndarray, dev_m1: np.ndarray) -> np.ndarray:
    """(n_layers, d_model) mean damage direction h_M1 - h_M0 on dev anchors."""
    if dev_m0.shape != dev_m1.shape:
        raise ValueError("dev activation stacks must align")
    return (dev_m1.astype(np.float64) - dev_m0.astype(np.float64)).mean(axis=0)


def paired_auc(proj_m0: np.ndarray, proj_m1: np.ndarray) -> float:
    """AUC of separating M1 from M0 boundary states by the projection."""
    m0 = np.asarray(proj_m0, dtype=np.float64)
    m1 = np.asarray(proj_m1, dtype=np.float64)
    greater = (m1[:, None] > m0[None, :]).sum()
    ties = (m1[:, None] == m0[None, :]).sum()
    return float((greater + 0.5 * ties) / (m1.size * m0.size))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
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

    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denominator) if denominator else math.nan


def evaluate_layers(
    *,
    directions: np.ndarray,
    holdout_m0: np.ndarray,
    holdout_m1: np.ndarray,
    holdout_ids: Sequence[str],
    margins_m1: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Per-layer holdout metrics for the frozen dev directions."""
    results = []
    for layer in range(directions.shape[0]):
        d = directions[layer]
        norm = float(np.linalg.norm(d))
        unit = d / norm if norm > 0 else d
        p0 = holdout_m0[:, layer, :].astype(np.float64) @ unit
        p1 = holdout_m1[:, layer, :].astype(np.float64) @ unit
        auc = paired_auc(p0, p1)
        margin_values = [margins_m1.get(aid) for aid in holdout_ids]
        pairs = [
            (float(proj), float(margin))
            for proj, margin in zip(p1, margin_values)
            if margin is not None
        ]
        rho = spearman([p for p, _ in pairs], [m for _, m in pairs]) if len(pairs) >= 8 else math.nan
        results.append(
            {
                "layer": layer,
                "direction_norm": norm,
                "holdout_auc": auc,
                "holdout_diff_mean": float((p1 - p0).mean()),
                "m1_margin_spearman": rho,
                "n_holdout": int(holdout_m0.shape[0]),
            }
        )
    return results


def choose_layer(layer_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """argmax holdout AUC; ties -> larger |margin rho| -> lower layer index."""
    def key(row: Mapping[str, Any]):
        rho = row["m1_margin_spearman"]
        rho_abs = abs(rho) if math.isfinite(rho) else -1.0
        return (round(float(row["holdout_auc"]), 6), rho_abs, -int(row["layer"]))

    best = max(layer_rows, key=key)
    return dict(best)


def gate_s2a(best: Mapping[str, Any]) -> dict[str, Any]:
    passed = float(best["holdout_auc"]) >= HOLDOUT_AUC_MIN
    return {
        "gate": "S2A_DIRECTION_READABLE",
        "layer": int(best["layer"]),
        "holdout_auc": float(best["holdout_auc"]),
        "threshold": HOLDOUT_AUC_MIN,
        "passed": bool(passed),
        "anchor_type": PRIMARY_ANCHOR_TYPE,
    }
