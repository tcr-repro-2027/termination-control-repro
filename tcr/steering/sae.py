"""Qwen-Scope SAE: minimal official-format loader + fresh numpy analysis.

Weight-format contract (referenced from the validated 02/08e loaders — the
FORMAT only, no previously identified features are ever read):

* per-layer files ``layer{n}.sae.pt`` with W_enc (d_sae, d_model),
  W_dec (d_model, d_sae), b_enc (d_sae,), b_dec (d_model,);
* ``config.json`` provides d_model / d_sae / k / layers / hook_point;
* encode: ``a = TopK_k(ReLU(x @ W_enc^T + b_enc))``;
* SAE layer n (0-based) reads the residual stream at the OUTPUT of decoder
  block n (forward hook on ``model.model.layers[n]``, never
  output_hidden_states — the final RMSNorm would contaminate the last layer).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import (
    SAE_MIN_ACTIVE_SHARE,
    SAE_TOPK_DIFFERENTIAL,
    SAE_TOPK_PROJECTION,
)


def load_sae_layer(sae_root: str | Path, layer: int) -> dict[str, np.ndarray | int]:
    """Load one layer's SAE weights as float32 numpy (torch required)."""
    import torch

    root = Path(sae_root)
    if not root.is_dir():
        raise FileNotFoundError(f"SAE directory does not exist: {root}")
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    candidates = [root / f"layer{layer}.sae.pt", root / f"layer{layer}.pt", root / f"layer_{layer}.pt"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        listing = sorted(p.name for p in root.iterdir())[:40]
        raise FileNotFoundError(f"SAE layer {layer} not found under {root}; entries={listing}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    missing = [k for k in ("W_enc", "W_dec", "b_enc", "b_dec") if k not in state]
    if missing:
        raise ValueError(f"{path} lacks keys {missing}")
    W_enc = state["W_enc"].float().numpy()
    W_dec = state["W_dec"].float().numpy()
    b_enc = state["b_enc"].float().numpy()
    b_dec = state["b_dec"].float().numpy()
    d_sae, d_model = W_enc.shape
    if W_dec.shape != (d_model, d_sae):
        raise ValueError(f"W_dec shape {W_dec.shape} != (d_model={d_model}, d_sae={d_sae})")
    k = int(config.get("k", 0)) or 100
    if config.get("d_model") not in (None, d_model) or config.get("d_sae") not in (None, d_sae):
        raise ValueError("SAE config.json dims disagree with layer weights")
    return {"W_enc": W_enc, "W_dec": W_dec, "b_enc": b_enc, "b_dec": b_dec, "k": k, "layer": layer}


def sae_encode_np(x: np.ndarray, sae: Mapping[str, Any]) -> np.ndarray:
    """TopK_k(ReLU(x @ W_enc^T + b_enc)) for (..., d_model) float inputs."""
    pre = x.astype(np.float32) @ np.asarray(sae["W_enc"]).T + np.asarray(sae["b_enc"])
    np.maximum(pre, 0.0, out=pre)
    k = int(sae["k"])
    if k < pre.shape[-1]:
        flat = pre.reshape(-1, pre.shape[-1])
        idx = np.argpartition(flat, -k, axis=-1)[:, :-k]
        np.put_along_axis(flat, idx, 0.0, axis=-1)
        pre = flat.reshape(pre.shape)
    return pre


def projection_view(d_stop: np.ndarray, sae: Mapping[str, Any], *, top_k: int = SAE_TOPK_PROJECTION) -> list[dict[str, Any]]:
    """Top features by |cos(d_stop, W_dec_f)| — the dictionary decomposition."""
    W_dec = np.asarray(sae["W_dec"], dtype=np.float64)
    d = np.asarray(d_stop, dtype=np.float64)
    d_norm = np.linalg.norm(d)
    col_norms = np.linalg.norm(W_dec, axis=0)
    cos = (W_dec.T @ d) / (np.maximum(col_norms, 1e-12) * max(d_norm, 1e-12))
    order = np.argsort(-np.abs(cos))[:top_k]
    return [
        {
            "feature_id": int(f),
            "view": "projection",
            "cos_with_d_stop": float(cos[f]),
            "decoder_norm": float(col_norms[f]),
        }
        for f in order
    ]


def differential_view(
    acts_m0: np.ndarray,
    acts_m1: np.ndarray,
    *,
    top_k: int = SAE_TOPK_DIFFERENTIAL,
    min_active_share: float = SAE_MIN_ACTIVE_SHARE,
) -> list[dict[str, Any]]:
    """Top features by |mean boundary activation M1 - M0| (fresh identification).

    Restricted to features active (nonzero) on at least ``min_active_share``
    of anchors in either model, so dead dictionary entries cannot rank.
    """
    if acts_m0.shape != acts_m1.shape:
        raise ValueError("activation stacks must align")
    active0 = (acts_m0 > 0).mean(axis=0)
    active1 = (acts_m1 > 0).mean(axis=0)
    eligible = np.where((active0 >= min_active_share) | (active1 >= min_active_share))[0]
    mean0 = acts_m0[:, eligible].mean(axis=0)
    mean1 = acts_m1[:, eligible].mean(axis=0)
    delta = mean1 - mean0
    order = np.argsort(-np.abs(delta))[:top_k]
    return [
        {
            "feature_id": int(eligible[i]),
            "view": "differential",
            "mean_act_m0": float(mean0[i]),
            "mean_act_m1": float(mean1[i]),
            "delta_m1_minus_m0": float(delta[i]),
            "active_share_m0": float(active0[eligible[i]]),
            "active_share_m1": float(active1[eligible[i]]),
        }
        for i in order
    ]


def merge_feature_views(projection: Sequence[Mapping[str, Any]], differential: Sequence[Mapping[str, Any]], *, cap: int) -> list[int]:
    seen: list[int] = []
    for row in list(projection) + list(differential):
        fid = int(row["feature_id"])
        if fid not in seen:
            seen.append(fid)
    return seen[:cap]


def checkpoint_trajectories(
    boundary_h_by_label: Mapping[str, np.ndarray],
    sae: Mapping[str, Any],
    feature_ids: Sequence[int],
    d_stop: np.ndarray,
) -> list[dict[str, Any]]:
    """Per-checkpoint mean activation of the fresh features + d_stop projection.

    ``boundary_h_by_label`` maps a model label (``M0``/``ckpt46``/…/``M1``) to
    the (n_anchors, d_model) boundary activations at the frozen layer.
    """
    d = np.asarray(d_stop, dtype=np.float64)
    unit = d / max(np.linalg.norm(d), 1e-12)
    rows: list[dict[str, Any]] = []
    for label, h in boundary_h_by_label.items():
        acts = sae_encode_np(np.asarray(h, dtype=np.float32), sae)
        row: dict[str, Any] = {
            "model_label": label,
            "n_anchors": int(h.shape[0]),
            "d_stop_projection_mean": float((h.astype(np.float64) @ unit).mean()),
        }
        for fid in feature_ids:
            row[f"feat_{fid}_mean_act"] = float(acts[:, fid].mean())
            row[f"feat_{fid}_active_share"] = float((acts[:, fid] > 0).mean())
        rows.append(row)
    return rows


LABEL_ORDER = re.compile(r"^ckpt(\d+)$")


def sort_labels(labels: Sequence[str], *, m1_step: int) -> list[str]:
    def step(label: str) -> int:
        if label == "M0":
            return 0
        if label == "M1":
            return m1_step
        match = LABEL_ORDER.match(label)
        if match:
            return int(match.group(1))
        return 10**9

    return sorted(labels, key=step)
