from __future__ import annotations

import math

import numpy as np
import pytest

from tcr.steering.constants import N_LAYERS
from tcr.steering.direction import (
    choose_layer,
    evaluate_layers,
    gate_s2a,
    layer_directions,
    paired_auc,
    split_prompts,
    stack_boundary,
)
from tcr.steering.sae import (
    differential_view,
    merge_feature_views,
    checkpoint_trajectories,
    projection_view,
    sae_encode_np,
    sort_labels,
)
from tcr.steering.steering import build_conditions


# ---------------------------------------------------------------- direction
def make_anchor(anchor_id: str, anchor_type: str, prompt: str) -> dict:
    return {"anchor_id": anchor_id, "anchor_type": anchor_type, "prompt_id": prompt}


def test_split_prompts_deterministic_and_disjoint():
    anchors = [make_anchor(f"A_{i}", "A", f"p{i}") for i in range(10)]
    dev1, hold1 = split_prompts(anchors, anchor_type="A")
    dev2, hold2 = split_prompts(anchors, anchor_type="A")
    assert dev1 == dev2 and hold1 == hold2
    assert dev1 | hold1 == {f"p{i}" for i in range(10)}
    assert not (dev1 & hold1)
    assert len(dev1) == 5


def test_direction_pipeline_finds_planted_layer():
    rng = np.random.default_rng(0)
    d_model = 16
    planted_layer = 7
    signal = np.zeros(d_model)
    signal[3] = 2.5
    anchors = [make_anchor(f"A_{i:02d}", "A", f"p{i:02d}") for i in range(20)]
    acts_m0, acts_m1 = {}, {}
    margins = {}
    for i, anchor in enumerate(anchors):
        base = rng.normal(size=(N_LAYERS, d_model))
        m1 = base + rng.normal(scale=0.1, size=(N_LAYERS, d_model))
        m1[planted_layer] += signal
        acts_m0[anchor["anchor_id"]] = base.astype(np.float16)
        acts_m1[anchor["anchor_id"]] = m1.astype(np.float16)
        margins[anchor["anchor_id"]] = -float(m1[planted_layer] @ (signal / np.linalg.norm(signal))) + rng.normal(scale=0.1)
    dev, hold = split_prompts(anchors, anchor_type="A")
    dev_m0, _ = stack_boundary(acts_m0, anchors, anchor_type="A", prompt_filter=dev)
    dev_m1, _ = stack_boundary(acts_m1, anchors, anchor_type="A", prompt_filter=dev)
    hold_m0, hold_ids = stack_boundary(acts_m0, anchors, anchor_type="A", prompt_filter=hold)
    hold_m1, _ = stack_boundary(acts_m1, anchors, anchor_type="A", prompt_filter=hold)
    directions = layer_directions(dev_m0, dev_m1)
    rows = evaluate_layers(
        directions=directions, holdout_m0=hold_m0, holdout_m1=hold_m1,
        holdout_ids=hold_ids, margins_m1=margins,
    )
    best = choose_layer(rows)
    assert best["layer"] == planted_layer
    assert best["holdout_auc"] > 0.9
    gate = gate_s2a(best)
    assert gate["passed"]


def test_paired_auc_values():
    assert paired_auc([0, 0, 0], [1, 1, 1]) == 1.0
    assert paired_auc([1, 1], [0, 0]) == 0.0
    assert paired_auc([0, 1], [0, 1]) == 0.5


# ---------------------------------------------------------------- SAE math
def small_sae():
    W_enc = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    W_dec = W_enc.T.copy()
    return {
        "W_enc": W_enc, "W_dec": W_dec,
        "b_enc": np.zeros(3), "b_dec": np.zeros(2), "k": 2, "layer": 0,
    }


def test_sae_encode_topk_and_relu():
    sae = small_sae()
    acts = sae_encode_np(np.array([[2.0, 1.0]]), sae)
    # pre = [2, 1, 3] -> top-2 keeps features 0 and 2
    assert acts.shape == (1, 3)
    assert acts[0].tolist() == [2.0, 0.0, 3.0]
    acts = sae_encode_np(np.array([[-1.0, 0.5]]), sae)
    # pre = [-1, .5, -.5] -> relu -> [0, .5, 0]
    assert acts[0].tolist() == [0.0, 0.5, 0.0]


def test_projection_and_differential_views():
    sae = small_sae()
    proj = projection_view(np.array([1.0, 0.0]), sae, top_k=2)
    assert proj[0]["feature_id"] == 0 and proj[0]["cos_with_d_stop"] == pytest.approx(1.0)
    m0 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m1 = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    diff = differential_view(m0, m1, top_k=2, min_active_share=0.5)
    assert diff[0]["feature_id"] == 2 and diff[0]["delta_m1_minus_m0"] == pytest.approx(2.0)
    assert diff[1]["feature_id"] == 0 and diff[1]["delta_m1_minus_m0"] == pytest.approx(-1.0)
    merged = merge_feature_views(proj, diff, cap=3)
    assert merged[0] == 0 and 2 in merged


def test_trajectories_and_label_order():
    sae = small_sae()
    h = {"M0": np.array([[1.0, 0.0]]), "ckpt46": np.array([[0.5, 0.0]]), "M1": np.array([[0.0, 0.0]])}
    rows = checkpoint_trajectories(h, sae, [0], np.array([1.0, 0.0]))
    by_label = {r["model_label"]: r for r in rows}
    assert by_label["M0"]["d_stop_projection_mean"] == pytest.approx(1.0)
    assert by_label["M1"]["feat_0_mean_act"] == pytest.approx(0.0)
    assert sort_labels(["M1", "ckpt46", "M0", "ckpt414"], m1_step=459) == ["M0", "ckpt46", "ckpt414", "M1"]


# ---------------------------------------------------------------- steering
def test_build_conditions_signs_and_norms():
    d = np.array([3.0, 4.0])
    m1 = build_conditions(d, model_tag="M1", alpha_grid=(2.0,), n_random=2)
    m0 = build_conditions(d, model_tag="M0", alpha_grid=(2.0,), n_random=2)
    assert m1[0]["condition"] == "baseline" and m1[0]["vector"] is None
    d_m1 = next(c for c in m1 if c["condition"] == "dstop_a2")
    d_m0 = next(c for c in m0 if c["condition"] == "dstop_a2")
    assert np.allclose(d_m1["vector"], -2.0 * d)
    assert np.allclose(d_m0["vector"], 2.0 * d)
    for cond in m1:
        if cond["family"] == "random":
            assert np.linalg.norm(cond["vector"]) == pytest.approx(2.0 * 5.0)
    r1 = [c["vector"] for c in m1 if c["family"] == "random"]
    r0 = [c["vector"] for c in m0 if c["family"] == "random"]
    assert np.allclose(r1[0], -r0[0])  # same seeded directions, opposite sign


# ---------------------------------------------------------------- report
def steer_row(anchor, condition, hazard, margin, prompt=None):
    return {
        "anchor_id": anchor, "anchor_type": "A", "prompt_id": prompt or anchor,
        "condition": condition, "stop_hazard": hazard,
        "margin_first_policy": margin, "margin_first_raw": margin * 0.9,
    }


def synth_steer_rows(n=16, d_effect=0.3, r_effect=0.05):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        anchor = f"A_{i:02d}"
        base_h, base_m = 0.05, -4.0
        rows.append(steer_row(anchor, "baseline", base_h, base_m))
        rows.append(steer_row(anchor, "dstop_a2", base_h + d_effect + rng.normal(0, 0.02), base_m + 3.0))
        for j in range(3):
            rows.append(steer_row(anchor, f"rand{j}_a2", base_h + r_effect + rng.normal(0, 0.02), base_m + 0.2))
    return rows






# ---------------------------------------------------------------- quality
