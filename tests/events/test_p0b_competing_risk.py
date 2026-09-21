from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tcr.events.io_utils import sha256_text, stable_json
from tcr.events.p0b_competing_risk import (
    GoldMapping,
    aggregate_competing,
    build_terminal_records,
    competing_decomposition,
    load_gold_mapping,
    make_progress_grid,
    prompt_competing_arrays,
    trace_first_seed_lineage,
)


def block(index: int, triple: str, *, segment: int = 0, complete: bool = True):
    return {
        "block_index_0based": index,
        "block_index_1based": index + 1,
        "sequence_segment": segment,
        "identity_complete": complete,
        "triple_hash": sha256_text(triple),
        "quad_hash": sha256_text(triple + ":q"),
        "token_start": index * 10,
        "token_end": (index + 1) * 10,
    }


def event_row(
    *,
    model: str,
    key: int,
    seed: int,
    triples: list[str],
    previous: int | None = None,
    current: int | None = None,
    hit_max: bool = False,
):
    exists = previous is not None and current is not None
    reuse = {"exists": False}
    if exists:
        reuse = {
            "exists": True,
            "previous_block_index_0based": previous,
            "block_index_0based": current,
            "block_index_1based": current + 1,
            "occurrence_number": 2,
        }
    return {
        "sample_id": f"{model}:{key}:{seed}",
        "stable_prompt_id": stable_json(key),
        "model_tag": model,
        "key": key,
        "seed": seed,
        "selection_roles": ["prevalence"],
        "n_blocks": len(triples),
        "block_index": [block(i, value) for i, value in enumerate(triples)],
        "first_nonempty_triple_reuse": reuse,
        "motif_capture_triple": {"exists": False},
        "legacy_orbit": {"exists": False, "hit_max_tokens": hit_max},
    }


def test_gold_mapping_prefers_index(tmp_path: Path):
    path = tmp_path / "gold.jsonl"
    rows = [
        {"index": 10, "n_output_dicts": 3},
        {"index": 11, "n_output_dicts": 4},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    mapping = load_gold_mapping(path, [10, 11])
    assert mapping.key_field == "index"
    assert mapping.values[stable_json(10)] == 3


def test_terminal_stop_is_competing_event_at_next_opportunity():
    rows = [
        event_row(model="M0", key=1, seed=0, triples=["A", "B"], hit_max=False),
        event_row(model="M1", key=1, seed=0, triples=["A", "B", "A"], previous=0, current=2),
    ]
    gold = GoldMapping("index", 1, {stable_json(1): 4})
    records = build_terminal_records(rows, gold)
    m0 = next(record for record in records if record.model_tag == "M0")
    m1 = next(record for record in records if record.model_tag == "M1")
    assert m0.event_kind == "normal_stop"
    assert m0.event_block == 3
    assert m0.event_progress == 0.75
    assert m1.event_kind == "seed"
    assert m1.event_block == 3


def test_lineage_tracks_same_motif_three_copies():
    # First copy A B, first seed starts the second A B, then a third A B.
    row = event_row(
        model="M1",
        key=1,
        seed=0,
        triples=["A", "B", "A", "B", "A", "B", "C"],
        previous=0,
        current=2,
    )
    record = trace_first_seed_lineage(row, gold_blocks=6)
    assert record.eligible
    assert record.motif_period_blocks == 2
    assert record.second_status == "completed"
    assert record.third_status == "completed"
    assert record.lineage_capture
    assert record.second_confirmation_block_1based == 4
    assert record.third_confirmation_block_1based == 6


def test_lineage_distinguishes_recovery_from_censor():
    mismatch = event_row(
        model="M0",
        key=1,
        seed=0,
        triples=["A", "B", "A", "X"],
        previous=0,
        current=2,
    )
    stopped = event_row(
        model="M0",
        key=1,
        seed=1,
        triples=["A", "B", "A"],
        previous=0,
        current=2,
        hit_max=False,
    )
    censored = event_row(
        model="M0",
        key=1,
        seed=2,
        triples=["A", "B", "A"],
        previous=0,
        current=2,
        hit_max=True,
    )
    assert trace_first_seed_lineage(mismatch, 4).second_status == "mismatch"
    assert trace_first_seed_lineage(stopped, 4).second_status == "normal_stop"
    assert trace_first_seed_lineage(censored, 4).second_status == "admin_censor"


def test_competing_risk_separates_seed_and_stop_hazards():
    rows = []
    # One prompt, two seeds/model. M0 stops early; M1 continues and seeds.
    rows.extend(
        [
            event_row(model="M0", key=1, seed=0, triples=["A"]),
            event_row(model="M0", key=1, seed=1, triples=["A"]),
            event_row(model="M1", key=1, seed=0, triples=["A", "B", "A"], previous=0, current=2),
            event_row(model="M1", key=1, seed=1, triples=["A", "B", "C"]),
        ]
    )
    gold = GoldMapping("index", 1, {stable_json(1): 2})
    terminals = build_terminal_records(rows, gold)
    edges = make_progress_grid(2.0, 0.5)
    risk, seed, stop = prompt_competing_arrays(terminals, [stable_json(1)], edges)
    result = aggregate_competing(risk, seed, stop)
    assert result["cif_stop"][0] > result["cif_stop"][1]
    assert result["cif_seed"][1] > result["cif_seed"][0]
    assert result["stop_exposure_component"] > 0
    assert abs(result["identity_error"]) < 1e-12


def test_shapley_components_sum_to_total():
    hs0 = np.array([0.05, 0.05, 0.02])
    ht0 = np.array([0.20, 0.20, 0.20])
    hs1 = np.array([0.06, 0.06, 0.03])
    ht1 = np.array([0.10, 0.10, 0.10])
    result = competing_decomposition(hs0, ht0, hs1, ht1)
    assert np.isclose(
        result["seed_hazard_component"] + result["stop_exposure_component"],
        result["total_diff"],
    )
    assert result["seed_hazard_component"] > 0
    assert result["stop_exposure_component"] > 0
