from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcr.events.audit import _motif_record, _signature_hash, audit_response
from tcr.events.block_parser import parse_relation_blocks
from tcr.events.io_utils import stable_json, write_json, write_jsonl
from tcr.events.motif_capture import find_motif_runs, primary_motif_event
from tcr.events.p0c_set_completion import (
    EVENT_CATEGORIES,
    EVENT_INDEX,
    GoldRecord,
    EventMeta,
    analyze_response,
    bootstrap_opportunity_rates,
    decision_from_results,
    load_gold_dataset,
    opportunity_contrasts,
)


def rel(source: str, target: str, relation: str, description: str = "d") -> dict[str, str]:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "description": description,
    }


def text_of(items) -> str:
    return json.dumps(items, ensure_ascii=False)


def make_meta(
    text: str,
    *,
    model: str = "M1",
    key: int = 1,
    seed: int = 0,
    hit_max: bool = False,
    legacy_structured_run=None,
) -> EventMeta:
    parsed = parse_relation_blocks(text)
    blocks = parsed.blocks
    raw_seen = {}
    first_seed = None
    for index, block in enumerate(blocks):
        if not block.identity_complete:
            continue
        if block.triple_signature in raw_seen:
            first_seed = index
            break
        raw_seen[block.triple_signature] = index
    runs = find_motif_runs(blocks, signature_kind="triple", min_repeats=3)
    semantic = [
        run for run in runs
        if all(all(value.strip() for value in signature) for signature in run.motif_signatures)
    ]
    primary = primary_motif_event(semantic, confirmation_repeats=3)
    capture = _motif_record(primary, blocks=blocks, confirmation_repeats=3)
    alignment = "no_legacy_orbit"
    evidence = {"matching_quad_run": None, "raw_onset_moved": False}
    legacy = {"exists": False, "hit_max_tokens": hit_max}
    if legacy_structured_run is not None:
        alignment = "multi_block_aligned"
        evidence = {
            "matching_quad_run": {
                "block_onset": legacy_structured_run.block_onset,
                "block_period": legacy_structured_run.block_period,
                "block_num_repeats": legacy_structured_run.block_num_repeats,
            },
            "raw_onset_moved": False,
        }
        legacy = {"exists": True, "hit_max_tokens": hit_max}
    return EventMeta(
        sample_id=f"{model}:{stable_json(key)}:{seed}",
        stable_prompt_id=stable_json(key),
        model_tag=model,
        key=key,
        key_id=stable_json(key),
        seed=seed,
        response_sha256=__import__("hashlib").sha256(text.encode()).hexdigest(),
        n_blocks=len(blocks),
        n_identity_complete_blocks=sum(block.identity_complete for block in blocks),
        block_triple_hashes=tuple(_signature_hash(block.triple_signature) for block in blocks),
        block_quad_hashes=tuple(_signature_hash(block.canonical_signature) for block in blocks),
        block_segments=tuple(block.sequence_segment for block in blocks),
        block_identity_complete=tuple(block.identity_complete for block in blocks),
        first_seed_index_0based=first_seed,
        capture=capture,
        legacy=legacy,
        alignment_type=alignment,
        alignment_evidence=evidence,
    )


def gold_record() -> GoldRecord:
    strict = {
        ("a", "b", "r1"),
        ("c", "d", "r2"),
    }
    return GoldRecord(
        key_id=stable_json(1),
        gold_blocks=5,
        strict_triples=frozenset(strict),
        relaxed_pairs=frozenset((s, t) for s, t, _r in strict),
        n_raw_relations=2,
        n_empty_identity=0,
    )


def test_block_classification_and_seed_state():
    text = text_of(
        [
            rel("A", "B", "R1"),                 # strict
            rel("C", "D", "wrong"),              # relaxed pair
            rel("A", "B", "R1"),                 # seen reuse / first seed
            rel("X", "Y", "z"),                  # unmatched
            rel("", "", ""),                     # invalid identity
        ]
    )
    analysis, cases = analyze_response(text=text, meta=make_meta(text), gold=gold_record())
    counts = np.asarray(analysis.opportunity_counts)
    assert counts[:, EVENT_INDEX["new_strict_gold"]].sum() == 1
    assert counts[:, EVENT_INDEX["new_relaxed_pair_only"]].sum() == 1
    assert counts[:, EVENT_INDEX["seen_triple_reuse"]].sum() == 1
    assert counts[:, EVENT_INDEX["new_unmatched"]].sum() == 1
    assert counts[:, EVENT_INDEX["invalid_identity"]].sum() == 1
    assert counts[:, EVENT_INDEX["normal_stop"]].sum() == 1
    utility = np.asarray(analysis.utility_counts)
    assert utility[:, 0].sum() == 1  # newly covered strict triple
    assert utility[:, 1].sum() == 2  # both strict and pair-only blocks add relaxed coverage
    assert analysis.first_seed_progress == 3 / 5
    assert analysis.first_seed_strict_coverage == 0.5
    assert analysis.first_seed_relaxed_coverage == 1.0
    assert analysis.no_new_relaxed_after_first_seed is True
    assert cases == []



def test_relaxed_utility_counts_unique_pair_once_even_with_multiple_gold_relations():
    text = text_of([rel("A", "B", "R1"), rel("A", "B", "R2")])
    gold = GoldRecord(
        key_id=stable_json(1),
        gold_blocks=2,
        strict_triples=frozenset({("a", "b", "r1"), ("a", "b", "r2")}),
        relaxed_pairs=frozenset({("a", "b")}),
        n_raw_relations=2,
        n_empty_identity=0,
    )
    analysis, _ = analyze_response(text=text, meta=make_meta(text), gold=gold)
    utility = np.asarray(analysis.utility_counts)
    assert utility[:, 0].sum() == 2  # two distinct strict triples
    assert utility[:, 1].sum() == 1  # one unique relaxed entity pair
    assert analysis.final_strict_coverage == 1.0
    assert analysis.final_relaxed_coverage == 1.0


def test_gold_pair_already_covered_variant_is_not_new_relaxed_utility():
    text = text_of([rel("A", "B", "wrong-1"), rel("A", "B", "wrong-2")])
    gold = GoldRecord(
        key_id=stable_json(1),
        gold_blocks=2,
        strict_triples=frozenset({("a", "b", "gold")}),
        relaxed_pairs=frozenset({("a", "b")}),
        n_raw_relations=1,
        n_empty_identity=0,
    )
    analysis, _ = analyze_response(text=text, meta=make_meta(text), gold=gold)
    counts = np.asarray(analysis.opportunity_counts)
    utility = np.asarray(analysis.utility_counts)
    assert counts[:, EVENT_INDEX["new_relaxed_pair_only"]].sum() == 1
    assert counts[:, EVENT_INDEX["gold_pair_already_covered_variant"]].sum() == 1
    assert utility[:, 1].sum() == 1

def test_normalized_reuse_can_precede_frozen_raw_reuse():
    text = text_of([rel(" A ", "B", "R"), rel("a", "b", "r")])
    meta = make_meta(text)
    assert meta.first_seed_index_0based is None  # raw exact signatures differ
    gold = GoldRecord(
        key_id=stable_json(1),
        gold_blocks=2,
        strict_triples=frozenset({("a", "b", "r")}),
        relaxed_pairs=frozenset({("a", "b")}),
        n_raw_relations=1,
        n_empty_identity=0,
    )
    analysis, _ = analyze_response(text=text, meta=meta, gold=gold)
    assert analysis.normalized_first_reuse_index_0based == 1
    assert analysis.frozen_seed_matches_normalized_first is False


def test_gold_pair_variant_and_overlapping_utility_are_distinct():
    text = text_of(
        [
            rel("A", "B", "wrong-1"),
            rel("A", "B", "wrong-2"),
            rel("A", "B", "R1"),
        ]
    )
    gold = GoldRecord(
        key_id=stable_json(1),
        gold_blocks=3,
        strict_triples=frozenset({("a", "b", "r1")}),
        relaxed_pairs=frozenset({("a", "b")}),
        n_raw_relations=1,
        n_empty_identity=0,
    )
    analysis, _ = analyze_response(text=text, meta=make_meta(text), gold=gold)
    counts = np.asarray(analysis.opportunity_counts)
    utility = np.asarray(analysis.utility_counts)
    assert counts[:, EVENT_INDEX["new_relaxed_pair_only"]].sum() == 1
    assert counts[:, EVENT_INDEX["gold_pair_already_covered_variant"]].sum() == 1
    assert counts[:, EVENT_INDEX["new_strict_gold"]].sum() == 1
    assert utility[:, 0].sum() == 1
    assert utility[:, 1].sum() == 1  # pair coverage is not counted again


def test_empty_gold_identity_is_audited_but_not_valid_utility(tmp_path: Path):
    path = tmp_path / "gold.jsonl"
    write_jsonl(
        path,
        [
            {
                "index": 0,
                "n_output_dicts": 2,
                "output": [rel("A", "B", "R1"), rel("", "", "")],
            }
        ],
    )
    gold = load_gold_dataset(path, [0])
    record = gold.records[stable_json(0)]
    assert record.gold_blocks == 2
    assert record.n_empty_identity == 1
    assert record.strict_triples == frozenset({("a", "b", "r1")})
    assert record.relaxed_pairs == frozenset({("a", "b")})


def test_capture_is_attributed_to_later_reuse_seed():
    items = [
        rel("X", "0", "rx"),
        rel("A", "1", "ra"),
        rel("Y", "2", "ry"),
        rel("X", "0", "rx"),       # reuse ordinal 1, no capture
        rel("A", "1", "ra"),       # reuse ordinal 2
        rel("B", "2", "rb"),
        rel("C", "3", "rc"),
        rel("A", "1", "ra"),       # capture second-copy start, ordinal 3
        rel("B", "2", "rb"),
        rel("C", "3", "rc"),
        rel("A", "1", "ra"),
        rel("B", "2", "rb"),
        rel("C", "3", "rc"),
    ]
    text = text_of(items)
    parsed = parse_relation_blocks(text)
    quad_runs = find_motif_runs(parsed.blocks, signature_kind="quad", min_repeats=3)
    run = next(run for run in quad_runs if run.block_onset == 4 and run.block_period == 3)
    meta = make_meta(text, hit_max=True, legacy_structured_run=run)
    strict = {tuple(value.lower() for value in item.values())[:3] for item in items[:7]}
    gold = GoldRecord(
        key_id=stable_json(1),
        gold_blocks=7,
        strict_triples=frozenset(strict),
        relaxed_pairs=frozenset((s, t) for s, t, _r in strict),
        n_raw_relations=7,
        n_empty_identity=0,
    )
    analysis, cases = analyze_response(text=text, meta=meta, gold=gold)
    assert analysis.semantic_capture is True
    assert analysis.capture_seed_ordinal == 3
    assert analysis.legacy_capture_seed_ordinal == 3
    assert {case.kind for case in cases} == {
        "primary_semantic_capture",
        "legacy_aligned_capture",
    }


def test_gold_auto_mapping_uses_line_1based(tmp_path: Path):
    path = tmp_path / "gold.jsonl"
    write_jsonl(
        path,
        [
            {"index": 0, "n_output_dicts": 1, "output": [rel("A", "B", "r")]},
            {"index": 1, "n_output_dicts": 1, "output": [rel("C", "D", "r")]},
        ],
    )
    gold = load_gold_dataset(path, [1, 2])
    assert gold.key_field == "line_1based"
    assert set(gold.records) == {stable_json(1), stable_json(2)}


def test_bootstrap_opportunity_contrasts_use_paired_prompt_draws():
    # Shape: prompt, model, bin, event.  Construct deterministic paired counts.
    counts = np.zeros((4, 2, 6, len(EVENT_CATEGORIES)), dtype=np.int64)
    # Pre range: both models mostly new-valid.
    counts[:, :, 1, EVENT_INDEX["new_strict_gold"]] = 8
    counts[:, :, 1, EVENT_INDEX["seen_triple_reuse"]] = 2
    # Post: M1 loses utility, gains reuse, and stops less.
    counts[:, 0, 3, EVENT_INDEX["new_strict_gold"]] = 4
    counts[:, 0, 3, EVENT_INDEX["seen_triple_reuse"]] = 2
    counts[:, 0, 3, EVENT_INDEX["normal_stop"]] = 4
    counts[:, 1, 3, EVENT_INDEX["new_strict_gold"]] = 1
    counts[:, 1, 3, EVENT_INDEX["seen_triple_reuse"]] = 8
    counts[:, 1, 3, EVENT_INDEX["normal_stop"]] = 1
    result = bootstrap_opportunity_rates(counts, n_bootstrap=200, random_seed=7, batch_size=50)
    contrasts = {row["contrast"]: row for row in opportunity_contrasts(result)}
    assert contrasts["M1_post_ge1_minus_pre_0.5_1.0::new_valid_any"]["ci_high"] < 0
    assert contrasts["M1_post_ge1_minus_pre_0.5_1.0::seen_triple_reuse"]["ci_low"] > 0


def test_end_to_end_cli(tmp_path: Path):
    result = tmp_path / "result"
    result.mkdir()
    responses = {}
    event_rows = []
    gold_rows = []
    for key in (1, 2):
        gold_items = [rel(f"A{key}", f"B{key}", "r"), rel(f"C{key}", f"D{key}", "r")]
        gold_rows.append({"index": key - 1, "n_output_dicts": 2, "output": gold_items})
        for model in ("M0", "M1"):
            items = gold_items if model == "M0" else gold_items + [gold_items[0], gold_items[0], gold_items[0]]
            text = text_of(items)
            response_row = responses.setdefault(model, {"key": key, "responses": []})
            response_row["responses"].append(
                {"seed": 0, "response": text, "reasoning": "", "finish_reason": "stop"}
            )
            token_ids = list(range(len(text)))
            offsets = [(index, index + 1) for index in range(len(text))]
            legacy_sample = {
                "seed": 0,
                "loop": 0,
                "hit_max_tokens": 0,
                "gen_len": len(token_ids),
            }
            event_rows.append(
                audit_response(
                    model_tag=model,
                    key=key,
                    seed=0,
                    text=text,
                    token_ids=token_ids,
                    offsets=offsets,
                    legacy_sample=legacy_sample,
                    selection_roles=["prevalence"],
                    capture_min_repeats=3,
                    max_motif_period_blocks=None,
                    legacy_min_repeats=50,
                )
            )
    response_paths = {}
    for model in ("M0", "M1"):
        path = tmp_path / f"{model}.jsonl"
        write_jsonl(path, [responses[model]])
        # Above dictionary only holds key=2 due setdefault design; rewrite full rows.
        rows = []
        for key in (1, 2):
            matching = [
                row for row in event_rows if row["model_tag"] == model and row["key"] == key
            ][0]
            # Recover exact text from the audit row's source construction.
            gold_items = [rel(f"A{key}", f"B{key}", "r"), rel(f"C{key}", f"D{key}", "r")]
            items = gold_items if model == "M0" else gold_items + [gold_items[0], gold_items[0], gold_items[0]]
            rows.append({"key": key, "responses": [{"seed": 0, "response": text_of(items), "reasoning": ""}]})
        write_jsonl(path, rows)
        response_paths[model] = path
    gold_path = tmp_path / "gold.jsonl"
    write_jsonl(gold_path, gold_rows)
    write_jsonl(result / "event_rows.jsonl", event_rows)
    selection = {"mode": "all"}
    write_json(result / "selection_manifest.json", selection)
    write_json(
        result / "run_manifest.json",
        {
            "protocol": {"target": "answer", "raw_onset_moved": False},
            "selection": {"mode": "all"},
            "inputs": {
                "responses_m0": str(response_paths["M0"]),
                "responses_m1": str(response_paths["M1"]),
            },
        },
    )
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "4_analysis" / "analyze_p0c_set_completion.py"),
            "--result-dir",
            str(result),
            "--gold-data",
            str(gold_path),
            "--output-dir",
            str(out),
            "--bootstrap",
            "100",
            "--bootstrap-batch-size",
            "25",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (out / "P0C_COMPACT_RETURN.txt").exists()
    assert (out / "P0C_REPORT.md").exists()
    assert len((out / "P0C_COMPACT_RETURN.txt").read_text(encoding="utf-8")) <= 1950
