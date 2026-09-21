from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from tcr.events.io_utils import stable_json
from tcr.events.p0d_episode_hazard import (
    BLOCK_EMPTY,
    BLOCK_NOVEL,
    BLOCK_REUSE,
    analyze_sequences,
    assign_outcome,
    build_slot_arrays,
    bootstrap_episode_analysis,
    classify_blocks,
    descriptive_episode_summary,
    distance_rows,
    episode_count_rows,
    episode_curve_rows,
    extract_block_sequence,
    response_metric_arrays,
    segment_episodes,
    validate_capture_identity,
)

ROOT = Path(__file__).resolve().parents[2]

EMPTY = "!"  # marker letter for identity-incomplete blocks
SEGBREAK = "|"  # marker for a sequence-segment increment before the next block


def _blocks(spec: str) -> tuple[list[dict], int]:
    """Build block_index entries from a compact spec like 'AB|BA!'.

    Letters are triple identities (quad follows the letter), '!' after a letter
    marks it identity-incomplete, '|' increments the sequence segment.
    """
    entries: list[dict] = []
    segment = 0
    i = 0
    while i < len(spec):
        ch = spec[i]
        if ch == SEGBREAK:
            segment += 1
            i += 1
            continue
        incomplete = i + 1 < len(spec) and spec[i + 1] == EMPTY
        entries.append(
            {
                "triple_hash": f"hash-{ch}",
                "quad_hash": f"quad-{ch}",
                "sequence_segment": segment,
                "identity_complete": not incomplete,
            }
        )
        i += 2 if incomplete else 1
    n_identity = sum(entry["identity_complete"] for entry in entries)
    return entries, n_identity


def _first_reuse_index(entries: list[dict]) -> int | None:
    seen: dict[str, int] = {}
    for index, entry in enumerate(entries):
        if not entry["identity_complete"]:
            continue
        signature = entry["triple_hash"]
        if signature in seen:
            return index
        seen[signature] = index
    return None


def make_row(
    spec: str,
    *,
    model: str = "M1",
    key: str = "k0",
    seed: int = 0,
    capture: tuple[int, int] | None = None,  # (onset, period)
    hit_max: bool = False,
    orbit: bool = False,
    alignment_type: str = "no_legacy_orbit",
    matching: tuple[int, int] | None = None,  # (block_onset, block_period)
    first_seed_override: int | None | str = "auto",
) -> dict:
    entries, n_identity = _blocks(spec)
    first = _first_reuse_index(entries) if first_seed_override == "auto" else first_seed_override
    if capture is not None:
        onset, period = capture
        capture_record = {
            "exists": True,
            "block_onset_0based": onset,
            "block_period": period,
            "second_copy_start_0based": onset + period,
            "confirmed_at_block_exclusive_0based": onset + 3 * period,
            "motif_triples": [["s", "t", "r"] for _ in range(period)],
        }
    else:
        capture_record = {"exists": False}
    evidence = {}
    if matching is not None:
        evidence["matching_quad_run"] = {
            "block_onset": matching[0],
            "block_period": matching[1],
            "block_num_repeats": 3,
        }
    return {
        "sample_id": f"{model}:{key}:{seed}",
        "stable_prompt_id": stable_json(key),
        "model_tag": model,
        "key": key,
        "seed": seed,
        "selection_roles": ["prevalence"],
        "response_sha256": f"sha-{model}-{key}-{seed}",
        "n_blocks": len(entries),
        "n_identity_complete_blocks": n_identity,
        "block_index": entries,
        "first_nonempty_triple_reuse": (
            {"exists": True, "block_index_0based": first} if first is not None else {"exists": False}
        ),
        "motif_capture_triple": capture_record,
        "legacy_orbit": {"exists": orbit, "hit_max_tokens": hit_max},
        "alignment_type": alignment_type,
        "alignment_evidence": evidence,
    }


def seq_of(spec: str, **kwargs):
    return extract_block_sequence(make_row(spec, **kwargs))


def episode_spans(spec: str, *, variant: str = "lineage", **kwargs):
    seq = seq_of(spec, **kwargs)
    kinds, prev = classify_blocks(seq)
    episodes = segment_episodes(seq, kinds, prev, variant=variant)
    return [(ep.start, ep.end, ep.closure_reason) for ep in episodes], kinds


def test_classify_blocks_and_frozen_seed_validation():
    seq = seq_of("ABA")
    kinds, prev = classify_blocks(seq)
    assert kinds == [BLOCK_NOVEL, BLOCK_NOVEL, BLOCK_REUSE]
    assert prev == [None, None, 0]

    with pytest.raises(ValueError, match="frozen seed"):
        classify_blocks(seq_of("ABA", first_seed_override=None))
    with pytest.raises(ValueError, match="frozen seed"):
        classify_blocks(seq_of("ABC", first_seed_override=1))


def test_empty_blocks_never_reuse_and_separate_episodes():
    # A A !A A : empty block closes the first episode; both A-reuses are events.
    spans, kinds = episode_spans("AAA!A")
    assert kinds == [BLOCK_NOVEL, BLOCK_REUSE, BLOCK_EMPTY, BLOCK_REUSE]
    assert spans == [(1, 1, "empty_block"), (3, 3, "response_end")]


def test_novel_and_segment_separators():
    spans, _ = episode_spans("ABCABCD")
    assert spans == [(3, 5, "novel_block")]
    spans, _ = episode_spans("AA|A")
    assert spans == [(1, 1, "segment_break"), (2, 2, "response_end")]


def test_lineage_split_motif_switch_and_relinking():
    # P Q R A A A A P Q R P Q R
    spans, _ = episode_spans("PQRAAAAPQRPQR")
    assert spans == [(4, 6, "motif_switch"), (7, 12, "response_end")]
    # Contiguous sensitivity variant merges the whole reuse run.
    spans, _ = episode_spans("PQRAAAAPQRPQR", variant="contiguous")
    assert spans == [(4, 12, "response_end")]


def test_lineage_hash_membership_keeps_cycle_together():
    # A B A B A B : period-2 cycle stays one episode (lag constant at 2).
    spans, _ = episode_spans("ABABAB", capture=(0, 2))
    assert spans == [(2, 5, "response_end")]


def test_outcome_capture_truncation_ordinal_and_distance():
    seq = seq_of("AAXBBB", capture=(3, 1))
    kinds, prev = classify_blocks(seq)
    episodes = segment_episodes(seq, kinds, prev, variant="lineage")
    outcome = assign_outcome(
        seq, kinds, episodes, capture_kind="primary", episode_variant="lineage"
    )
    assert outcome.terminal_event == "capture"
    assert outcome.captured_episode_ordinal == 2
    assert (outcome.terminal_stage, outcome.terminal_ordinal) == ("episode", 2)
    assert outcome.recovered_episodes == 1
    assert outcome.n_episodes_experienced == 2
    assert outcome.captured_episode_start == 4
    assert outcome.last_new_triple_before_capture == 3
    assert outcome.distance_last_new_triple_blocks == 1
    assert outcome.episodes_between_last_new_and_capture == 0
    assert outcome.capture_is_first_episode_after_last_new is True
    assert outcome.post_capture_novel_blocks == 0
    assert outcome.closure_reasons[-1] == "capture_truncation"


def test_outcome_drops_post_capture_episodes_and_counts_novelty():
    # Capture in the first episode, then recovery and a later reuse episode.
    seq = seq_of("AAAAXBYB", capture=(0, 1), first_seed_override=1)
    kinds, prev = classify_blocks(seq)
    episodes = segment_episodes(seq, kinds, prev, variant="lineage")
    assert [(ep.start, ep.end) for ep in episodes] == [(1, 3), (7, 7)]
    outcome = assign_outcome(
        seq, kinds, episodes, capture_kind="primary", episode_variant="lineage"
    )
    assert outcome.terminal_event == "capture"
    assert outcome.captured_episode_ordinal == 1
    assert outcome.n_episodes_experienced == 1
    assert outcome.post_capture_novel_blocks == 3  # X(4), B(5), Y(6); B at 7 is a reuse


def test_outcome_stop_and_censor_slots():
    def outcome_for(spec: str, **kwargs):
        seq = seq_of(spec, **kwargs)
        kinds, prev = classify_blocks(seq)
        episodes = segment_episodes(seq, kinds, prev, variant="lineage")
        return assign_outcome(
            seq, kinds, episodes, capture_kind="primary", episode_variant="lineage"
        )

    o = outcome_for("ABC")
    assert (o.terminal_event, o.terminal_stage, o.terminal_ordinal) == ("stop", "gap", 1)
    assert o.stop_location == "between_episodes"
    assert o.n_episodes_experienced == 0

    o = outcome_for("ABC", hit_max=True, orbit=True, alignment_type="non_structured")
    assert (o.terminal_event, o.terminal_stage, o.terminal_ordinal) == ("censor", "gap", 1)

    o = outcome_for("AA")
    assert (o.terminal_event, o.terminal_stage, o.terminal_ordinal) == ("stop", "episode", 1)
    assert o.stop_location == "in_episode"
    assert o.recovered_episodes == 0

    o = outcome_for("AAB")
    assert (o.terminal_event, o.terminal_stage, o.terminal_ordinal) == ("stop", "gap", 2)
    assert o.recovered_episodes == 1

    o = outcome_for("")
    assert (o.terminal_event, o.terminal_stage, o.terminal_ordinal) == ("stop", "gap", 1)
    assert o.n_episodes_experienced == 0


def test_capture_coordinate_validation_fails_on_broken_motif():
    with pytest.raises(ValueError, match="three exact motif copies"):
        seq_of("AABAB", capture=(0, 1))
    with pytest.raises(ValueError, match="capture confirmation"):
        seq_of("AAA", capture=(0, 2))


def test_legacy_mapping_and_unmapped_orbit():
    seq = seq_of(
        "ABABAB",
        capture=(0, 2),
        orbit=True,
        hit_max=True,
        alignment_type="multi_block_aligned",
        matching=(0, 2),
    )
    assert seq.legacy_capture_start == 2
    assert seq.primary_capture_start == 2
    assert not seq.legacy_orbit_unmapped

    kinds, prev = classify_blocks(seq)
    episodes = segment_episodes(seq, kinds, prev, variant="lineage")
    legacy_outcome = assign_outcome(
        seq, kinds, episodes, capture_kind="legacy", episode_variant="lineage"
    )
    assert legacy_outcome.terminal_event == "capture"
    assert legacy_outcome.captured_episode_ordinal == 1

    unmapped = seq_of("AAB", orbit=True, hit_max=False, alignment_type="non_structured")
    assert unmapped.legacy_capture_start is None
    assert unmapped.legacy_orbit_unmapped
    kinds, prev = classify_blocks(unmapped)
    episodes = segment_episodes(unmapped, kinds, prev, variant="lineage")
    outcome = assign_outcome(
        unmapped, kinds, episodes, capture_kind="legacy", episode_variant="lineage"
    )
    assert outcome.terminal_event == "censor"  # cannot rule capture in or out
    primary = assign_outcome(
        unmapped, kinds, episodes, capture_kind="primary", episode_variant="lineage"
    )
    assert primary.terminal_event == "stop"


def test_pair_distance_uses_supplied_flags():
    seq = seq_of("AAXBBB", capture=(3, 1))
    kinds, prev = classify_blocks(seq)
    episodes = segment_episodes(seq, kinds, prev, variant="lineage")
    flags = [True, False, True, False, False, False]  # last new pair at block 2
    outcome = assign_outcome(
        seq,
        kinds,
        episodes,
        capture_kind="primary",
        episode_variant="lineage",
        pair_first_flags=flags,
    )
    assert outcome.last_new_pair_before_capture == 2
    assert outcome.distance_last_new_pair_blocks == 2


def test_slot_arrays_hazards_and_shapley_identity():
    rows = [
        make_row("AAXBBB", model="M0", key="p1", capture=(3, 1)),  # capture at slot 2
        make_row("AAB", model="M1", key="p1"),  # stop at slot 2
        make_row("ABC", model="M0", key="p2"),  # stop at slot 1
        make_row("CCC", model="M1", key="p2", capture=(0, 1)),  # capture at slot 1
    ]
    sequences = [extract_block_sequence(row) for row in rows]
    outcomes = analyze_sequences(sequences, variant="lineage", capture_kind="primary")
    validate_capture_identity(sequences, outcomes)
    prompt_ids = sorted({seq.prompt_id for seq in sequences})
    risk, capture, stop = build_slot_arrays(outcomes, prompt_ids)
    # Interleaved axis: [G1, E1, G2, E2]; capture at episode 2 -> index 3.
    assert risk.shape[-1] == 4
    # M0: capture at E2 (at risk G1,E1,G2,E2) + gap-stop at G1.
    assert np.sum(risk, axis=0)[0].tolist() == [2.0, 1.0, 1.0, 1.0]
    assert np.sum(capture, axis=0)[0].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert np.sum(stop, axis=0)[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    # M1: gap-stop at G2 + capture at E1.
    assert np.sum(risk, axis=0)[1].tolist() == [2.0, 2.0, 1.0, 0.0]
    assert np.sum(capture, axis=0)[1].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert np.sum(stop, axis=0)[1].tolist() == [0.0, 0.0, 1.0, 0.0]

    sums, counts, names = response_metric_arrays(outcomes, prompt_ids)
    results = bootstrap_episode_analysis(
        risk=risk,
        capture=capture,
        stop=stop,
        metric_sums=sums,
        metric_counts=counts,
        metric_names=names,
        n_bootstrap=100,
        random_seed=7,
        batch_size=50,
        tail_pool_min=1,
    )
    slot = results["slot"]
    assert abs(slot["identity_error"]["value"]) < 1e-9
    reconstructed = slot["propensity_component"]["value"] + slot["exposure_component"]["value"]
    assert math.isclose(reconstructed, slot["decomposition_total"]["value"], abs_tol=1e-9)
    assert math.isclose(
        slot["decomposition_total"]["value"],
        slot["cif_capture_final"]["m1"] - slot["cif_capture_final"]["m0"],
        abs_tol=1e-9,
    )
    assert results["response"]["capture_rate"]["m0"] == 0.5
    assert results["response"]["capture_rate"]["m1"] == 0.5
    # Conditional per-episode hazards: M0 1 capture / 2 episodes started,
    # M1 1 capture / 2 episodes started.
    assert results["slot"]["pooled_capture_hazard"]["m0"] == 0.5
    assert results["slot"]["pooled_capture_hazard"]["m1"] == 0.5
    assert results["kmax"] == 2

    curve = episode_curve_rows(risk, capture, stop, variant="lineage", capture_kind="primary")
    k1_m0 = next(r for r in curve if r["model"] == "M0" and r["episode_ordinal"] == 1)
    assert k1_m0["gap_at_risk"] == 2 and k1_m0["gap_stop_events"] == 1
    assert k1_m0["episodes_started"] == 1 and k1_m0["capture_events"] == 0
    counts_rows = episode_count_rows(outcomes, variant="lineage", capture_kind="primary")
    assert sum(row["n_responses"] for row in counts_rows if row["model"] == "M0") == 2
    desc = descriptive_episode_summary(outcomes)
    assert desc["M0"]["n_responses"] == 2
    dist = distance_rows(outcomes, variant="lineage", capture_kind="primary")
    m0_dist = next(r for r in dist if r["model"] == "M0")
    assert m0_dist["n_captured"] == 1
    assert m0_dist["dist_last_new_triple_blocks_median"] == 1.0


def test_equal_conditional_propensity_yields_zero_propensity_component():
    """Ground truth: per-slot h(capture|episode) equal for M0/M1, only the
    stop hazard differs.  The Shapley propensity component must be exactly 0 —
    this is the confound the interleaved gap/episode axis exists to remove."""
    m0_specs = (
        [("ABC", None)] * 4
        + [("ABBB", (1, 1))] * 2
        + [("AAB", None)]
        + [("AAXBBB", (3, 1))]
    )
    m1_specs = [("ABBB", (1, 1))] * 4 + [("AAXBBB", (3, 1))] * 4
    rows = []
    for index, ((spec0, cap0), (spec1, cap1)) in enumerate(zip(m0_specs, m1_specs)):
        rows.append(make_row(spec0, model="M0", key=f"p{index}", capture=cap0))
        rows.append(make_row(spec1, model="M1", key=f"p{index}", capture=cap1))
    sequences = [extract_block_sequence(row) for row in rows]
    outcomes = analyze_sequences(sequences, variant="lineage", capture_kind="primary")
    prompt_ids = sorted({seq.prompt_id for seq in sequences})
    risk, capture, stop = build_slot_arrays(outcomes, prompt_ids)
    sums, counts, names = response_metric_arrays(outcomes, prompt_ids)
    results = bootstrap_episode_analysis(
        risk=risk,
        capture=capture,
        stop=stop,
        metric_sums=sums,
        metric_counts=counts,
        metric_names=names,
        n_bootstrap=100,
        random_seed=11,
        batch_size=50,
        tail_pool_min=1,
    )
    slot = results["slot"]
    # Per-slot conditional capture hazards: E1=0.5, E2=1.0 for both models.
    assert slot["hazard_k1"]["m0"] == pytest.approx(0.5)
    assert slot["hazard_k1"]["m1"] == pytest.approx(0.5)
    assert slot["hazard_k2"]["m0"] == pytest.approx(1.0)
    assert slot["hazard_k2"]["m1"] == pytest.approx(1.0)
    # Stop hazard differs: M0 0.5 per gap, M1 0.
    assert slot["pooled_gap_stop_hazard"]["m0"] == pytest.approx(0.5)
    assert slot["pooled_gap_stop_hazard"]["m1"] == pytest.approx(0.0)
    # CIFs and the decomposition: the entire difference is exposure.
    assert slot["cif_capture_final"]["m0"] == pytest.approx(0.375)
    assert slot["cif_capture_final"]["m1"] == pytest.approx(1.0)
    assert slot["propensity_component"]["value"] == pytest.approx(0.0, abs=1e-12)
    assert slot["exposure_component"]["value"] == pytest.approx(0.625, abs=1e-12)
    assert slot["decomposition_total"]["value"] == pytest.approx(0.625, abs=1e-12)


def test_reparse_pair_novelty_against_real_parser(tmp_path):
    from tcr.events.audit import _signature_hash
    from tcr.events.block_parser import parse_relation_blocks
    from tcr.events.io_utils import sha256_text
    from tcr.events.p0d_episode_hazard import reparse_pair_novelty

    def block(source, target, relation, description="d"):
        return (
            '{"source": "%s", "target": "%s", "relation": "%s", "description": "%s"}'
            % (source, target, relation, description)
        )

    # new pair | same pair new relation (new triple) | new pair | raw triple reuse
    text = "[" + ", ".join(
        [
            block("s1", "t1", "r1"),
            block("s1", "t1", "r2"),
            block("s2", "t2", "r1"),
            block("s1", "t1", "r1"),
        ]
    ) + "]"
    parsed = parse_relation_blocks(text, offsets=None, reject_extra_fields=True)
    assert len(parsed.blocks) == 4
    entries = [
        {
            "triple_hash": _signature_hash(b.triple_signature),
            "quad_hash": _signature_hash(b.canonical_signature),
            "sequence_segment": b.sequence_segment,
            "identity_complete": b.identity_complete,
        }
        for b in parsed.blocks
    ]
    row = make_row("A", model="M0", key="p1")  # placeholder, then overwrite blocks
    row.update(
        {
            "n_blocks": 4,
            "n_identity_complete_blocks": 4,
            "block_index": entries,
            "first_nonempty_triple_reuse": {"exists": True, "block_index_0based": 3},
            "response_sha256": sha256_text(text),
        }
    )
    m1_text = "[" + block("x", "y", "z") + "]"
    m1_parsed = parse_relation_blocks(m1_text, offsets=None, reject_extra_fields=True)
    m1_row = make_row("A", model="M1", key="p1")
    m1_row.update(
        {
            "n_blocks": 1,
            "n_identity_complete_blocks": 1,
            "block_index": [
                {
                    "triple_hash": _signature_hash(b.triple_signature),
                    "quad_hash": _signature_hash(b.canonical_signature),
                    "sequence_segment": b.sequence_segment,
                    "identity_complete": b.identity_complete,
                }
                for b in m1_parsed.blocks
            ],
            "first_nonempty_triple_reuse": {"exists": False},
            "response_sha256": sha256_text(m1_text),
        }
    )
    sequences = [extract_block_sequence(row), extract_block_sequence(m1_row)]
    m0_path = tmp_path / "m0.jsonl"
    m1_path = tmp_path / "m1.jsonl"
    m0_path.write_text(
        json.dumps({"key": "p1", "responses": [{"seed": 0, "response": text}]}) + "\n",
        encoding="utf-8",
    )
    m1_path.write_text(
        json.dumps({"key": "p1", "responses": [{"seed": 0, "response": m1_text}]}) + "\n",
        encoding="utf-8",
    )
    flags = reparse_pair_novelty(
        sequences, responses_m0=m0_path, responses_m1=m1_path, target="answer"
    )
    assert flags[sequences[0].sample_id] == [True, False, True, False]
    assert flags[sequences[1].sample_id] == [True]

    # A tampered response must be rejected by the SHA check.
    m0_path.write_text(
        json.dumps({"key": "p1", "responses": [{"seed": 0, "response": text + " "}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA mismatch"):
        reparse_pair_novelty(
            sequences, responses_m0=m0_path, responses_m1=m1_path, target="answer"
        )


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_p0d_episode_hazard",
        ROOT / "experiments" / "4_analysis" / "analyze_p0d_episode_hazard.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_end_to_end_smoke(tmp_path, monkeypatch, capsys):
    rows = [
        # prompt 1
        make_row("ABC", model="M0", key="p1"),
        make_row(
            "ABBB",
            model="M1",
            key="p1",
            capture=(1, 1),
            hit_max=True,
            orbit=True,
            alignment_type="single_block_aligned",
            matching=(1, 1),
        ),
        # prompt 2
        make_row("AAB", model="M0", key="p2"),
        make_row(
            "ABABAB",
            model="M1",
            key="p2",
            capture=(0, 2),
            hit_max=True,
            orbit=True,
            alignment_type="multi_block_aligned",
            matching=(0, 2),
        ),
        # prompt 3 (M0 has no parsed blocks at all)
        make_row("", model="M0", key="p3"),
        make_row("ABBA", model="M1", key="p3", hit_max=True),
        # prompt 4 (empty-identity separator; M1 captured without hit-max)
        make_row("AA!A", model="M0", key="p4"),
        make_row("CCCCC", model="M1", key="p4", capture=(0, 1)),
    ]
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    with open(result_dir / "event_rows.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (result_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "selection": {"mode": "all"},
                "protocol": {"raw_onset_moved": False, "target": "answer"},
                "inputs": {},
            }
        ),
        encoding="utf-8",
    )

    module = _load_script_module()
    out_dir = tmp_path / "p0d_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_p0d_episode_hazard.py",
            "--result-dir",
            str(result_dir),
            "--output-dir",
            str(out_dir),
            "--bootstrap",
            "100",
            "--bootstrap-batch-size",
            "50",
        ],
    )
    assert module.main() == 0
    for name in (
        "P0D_EPISODE_HAZARD_REPORT.md",
        "P0D_COMPACT_RETURN.txt",
        "P0D_EPISODE_SUMMARY.csv",
        "P0D_KEY_METRICS.csv",
        "P0D_SHAPLEY.csv",
        "P0D_EPISODE_COUNT_DISTRIBUTION.csv",
        "P0D_DISTANCE_TO_CAPTURE.csv",
        "P0D_RESPONSE_EPISODES.jsonl",
        "P0D_SENSITIVITY.csv",
        "P0D_MANIFEST.json",
    ):
        assert (out_dir / name).exists(), name
    compact = (out_dir / "P0D_COMPACT_RETURN.txt").read_text(encoding="utf-8")
    assert len(compact) <= 1950
    assert compact.startswith("[01b-P0d COMPACT]")
    manifest = json.loads((out_dir / "P0D_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["decision"]["label"].startswith("P0D_")
    assert manifest["counts"]["responses_m0"] == 4
    assert manifest["counts"]["responses_m1"] == 4
    episodes_lines = (
        (out_dir / "P0D_RESPONSE_EPISODES.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(episodes_lines) == 16  # 8 responses x {primary, legacy}


def test_sample_mode_rejected_without_flag(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "event_rows.jsonl").write_text("", encoding="utf-8")
    (result_dir / "run_manifest.json").write_text(
        json.dumps({"selection": {"mode": "sample"}, "protocol": {"raw_onset_moved": False}}),
        encoding="utf-8",
    )
    from tcr.events.p0d_episode_hazard import load_event_rows

    with pytest.raises(ValueError, match="all-mode"):
        load_event_rows(result_dir)
