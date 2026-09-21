from __future__ import annotations

import json
import multiprocessing
import runpy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from conftest import CharTokenizer, relation, relation_surface
from tcr.motif.detector_v2.block_parser import parse_relation_blocks
from tcr.motif.probes.controls import select_nonrecent_motif
from tcr.motif.probes.motif_bank import anchor_region, build_probe_bank
from tcr.motif.probes.serialization import assemble_surfaces
from tcr.motif.probes.splits import assign_split, stable_sample_id
from tcr.motif.probes.token_masks import TokenAlignmentError, build_candidate_branch




@pytest.mark.parametrize("m", [1, 2, 5])
def test_c1_c2_repeat_recovery_exact_surface(m):
    rows = [relation(index) for index in range(8)]
    text = relation_surface(rows, separator=",\n  ")
    parsed = parse_relation_blocks(text)
    k = m - 1
    surfaces = assemble_surfaces(text, parsed.blocks, k_0based=k, m=m)
    assert surfaces["C2"] == surfaces["C1"] + surfaces["U"]
    assert surfaces["C3"] == surfaces["C2"] + surfaces["U"]
    assert json.loads(surfaces["R"].lstrip(",\n ")) == rows[0]
    assert json.loads(surfaces["N"].lstrip(",\n ")) == rows[k + 1]
    assert not surfaces["U"].lstrip().startswith("[")


def test_motif_starting_at_first_block_borrows_real_separator():
    text = relation_surface([relation(0), relation(1), relation(2)], separator=",\n    ")
    parsed = parse_relation_blocks(text)
    surfaces = assemble_surfaces(text, parsed.blocks, k_0based=0, m=1)
    assert surfaces["U"].startswith(",\n    {")
    assert surfaces["R"].startswith(",\n    {")
    assert surfaces["C2"].count("[") == surfaces["C1"].count("[")


def test_candidate_mask_selects_only_three_field_values_and_shifts_causally():
    tokenizer = CharTokenizer()
    candidate = ", " + json.dumps(
        relation(0, source="SRC", target="TARGET", relation_name="REL", description="DESCRIPTION"),
        ensure_ascii=False,
    )
    branch, audit = build_candidate_branch(
        tokenizer,
        probe_id="p",
        context_name="C1",
        candidate_name="R",
        chatml_input="PROMPT",
        response_context="[",
        candidate_surface=candidate,
        max_length=1000,
    )
    target_chars = "".join(branch.full_text[index] for index in branch.target_indices)
    assert target_chars == "SRCTARGETREL"
    assert "DESCRIPTION" not in target_chars
    assert all(predictor == target - 1 for predictor, target in zip(branch.predictor_indices, branch.target_indices))
    assert len(branch.target_indices) == 3 + 6 + 3
    assert audit["patch_predictor_start"] <= min(branch.predictor_indices)


def test_empty_identity_value_and_overlength_fail_without_truncation():
    tokenizer = CharTokenizer()
    candidate = json.dumps(relation(0, source=""), ensure_ascii=False)
    with pytest.raises(TokenAlignmentError, match="empty_identity_mask"):
        build_candidate_branch(
            tokenizer,
            probe_id="p",
            context_name="C1",
            candidate_name="R",
            chatml_input="prompt",
            response_context="[",
            candidate_surface=candidate,
            max_length=1000,
        )
    with pytest.raises(TokenAlignmentError, match="probe_exceeds_max_length"):
        build_candidate_branch(
            tokenizer,
            probe_id="p",
            context_name="C1",
            candidate_name="R",
            chatml_input="prompt",
            response_context="[",
            candidate_surface=json.dumps(relation(0)),
            max_length=10,
        )


def test_probe_bank_rejects_prior_motif_and_equal_repeat_recovery():
    tokenizer = CharTokenizer()
    prior = [relation(0), relation(1), relation(7), relation(0), relation(1), relation(8)]
    text = relation_surface(prior)
    _, rejected = build_probe_bank(
        stable_prompt_id="x",
        split="construct",
        chatml_input="P",
        output_text=text,
        parsed=parse_relation_blocks(text),
        m_values=[2],
        tokenizer=tokenizer,
        protocol_hash="h",
        source_hashes={},
    )
    assert "motif_seen_earlier" in {row["reason"] for row in rejected}

    equal = [relation(0), relation(1), relation(0)]
    text = relation_surface(equal)
    probes, rejected = build_probe_bank(
        stable_prompt_id="y",
        split="construct",
        chatml_input="P",
        output_text=text,
        parsed=parse_relation_blocks(text),
        m_values=[2],
        tokenizer=tokenizer,
        protocol_hash="h",
        source_hashes={},
    )
    assert not probes
    assert rejected[0]["reason"] == "repeat_recovery_triple_equal"


def test_probe_cannot_cross_invalid_candidate_gap():
    good0 = json.dumps(relation(0))
    bad = json.dumps(relation(9, extra={"x": 1}))
    good1 = json.dumps(relation(1))
    good2 = json.dumps(relation(2))
    text = f"[{good0}, {bad}, {good1}, {good2}]"
    probes, rejected = build_probe_bank(
        stable_prompt_id="gap",
        split="construct",
        chatml_input="P",
        output_text=text,
        parsed=parse_relation_blocks(text),
        m_values=[1, 2],
        tokenizer=CharTokenizer(),
        protocol_hash="h",
        source_hashes={},
    )
    assert "invalid_candidate_gap" in {row["reason"] for row in rejected}
    assert all(probe.k_0based >= 1 for probe in probes)


def test_nonrecent_control_is_deterministic_and_length_matched():
    rows = [relation(index) for index in range(8)]
    parsed = parse_relation_blocks(relation_surface(rows))
    left = select_nonrecent_motif(parsed.blocks, k_0based=6, m=2, tokenizer=CharTokenizer())
    right = select_nonrecent_motif(parsed.blocks, k_0based=6, m=2, tokenizer=CharTokenizer())
    assert left == right
    assert left[0] <= 3


def test_stable_split_is_order_independent_and_regions_are_fixed():
    ids = [f"index:{index}" for index in range(100)]
    first = {value: assign_split(value, salt="frozen") for value in ids}
    second = {value: assign_split(value, salt="frozen") for value in reversed(ids)}
    assert first == second
    assert stable_sample_id({"key": 1}) == "index:0"
    assert anchor_region(0, 10) == "early"
    assert anchor_region(4, 10) == "mid"
    assert anchor_region(8, 10) == "late"
