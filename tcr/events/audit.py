"""Core response-level audit orchestration."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from typing import Any, Sequence

from .alignment import classify_legacy_alignment
from .block_parser import ParsedBlock, parse_relation_blocks
from .constants import SCHEMA_VERSION
from .events import ReuseEvent, find_first_reuse
from .io_utils import sha256_text, short, stable_json
from .legacy import LegacyOrbit, adapt_legacy_sample, token_boundary_char
from .motif_capture import MotifRun, find_motif_runs, primary_motif_event


def _signature_hash(signature: Sequence[str]) -> str:
    return sha256_text(stable_json(list(signature)))


def _block_record(block: ParsedBlock) -> dict[str, Any]:
    return {
        "block_index_0based": block.block_index,
        "block_index_1based": block.block_index + 1,
        "char_start": block.char_start,
        "char_end": block.char_end,
        "token_start": block.token_start,
        "token_end": block.token_end,
        "sequence_segment": block.sequence_segment,
        "identity_complete": block.identity_complete,
        "triple_hash": _signature_hash(block.triple_signature),
        "quad_hash": _signature_hash(block.canonical_signature),
    }


def _reuse_record(event: ReuseEvent | None, blocks: Sequence[ParsedBlock]) -> dict[str, Any]:
    if event is None:
        return {"exists": False}
    current = blocks[event.block_index]
    previous = blocks[event.previous_block_index]
    return {
        "exists": True,
        "signature_kind": event.signature_kind,
        "block_index_0based": event.block_index,
        "block_index_1based": event.block_index + 1,
        "previous_block_index_0based": event.previous_block_index,
        "previous_block_index_1based": event.previous_block_index + 1,
        "first_occurrence_block_index_0based": event.first_occurrence_block_index,
        "first_occurrence_block_index_1based": event.first_occurrence_block_index + 1,
        "occurrence_number": event.occurrence_number,
        "distance_from_previous_blocks": event.distance_from_previous_blocks,
        "same_sequence_segment_as_previous": event.same_sequence_segment_as_previous,
        "char_start": event.char_start,
        "char_end": event.char_end,
        "token_start": event.token_start,
        "confirmed_at_token_exclusive": event.token_end,
        "identity_complete": event.identity_complete,
        "description_changed": event.description_changed,
        "triple": list(current.triple_signature),
        "previous_description": previous.canonical_signature[3],
        "current_description": current.canonical_signature[3],
    }


def _motif_record(
    run: MotifRun | None,
    *,
    blocks: Sequence[ParsedBlock],
    confirmation_repeats: int,
) -> dict[str, Any]:
    if run is None:
        return {"exists": False}
    confirmation_exclusive = run.confirmation_block_exclusive(confirmation_repeats)
    if confirmation_exclusive > len(blocks):
        raise AssertionError("motif confirmation exceeds parsed block sequence")
    final_block = blocks[confirmation_exclusive - 1]
    motif_triples = [
        list(blocks[index].triple_signature)
        for index in range(run.block_onset, run.block_onset + run.block_period)
    ]
    return {
        "exists": True,
        "signature_kind": run.signature_kind,
        "sequence_segment": run.sequence_segment,
        "block_onset_0based": run.block_onset,
        "block_onset_1based": run.block_onset + 1,
        "block_period": run.block_period,
        "observed_block_num_repeats": run.block_num_repeats,
        "second_copy_start_0based": run.second_copy_start,
        "second_copy_start_1based": run.second_copy_start + 1,
        "confirmed_after_repeats": confirmation_repeats,
        "confirmed_at_block_exclusive_0based": confirmation_exclusive,
        "confirmed_at_block_1based": confirmation_exclusive,
        "confirmed_at_char_exclusive": final_block.char_end,
        "confirmed_at_token_exclusive": final_block.token_end,
        "strict_block_end_0based": run.strict_block_end,
        "motif_triples": motif_triples,
    }


def _legacy_record(
    legacy: LegacyOrbit,
    *,
    offsets: Sequence[tuple[int, int]],
    text_length: int,
) -> dict[str, Any]:
    record = dataclasses.asdict(legacy)
    if legacy.exists:
        assert legacy.raw_onset_token is not None
        assert legacy.orbit_confirmed_at_token_exclusive is not None
        record["raw_onset_char"] = token_boundary_char(offsets, legacy.raw_onset_token, text_length)
        record["orbit_confirmed_at_char_exclusive"] = token_boundary_char(
            offsets, legacy.orbit_confirmed_at_token_exclusive, text_length
        )
    else:
        record["raw_onset_char"] = None
        record["orbit_confirmed_at_char_exclusive"] = None
    return record


def _event_chain(seed: bool, capture: bool, orbit: bool, hit_max: bool) -> str:
    if not seed and not capture and not orbit:
        return "clean_progress"
    if seed and not capture and not orbit:
        return "seed_only"
    if seed and capture and not orbit:
        return "seed_to_capture"
    if seed and not capture and orbit:
        return "seed_to_orbit_without_complete_capture"
    if seed and capture and orbit:
        return "seed_to_capture_to_orbit_to_hitmax" if hit_max else "seed_to_capture_to_orbit"
    if orbit and not seed:
        return "legacy_orbit_without_complete_seed"
    if capture and not seed:
        return "capture_without_seed_invariant_violation"
    return "other"


def _temporal_record(
    reuse: dict[str, Any], capture: dict[str, Any], legacy: dict[str, Any]
) -> dict[str, Any]:
    seed_t = reuse.get("confirmed_at_token_exclusive") if reuse.get("exists") else None
    capture_t = capture.get("confirmed_at_token_exclusive") if capture.get("exists") else None
    orbit_t = legacy.get("orbit_confirmed_at_token_exclusive") if legacy.get("exists") else None
    raw_onset = legacy.get("raw_onset_token") if legacy.get("exists") else None
    complete = seed_t is not None and capture_t is not None and orbit_t is not None
    valid = bool(complete and seed_t <= capture_t <= orbit_t)
    return {
        "seed_confirmed_at_token_exclusive": seed_t,
        "capture_confirmed_at_token_exclusive": capture_t,
        "orbit_confirmed_at_token_exclusive": orbit_t,
        "legacy_raw_onset_token_retrospective": raw_onset,
        "complete_chain_available": complete,
        "online_confirmation_order_valid": valid if complete else None,
        "raw_onset_precedes_seed_confirmation": (
            raw_onset <= seed_t if raw_onset is not None and seed_t is not None else None
        ),
    }


def _snippets(
    text: str,
    blocks: Sequence[ParsedBlock],
    reuse: dict[str, Any],
    capture: dict[str, Any],
    legacy: dict[str, Any],
) -> dict[str, str]:
    snippets: dict[str, str] = {
        "parser_blocks": "",
        "first_reuse": "",
        "capture": "",
        "legacy": "",
    }
    if blocks:
        sample_indices = sorted({0, len(blocks) // 2, len(blocks) - 1})
        snippets["parser_blocks"] = " || BLOCK || ".join(
            f"B{index + 1}:" + short(blocks[index].raw_text, 350)
            for index in sample_indices
        )
    if reuse.get("exists"):
        previous = blocks[int(reuse["previous_block_index_0based"])]
        current = blocks[int(reuse["block_index_0based"])]
        snippets["first_reuse"] = short(previous.raw_text, 450) + " || REUSE || " + short(current.raw_text, 450)
    if capture.get("exists"):
        onset = int(capture["block_onset_0based"])
        end = int(capture["confirmed_at_block_exclusive_0based"])
        snippets["capture"] = short(text[blocks[onset].char_start:blocks[end - 1].char_end], 1000)
    if legacy.get("exists"):
        onset_char = int(legacy["raw_onset_char"])
        period_text = legacy.get("loop_content") or ""
        snippets["legacy"] = short(text[max(0, onset_char - 300):onset_char] + " || RAW || " + period_text, 1000)
    return snippets


def audit_response(
    *,
    model_tag: str,
    key: Any,
    seed: int,
    text: str,
    token_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    legacy_sample: dict[str, Any],
    selection_roles: Sequence[str],
    capture_min_repeats: int,
    max_motif_period_blocks: int | None,
    legacy_min_repeats: int,
    reject_extra_fields: bool = True,
) -> dict[str, Any]:
    parsed = parse_relation_blocks(text, offsets, reject_extra_fields=reject_extra_fields)
    blocks = parsed.blocks
    triple_reuse = _reuse_record(find_first_reuse(blocks, signature_kind="triple"), blocks)
    nonempty_reuse = _reuse_record(
        find_first_reuse(blocks, signature_kind="triple", require_nonempty_identity=True), blocks
    )
    quad_reuse = _reuse_record(find_first_reuse(blocks, signature_kind="quad"), blocks)

    adjacent_triple_runs = find_motif_runs(
        blocks,
        signature_kind="triple",
        min_repeats=2,
        max_period_blocks=max_motif_period_blocks,
    )
    adjacent_quad_runs = find_motif_runs(
        blocks,
        signature_kind="quad",
        min_repeats=2,
        max_period_blocks=max_motif_period_blocks,
    )
    capture_triple_runs = find_motif_runs(
        blocks,
        signature_kind="triple",
        min_repeats=capture_min_repeats,
        max_period_blocks=max_motif_period_blocks,
    )
    capture_quad_runs = find_motif_runs(
        blocks,
        signature_kind="quad",
        min_repeats=capture_min_repeats,
        max_period_blocks=max_motif_period_blocks,
    )

    adjacent_triple = _motif_record(
        primary_motif_event(adjacent_triple_runs, confirmation_repeats=2),
        blocks=blocks,
        confirmation_repeats=2,
    )
    adjacent_quad = _motif_record(
        primary_motif_event(adjacent_quad_runs, confirmation_repeats=2),
        blocks=blocks,
        confirmation_repeats=2,
    )
    capture_triple = _motif_record(
        primary_motif_event(capture_triple_runs, confirmation_repeats=capture_min_repeats),
        blocks=blocks,
        confirmation_repeats=capture_min_repeats,
    )
    capture_quad = _motif_record(
        primary_motif_event(capture_quad_runs, confirmation_repeats=capture_min_repeats),
        blocks=blocks,
        confirmation_repeats=capture_min_repeats,
    )

    legacy = adapt_legacy_sample(
        legacy_sample,
        min_repeats=legacy_min_repeats,
        n_tokens=len(token_ids),
        token_ids=token_ids,
    )
    legacy_record = _legacy_record(legacy, offsets=offsets, text_length=len(text))
    alignment_type, alignment_evidence = classify_legacy_alignment(
        blocks=blocks,
        quad_capture_runs=capture_quad_runs,
        triple_capture_runs=capture_triple_runs,
        diagnostics=parsed.diagnostics,
        legacy=legacy,
    )

    seed_exists = bool(triple_reuse["exists"])
    capture_exists = bool(capture_triple["exists"])
    orbit_exists = bool(legacy_record["exists"])
    hit_max = bool(legacy_record["hit_max_tokens"])
    if capture_exists and not seed_exists:
        raise AssertionError("a repeated triple motif must imply an earlier triple reuse")

    temporal = _temporal_record(triple_reuse, capture_triple, legacy_record)
    diagnostics = [dataclasses.asdict(value) for value in parsed.diagnostics]
    diagnostic_counts = Counter(value["parse_status"] for value in diagnostics)
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"{model_tag}:{stable_json(key)}:{int(seed)}",
        "stable_prompt_id": stable_json(key),
        "model_tag": model_tag,
        "key": key,
        "seed": int(seed),
        "selection_roles": sorted(set(selection_roles)),
        "response_sha256": sha256_text(text),
        "gen_tokens": len(token_ids),
        "n_blocks": len(blocks),
        "n_identity_complete_blocks": sum(block.identity_complete for block in blocks),
        "top_level_list_complete": parsed.top_level_list_complete,
        "full_json_list_valid": parsed.full_json_list_valid,
        "has_incomplete_tail": parsed.incomplete_tail_start is not None,
        "incomplete_tail_char_start": parsed.incomplete_tail_start,
        "parse_diagnostic_counts": dict(diagnostic_counts),
        "parse_diagnostics": diagnostics,
        "block_index": [_block_record(block) for block in blocks],
        "first_triple_reuse": triple_reuse,
        "first_nonempty_triple_reuse": nonempty_reuse,
        "first_exact_quad_reuse": quad_reuse,
        "first_adjacent_motif_repeat_triple": adjacent_triple,
        "first_adjacent_motif_repeat_quad": adjacent_quad,
        "motif_capture_triple": capture_triple,
        "motif_capture_quad": capture_quad,
        "legacy_orbit": legacy_record,
        "alignment_type": alignment_type,
        "alignment_evidence": alignment_evidence,
        "temporal_alignment": temporal,
        "stage_flags": {
            "seed_reuse": seed_exists,
            "capture": capture_exists,
            "stable_orbit": orbit_exists,
            "hit_max": hit_max,
            "runaway_legacy": orbit_exists,
            "runaway_block_hitmax": bool(capture_quad["exists"]) and hit_max,
            "runaway_any": orbit_exists or (bool(capture_quad["exists"]) and hit_max),
        },
        "stage_chain": _event_chain(seed_exists, capture_exists, orbit_exists, hit_max),
    }
    record["audit_snippets"] = _snippets(
        text, blocks, triple_reuse, capture_triple, legacy_record
    )
    return record
