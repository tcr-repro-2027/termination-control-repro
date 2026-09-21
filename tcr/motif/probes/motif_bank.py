"""Multi-block motif bank construction with frozen deterministic anchors."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from ..detector_v2.block_parser import ParseResult
from ..schemas import MotifProbe, ParsedBlock, as_record
from .controls import select_nonrecent_motif
from .serialization import assemble_surfaces
from .token_masks import TokenAlignmentError, build_candidate_branch


def _motif_seen_earlier(signatures: Sequence[tuple], *, k_0based: int, m: int) -> bool:
    motif = tuple(signatures[k_0based - m + 1 : k_0based + 1])
    earlier = signatures[: k_0based - m + 1]
    return any(tuple(earlier[start : start + m]) == motif for start in range(0, len(earlier) - m + 1))


def anchor_region(k_0based: int, n_blocks: int) -> str:
    ratio = k_0based / (n_blocks - 1)
    if ratio < 1 / 3:
        return "early"
    if ratio < 2 / 3:
        return "mid"
    return "late"


def _anchor_hash(salt: str, prompt_id: str, m: int, k_0based: int) -> str:
    return hashlib.sha256(f"{salt}|{prompt_id}|{m}|{k_0based}".encode()).hexdigest()


def _candidate_structural_validity(blocks: Sequence[ParsedBlock], k: int, m: int) -> str | None:
    if m < 1 or k < m - 1 or k + 1 >= len(blocks):
        return "anchor_bounds"
    if blocks[k - m + 1].triple_signature == blocks[k + 1].triple_signature:
        return "repeat_recovery_triple_equal"
    involved = blocks[k - m + 1 : k + 2]
    if len({block.sequence_segment for block in involved}) != 1:
        return "invalid_candidate_gap"
    signatures = [block.canonical_signature for block in blocks]
    if _motif_seen_earlier(signatures, k_0based=k, m=m):
        return "motif_seen_earlier"
    return None


def _tokenize_probe(
    tokenizer,
    *,
    probe_id: str,
    chatml_input: str,
    surfaces: dict[str, str],
    max_length: int,
    control_surface: str | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    token_data: dict[str, Any] = {"branches": {}}
    lengths: dict[str, int] = {}
    for context_name in ("C1", "C2", "C3"):
        for candidate_name in ("R", "N"):
            branch, audit = build_candidate_branch(
                tokenizer,
                probe_id=probe_id,
                context_name=context_name,
                candidate_name=candidate_name,
                chatml_input=chatml_input,
                response_context=surfaces[context_name],
                candidate_surface=surfaces[candidate_name],
                max_length=max_length,
            )
            key = f"{context_name}_{candidate_name}"
            token_data["branches"][key] = as_record(branch)
            token_data["branches"][key]["audit"] = audit
            lengths[key] = audit["sequence_length"]
    if control_surface is not None:
        control_context = surfaces["C1"] + control_surface
        for candidate_name in ("R", "N"):
            branch, audit = build_candidate_branch(
                tokenizer,
                probe_id=probe_id,
                context_name="CONTROL",
                candidate_name=candidate_name,
                chatml_input=chatml_input,
                response_context=control_context,
                candidate_surface=surfaces[candidate_name],
                max_length=max_length,
            )
            key = f"CONTROL_{candidate_name}"
            token_data["branches"][key] = as_record(branch)
            token_data["branches"][key]["audit"] = audit
            lengths[key] = audit["sequence_length"]
    return token_data, lengths


def build_probe_bank(
    *,
    stable_prompt_id: str,
    split: str,
    chatml_input: str,
    output_text: str,
    parsed: ParseResult,
    m_values: Sequence[int],
    tokenizer,
    protocol_hash: str,
    source_hashes: dict[str, str],
    max_length: int = 32768,
    probe_salt: str = "tcr.motif-probe-v1.1",
) -> tuple[list[MotifProbe], list[dict[str, Any]]]:
    blocks = parsed.blocks
    candidates: dict[tuple[int, str], list[tuple[str, MotifProbe]]] = defaultdict(list)
    rejections: list[dict[str, Any]] = []
    for m in sorted(set(int(value) for value in m_values)):
        if m < 1:
            raise ValueError("motif lengths must be positive")
        for k in range(m - 1, len(blocks) - 1):
            reason = _candidate_structural_validity(blocks, k, m)
            if reason:
                rejections.append({"stable_prompt_id": stable_prompt_id, "m": m, "k_0based": k, "reason": reason})
                continue
            surfaces = assemble_surfaces(output_text, blocks, k_0based=k, m=m)
            control = select_nonrecent_motif(blocks, k_0based=k, m=m, tokenizer=tokenizer)
            control_surface = control[1] if control else None
            region = anchor_region(k, len(blocks))
            probe_id = hashlib.sha256(
                f"{protocol_hash}|{stable_prompt_id}|m={m}|k={k}".encode()
            ).hexdigest()[:24]
            try:
                token_data, lengths = _tokenize_probe(
                    tokenizer,
                    probe_id=probe_id,
                    chatml_input=chatml_input,
                    surfaces=surfaces,
                    max_length=max_length,
                    control_surface=control_surface,
                )
            except TokenAlignmentError as exc:
                rejections.append(
                    {"stable_prompt_id": stable_prompt_id, "m": m, "k_0based": k, "reason": str(exc)}
                )
                continue
            # Exact construction assertions freeze the scientific object.
            if surfaces["C2"] != surfaces["C1"] + surfaces["U"]:
                raise AssertionError("C2 must equal C1 plus exactly one motif")
            if surfaces["C3"] != surfaces["C2"] + surfaces["U"]:
                raise AssertionError("C3 must equal C2 plus exactly one motif")
            probe = MotifProbe(
                probe_id=probe_id,
                stable_prompt_id=stable_prompt_id,
                split=split,
                m=m,
                k_1based=k + 1,
                k_0based=k,
                anchor_region=region,
                motif_block_indices_1based=list(range(k - m + 2, k + 2)),
                c1_response=surfaces["C1"],
                c2_response=surfaces["C2"],
                c3_response=surfaces["C3"],
                repeat_surface=surfaces["R"],
                recovery_surface=surfaces["N"],
                motif_surface=surfaces["U"],
                nonrecent_motif_surface=control_surface,
                serialization_mode="exact_surface",
                lengths=lengths,
                token_data=token_data,
                source_hashes=source_hashes,
                protocol_hash=protocol_hash,
            )
            candidates[(m, region)].append((_anchor_hash(probe_salt, stable_prompt_id, m, k), probe))
    selected = [min(values, key=lambda item: item[0])[1] for values in candidates.values()]
    selected.sort(key=lambda probe: (probe.m, ("early", "mid", "late").index(probe.anchor_region)))
    return selected, rejections
