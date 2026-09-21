"""One-shot full-text tokenization and identity-field masks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..constants import IDENTITY_FIELDS, MAX_LENGTH
from ..detector_v2.block_parser import parse_relation_blocks
from ..schemas import CandidateBranch


class TokenAlignmentError(ValueError):
    pass


def _tokenizer_call(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    if getattr(tokenizer, "is_fast", True) is False:
        raise TokenAlignmentError("tokenizer must be fast and support offset mappings")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None or len(offsets) != len(ids):
        raise TokenAlignmentError("tokenizer did not return a valid offset_mapping")
    return [int(x) for x in ids], [(int(a), int(b)) for a, b in offsets]


def overlapping_token_indices(
    offsets: Sequence[tuple[int, int]], span: tuple[int, int]
) -> tuple[int, ...]:
    char_start, char_end = span
    if char_end <= char_start:
        return ()
    return tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if (start, end) != (0, 0) and end > start and start < char_end and end > char_start
    )


def build_candidate_branch(
    tokenizer,
    *,
    probe_id: str,
    context_name: str,
    candidate_name: str,
    chatml_input: str,
    response_context: str,
    candidate_surface: str,
    max_length: int = MAX_LENGTH,
) -> tuple[CandidateBranch, dict[str, Any]]:
    full_text = chatml_input + response_context + candidate_surface
    ids, offsets = _tokenizer_call(tokenizer, full_text)
    if len(ids) > max_length:
        raise TokenAlignmentError(f"probe_exceeds_max_length:{len(ids)}>{max_length}")
    candidate_start = len(chatml_input) + len(response_context)
    parsed = parse_relation_blocks(candidate_surface)
    if not parsed.blocks:
        raise TokenAlignmentError("candidate_surface_has_no_strict_relation_block")
    local_block = parsed.blocks[0]
    field_tokens: dict[str, tuple[int, ...]] = {}
    for field in IDENTITY_FIELDS:
        local_span = local_block.field_value_char_spans[field]
        absolute_span = (candidate_start + local_span[0], candidate_start + local_span[1])
        field_tokens[field] = overlapping_token_indices(offsets, absolute_span)
        if not field_tokens[field]:
            raise TokenAlignmentError(f"empty_identity_mask:{field}")
    target_indices = tuple(sorted({i for indices in field_tokens.values() for i in indices}))
    if not target_indices or target_indices[0] <= 0:
        raise TokenAlignmentError("identity target has no causal predictor")
    predictor_indices = tuple(index - 1 for index in target_indices)
    candidate_tokens = overlapping_token_indices(offsets, (candidate_start, len(full_text)))
    if not candidate_tokens or candidate_tokens[0] <= 0:
        raise TokenAlignmentError("candidate has no predictor token")
    patch_start = candidate_tokens[0] - 1
    patch_end = max(predictor_indices)
    branch = CandidateBranch(
        probe_id=probe_id,
        context_name=context_name,  # type: ignore[arg-type]
        candidate_name=candidate_name,  # type: ignore[arg-type]
        full_text=full_text,
        input_ids=tuple(ids),
        target_indices=target_indices,
        predictor_indices=predictor_indices,
        field_token_indices=field_tokens,
    )
    audit = {
        "sequence_length": len(ids),
        "candidate_token_start": candidate_tokens[0],
        "candidate_token_end": candidate_tokens[-1] + 1,
        "patch_predictor_start": patch_start,
        "patch_predictor_end_inclusive": patch_end,
        "field_token_counts": {field: len(indices) for field, indices in field_tokens.items()},
    }
    return branch, audit
