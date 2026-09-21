"""First semantic reuse events over complete relation blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

from .block_parser import ParsedBlock


@dataclass(frozen=True)
class ReuseEvent:
    signature_kind: str
    block_index: int
    previous_block_index: int
    first_occurrence_block_index: int
    occurrence_number: int
    distance_from_previous_blocks: int
    same_sequence_segment_as_previous: bool
    char_start: int
    char_end: int
    token_start: int | None
    token_end: int | None
    identity_complete: bool
    description_changed: bool | None


def _signature(block: ParsedBlock, kind: str) -> Hashable:
    if kind == "triple":
        return block.triple_signature
    if kind == "quad":
        return block.canonical_signature
    raise ValueError(f"signature kind must be triple or quad, got {kind!r}")


def find_first_reuse(
    blocks: Sequence[ParsedBlock],
    *,
    signature_kind: str = "triple",
    require_nonempty_identity: bool = False,
) -> ReuseEvent | None:
    """Return the first complete block whose signature appeared earlier."""
    seen: dict[Hashable, list[int]] = {}
    for index, block in enumerate(blocks):
        if require_nonempty_identity and not block.identity_complete:
            continue
        signature = _signature(block, signature_kind)
        previous = seen.get(signature)
        if previous:
            previous_index = previous[-1]
            first_index = previous[0]
            previous_block = blocks[previous_index]
            description_changed: bool | None = None
            if signature_kind == "triple":
                description_changed = (
                    previous_block.canonical_signature[3]
                    != block.canonical_signature[3]
                )
            return ReuseEvent(
                signature_kind=signature_kind,
                block_index=index,
                previous_block_index=previous_index,
                first_occurrence_block_index=first_index,
                occurrence_number=len(previous) + 1,
                distance_from_previous_blocks=index - previous_index,
                same_sequence_segment_as_previous=(
                    previous_block.sequence_segment == block.sequence_segment
                ),
                char_start=block.char_start,
                char_end=block.char_end,
                token_start=block.token_start,
                token_end=block.token_end,
                identity_complete=block.identity_complete,
                description_changed=description_changed,
            )
        seen.setdefault(signature, []).append(index)
    return None
