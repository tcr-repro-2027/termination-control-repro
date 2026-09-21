"""Exact-surface assembly for C1/C2/C3 and candidate blocks."""

from __future__ import annotations

from collections.abc import Sequence

from ..schemas import ParsedBlock


def append_separator(blocks: Sequence[ParsedBlock], index: int) -> str:
    """Return a real inter-object separator for appending ``blocks[index]``.

    The first parsed block carries the response/list prefix (often ``[``), so
    reusing its ``separator_before`` would nest a second list in C2.  When a
    motif starts at block zero, borrow the separator observed before block one.
    """
    if index < 0 or index >= len(blocks):
        raise IndexError(index)
    if index > 0:
        return blocks[index].separator_before
    if len(blocks) > 1:
        return blocks[1].separator_before
    return ", "


def surface_with_separator(blocks: Sequence[ParsedBlock], index: int) -> str:
    return append_separator(blocks, index) + blocks[index].raw_text


def response_prefix_through(output_text: str, block: ParsedBlock) -> str:
    return output_text[: block.char_end]


def motif_surface(blocks: Sequence[ParsedBlock], *, start: int, length: int) -> str:
    if length < 1:
        raise ValueError("motif cannot be empty")
    if start < 0 or start + length > len(blocks):
        raise ValueError("motif outside parsed block sequence")
    return "".join(surface_with_separator(blocks, index) for index in range(start, start + length))


def assemble_surfaces(
    output_text: str, blocks: Sequence[ParsedBlock], *, k_0based: int, m: int
) -> dict[str, str]:
    if m < 1 or k_0based < m - 1 or k_0based + 1 >= len(blocks):
        raise ValueError("invalid motif anchor")
    motif_start = k_0based - m + 1
    c1 = response_prefix_through(output_text, blocks[k_0based])
    unit = motif_surface(blocks, start=motif_start, length=m)
    repeat = surface_with_separator(blocks, motif_start)
    recovery = surface_with_separator(blocks, k_0based + 1)
    return {
        "C1": c1,
        "C2": c1 + unit,
        "C3": c1 + unit + unit,
        "U": unit,
        "R": repeat,
        "N": recovery,
    }
