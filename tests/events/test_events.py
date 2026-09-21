from __future__ import annotations

from conftest import relation, surface
from tcr.events.block_parser import parse_relation_blocks
from tcr.events.events import find_first_reuse


def test_first_triple_reuse_precedes_exact_quad_when_description_changes():
    rows = [
        relation(0, description="first"),
        relation(1),
        relation(0, description="changed"),
        relation(0, description="first"),
    ]
    blocks = parse_relation_blocks(surface(rows)).blocks
    triple = find_first_reuse(blocks, signature_kind="triple")
    quad = find_first_reuse(blocks, signature_kind="quad")
    assert triple is not None and triple.block_index == 2
    assert triple.previous_block_index == 0
    assert triple.description_changed is True
    assert quad is not None and quad.block_index == 3


def test_nonempty_reuse_can_be_reported_separately():
    empty = relation(0, source="", target="", relation="")
    rows = [empty, empty, relation(1), relation(1)]
    blocks = parse_relation_blocks(surface(rows)).blocks
    any_reuse = find_first_reuse(blocks, signature_kind="triple")
    nonempty = find_first_reuse(
        blocks, signature_kind="triple", require_nonempty_identity=True
    )
    assert any_reuse is not None and any_reuse.block_index == 1
    assert nonempty is not None and nonempty.block_index == 3
