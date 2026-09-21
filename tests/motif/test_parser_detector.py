from __future__ import annotations

import json

import pytest

from conftest import CharTokenizer, relation, relation_surface
from tcr.motif.detector_v2.block_parser import parse_relation_blocks
from tcr.motif.schemas import RawTokenLoop


def test_parser_handles_strings_escapes_unicode_and_multiline():
    row = relation(
        0,
        source='a"b\\c',
        target="中文",
        description="contains {braces}, [brackets], and\nmultiple lines",
    )
    text = relation_surface([row], separator=",\n  ")
    tokenizer = CharTokenizer()
    offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    parsed = parse_relation_blocks(text, offsets)
    assert parsed.top_level_list_complete
    assert not parsed.diagnostics
    assert len(parsed.blocks) == 1
    block = parsed.blocks[0]
    assert block.canonical_signature == tuple(row[field] for field in ("source", "target", "relation", "description"))
    for field, (start, end) in block.field_value_char_spans.items():
        assert json.loads(text[start - 1 : end + 1]) == row[field]
    assert block.token_start == block.char_start
    assert block.token_end == block.char_end


def test_extra_fields_are_diagnostics_only_in_compatibility_mode():
    text = relation_surface([relation(0, extra={"extra": "x"})])
    strict = parse_relation_blocks(text, reject_extra_fields=True)
    compatible = parse_relation_blocks(text, reject_extra_fields=False)
    assert not strict.blocks and not compatible.blocks
    assert strict.diagnostics[0].parse_status == "invalid_schema"
    assert compatible.diagnostics[0].parse_status == "compatible_extra_fields"


def test_incomplete_tail_and_invalid_middle_break_continuity():
    good0 = json.dumps(relation(0), ensure_ascii=False)
    bad = json.dumps(relation(9, extra={"extra": 1}), ensure_ascii=False)
    good1 = json.dumps(relation(0), ensure_ascii=False)
    text = f"[{good0}, {bad}, {good1}, {{\"source\": \"tail\""
    parsed = parse_relation_blocks(text)
    assert len(parsed.blocks) == 2
    assert parsed.blocks[0].sequence_segment == 0
    assert parsed.blocks[1].sequence_segment == 1
    assert bad not in parsed.blocks[1].separator_before
    assert {row.parse_status for row in parsed.diagnostics} == {"invalid_schema", "incomplete_tail"}
