from __future__ import annotations

import json

from conftest import char_offsets, relation, surface
from tcr.events.block_parser import parse_relation_blocks
from tcr.events.motif_capture import find_motif_runs


def test_parser_handles_escaped_braces_quotes_unicode_and_multiline():
    row = relation(
        0,
        source='a"b\\c',
        target="中文",
        description="contains {braces}, [brackets], and\nmultiple lines",
    )
    text = surface([row], separator=",\n  ")
    parsed = parse_relation_blocks(text, char_offsets(text))
    assert parsed.top_level_list_complete
    assert parsed.full_json_list_valid
    assert not parsed.diagnostics
    assert len(parsed.blocks) == 1
    block = parsed.blocks[0]
    assert block.canonical_signature == tuple(
        row[field] for field in ("source", "target", "relation", "description")
    )
    assert block.token_start == block.char_start
    assert block.token_end == block.char_end


def test_incomplete_tail_is_diagnostic_and_never_becomes_block():
    good = json.dumps(relation(0), ensure_ascii=False)
    text = f"[{good}, {{\"source\": \"tail\""
    parsed = parse_relation_blocks(text, char_offsets(text))
    assert len(parsed.blocks) == 1
    assert not parsed.top_level_list_complete
    assert {d.parse_status for d in parsed.diagnostics} == {"incomplete_tail"}
    assert not find_motif_runs(parsed.blocks, signature_kind="quad", min_repeats=2)


def test_invalid_middle_and_arbitrary_text_break_motif_continuity():
    good0 = json.dumps(relation(0), ensure_ascii=False)
    bad = json.dumps({**relation(9), "extra": "x"}, ensure_ascii=False)
    good1 = json.dumps(relation(0), ensure_ascii=False)
    text = f"[{good0}, {bad}, {good1}]"
    parsed = parse_relation_blocks(text, char_offsets(text))
    assert len(parsed.blocks) == 2
    assert parsed.blocks[0].sequence_segment != parsed.blocks[1].sequence_segment
    assert not find_motif_runs(parsed.blocks, signature_kind="quad", min_repeats=2)

    text2 = f"[{good0} explanatory text {good1}]"
    parsed2 = parse_relation_blocks(text2, char_offsets(text2))
    assert len(parsed2.blocks) == 2
    assert parsed2.blocks[0].sequence_segment != parsed2.blocks[1].sequence_segment
    assert any(d.parse_status == "continuity_break" for d in parsed2.diagnostics)


def test_duplicate_json_keys_are_rejected_from_strict_blocks():
    text = '[{"source":"S0","source":"S1","target":"T","relation":"R","description":"D"}]'
    parsed = parse_relation_blocks(text, char_offsets(text))
    assert not parsed.blocks
    assert any(d.parse_status == "duplicate_keys" for d in parsed.diagnostics)
