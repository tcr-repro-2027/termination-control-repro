# coding=utf-8
"""Tests for response parsing + the JSON-validity flag.

Run from the project dir:   python tests/test_parsing.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tcr.extraction.evaluation.parsing import parse_response

FULL_ITEM = ('{"source": "A", "target": "B", "relation": "R", '
             '"description": "D"}')


def test_plain_array_valid():
    parsed = parse_response(f"[{FULL_ITEM}]")
    assert parsed.json_valid is True
    assert parsed.relations == [{"source": "A", "target": "B",
                                 "relation": "R", "description": "D"}]
    print("PASS test_plain_array_valid")


def test_fenced_array_valid():
    parsed = parse_response(f"some prose\n```json\n[{FULL_ITEM}]\n```\nmore prose")
    assert parsed.json_valid is True and len(parsed.relations) == 1
    print("PASS test_fenced_array_valid")


def test_leading_think_tag_stripped():
    parsed = parse_response(f"...thinking...</think>\n[{FULL_ITEM}]")
    assert parsed.json_valid is True and len(parsed.relations) == 1
    print("PASS test_leading_think_tag_stripped")


def test_bare_object_scored_but_invalid():
    # A single object is not a JSON LIST -> invalid, but still scoreable.
    parsed = parse_response(FULL_ITEM)
    assert parsed.json_valid is False
    assert len(parsed.relations) == 1
    print("PASS test_bare_object_scored_but_invalid")


def test_missing_field_invalid():
    item = '{"source": "A", "target": "B", "relation": "R"}'   # no description
    parsed = parse_response(f"[{item}]")
    assert parsed.json_valid is False
    assert parsed.relations[0]["description"] == ""             # still scoreable
    print("PASS test_missing_field_invalid")


def test_truncated_looping_response_invalid():
    truncated = '[{"source": "A", "target": "B", "rel' + "ation" * 200
    parsed = parse_response(truncated)
    assert parsed.json_valid is False and parsed.relations == []
    print("PASS test_truncated_looping_response_invalid")


def test_empty_array_valid():
    parsed = parse_response("[]")
    assert parsed.json_valid is True and parsed.relations == []
    print("PASS test_empty_array_valid")


def test_empty_response_invalid():
    parsed = parse_response("")
    assert parsed.json_valid is False and parsed.relations == []
    print("PASS test_empty_response_invalid")




if __name__ == "__main__":
    test_plain_array_valid()
    test_fenced_array_valid()
    test_leading_think_tag_stripped()
    test_bare_object_scored_but_invalid()
    test_missing_field_invalid()
    test_truncated_looping_response_invalid()
    test_empty_array_valid()
    test_empty_response_invalid()
    test_split_reasoning()
    print("\nAll parsing tests passed.")
