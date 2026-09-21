"""Strict state-machine parser for complete four-field relation objects.

The parser intentionally does not require the response's final list or last
object to be complete.  Every complete object is validated independently, and
an unterminated tail is retained only as a diagnostic.  Motif continuity is
broken by invalid objects or by non-list text between otherwise valid objects.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from .constants import RELATION_FIELDS


@dataclass(frozen=True)
class ParsedBlock:
    block_index: int
    char_start: int
    char_end: int
    token_start: int | None
    token_end: int | None
    raw_text: str
    canonical_signature: tuple[str, str, str, str]
    triple_signature: tuple[str, str, str]
    field_value_char_spans: dict[str, tuple[int, int]]
    separator_before: str
    sequence_segment: int

    @property
    def identity_complete(self) -> bool:
        return all(value.strip() for value in self.triple_signature)


@dataclass(frozen=True)
class ParseDiagnostic:
    char_start: int
    char_end: int
    raw_text: str
    parse_status: str
    reason: str


@dataclass(frozen=True)
class ParseResult:
    blocks: tuple[ParsedBlock, ...]
    diagnostics: tuple[ParseDiagnostic, ...]
    top_level_list_complete: bool
    full_json_list_valid: bool
    incomplete_tail_start: int | None


class _ObjectPairs(list):
    """Marker returned by object_pairs_hook so duplicate keys remain visible."""


def _loads_object_strict(raw: str) -> tuple[dict[str, object], list[str]]:
    parsed = json.loads(raw, object_pairs_hook=_ObjectPairs)
    if not isinstance(parsed, _ObjectPairs):
        raise TypeError("candidate is not an object")
    keys = [str(key) for key, _value in parsed]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    return dict(parsed), duplicates


def _candidate_object_spans(text: str) -> tuple[list[tuple[int, int]], int | None]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    brace_depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if brace_depth == 0:
                start = index
            brace_depth += 1
        elif char == "}" and brace_depth > 0:
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                spans.append((start, index + 1))
                start = None
    return spans, start


def _top_level_list_complete(text: str) -> bool:
    in_string = False
    escaped = False
    square_depth = 0
    curly_depth = 0
    saw_open = False
    closed_at: int | None = None
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            if square_depth == 0 and not saw_open:
                saw_open = True
            square_depth += 1
        elif char == "]":
            square_depth -= 1
            if square_depth < 0:
                return False
            if saw_open and square_depth == 0:
                closed_at = index
        elif char == "{":
            curly_depth += 1
        elif char == "}":
            curly_depth -= 1
            if curly_depth < 0:
                return False
    if not (saw_open and closed_at is not None and square_depth == 0 and curly_depth == 0 and not in_string):
        return False
    # Only whitespace may follow the completed top-level list.
    return not text[closed_at + 1 :].strip()


def _full_json_list_valid(text: str) -> bool:
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(value, list)


def _json_string_tokens(raw: str) -> list[tuple[int, int, str]]:
    tokens: list[tuple[int, int, str]] = []
    index = 0
    while index < len(raw):
        if raw[index] != '"':
            index += 1
            continue
        start = index
        index += 1
        escaped = False
        while index < len(raw):
            char = raw[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                end = index + 1
                token = raw[start:end]
                try:
                    value = json.loads(token)
                except json.JSONDecodeError:
                    value = ""
                tokens.append((start, end, value))
                index = end
                break
            index += 1
        else:
            break
    return tokens


def _field_value_spans(raw: str, object_start: int) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    tokens = _json_string_tokens(raw)
    token_by_start = {start: (start, end, value) for start, end, value in tokens}
    for _start, end, value in tokens:
        if value not in RELATION_FIELDS or value in result:
            continue
        cursor = end
        while cursor < len(raw) and raw[cursor].isspace():
            cursor += 1
        if cursor >= len(raw) or raw[cursor] != ":":
            continue
        cursor += 1
        while cursor < len(raw) and raw[cursor].isspace():
            cursor += 1
        candidate = token_by_start.get(cursor)
        if candidate is None:
            continue
        q_start, q_end, _ = candidate
        result[value] = (object_start + q_start + 1, object_start + q_end - 1)
    return result


def _overlapping_token_span(
    offsets: Sequence[tuple[int, int]] | None,
    char_start: int,
    char_end: int,
) -> tuple[int | None, int | None]:
    if offsets is None:
        return None, None
    first: int | None = None
    last: int | None = None
    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if start < char_end and end > char_start:
            if first is None:
                first = index
            last = index + 1
        elif first is not None and start >= char_end:
            break
    return first, last


def _valid_list_separator(separator: str, *, before_first: bool) -> bool:
    if before_first:
        # Prefix prose is allowed for parsing, but does not affect continuity
        # because there is no previous valid block to connect to.
        return True
    return re.fullmatch(r"\s*,\s*", separator) is not None


def parse_relation_blocks(
    text: str,
    offsets: Sequence[tuple[int, int]] | None = None,
    *,
    reject_extra_fields: bool = True,
) -> ParseResult:
    """Parse complete strict relation blocks without admitting a partial tail."""
    candidates, incomplete_start = _candidate_object_spans(text)
    blocks: list[ParsedBlock] = []
    diagnostics: list[ParseDiagnostic] = []
    required = set(RELATION_FIELDS)
    sequence_segment = 0
    previous_candidate_end = 0
    previous_valid_end: int | None = None

    for char_start, char_end in candidates:
        raw = text[char_start:char_end]
        if previous_valid_end is not None:
            separator = text[previous_valid_end:char_start]
            if not _valid_list_separator(separator, before_first=False):
                sequence_segment += 1
                diagnostics.append(
                    ParseDiagnostic(
                        previous_valid_end,
                        char_start,
                        separator,
                        "continuity_break",
                        "text between complete objects is not a JSON-list comma separator",
                    )
                )
        else:
            separator = text[previous_candidate_end:char_start]

        try:
            value, duplicate_keys = _loads_object_strict(raw)
        except json.JSONDecodeError as exc:
            diagnostics.append(ParseDiagnostic(char_start, char_end, raw, "invalid_json", str(exc)))
            sequence_segment += 1
            previous_candidate_end = char_end
            previous_valid_end = None
            continue
        except TypeError as exc:
            diagnostics.append(ParseDiagnostic(char_start, char_end, raw, "not_object", str(exc)))
            sequence_segment += 1
            previous_candidate_end = char_end
            previous_valid_end = None
            continue
        if duplicate_keys:
            diagnostics.append(
                ParseDiagnostic(
                    char_start, char_end, raw, "duplicate_keys", f"duplicates={duplicate_keys}"
                )
            )
            sequence_segment += 1
            previous_candidate_end = char_end
            previous_valid_end = None
            continue

        keys = set(value)
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        non_string = sorted(key for key in required & keys if not isinstance(value[key], str))
        if missing or non_string or (extra and reject_extra_fields):
            diagnostics.append(
                ParseDiagnostic(
                    char_start,
                    char_end,
                    raw,
                    "invalid_schema",
                    f"missing={missing}; extra={extra}; non_string={non_string}",
                )
            )
            sequence_segment += 1
            previous_candidate_end = char_end
            previous_valid_end = None
            continue
        if extra:
            diagnostics.append(
                ParseDiagnostic(
                    char_start,
                    char_end,
                    raw,
                    "compatible_extra_fields",
                    f"extra={extra}; object excluded from strict event sequence",
                )
            )
            sequence_segment += 1
            previous_candidate_end = char_end
            previous_valid_end = None
            continue

        spans = _field_value_spans(raw, char_start)
        if set(spans) != required:
            diagnostics.append(
                ParseDiagnostic(
                    char_start,
                    char_end,
                    raw,
                    "field_span_failure",
                    f"found={sorted(spans)}",
                )
            )
            sequence_segment += 1
            previous_candidate_end = char_end
            previous_valid_end = None
            continue

        token_start, token_end = _overlapping_token_span(offsets, char_start, char_end)
        signature = tuple(value[field] for field in RELATION_FIELDS)
        block = ParsedBlock(
            block_index=len(blocks),
            char_start=char_start,
            char_end=char_end,
            token_start=token_start,
            token_end=token_end,
            raw_text=raw,
            canonical_signature=signature,  # type: ignore[arg-type]
            triple_signature=signature[:3],  # type: ignore[arg-type]
            field_value_char_spans=spans,
            separator_before=separator,
            sequence_segment=sequence_segment,
        )
        blocks.append(block)
        previous_candidate_end = char_end
        previous_valid_end = char_end

    if incomplete_start is not None:
        diagnostics.append(
            ParseDiagnostic(
                incomplete_start,
                len(text),
                text[incomplete_start:],
                "incomplete_tail",
                "unterminated object at end of response; excluded from block events",
            )
        )

    return ParseResult(
        blocks=tuple(blocks),
        diagnostics=tuple(diagnostics),
        top_level_list_complete=_top_level_list_complete(text),
        full_json_list_valid=_full_json_list_valid(text),
        incomplete_tail_start=incomplete_start,
    )
