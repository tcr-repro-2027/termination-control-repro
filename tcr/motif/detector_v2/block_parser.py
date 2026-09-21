"""String-state-machine parser for exact four-field relation objects."""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence

from ..constants import RELATION_FIELDS
from ..schemas import ParseDiagnostic, ParsedBlock


@dataclass(frozen=True)
class ParseResult:
    blocks: tuple[ParsedBlock, ...]
    diagnostics: tuple[ParseDiagnostic, ...]
    top_level_list_complete: bool


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
        elif char == "}" and brace_depth:
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                spans.append((start, index + 1))
                start = None
    return spans, start


def _top_level_list_complete(text: str) -> bool:
    in_string = escaped = False
    square = curly = 0
    saw_list = False
    closed = False
    for char in text:
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
            square += 1
            saw_list = True
        elif char == "]":
            square -= 1
            if square == 0:
                closed = True
        elif char == "{":
            curly += 1
        elif char == "}":
            curly -= 1
        if square < 0 or curly < 0:
            return False
    return bool(saw_list and closed and square == 0 and curly == 0 and not in_string)


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
    for start, end, value in tokens:
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
        candidate = next((t for t in tokens if t[0] == cursor), None)
        if candidate is None:
            continue
        q_start, q_end, _ = candidate
        result[value] = (object_start + q_start + 1, object_start + q_end - 1)
    return result


def _overlapping_token_span(
    offsets: Sequence[tuple[int, int]] | None, char_start: int, char_end: int
) -> tuple[int | None, int | None]:
    if offsets is None:
        return None, None
    indices = [i for i, (start, end) in enumerate(offsets) if end > start and start < char_end and end > char_start]
    return (min(indices), max(indices) + 1) if indices else (None, None)


def parse_relation_blocks(
    text: str,
    offsets: Sequence[tuple[int, int]] | None = None,
    *,
    reject_extra_fields: bool = True,
) -> ParseResult:
    """Parse candidates independently; only strict valid objects enter ``blocks``."""
    candidates, incomplete_start = _candidate_object_spans(text)
    blocks: list[ParsedBlock] = []
    diagnostics: list[ParseDiagnostic] = []
    previous_end = 0
    sequence_segment = 0
    required = set(RELATION_FIELDS)
    for char_start, char_end in candidates:
        raw = text[char_start:char_end]
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            diagnostics.append(ParseDiagnostic(char_start, char_end, raw, "invalid_json", str(exc)))
            sequence_segment += 1
            previous_end = char_end
            continue
        if not isinstance(value, dict):
            diagnostics.append(ParseDiagnostic(char_start, char_end, raw, "not_object", "candidate is not an object"))
            sequence_segment += 1
            previous_end = char_end
            continue
        keys = set(value)
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        non_string = sorted(key for key in required & keys if not isinstance(value[key], str))
        # Compatibility mode only changes the diagnostic label.  Extra-field
        # objects must never enter the scientific sample set.
        if missing or non_string or extra:
            reason = f"missing={missing}; extra={extra}; non_string={non_string}"
            status = (
                "compatible_extra_fields"
                if extra and not reject_extra_fields and not missing and not non_string
                else "invalid_schema"
            )
            diagnostics.append(ParseDiagnostic(char_start, char_end, raw, status, reason))
            sequence_segment += 1
            previous_end = char_end
            continue
        spans = _field_value_spans(raw, char_start)
        if set(spans) != required:
            diagnostics.append(
                ParseDiagnostic(char_start, char_end, raw, "field_span_failure", f"found={sorted(spans)}")
            )
            sequence_segment += 1
            previous_end = char_end
            continue
        token_start, token_end = _overlapping_token_span(offsets, char_start, char_end)
        signature = tuple(value[field] for field in RELATION_FIELDS)
        separator = text[previous_end:char_start]
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
        previous_end = char_end
    if incomplete_start is not None:
        diagnostics.append(
            ParseDiagnostic(
                incomplete_start,
                len(text),
                text[incomplete_start:],
                "incomplete_tail",
                "unterminated object at end of response",
            )
        )
    return ParseResult(tuple(blocks), tuple(diagnostics), _top_level_list_complete(text))


def token_index_for_char(offsets: Sequence[tuple[int, int]], char_pos: int) -> int | None:
    starts = [start for start, _ in offsets]
    index = bisect_right(starts, char_pos) - 1
    while index >= 0:
        start, end = offsets[index]
        if end > start and start <= char_pos < end:
            return index
        if end <= char_pos:
            break
        index -= 1
    return None
