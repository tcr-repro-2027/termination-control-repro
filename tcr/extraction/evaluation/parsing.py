# coding=utf-8
"""Parse a (possibly messy) model response into relations + a validity flag.

Real responses wrap the JSON array in ```json fences, prepend a leftover
``</think>`` tag, or surround it with prose; and -- crucially for this task -- a
LOOPING response is often truncated mid-array and cannot be parsed at all.
Extraction is therefore lenient on structure but strict on validity:

* ``relations``  -- best-effort: every dict item found in the first parseable
  JSON candidate (a bare object counts as a one-element list), so partial
  answers still score what they did produce;
* ``json_valid`` -- the protocol metric: 1 iff the response parses to a JSON
  LIST whose every item is a dict carrying ALL FOUR fields
  (source / target / relation / description).
"""

import dataclasses
import json
import re
from typing import Any, Dict, List, Optional

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_LEADING_TAGS = ("</think>", "</tool_call>", "<tool_call>")

REQUIRED_FIELDS = ("source", "target", "relation", "description")


@dataclasses.dataclass(frozen=True)
class ParsedResponse:
    relations: List[Dict[str, str]]   # best-effort relations for scoring
    json_valid: bool                  # the strict protocol validity flag


def _strip_leading_tags(s: str) -> str:
    """Keep only the text after the last reasoning / tool-call tag, if any."""
    for tag in _LEADING_TAGS:
        if tag in s:
            s = s.rsplit(tag, 1)[-1]
    return s.strip()


def extract_json_value(text: str) -> Optional[Any]:
    """Return the first parseable JSON value (list or dict) found in ``text``.

    Candidates, in order: the first ```...``` fence body, the whole text, the
    outermost ``[...]`` span.  ``None`` when nothing parses.
    """
    if not text or not isinstance(text, str):
        return None

    s = _strip_leading_tags(text.strip())

    candidates = []
    fence = _FENCE_RE.search(s)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(s)
    left, right = s.find("["), s.rfind("]")
    if left != -1 and right > left:
        candidates.append(s[left:right + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, (list, dict)):
            return data
    return None


def parse_response(text: str) -> ParsedResponse:
    """Parse one response into scoring relations + the JSON-validity flag."""
    value = extract_json_value(text)
    if value is None:
        return ParsedResponse(relations=[], json_valid=False)

    items = value if isinstance(value, list) else [value]
    relations = [
        {field: str(item.get(field, "")).strip() for field in REQUIRED_FIELDS}
        for item in items if isinstance(item, dict)
    ]
    json_valid = isinstance(value, list) and all(
        isinstance(item, dict) and all(field in item for field in REQUIRED_FIELDS)
        for item in value
    )
    return ParsedResponse(relations=relations, json_valid=json_valid)
