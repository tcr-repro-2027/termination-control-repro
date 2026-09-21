from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


class CharTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, truncation=False, **kwargs):
        assert isinstance(text, str)
        result = {"input_ids": [ord(char) + 1 for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def relation(index, *, description=None, source=None, target=None, relation_name=None, extra=None):
    row = {
        "source": source if source is not None else f"s{index}",
        "target": target if target is not None else f"t{index}",
        "relation": relation_name if relation_name is not None else f"r{index}",
        "description": description if description is not None else f"d{index}",
    }
    if extra:
        row.update(extra)
    return row


def relation_surface(rows, separator=", "):
    return "[" + separator.join(json.dumps(row, ensure_ascii=False) for row in rows) + "]"
