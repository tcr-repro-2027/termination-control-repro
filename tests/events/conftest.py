from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def relation(i: int, *, description: str | None = None, **updates):
    row = {
        "source": f"S{i}",
        "target": f"T{i}",
        "relation": f"R{i}",
        "description": description if description is not None else f"D{i}",
    }
    row.update(updates)
    return row


def surface(rows, *, separator=", "):
    return "[" + separator.join(json.dumps(row, ensure_ascii=False) for row in rows) + "]"


def char_offsets(text: str):
    return [(i, i + 1) for i in range(len(text))]
