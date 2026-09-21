# coding: utf-8
"""JSONL / CSV / compact-summary plumbing shared by the closeout entries."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

COMPACT_MAX_CHARS = 1900


def log(prefix: str, message: str) -> None:
    print(f"[{prefix} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write through a `.partial` file so an interrupted run leaves nothing
    a later step could mistake for a finished artefact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".partial")
    count = 0
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temp.replace(target)
    return count


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]],
              fields: Sequence[str] | None = None) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return 0
    names = list(fields) if fields else list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_table(path_stem: str | Path, rows: Sequence[Mapping[str, Any]],
                *, fields: Sequence[str] | None = None,
                caption: str = "", align_right: Sequence[str] = ()) -> None:
    """One table in the three forms the paper needs: CSV, LaTeX, Markdown.

    The CSV keeps full precision for anything recomputed later; the LaTeX and
    Markdown views are what actually gets pasted, so they carry the caption and
    the column alignment with them.
    """
    stem = Path(path_stem)
    names = list(fields) if fields else (list(rows[0].keys()) if rows else [])
    write_csv(stem.with_suffix(".csv"), rows, names)
    if not rows:
        stem.with_suffix(".tex").write_text("", encoding="utf-8")
        stem.with_suffix(".md").write_text("", encoding="utf-8")
        return

    right = set(align_right)

    def cell(value: Any) -> str:
        # A missing measurement prints as an em dash, never as 0 and never as
        # the string "nan": a reader must not be able to mistake "not measured"
        # for "measured and zero".
        if value is None or value == "":
            return "--"
        if isinstance(value, str):
            # Rows read back from a CSV arrive as strings; a number that came
            # from one should still print like a number rather than as
            # seventeen digits of float repr.
            try:
                value = float(value)
            except ValueError:
                return value
        if isinstance(value, float):
            return "--" if value != value else f"{value:.4g}"
        return str(value)

    header = "| " + " | ".join(names) + " |"
    ruler = "|" + "|".join("---:" if name in right else "---" for name in names) + "|"
    body = ["| " + " | ".join(cell(row.get(name)) for name in names) + " |"
            for row in rows]
    md = "\n".join([f"**{caption}**", "", header, ruler, *body]) if caption \
        else "\n".join([header, ruler, *body])
    stem.with_suffix(".md").write_text(md + "\n", encoding="utf-8")

    def tex_escape(value: str) -> str:
        for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                         ("&", r"\&"), ("#", r"\#")):
            value = value.replace(old, new)
        return value

    spec = "".join("r" if name in right else "l" for name in names)
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule",
             " & ".join(tex_escape(name) for name in names) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(tex_escape(cell(row.get(name))) for name in names)
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines = [r"\caption{" + tex_escape(caption) + "}"] + lines
    stem.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compact(path: str | Path, title: str, lines: Sequence[str],
                  *, max_chars: int = COMPACT_MAX_CHARS) -> Path:
    """A low-bandwidth summary, split rather than truncated when it overflows.

    Cutting a summary mid-number is how a reader ends up quoting half an
    interval, so an overlong summary becomes `<name>_2.txt`, `_3.txt` and so on.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    parts: list[list[str]] = [[]]
    length = len(title) + 1
    for line in lines:
        if length + len(line) + 1 > max_chars and parts[-1]:
            parts.append([])
            length = len(title) + 1
        parts[-1].append(line)
        length += len(line) + 1
    written = target
    for index, part in enumerate(parts):
        name = target if index == 0 else target.with_name(
            f"{target.stem}_{index + 1}{target.suffix}")
        suffix = "" if len(parts) == 1 else f" ({index + 1}/{len(parts)})"
        name.write_text(f"{title}{suffix}\n" + "\n".join(part) + "\n",
                        encoding="utf-8")
        written = name
    return written


def fmt_pct(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if number != number:  # NaN
        return "NA"
    return f"{100.0 * number:.{digits}f}%"


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if number != number:
        return "NA"
    return f"{number:.{digits}f}"


def fmt_ci(estimate: Any, low: Any, high: Any, *, pct: bool = False,
           digits: int = 2) -> str:
    render = (lambda v: fmt_pct(v, digits)) if pct else (lambda v: fmt(v, digits + 1))
    return f"{render(estimate)} [{render(low)}, {render(high)}]"
