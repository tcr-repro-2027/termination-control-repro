# coding=utf-8
"""JSONL / CSV / locking helpers, plus sibling-project path resolution.

Copied from `tcr/evaluation/tcr/evaluation/io_utils.py` with only the sibling map changed:
E2 writes the same kinds of artefact (append-only jsonl, atomic summaries,
locked CSV appends) and there is no value in a second, subtly different set of
helpers for them.

Two things here are worth reading before use:

* every writer that produces a FINAL artefact writes to a temporary file in
  the same directory and renames it into place, so a killed worker can never
  leave a half-written summary that a later stage would read as complete;
  append-only writers (responses, event rows) flush per line instead, because
  their whole point is to survive an interruption mid-file;
* :func:`append_csv_row` takes a cross-platform exclusive lock.  The
  orchestrator appends the unified CSV from a single process, but `analyze.py`
  can also be run by hand on several models at once, and a torn CSV line in an
  overnight run is not worth the two dozen lines it costs to prevent.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Set

# ----------------------------------------------------------------- filesystem

def ensure_dir(path: str | os.PathLike[str]) -> None:
    if str(path):
        os.makedirs(path, exist_ok=True)


def ensure_parent(path: str | os.PathLike[str]) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Turn an arbitrary tag into a safe file-name component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


# ---------------------------------------------------------------------- jsonl

def read_jsonl(path: str | os.PathLike[str]) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            yield value


def append_jsonl(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Append one record and flush it to the OS, so a kill loses at most the
    record in flight."""
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        handle.flush()


class JsonlWriter:
    """Append-mode jsonl writer that keeps the handle open and flushes lines.

    `append_jsonl` reopens per record; keeping the handle open avoids that cost
    on a streamed file.  The FLUSH still happens per line, and that is not the
    same trade-off: without it a killed worker leaves a half-written line in the
    middle of a resumable file.  `load_done_keys` tolerates such a line and the
    cell is regenerated -- but the torn text stays in the file, and the analysis
    stage's `read_jsonl` then dies on it, on a file whose data is actually
    complete.  One `flush()` per line is a buffer copy, not a disk sync; over
    2560 rows it costs nothing measurable.
    """

    def __init__(self, path: str | os.PathLike[str], *, mode: str = "a") -> None:
        self.path = Path(path)
        self._mode = mode
        self._handle = None

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # newline="": write the line feed verbatim.  In text mode Windows
        # expands it to CRLF, so an anchor set built on Windows and the
        # byte-identical one built on Linux hash differently -- which is exactly
        # what happened (1043d3b2 vs 0f8dcf02 for the same 384 anchors, same
        # counts, same donors).  A provenance check built on that hash is
        # worthless if the operating system can change it.
        self._handle = open(self.path, self._mode, encoding="utf-8", newline="")
        return self

    def write(self, record: Mapping[str, Any]) -> None:
        assert self._handle is not None, "JsonlWriter used outside its context"
        self._handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        self._handle.flush()

    def __exit__(self, *_exc: Any) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Atomic pretty-printed JSON write (temp file + rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + f".partial.{os.getpid()}")
    with open(temp, "w", encoding="utf-8", newline="") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_done_keys(path: str | os.PathLike[str],
                   key_field: str = "key") -> Set[Any]:
    """Keys already present in an append-only file; empty set if absent.

    A trailing torn line (killed mid-write) is tolerated and ignored: the
    record it belonged to is simply regenerated.
    """
    done: Set[Any] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get(key_field) is not None:
                done.add(record[key_field])
    return done


# ----------------------------------------------------------------- csv + lock

class FileLock:
    """Exclusive lock via ``O_CREAT | O_EXCL``; works on Linux and Windows.

    A stale lock (holder killed) is broken after ``stale_after`` seconds so an
    unattended run cannot deadlock on a dead process.
    """

    def __init__(self, path: str | os.PathLike[str], *, timeout: float = 60.0,
                 poll: float = 0.05, stale_after: float = 300.0) -> None:
        self.path = Path(str(path) + ".lock")
        self.timeout = timeout
        self.poll = poll
        self.stale_after = stale_after
        self._fd: int | None = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.timeout
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode("ascii"))
                return self
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_after:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.time() > deadline:
                    raise TimeoutError(f"could not acquire {self.path}")
                time.sleep(self.poll)

    def __exit__(self, *_exc: Any) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.path.unlink(missing_ok=True)


def append_csv_row(path: str | os.PathLike[str], row: Mapping[str, Any],
                   fieldnames: Sequence[str]) -> None:
    """Append one row under an exclusive lock, writing the header if new.

    Columns absent from ``row`` are written empty and columns absent from
    ``fieldnames`` are dropped, so adding a metric never invalidates a CSV that
    is already half written -- rerun `scripts/e1_collect.py` to widen it.
    """
    target = Path(path)
    with FileLock(target):
        exists = target.is_file() and target.stat().st_size > 0
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames),
                                    extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in fieldnames})
            handle.flush()


def write_csv(path: str | os.PathLike[str], rows: Sequence[Mapping[str, Any]],
              fieldnames: Sequence[str] | None = None) -> None:
    """Atomic full-file CSV write (used when rebuilding from summaries)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names: List[str] = []
        for row in rows:
            for name in row:
                if name not in names:
                    names.append(name)
        fieldnames = names
    temp = target.with_name(target.name + f".partial.{os.getpid()}")
    with open(temp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames),
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


def read_csv_rows(path: str | os.PathLike[str]) -> List[Dict[str, str]]:
    if not Path(path).is_file():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def count_lines(path: str | os.PathLike[str]) -> int:
    if not Path(path).is_file():
        return 0
    total = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            total += chunk.count(b"\n")
    return total


def chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]
