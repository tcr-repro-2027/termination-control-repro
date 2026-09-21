# coding: utf-8
"""Shared helpers for the controlled SCS construction."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FIELDS = ("source", "target", "relation", "description")


def norm(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.strip()).casefold()


def contains(text: str, value: str) -> bool:
    if not value:
        return False
    if value in text:
        return True
    return norm(value) in norm(text)


def stable_hash(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(compact_json(row))
            handle.write("\n")
            count += 1
    return count


def parse_entities(value: str) -> list[str]:
    return [str(x) for x in ast.literal_eval(value)]


def shape(value: str) -> str:
    if any("㐀" <= ch <= "鿿" for ch in value):
        return "cjk"
    if value and all(ch.isascii() and (ch.isalnum() or ch.isspace() or ch in "-_.") for ch in value):
        return "latin"
    return "mixed"


class TokenCounter:
    """Tokenizer-length cache; Qwen3 shares one tokenizer across 1.7B/4B/8B."""

    def __init__(self, tokenizer_dir: Path) -> None:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
        self._cache: dict[str, int] = {}

    def length(self, text: str, cache: bool = True) -> int:
        if not cache:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        cached = self._cache.get(text)
        if cached is None:
            cached = len(self.tokenizer.encode(text, add_special_tokens=False))
            self._cache[text] = cached
        return cached

    def lengths(self, texts: Sequence[str], batch_size: int = 4096) -> list[int]:
        """Return cached token lengths, encoding cache misses in batches."""
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            encoded = self.tokenizer(
                batch, add_special_tokens=False, padding=False, truncation=False,
                return_attention_mask=False, return_token_type_ids=False,
            )["input_ids"]
            for text, token_ids in zip(batch, encoded):
                self._cache[text] = len(token_ids)
        return [self._cache[text] for text in texts]


class EntityPool:
    """Replacement entities bucketed by (token length, script shape)."""

    def __init__(self, entities: Iterable[str], counter: TokenCounter) -> None:
        self.counter = counter
        self.by_key: dict[tuple[int, str], list[str]] = defaultdict(list)
        seen: set[str] = set()
        unique: list[str] = []
        for entity in entities:
            key = norm(entity)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(entity)
        for entity, token_length in zip(unique, counter.lengths(unique)):
            self.by_key[(token_length, shape(entity))].append(entity)
        for values in self.by_key.values():
            values.sort(key=lambda value: (norm(value), value))

    def pick(self, text: str, banned: set[str], token_length: int,
             entity_shape: str, salt: str) -> str | None:
        found = self.pick_many(text, banned, token_length, entity_shape, salt, 1)
        return found[0] if found else None

    def pick_many(self, text: str, banned: set[str], token_length: int,
                  entity_shape: str, salt: str, limit: int) -> list[str]:
        """Replacements absent from `text` and from `banned`, best-shape first.

        `banned` holds normalized strings.  Falls back to the same token length
        with a different shape before giving up, so length parity is preserved.
        """
        found: list[str] = []
        text_folded = norm(text)
        for key in ((token_length, entity_shape),
                    *[(token_length, other) for other in ("cjk", "latin", "mixed")
                      if other != entity_shape]):
            values = self.by_key.get(key)
            if not values:
                continue
            start = stable_hash(salt, key[1]) % len(values)
            for offset in range(len(values)):
                candidate = values[(start + offset) % len(values)]
                folded = norm(candidate)
                if folded in banned:
                    continue
                if candidate in text or folded in text_folded:
                    continue
                found.append(candidate)
                if len(found) >= limit:
                    return found
            if found:
                return found
        return found


def nested_keys(entities: Sequence[str]) -> set[str]:
    keys = list(dict.fromkeys(norm(entity) for entity in entities if norm(entity)))
    return {key for key in keys if any(key != other and key in other for other in keys)}


def support_state(text: str, entity_set: set[str],
                  blocks: Sequence[Mapping[str, str]]) -> list[tuple[bool, bool]]:
    """Per-block (admissible, evidenced) flags used to detect collateral damage."""
    state: list[tuple[bool, bool]] = []
    for block in blocks:
        admissible = block["source"] in entity_set and block["target"] in entity_set
        evidenced = contains(text, block["source"]) and contains(text, block["target"])
        state.append((admissible, evidenced))
    return state


def joint_label(admissible: bool, evidenced: bool) -> str:
    if admissible and evidenced:
        return "Valid"
    if not admissible and not evidenced:
        return "AE"
    return "A" if not admissible else "E"


def position_bucket(index: int, total: int) -> str:
    relative = index / max(total - 1, 1)
    return "early" if relative < 0.25 else "middle" if relative < 0.75 else "late"
