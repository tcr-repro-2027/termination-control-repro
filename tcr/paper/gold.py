# coding: utf-8
"""Reference relations, under the one normalisation the whole project uses.

`strip -> lowercase -> collapse whitespace` is the frozen protocol-v1.0 field
normalisation; E1's F1, P0c's coverage and S2's quality check all use it, so a
closeout number computed here lands on the same scale as the E1 CSV rather than
on a private one.  The strict identity is `(source, target, relation)` and the
relaxed identity is `(source, target)`; a block missing either endpoint has no
identity at all and is excluded rather than counted as a miss.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

Triple = tuple[str, str, str]
Pair = tuple[str, str]


def norm_field(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def canonical_key(value: Any) -> str:
    """The key form 01b freezes into `stable_prompt_id`."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def parse_entities(entities_str: Any) -> list[str]:
    if isinstance(entities_str, (list, tuple)):
        return [str(value) for value in entities_str]
    text = str(entities_str or "").strip()
    if not text:
        return []
    for loader in (ast.literal_eval, json.loads):
        try:
            value = loader(text)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
    return []


def block_identity(block: Mapping[str, Any]) -> Triple | None:
    triple = (norm_field(block.get("source")), norm_field(block.get("target")),
              norm_field(block.get("relation")))
    return triple if triple[0] and triple[1] else None


@dataclass(frozen=True)
class GoldRecord:
    key: Any
    key_id: str
    n_blocks: int
    triples: frozenset[Triple]
    pairs: frozenset[Pair]
    candidates: frozenset[str]
    text: str
    entities_str: str
    #: The reference blocks themselves, kept because X1 builds its probe
    #: contexts out of the exact serialised answer, not out of identities.
    blocks: tuple[Mapping[str, Any], ...] = ()


class GoldSet:
    """Every evaluation record, addressable by `key` or by `stable_prompt_id`."""

    def __init__(self, records: Sequence[GoldRecord]) -> None:
        self.records = list(records)
        self._by_key_id = {record.key_id: record for record in self.records}
        self._by_key = {repr(record.key): record for record in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def by_key(self, key: Any) -> GoldRecord:
        record = self._by_key_id.get(canonical_key(key)) or self._by_key.get(repr(key))
        if record is None:
            raise KeyError(f"no gold record for key={key!r}")
        return record

    def by_prompt_id(self, prompt_id: str) -> GoldRecord:
        record = self._by_key_id.get(str(prompt_id))
        if record is None:
            raise KeyError(f"no gold record for stable_prompt_id={prompt_id!r}")
        return record

    def get(self, key: Any) -> GoldRecord | None:
        try:
            return self.by_key(key)
        except KeyError:
            return None


def load_gold(path: str | Path) -> GoldSet:
    records: list[GoldRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            output = row.get("output", [])
            if isinstance(output, str):
                output = json.loads(output)
            triples: set[Triple] = set()
            pairs: set[Pair] = set()
            for block in output if isinstance(output, list) else []:
                if not isinstance(block, Mapping):
                    continue
                identity = block_identity(block)
                if identity is None:
                    continue
                triples.add(identity)
                pairs.add(identity[:2])
            entities = parse_entities(row.get("entities_str", ""))
            records.append(GoldRecord(
                key=row.get("key"),
                key_id=canonical_key(row.get("key")),
                n_blocks=len(output) if isinstance(output, list) else 0,
                triples=frozenset(triples),
                pairs=frozenset(pairs),
                candidates=frozenset(norm_field(name) for name in entities),
                text=str(row.get("text", "")),
                entities_str=str(row.get("entities_str", "")),
                blocks=tuple(block for block in
                             (output if isinstance(output, list) else [])
                             if isinstance(block, Mapping)),
            ))
    if not records:
        raise ValueError(f"{path} holds no gold record")
    return GoldSet(records)


def blocks_from_spans(text: str, spans: Iterable[Mapping[str, Any]]
                      ) -> list[dict[str, Any] | None]:
    """Re-read each frozen block span out of the response text.

    The event rows carry char spans and hashes, not the field values; the
    values are needed to say which reference relations a prefix already covers.
    A span that no longer parses yields None rather than an exception, so one
    malformed block cannot lose a whole response.
    """
    out: list[dict[str, Any] | None] = []
    for span in spans:
        start, end = int(span["char_start"]), int(span["char_end"])
        try:
            value = json.loads(text[start:end])
        except (ValueError, TypeError):
            out.append(None)
            continue
        out.append(value if isinstance(value, dict) else None)
    return out


def coverage_prefix(blocks: Sequence[Mapping[str, Any] | None], record: GoldRecord
                    ) -> list[dict[str, int]]:
    """Running reference coverage after each block of a response.

    Element `i` describes the state after blocks `0..i`, which is what a
    boundary cut just past block `i` conditions on.
    """
    seen_triples: set[Triple] = set()
    strict: set[Triple] = set()
    relaxed: set[Pair] = set()
    running: list[dict[str, int]] = []
    for block in blocks:
        identity = block_identity(block) if isinstance(block, Mapping) else None
        if identity is not None:
            if identity in record.triples:
                strict.add(identity)
            if identity[:2] in record.pairs:
                relaxed.add(identity[:2])
            seen_triples.add(identity)
        running.append({
            "gold_triples_hit": len(strict),
            "gold_pairs_hit": len(relaxed),
            "unique_triples": len(seen_triples),
            "remaining_gold_pairs": len(record.pairs) - len(relaxed),
            "remaining_gold_triples": len(record.triples) - len(strict),
        })
    return running
