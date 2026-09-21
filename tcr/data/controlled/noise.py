# coding: utf-8
"""GenericNoise content degradation for relation *and* description.

Section 4.4 confines the noise to the `relation`/`description` fields and names
three mechanisms: relation abstraction degradation, clause reordering, and
same-domain content substitution.  Both fields are degraded on every anchor
block, because the baseline only answers its question ("is support conflict
worse than an equal amount of content noise that leaves block existence
intact?") if the amount of noise is comparable to what ISC changes.

The description operation is intentionally conservative: it only reorders
clauses already present in the target block.  No description material is ever
borrowed from a different block, so unreviewed construction cannot introduce a
cross-subtopic statement. Relations are selected from the same source record.
Endpoint literals are preserved exactly.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .common import TokenCounter, norm, stable_hash

# Clause boundaries in the synthetic Chinese descriptions.  The delimiters are
# kept so the rebuilt description has the identical punctuation pattern.
CLAUSE_SPLIT = re.compile(r"([，,；;。])")
POOL_CANDIDATES = 24


def split_clauses(text: str) -> tuple[list[str], list[str]]:
    """Return (clauses, delimiters) with len(delimiters) == len(clauses)."""
    parts = CLAUSE_SPLIT.split(text)
    clauses: list[str] = []
    delimiters: list[str] = []
    for index in range(0, len(parts), 2):
        clause = parts[index]
        delimiter = parts[index + 1] if index + 1 < len(parts) else ""
        if not clause.strip():
            # A stray delimiter with no text before it stays attached to the
            # previous clause so nothing is silently dropped.
            if delimiters:
                delimiters[-1] += clause + delimiter
            continue
        clauses.append(clause)
        delimiters.append(delimiter)
    return clauses, delimiters


def join_clauses(clauses: Sequence[str], delimiters: Sequence[str]) -> str:
    return "".join(clause + delimiter for clause, delimiter in zip(clauses, delimiters))


class DomainPool(dict[int, list[str]]):
    """Per-record semantic material for GenericNoise.

    A cleanv2 record is one source document.  Restricting every replacement to
    that document is deliberately conservative: a substitution can be noisy,
    but cannot inject a clause or relation from an unrelated subject area.
    The dict base remains only for the old build script's diagnostic count.
    """

    def __init__(self) -> None:
        super().__init__()
        self.clauses: dict[int, dict[int, list[tuple[str, int]]]] = {}
        self.relations: dict[int, dict[int, list[tuple[str, int]]]] = {}

    def pick_clause(self, line: int, length: int, banned: set[str],
                    forbidden_substrings: Sequence[str], salt: str,
                    bias: int) -> tuple[str, int] | None:
        return _pick_local(
            self.clauses.get(line, {}), length, banned, forbidden_substrings,
            salt, bias)

    def pick_relation(self, line: int, original: str, counter: TokenCounter,
                      salt: str) -> tuple[str, int] | None:
        """Pick a relation from this document, preferring equal token length."""
        values_by_length = self.relations.get(line, {})
        original_key = norm(original)
        length = counter.length(original)
        lengths = [length]
        for offset in range(1, 5):
            lengths.extend((length - offset, length + offset))
        for candidate_length in lengths:
            values = values_by_length.get(candidate_length)
            if not values:
                continue
            start = stable_hash(salt, candidate_length, "relation") % len(values)
            for offset in range(len(values)):
                candidate, block_index = values[(start + offset) % len(values)]
                if norm(candidate) != original_key:
                    return candidate, block_index
        return None


def build_clause_pool(records: Iterable[Mapping[str, Any]],
                      counter: TokenCounter) -> DomainPool:
    """Build a document-local pool, retaining source block provenance."""
    pool = DomainPool()
    all_by_length: dict[int, list[str]] = defaultdict(list)
    for line, record in enumerate(records):
        relations_by_length: dict[int, list[tuple[str, int]]] = defaultdict(list)
        seen_relations: set[str] = set()
        for block_index, block in enumerate(record["output"]):
            relation = str(block["relation"])
            relation_key = norm(relation)
            if relation_key and relation_key not in seen_relations:
                seen_relations.add(relation_key)
                relations_by_length[counter.length(relation)].append((relation, block_index))
        for values in relations_by_length.values():
            values.sort(key=lambda item: (norm(item[0]), item[0], item[1]))
        pool.clauses[line] = {}
        pool.relations[line] = dict(relations_by_length)
    for length, values in all_by_length.items():
        pool[length] = values
    return pool


def _pick_local(pool: Mapping[int, Sequence[tuple[str, int]]], length: int,
                banned: set[str], forbidden_substrings: Sequence[str], salt: str,
                bias: int) -> tuple[str, int] | None:
    """Pick a same-document clause with nearest token length."""
    lengths = [length]
    for offset in (1, 2, 3):
        lengths += [length - offset, length + offset]
    for candidate_length in lengths:
        if candidate_length <= 0:
            continue
        # `bias` is the record's running token surplus; prefer a length that
        # pulls it back toward zero when the exact length is unavailable.
        values = pool.get(candidate_length)
        if not values:
            continue
        start = stable_hash(salt, candidate_length) % len(values)
        for step in range(min(POOL_CANDIDATES, len(values))):
            candidate, block_index = values[(start + step) % len(values)]
            if norm(candidate) in banned:
                continue
            if any(token and token in candidate for token in forbidden_substrings):
                continue
            return candidate, block_index
    return None


def degrade_description(description: str, source: str, target: str,
                        record_entities: Sequence[str],
                        pool: DomainPool, line: int, counter: TokenCounter,
                        salt: str, bias: int = 0) -> tuple[str, str, list[int]]:
    """Return ``(new_description, mechanism, source_block_indices)``.

    The only operation is a deterministic reorder of clauses that already
    belong to this block. This keeps all factual material local while still
    degrading the order in which the evidence is presented.
    """
    clauses, delimiters = split_clauses(description)
    if not clauses:
        return description, "unchanged_no_clause", []

    reordered = _reorder(clauses, salt)
    changed_order = reordered != clauses
    result = join_clauses(reordered, delimiters)

    # An endpoint mention can straddle a clause boundary (the entity itself
    # contains a comma or period).  Rewriting or moving clauses would then cut
    # the literal in half, so verify and fall back rather than corrupt support.
    for endpoint in (source, target):
        if endpoint and description.count(endpoint) != result.count(endpoint):
            return description, "unchanged_endpoint_spans_clause", []

    if changed_order:
        mechanism = "reorder"
    else:
        mechanism = "unchanged_single_clause"
    return result, mechanism, []


def _reorder(clauses: Sequence[str], salt: str) -> list[str]:
    """Deterministic derangement; falls back to a rotation for 2 clauses."""
    count = len(clauses)
    if count < 2:
        return list(clauses)
    if count == 2:
        return [clauses[1], clauses[0]]
    order = sorted(range(count), key=lambda i: stable_hash(salt, "order", i))
    if all(order[i] == i for i in range(count)):
        order = order[1:] + order[:1]
    # Guarantee at least one moved clause even if the hash happens to be sorted.
    if order == list(range(count)):
        order[0], order[-1] = order[-1], order[0]
    return [clauses[i] for i in order]


def apply_generic_noise(record: Mapping[str, Any], anchors: Sequence[Any],
                        pool: DomainPool, counter: TokenCounter,
                        stats: Counter | None = None,
                        rows: list[dict[str, Any]] | None = None,
                        budget_ratio: float = 0.01) -> list[dict[str, str]]:
    """Degrade relation and description on every anchor block of one record."""
    blocks = [dict(block) for block in record["output"]]
    entities = record.get("entities_str", "")
    running = 0
    for anchor in anchors:
        block = blocks[anchor.block_index]
        before = counter.length(block["relation"]) + counter.length(block["description"])
        relation_pick = pool.pick_relation(
            anchor.line, block["relation"], counter,
            f"{anchor.rec_id}:{anchor.block_index}")
        if relation_pick is None:
            relation_after = block["relation"]
            relation_source_block = None
            relation_mechanism = "unchanged_no_local_alternative"
        else:
            relation_after, relation_source_block = relation_pick
            relation_mechanism = "same_record_substitution"
        block["relation"] = relation_after
        description, mechanism, description_source_blocks = degrade_description(
            block["description"], block["source"], block["target"], entities,
            pool, anchor.line, counter, f"{anchor.rec_id}:{anchor.block_index}", running)
        block["description"] = description
        after = counter.length(block["relation"]) + counter.length(block["description"])
        running += after - before
        if stats is not None:
            stats[mechanism] += 1
            stats[f"relation:{relation_mechanism}"] += 1
        if rows is not None:
            rows.append({
                "rec_id": anchor.rec_id, "line": anchor.line,
                "block_index": anchor.block_index, "mechanism": mechanism,
                "domain_scope": "same_cleanv2_record",
                "relation_mechanism": relation_mechanism,
                "relation_source_line": anchor.line if relation_source_block is not None else None,
                "relation_source_block_index": relation_source_block,
                "description_source_line": anchor.line if description_source_blocks else None,
                "description_source_block_indices": description_source_blocks,
                "relation_before": record["output"][anchor.block_index]["relation"],
                "relation_after": block["relation"],
                "description_before": record["output"][anchor.block_index]["description"],
                "description_after": block["description"],
                "token_delta": after - before,
            })
    return blocks
