# coding: utf-8
"""Observed-Block Reinsertion helpers.

An OBR pair always re-inserts a filter->clean problem-4 block into the cleanv2
version of the *same* record. Position bucket is hard-constrained. Graph
degree, relative position and block-token length are jointly matched rather
than turning exact degree into an artificial capacity ceiling.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .common import TokenCounter, compact_json, contains, norm, parse_entities, position_bucket, stable_hash


@dataclass
class HostBlock:
    block_index: int
    bucket: str
    relative_position: float
    token_length: int
    source_only: bool
    target_only: bool
    both_endpoints: bool
    source_nested: bool
    target_nested: bool
    source_degree: int
    target_degree: int


@dataclass
class ObrPair:
    line: int
    rec_id: str
    host_block_index: int
    filter_block_index: int
    block: dict[str, str]
    joint_type: str
    host_bucket: str
    drop_bucket: str
    token_delta: int
    degree_distance: float
    position_distance: float
    host_source_degree: int
    host_target_degree: int
    host_relative_position: float
    token_balancing_move: bool


def describe_host(record: Mapping[str, Any], counter: TokenCounter, nested: set[str]) -> list[HostBlock]:
    blocks = record["output"]
    total = len(blocks)
    degrees: Counter[str] = Counter()
    for block in blocks:
        degrees[norm(block["source"])] += 1
        degrees[norm(block["target"])] += 1
    hosts: list[HostBlock] = []
    for index, block in enumerate(blocks):
        has_source = contains(block["description"], block["source"])
        has_target = contains(block["description"], block["target"])
        hosts.append(HostBlock(
            block_index=index,
            bucket=position_bucket(index, total),
            relative_position=index / max(total - 1, 1),
            token_length=counter.length(compact_json(dict(block)), cache=False),
            source_only=has_source and not has_target,
            target_only=has_target and not has_source,
            both_endpoints=has_source and has_target,
            source_nested=norm(block["source"]) in nested,
            target_nested=norm(block["target"]) in nested,
            source_degree=degrees[norm(block["source"])],
            target_degree=degrees[norm(block["target"])],
        ))
    return hosts


def degree_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (str(row["position_bucket"]), int(row["source_degree"]), int(row["target_degree"]))


def host_degree_key(host: HostBlock) -> tuple[Any, ...]:
    return (host.bucket, host.source_degree, host.target_degree)


def profile_penalty(drop: Mapping[str, Any], host: HostBlock) -> int:
    return sum((bool(drop[flag]) != bool(getattr(host, flag))) for flag in (
        "source_only", "target_only", "both_endpoints", "source_nested", "target_nested"))


def degree_distance(drop: Mapping[str, Any], host: HostBlock) -> float:
    """Symmetric log-distance between the filter and host graph degrees."""
    return (abs(math.log1p(int(drop["source_degree"])) - math.log1p(host.source_degree)) +
            abs(math.log1p(int(drop["target_degree"])) - math.log1p(host.target_degree)))


def position_distance(drop: Mapping[str, Any], host: HostBlock) -> float:
    return abs(float(drop.get("relative_position", 0.5)) - host.relative_position)


def allocate(capacities: Mapping[str, int], target: int) -> dict[str, int]:
    """Largest-remainder allocation proportional to capacity."""
    total = sum(capacities.values())
    if total <= target:
        return dict(capacities)
    exact = {key: value * target / total for key, value in capacities.items()}
    quota = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = target - sum(quota.values())
    for key in sorted(capacities, key=lambda key: (-(exact[key] - quota[key]), key)):
        if remaining <= 0:
            break
        if quota[key] < capacities[key]:
            quota[key] += 1
            remaining -= 1
    return quota


def pair_record(line: int, rec_id: str, record: Mapping[str, Any],
                drops: Sequence[Mapping[str, Any]], quota: int,
                counter: TokenCounter, budget_ratio: float = 0.01) -> list[ObrPair]:
    """Pair selected OBR drops to unused clean blocks in one record.

    Record identity and position bucket are never relaxed. Within a bucket,
    exact `(source_degree, target_degree)` equality is consumed first, which
    maximizes exact degree pairs. Remaining pairs use nearest log degree, then
    relative position, token length and coverage/nesting profile.
    """
    del budget_ratio
    if quota <= 0 or not drops:
        return []
    entities = parse_entities(record["entities_str"])
    keys = list(dict.fromkeys(norm(e) for e in entities if norm(e)))
    nested = {key for key in keys if any(key != other and key in other for other in keys)}
    hosts = describe_host(record, counter, nested)
    prepared = []
    for drop in sorted(drops, key=lambda row: stable_hash(rec_id, row["filter_block_index"]))[:quota]:
        block = {name: str(drop["block"][name]) for name in ("source", "target", "relation", "description")}
        prepared.append((drop, block, counter.length(compact_json(block), cache=False)))

    chosen: list[tuple[Mapping[str, Any], dict[str, str], int, HostBlock]] = []
    for bucket in ("early", "middle", "late"):
        pending = [item for item in prepared if str(item[0]["position_bucket"]) == bucket]
        available = [host for host in hosts if host.bucket == bucket]
        if len(pending) > len(available):
            raise ValueError(f"{rec_id}: selected {len(pending)} {bucket} drops but only {len(available)} same-bucket hosts")
        drops_by_degree: dict[tuple[int, int], list] = defaultdict(list)
        hosts_by_degree: dict[tuple[int, int], list[HostBlock]] = defaultdict(list)
        for item in pending:
            drop = item[0]
            drops_by_degree[(int(drop["source_degree"]), int(drop["target_degree"]))].append(item)
        for host in available:
            hosts_by_degree[(host.source_degree, host.target_degree)].append(host)
        matched: set[int] = set()
        used: set[int] = set()
        for key, values in drops_by_degree.items():
            free = list(hosts_by_degree.get(key, []))
            for drop, block, length in sorted(values, key=lambda item: (-item[2], stable_hash(rec_id, item[0]["filter_block_index"]))):
                if not free:
                    break
                host = min(free, key=lambda candidate: (
                    abs(candidate.token_length - length), position_distance(drop, candidate),
                    profile_penalty(drop, candidate), candidate.block_index))
                free.remove(host)
                used.add(host.block_index)
                matched.add(int(drop["filter_block_index"]))
                chosen.append((drop, block, length, host))
        remainder = [item for item in pending if int(item[0]["filter_block_index"]) not in matched]
        free = [host for host in available if host.block_index not in used]
        for drop, block, length in sorted(remainder, key=lambda item: (
                -int(item[0]["source_degree"]) - int(item[0]["target_degree"]),
                stable_hash(rec_id, item[0]["filter_block_index"]))):
            nearest_degree = min(degree_distance(drop, candidate) for candidate in free)
            degree_candidates = [candidate for candidate in free if degree_distance(
                drop, candidate) <= nearest_degree + 0.45]
            nearest_position = min(position_distance(drop, candidate)
                                   for candidate in degree_candidates)
            candidates = [candidate for candidate in degree_candidates if position_distance(
                drop, candidate) <= nearest_position + 0.06]
            host = min(candidates, key=lambda candidate: (
                abs(candidate.token_length - length), degree_distance(drop, candidate),
                position_distance(drop, candidate), profile_penalty(drop, candidate), candidate.block_index))
            free.remove(host)
            chosen.append((drop, block, length, host))
    # The greedy per-drop selection has no record-level token objective.  Pull
    # the signed record delta toward zero using only unused hosts in the same
    # position bucket. Exact-degree pairs cannot be degraded; residual pairs
    # may move only within a small degree/relative-position envelope.
    moved_filter_indices: set[int] = set()
    free_by_bucket = {
        bucket: [host for host in hosts if host.bucket == bucket and host.block_index not in {
            item[3].block_index for item in chosen}]
        for bucket in ("early", "middle", "late")
    }
    record_delta = sum(length - host.token_length for _, _, length, host in chosen)
    for _ in range(12):
        best: tuple[float, int, HostBlock, float] | None = None
        for index, (drop, block, length, host) in enumerate(chosen):
            exact = (int(drop["source_degree"]) == host.source_degree and
                     int(drop["target_degree"]) == host.target_degree)
            old_degree = degree_distance(drop, host)
            old_position = position_distance(drop, host)
            for candidate in free_by_bucket[host.bucket]:
                if exact:
                    if (candidate.source_degree, candidate.target_degree) != (
                            int(drop["source_degree"]), int(drop["target_degree"])):
                        continue
                elif (degree_distance(drop, candidate) > old_degree + 0.12 or
                      position_distance(drop, candidate) > old_position + 0.04):
                    continue
                new_delta = record_delta - (length - host.token_length) + (
                    length - candidate.token_length)
                gain = abs(record_delta) - abs(new_delta)
                if gain <= 0:
                    continue
                contender = (gain, index, candidate, new_delta)
                if best is None or contender[0] > best[0] or (
                        contender[0] == best[0] and candidate.block_index < best[2].block_index):
                    best = contender
        if best is None:
            break
        _, index, candidate, record_delta = best
        drop, block, length, host = chosen[index]
        free_by_bucket[host.bucket].remove(candidate)
        free_by_bucket[host.bucket].append(host)
        chosen[index] = (drop, block, length, candidate)
        moved_filter_indices.add(int(drop["filter_block_index"]))
    return [ObrPair(
        line=line, rec_id=rec_id, host_block_index=host.block_index,
        filter_block_index=int(drop["filter_block_index"]), block=block,
        joint_type=str(drop["joint_type"]), host_bucket=host.bucket,
        drop_bucket=str(drop["position_bucket"]), token_delta=length - host.token_length,
        degree_distance=round(degree_distance(drop, host), 6),
        position_distance=round(position_distance(drop, host), 6),
        host_source_degree=host.source_degree,
        host_target_degree=host.target_degree,
        host_relative_position=host.relative_position,
        token_balancing_move=int(drop["filter_block_index"]) in moved_filter_indices,
    ) for drop, block, length, host in sorted(chosen, key=lambda item: item[3].block_index)]


def apply_obr(record: Mapping[str, Any], pairs: Sequence[ObrPair]) -> list[dict[str, str]]:
    blocks = [dict(block) for block in record["output"]]
    for pair in pairs:
        blocks[pair.host_block_index] = dict(pair.block)
    return blocks


def composition(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    counts = Counter(str(row[key]) for row in rows)
    total = max(sum(counts.values()), 1)
    return {name: round(value / total, 4) for name, value in sorted(counts.items())}
