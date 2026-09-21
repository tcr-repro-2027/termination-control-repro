# coding: utf-8
"""Staged reconstruction of base / dedup / filter / clean / cleanv2.

One record universe, one identity rule.  Every stage is a strict subset of the
previous one, so record and block counts are monotonically non-increasing:

    base  >=  dedup  >=  filter  >=  clean  >=  cleanv2

Stage definitions:

base     structural load only; cross-split text leakage removed from train.
dedup    exact duplicate records dropped + exact (source,target,relation)
         duplicates dropped inside each record.  Reproduces the historical
         train_dedup.jsonl exactly (9,778 records / 922,570 edges).
filter   problems 1-3: output schema repair, entities_str repair,
         ordered (source,target) deduplication.
clean    problem 4: source/target must be in the candidate list and in text.
cleanv2  problems 5,6,7,9,10,11.  Problem 8 (nested entities) is kept.

Record identity is `rec_id = "<split>:row_<n>"` taken from the original file
and never recomputed, so filter->clean block diffs pair exactly.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FIELDS = ("source", "target", "relation", "description")

KEY_ALIASES = {
    "描述": "description",      # 描述
    "关系": "relation",         # 关系
    "起始实体": "source",   # 起始实体
    "终止实体": "target",   # 终止实体
}

STAGES = ("base", "dedup", "filter", "clean", "cleanv2")

# cleanv2 writes the frozen names used everywhere in the planning document.
STAGE_FILENAMES = {
    "base": "{split}_base.jsonl",
    "dedup": "{split}_dedup.jsonl",
    "filter": "{split}_filter.jsonl",
    "clean": "{split}_clean.jsonl",
    "cleanv2": "{split}_supportclean_keep8.jsonl",
}


def norm(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.strip()).casefold()


def contains(text: str, value: str) -> bool:
    if not value:
        return False
    if value in text:
        return True
    return norm(value) in norm(text)


def is_low_quality_entity(value: str) -> bool:
    """Problem 10: single characters, short bare numbers, punctuation-only."""
    text = value.strip()
    if len(text) <= 1:
        return True
    if re.fullmatch(r"[+-]?\d{1,4}(?:\.\d+)?", text):
        return True
    return not any(ch.isalnum() or "㐀" <= ch <= "鿿" for ch in text)


def _try_literal(value: str) -> tuple[Any, str | None]:
    for name, parser in (("python_literal", ast.literal_eval), ("json", json.loads)):
        try:
            return parser(value), name
        except (ValueError, SyntaxError, TypeError):
            continue
    return None, None


def parse_entities(raw: Any) -> tuple[list[str] | None, str]:
    """Problem 2: parse the entity list, repairing the known truncated tail."""
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw], "native"
    if not isinstance(raw, str) or not raw.strip():
        return None, "not_a_string"
    text = raw.strip()
    value, method = _try_literal(text)
    if method and isinstance(value, (list, tuple)):
        return [str(x) for x in value], method
    for suffix in ("']", '"]', "' ]", '" ]', "]"):
        value, method = _try_literal(text + suffix)
        if method and isinstance(value, (list, tuple)):
            return [str(x) for x in value], f"{method}+tail_suffix"
    return None, "parse_failed"


def canonical_block(item: Any) -> tuple[dict[str, str] | None, list[str], str | None]:
    """Problem 1: alias repair, then reject blocks that stay unusable."""
    if not isinstance(item, Mapping):
        return None, [], "block_not_object"
    source = dict(item)
    repairs: list[str] = []
    for alias, name in KEY_ALIASES.items():
        if name not in source and alias in source:
            source[name] = source[alias]
            repairs.append(f"field_alias:{alias}->{name}")
    missing = [name for name in FIELDS if name not in source]
    if missing:
        return None, repairs, f"missing_fields:{','.join(missing)}"
    block: dict[str, str] = {}
    for name in FIELDS:
        value = source[name]
        if value is None or isinstance(value, (dict, list, tuple, set)):
            return None, repairs, f"invalid_value:{name}"
        block[name] = str(value).strip()
        if not block[name]:
            return None, repairs, f"empty_field:{name}"
    return block, repairs, None


def relation_leak(block: Mapping[str, str]) -> tuple[str | None, bool]:
    """Problem 7, using the frozen union definition from the analysis report.

    Returns (reason, swapped).  `swapped` means relation/description are clearly
    inverted and can be repaired instead of dropped.
    """
    relation, description = block["relation"], block["description"]
    if norm(relation) == norm(description):
        return "relation_equals_description", False
    has_punctuation = any(ch in relation for ch in ",，.。")
    too_long = len(relation) > 10
    if not has_punctuation and not too_long:
        return None, False
    # A long sentence in `relation` next to a short label in `description`
    # is the observed swap; repair it instead of discarding supervision.
    if len(relation) > 30 and len(description) <= 20:
        return "relation_description_swapped", True
    return "relation_leak", False


def block_quality(block: Mapping[str, str], index: int) -> tuple[int, int, int, int]:
    """Deterministic tie-break for deduplication; never uses support state."""
    description = block["description"]
    grounded = int(contains(description, block["source"])) + int(
        contains(description, block["target"]))
    return grounded, len(description), len(block["relation"]), -index


@dataclass
class Record:
    rec_id: str
    split: str
    row_index: int
    extra: dict[str, Any]
    text: str
    entities: list[str]
    entities_str: str
    blocks: list[dict[str, str]]
    # block_index inside the *filter* stage, kept so the filter->clean diff and
    # OBR reinsertion can address the very same position.
    filter_index: list[int] = field(default_factory=list)


@dataclass
class StageStats:
    records_in: int = 0
    records_out: int = 0
    blocks_in: int = 0
    blocks_out: int = 0
    record_drops: Counter = field(default_factory=Counter)
    block_drops: Counter = field(default_factory=Counter)
    repairs: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "records_in": self.records_in, "records_out": self.records_out,
            "blocks_in": self.blocks_in, "blocks_out": self.blocks_out,
            "record_drops": dict(self.record_drops),
            "block_drops": dict(self.block_drops),
            "repairs": dict(self.repairs),
        }


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: top-level value is not an object")
            yield value


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def to_payload(record: Record) -> dict[str, Any]:
    """Emit the exact training schema: no bookkeeping fields leak into the data."""
    payload = dict(record.extra)
    payload["text"] = record.text
    payload["entities_str"] = record.entities_str
    payload["output"] = record.blocks
    return payload


# --------------------------------------------------------------------------
# stage 0: base
# --------------------------------------------------------------------------

def load_base(path: Path, split: str, stats: StageStats) -> list[Record]:
    records: list[Record] = []
    for row_index, row in enumerate(read_jsonl(path)):
        stats.records_in += 1
        text = row.get("text")
        output = row.get("output")
        if isinstance(output, Mapping):
            output = [output]
        if isinstance(output, list):
            stats.blocks_in += len(output)
        if not isinstance(text, str) or not text.strip():
            stats.record_drops["missing_text"] += 1
            continue
        if not isinstance(output, list) or not output:
            stats.record_drops["missing_or_empty_output"] += 1
            continue
        extra = {k: v for k, v in row.items()
                 if k not in ("text", "entities_str", "output")}
        entities, _ = parse_entities(row.get("entities_str"))
        records.append(Record(
            rec_id=f"{split}:row_{row_index + 1:06d}",
            split=split,
            row_index=row_index,
            extra=extra,
            text=text,
            entities=list(entities or []),
            entities_str=str(row.get("entities_str", "")),
            # Non-dict entries are left untouched here; the filter stage is
            # where problem 1 decides to repair or drop them.
            blocks=list(output),
        ))
    stats.records_out = len(records)
    stats.blocks_out = sum(len(r.blocks) for r in records)
    return records


def drop_cross_split_text(train: list[Record], eval_rows: list[Record],
                          stats: StageStats) -> list[Record]:
    """Train records whose document also appears in eval are removed."""
    eval_texts = {norm(r.text) for r in eval_rows}
    kept = []
    for record in train:
        if norm(record.text) in eval_texts:
            stats.record_drops["cross_split_text_overlap"] += 1
            stats.blocks_out -= len(record.blocks)
            continue
        kept.append(record)
    stats.records_out = len(kept)
    return kept


def drop_near_duplicate_text(train: list[Record], removals: set[str],
                             stats: StageStats) -> list[Record]:
    """Remove train records whose document is largely contained in an eval one.

    Exact-match removal only catches byte-identical documents.  `removals` is
    produced by `near_duplicate_scan.py --source raw` and holds the train
    `rec_id`s whose text is at least the gate fraction contained in some
    evaluation document, which is the form of leakage that actually lets a
    model memorise an evaluation item.
    """
    kept = []
    for record in train:
        if record.rec_id in removals:
            stats.record_drops["near_duplicate_of_eval"] += 1
            stats.blocks_out -= len(record.blocks)
            continue
        kept.append(record)
    stats.records_out = len(kept)
    return kept


# --------------------------------------------------------------------------
# stage 1: dedup
# --------------------------------------------------------------------------

def stage_dedup(records: Sequence[Record], stats: StageStats) -> list[Record]:
    stats.records_in = len(records)
    stats.blocks_in = sum(len(r.blocks) for r in records)
    seen: set[tuple[str, str]] = set()
    kept: list[Record] = []
    for record in records:
        key = (record.text, record.entities_str)
        if key in seen:
            stats.record_drops["duplicate_record"] += 1
            continue
        seen.add(key)
        survivors: list[dict[str, str]] = []
        seen_triples: set[tuple[str, str, str]] = set()
        for block in record.blocks:
            if not isinstance(block, dict):
                survivors.append(block)
                continue
            triple = (str(block.get("source", "")), str(block.get("target", "")),
                      str(block.get("relation", "")))
            if triple in seen_triples:
                stats.block_drops["duplicate_source_target_relation"] += 1
                continue
            seen_triples.add(triple)
            survivors.append(block)
        record.blocks = survivors
        if not survivors:
            stats.record_drops["no_blocks_after_dedup"] += 1
            continue
        kept.append(record)
    stats.records_out = len(kept)
    stats.blocks_out = sum(len(r.blocks) for r in kept)
    return kept


# --------------------------------------------------------------------------
# stage 2: filter (problems 1-3)
# --------------------------------------------------------------------------

def stage_filter(records: Sequence[Record], stats: StageStats) -> list[Record]:
    stats.records_in = len(records)
    stats.blocks_in = sum(len(r.blocks) for r in records)
    kept: list[Record] = []
    for record in records:
        # problem 2: repair the serialized entity list, drop duplicates.
        parsed, method = parse_entities(record.entities_str)
        if parsed is None:
            stats.record_drops["entities_parse_failed"] += 1
            continue
        if method.endswith("tail_suffix"):
            stats.repairs["entities_tail_suffix"] += 1
        entities: list[str] = []
        seen_entity: set[str] = set()
        for value in parsed:
            entity = str(value).strip()
            key = norm(entity)
            if not entity or key in seen_entity:
                if entity:
                    stats.repairs["duplicate_entity_removed"] += 1
                continue
            seen_entity.add(key)
            entities.append(entity)

        # problem 1: alias repair, then drop blocks that remain unusable.
        canonical: list[dict[str, str]] = []
        for block in record.blocks:
            item, repairs, error = canonical_block(block)
            for repair in repairs:
                stats.repairs[repair] += 1
            if item is None:
                stats.block_drops[error or "block_invalid"] += 1
                continue
            canonical.append(item)

        # problem 3: ordered (source, target) redundancy.
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, block in enumerate(canonical):
            groups[(norm(block["source"]), norm(block["target"]))].append(index)
        keep_indices: set[int] = set()
        for indices in groups.values():
            best = max(indices, key=lambda i: block_quality(canonical[i], i))
            keep_indices.add(best)
            if len(indices) > 1:
                stats.block_drops["duplicate_source_target"] += len(indices) - 1
        survivors = [block for index, block in enumerate(canonical) if index in keep_indices]

        if not survivors:
            stats.record_drops["no_blocks_after_filter"] += 1
            continue
        record.entities = entities
        record.entities_str = repr(entities)
        record.blocks = survivors
        record.filter_index = list(range(len(survivors)))
        kept.append(record)
    stats.records_out = len(kept)
    stats.blocks_out = sum(len(r.blocks) for r in kept)
    return kept


# --------------------------------------------------------------------------
# stage 3: clean (problem 4)
# --------------------------------------------------------------------------

def joint_type(source_ok_list: bool, target_ok_list: bool,
               source_ok_text: bool, target_ok_text: bool) -> str:
    admissibility = not (source_ok_list and target_ok_list)
    evidence = not (source_ok_text and target_ok_text)
    if admissibility and evidence:
        return "AE"
    if admissibility:
        return "A"
    if evidence:
        return "E"
    return "Valid"


def stage_clean(records: Sequence[Record], stats: StageStats) -> tuple[list[Record], list[dict[str, Any]]]:
    """Drop out-of-candidate / out-of-text endpoints and log every removal."""
    stats.records_in = len(records)
    stats.blocks_in = sum(len(r.blocks) for r in records)
    kept: list[Record] = []
    diff_rows: list[dict[str, Any]] = []
    for record in records:
        entity_set = set(record.entities)
        entity_keys = {norm(e) for e in record.entities}
        nested_keys = {key for key in entity_keys
                       if any(key != other and key in other for other in entity_keys)}
        degrees: Counter[str] = Counter()
        for block in record.blocks:
            degrees[norm(block["source"])] += 1
            degrees[norm(block["target"])] += 1
        denominator = max(len(record.blocks) - 1, 1)
        survivors: list[dict[str, str]] = []
        survivor_index: list[int] = []
        for index, block in enumerate(record.blocks):
            source_in_list = block["source"] in entity_set
            target_in_list = block["target"] in entity_set
            source_in_text = contains(record.text, block["source"])
            target_in_text = contains(record.text, block["target"])
            kind = joint_type(source_in_list, target_in_list, source_in_text, target_in_text)
            if kind == "Valid":
                survivors.append(block)
                survivor_index.append(record.filter_index[index])
                continue
            stats.block_drops[f"problem4_{kind}"] += 1
            relative = index / denominator
            diff_rows.append({
                "rec_id": record.rec_id,
                "split": record.split,
                "filter_block_index": record.filter_index[index],
                "block": block,
                "joint_type": kind,
                "source_in_list": source_in_list, "target_in_list": target_in_list,
                "source_in_text": source_in_text, "target_in_text": target_in_text,
                "source_only": contains(block["description"], block["source"])
                and not contains(block["description"], block["target"]),
                "target_only": contains(block["description"], block["target"])
                and not contains(block["description"], block["source"]),
                "both_endpoints": contains(block["description"], block["source"])
                and contains(block["description"], block["target"]),
                "source_nested": norm(block["source"]) in nested_keys,
                "target_nested": norm(block["target"]) in nested_keys,
                "source_degree": degrees[norm(block["source"])],
                "target_degree": degrees[norm(block["target"])],
                "relative_position": relative,
                "position_bucket": ("early" if relative < 0.25
                                    else "middle" if relative < 0.75 else "late"),
            })
        if not survivors:
            stats.record_drops["no_blocks_after_clean"] += 1
            continue
        record.blocks = survivors
        record.filter_index = survivor_index
        kept.append(record)
    stats.records_out = len(kept)
    stats.blocks_out = sum(len(r.blocks) for r in kept)
    return kept, diff_rows


# --------------------------------------------------------------------------
# stage 4: cleanv2 (problems 5,6,7,9,10,11; problem 8 kept)
# --------------------------------------------------------------------------

def stage_cleanv2(records: Sequence[Record], stats: StageStats,
                  protected_descriptions: set[str]) -> list[Record]:
    stats.records_in = len(records)
    stats.blocks_in = sum(len(r.blocks) for r in records)
    kept: list[Record] = []
    for record in records:
        # problem 10: drop low-quality entities from the candidate list.
        entities: list[str] = []
        for entity in record.entities:
            if is_low_quality_entity(entity):
                stats.repairs["low_quality_entity_removed"] += 1
                continue
            entities.append(entity)
        entity_set = set(entities)

        survivors: list[dict[str, str]] = []
        survivor_index: list[int] = []
        for index, block in enumerate(record.blocks):
            block = dict(block)
            # problem 7: relation/description leakage.
            reason, swapped = relation_leak(block)
            if swapped:
                block["relation"], block["description"] = block["description"], block["relation"]
                stats.repairs["relation_description_swapped"] += 1
                reason, swapped = relation_leak(block)
            if reason:
                stats.block_drops[f"problem7_{reason}"] += 1
                continue
            # problem 10 fallout: an endpoint whose candidate entry is gone.
            if block["source"] not in entity_set or block["target"] not in entity_set:
                stats.block_drops["problem10_endpoint_entity_removed"] += 1
                continue
            # problem 5: self loop.
            if norm(block["source"]) == norm(block["target"]):
                stats.block_drops["problem5_self_loop"] += 1
                continue
            # problem 9: description grounded in neither endpoint.
            if not contains(block["description"], block["source"]) and not contains(
                    block["description"], block["target"]):
                stats.block_drops["problem9_weak_description"] += 1
                continue
            # problem 11: train description reused from eval.
            if norm(block["description"]) in protected_descriptions:
                stats.block_drops["problem11_cross_split_description"] += 1
                continue
            survivors.append(block)
            survivor_index.append(record.filter_index[index])

        # problem 6: reverse-direction duplicates.
        groups: dict[frozenset[str], list[int]] = defaultdict(list)
        for index, block in enumerate(survivors):
            groups[frozenset((norm(block["source"]), norm(block["target"])))].append(index)
        keep_indices: set[int] = set()
        for indices in groups.values():
            keep_indices.add(max(indices, key=lambda i: block_quality(survivors[i], i)))
            if len(indices) > 1:
                stats.block_drops["problem6_reverse_duplicate"] += len(indices) - 1

        # problem 11: duplicate description inside one record.
        by_description: dict[str, list[int]] = defaultdict(list)
        for index in sorted(keep_indices):
            by_description[norm(survivors[index]["description"])].append(index)
        final_indices: set[int] = set()
        for indices in by_description.values():
            final_indices.add(max(indices, key=lambda i: block_quality(survivors[i], i)))
            if len(indices) > 1:
                stats.block_drops["problem11_duplicate_description"] += len(indices) - 1

        final = [(survivors[i], survivor_index[i]) for i in sorted(final_indices)]
        if not final:
            stats.record_drops["no_blocks_after_cleanv2"] += 1
            continue
        record.entities = entities
        record.entities_str = repr(entities)
        record.blocks = [item[0] for item in final]
        record.filter_index = [item[1] for item in final]
        kept.append(record)
    stats.records_out = len(kept)
    stats.blocks_out = sum(len(r.blocks) for r in kept)
    return kept


def cleanv2_descriptions(records: Sequence[Record]) -> set[str]:
    return {norm(block["description"]) for record in records for block in record.blocks}
