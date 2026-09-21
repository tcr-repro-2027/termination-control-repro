"""P0c set-completion utility, overrun, and multi-seed capture audit.

This module consumes the frozen all-mode 01b event audit, reparses the exact
M0/M1 response files named by ``run_manifest.json``, and joins each response to
its gold relation set.  It does not load an LLM/SAE, regenerate text, move the
legacy raw onset, or change any 01b event definition.

The gold matching normalisation intentionally follows the frozen
``tcr.extraction.protocol`` v1.0, while P0c excludes empty identities from
the *valid relation utility* sets:

* strict: normalized ``(source, target, relation)``;
* relaxed: normalized ``(source, target)``;
* normalization: strip, lowercase, collapse whitespace;
* empty source/target/relation: audited but not counted as valid utility.
"""

from __future__ import annotations

import bisect
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .audit import _signature_hash
from .block_parser import ParsedBlock, ParseResult, parse_relation_blocks
from .io_utils import canonical_key, iter_jsonl, sha256_text, stable_json

MODEL_TAGS = ("M0", "M1")
MODEL_INDEX = {model: index for index, model in enumerate(MODEL_TAGS)}

# Frozen gold-normalized relation-opportunity bins.  A block at opportunity j
# is assigned by j / G.  A normal stop after n complete blocks is assigned to
# opportunity (n + 1) / G.  Hit-max is administrative censor, not a natural
# stop event.
PROGRESS_BINS: tuple[tuple[str, float, float], ...] = (
    ("<0.50G", 0.0, 0.50),
    ("0.50-0.75G", 0.50, 0.75),
    ("0.75-1.00G", 0.75, 1.00),
    ("1.00-1.25G", 1.00, 1.25),
    ("1.25-1.50G", 1.25, 1.50),
    (">=1.50G", 1.50, math.inf),
)
BIN_INDEX = {name: index for index, (name, _left, _right) in enumerate(PROGRESS_BINS)}

# Base mutually exclusive observed events at a relation opportunity.
EVENT_CATEGORIES: tuple[str, ...] = (
    "new_strict_gold",
    "new_relaxed_pair_only",
    "seen_triple_reuse",
    "gold_pair_already_covered_variant",
    "new_unmatched",
    "invalid_identity",
    "normal_stop",
    "admin_censor",
)
EVENT_INDEX = {name: index for index, name in enumerate(EVENT_CATEGORIES)}
NATURAL_EVENT_CATEGORIES = EVENT_CATEGORIES[:-1]  # administrative censor excluded
UTILITY_NAMES = ("new_strict_gold", "new_relaxed_gold_pair")
UTILITY_INDEX = {name: index for index, name in enumerate(UTILITY_NAMES)}

INVALID_OBJECT_STATUSES = {
    "invalid_json",
    "not_object",
    "duplicate_keys",
    "invalid_schema",
    "compatible_extra_fields",
    "field_span_failure",
}
STRUCTURED_ALIGNMENT_TYPES = {
    "single_block_aligned",
    "multi_block_aligned",
    "phase_rotated_structured",
}


@dataclass(frozen=True)
class GoldRecord:
    key_id: str
    gold_blocks: int
    strict_triples: frozenset[tuple[str, str, str]]
    relaxed_pairs: frozenset[tuple[str, str]]
    n_raw_relations: int
    n_empty_identity: int


@dataclass(frozen=True)
class GoldDataset:
    key_field: str
    n_rows: int
    records: dict[str, GoldRecord]


@dataclass(frozen=True)
class EventMeta:
    sample_id: str
    stable_prompt_id: str
    model_tag: str
    key: Any
    key_id: str
    seed: int
    response_sha256: str
    n_blocks: int
    n_identity_complete_blocks: int
    block_triple_hashes: tuple[str, ...]
    block_quad_hashes: tuple[str, ...]
    block_segments: tuple[int, ...]
    block_identity_complete: tuple[bool, ...]
    first_seed_index_0based: int | None
    capture: Mapping[str, Any]
    legacy: Mapping[str, Any]
    alignment_type: str
    alignment_evidence: Mapping[str, Any]


@dataclass(frozen=True)
class ResponseAnalysis:
    sample_id: str
    stable_prompt_id: str
    model_tag: str
    key: Any
    seed: int
    gold_blocks: int
    n_blocks: int
    output_gold_ratio: float
    hit_max: bool
    full_json_list_valid: bool
    has_incomplete_tail: bool
    parser_invalid_objects: int
    parser_continuity_breaks: int
    normalized_first_reuse_index_0based: int | None
    frozen_first_seed_index_0based: int | None
    frozen_seed_matches_normalized_first: bool | None
    final_strict_coverage: float
    final_relaxed_coverage: float
    strict_coverage_at_1g: float
    relaxed_coverage_at_1g: float
    post_1g_strict_coverage_gain: float
    post_1g_relaxed_coverage_gain: float
    final_unique_nonempty_triple_ratio: float
    first_seed_progress: float | None
    first_seed_strict_coverage: float | None
    first_seed_relaxed_coverage: float | None
    first_seed_strict_fraction_of_final: float | None
    first_seed_relaxed_fraction_of_final: float | None
    first_seed_unique_nonempty_triple_ratio: float | None
    no_new_strict_after_first_seed: bool | None
    no_new_relaxed_after_first_seed: bool | None
    n_new_relaxed_after_first_seed: int | None
    first_seed_after_last_new_relaxed: bool | None
    first_seed_ge_1g: bool | None
    first_seed_ge_1_25g: bool | None
    semantic_capture: bool
    capture_seed_ordinal: int | None
    capture_progress: float | None
    capture_strict_coverage: float | None
    capture_relaxed_coverage: float | None
    capture_relaxed_fraction_of_final: float | None
    no_new_relaxed_after_capture_seed: bool | None
    n_new_relaxed_after_capture_seed: int | None
    capture_after_last_new_relaxed: bool | None
    capture_seed_ge_1g: bool | None
    capture_seed_ge_1_25g: bool | None
    stable_orbit: bool
    legacy_structured: bool
    legacy_capture_seed_ordinal: int | None
    legacy_capture_progress: float | None
    legacy_capture_after_last_new_relaxed: bool | None
    legacy_capture_seed_ge_1g: bool | None
    n_nonempty_reuse_events: int
    last_new_strict_block_1based: int | None
    last_new_relaxed_block_1based: int | None
    opportunity_counts: tuple[tuple[int, ...], ...]
    utility_counts: tuple[tuple[int, ...], ...]
    invalid_object_bin_counts: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        value = self.__dict__.copy()
        value["opportunity_counts"] = [list(row) for row in self.opportunity_counts]
        value["utility_counts"] = [list(row) for row in self.utility_counts]
        value["invalid_object_bin_counts"] = list(self.invalid_object_bin_counts)
        return value


@dataclass(frozen=True)
class CaptureCase:
    sample_id: str
    stable_prompt_id: str
    model_tag: str
    seed: int
    kind: str  # primary_semantic_capture | legacy_aligned_capture
    seed_ordinal: int
    prior_reuse_events: int
    progress: float
    after_1g: bool
    after_1_25g: bool
    after_last_new_relaxed: bool | None
    relaxed_coverage: float
    relaxed_fraction_of_final: float | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def norm_field(value: Any) -> str:
    """Frozen protocol-v1.0 field normalization."""
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def normalized_triple(values: Sequence[Any]) -> tuple[str, str, str]:
    if len(values) < 3:
        raise ValueError("triple requires three fields")
    return tuple(norm_field(value) for value in values[:3])  # type: ignore[return-value]


def normalized_pair(values: Sequence[Any]) -> tuple[str, str]:
    if len(values) < 2:
        raise ValueError("pair requires two fields")
    return tuple(norm_field(value) for value in values[:2])  # type: ignore[return-value]


def _parse_gold_output(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = row.get("output")
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ValueError("gold output is an unparseable JSON string") from exc
    if not isinstance(output, list):
        raise ValueError("gold row output must be a list or JSON-encoded list")
    return [item for item in output if isinstance(item, dict)]


def _gold_record(key_id: str, row: Mapping[str, Any]) -> GoldRecord:
    relations = _parse_gold_output(row)
    declared = row.get("n_output_dicts")
    gold_blocks = int(declared) if declared is not None else len(relations)
    if gold_blocks <= 0:
        raise ValueError("gold n_output_dicts must be positive")
    if declared is not None and len(relations) != gold_blocks:
        raise ValueError(
            f"gold n_output_dicts={gold_blocks} disagrees with output list length={len(relations)}"
        )
    normalized = [
        normalized_triple((rel.get("source"), rel.get("target"), rel.get("relation")))
        for rel in relations
    ]
    empty = sum(not all(triple) for triple in normalized)
    triples = {triple for triple in normalized if all(triple)}
    pairs = {triple[:2] for triple in normalized if all(triple)}
    return GoldRecord(
        key_id=key_id,
        gold_blocks=gold_blocks,
        strict_triples=frozenset(triples),
        relaxed_pairs=frozenset(pairs),
        n_raw_relations=len(relations),
        n_empty_identity=empty,
    )


def load_gold_dataset(
    gold_path: str | Path,
    event_keys: Sequence[Any],
    *,
    key_field: str = "auto",
) -> GoldDataset:
    """Load full gold relations with the same strict key-mapping audit as P0b."""
    rows = list(iter_jsonl(gold_path))
    if not rows:
        raise ValueError(f"gold dataset is empty: {gold_path}")
    event_key_ids = {canonical_key(value) for value in event_keys}

    candidate_rows: list[tuple[str, dict[str, Mapping[str, Any]]]] = []
    for field in ("key", "index", "id"):
        mapping: dict[str, Mapping[str, Any]] = {}
        valid = True
        for row in rows:
            if field not in row:
                valid = False
                break
            key_id = canonical_key(row[field])
            if key_id in mapping and stable_json(mapping[key_id]) != stable_json(row):
                raise ValueError(f"duplicate gold {field}={row[field]!r} has conflicting rows")
            mapping[key_id] = row
        if valid:
            candidate_rows.append((field, mapping))
    candidate_rows.append(
        (
            "line_0based",
            {canonical_key(index): row for index, row in enumerate(rows)},
        )
    )
    candidate_rows.append(
        (
            "line_1based",
            {canonical_key(index + 1): row for index, row in enumerate(rows)},
        )
    )

    if key_field != "auto":
        matches = [item for item in candidate_rows if item[0] == key_field]
        if not matches:
            raise ValueError(
                f"unsupported/unavailable gold key field {key_field!r}; "
                f"available={[name for name, _ in candidate_rows]}"
            )
        chosen_name, chosen_rows = matches[0]
        missing = sorted(event_key_ids - set(chosen_rows))
        if missing:
            raise ValueError(
                f"gold key field {chosen_name!r} misses {len(missing)} event keys; first={missing[:5]}"
            )
    else:
        full = [(name, mapping) for name, mapping in candidate_rows if event_key_ids <= set(mapping)]
        if not full:
            coverage = sorted(
                ((name, len(event_key_ids & set(mapping))) for name, mapping in candidate_rows),
                key=lambda item: (-item[1], item[0]),
            )
            raise ValueError(f"no gold-key mapping covers all event keys; coverage={coverage}")
        priority = {"key": 0, "index": 1, "id": 2, "line_0based": 3, "line_1based": 4}
        full.sort(key=lambda item: priority[item[0]])
        chosen_name, chosen_rows = full[0]
        # Semantic identifier mappings must not silently disagree.
        chosen_counts = {
            key: int(chosen_rows[key].get("n_output_dicts", len(_parse_gold_output(chosen_rows[key]))))
            for key in event_key_ids
        }
        for other_name, other_rows in full[1:]:
            if priority[other_name] > 2:
                continue
            other_counts = {
                key: int(other_rows[key].get("n_output_dicts", len(_parse_gold_output(other_rows[key]))))
                for key in event_key_ids
            }
            if other_counts != chosen_counts:
                raise ValueError(
                    f"ambiguous automatic gold mapping: {chosen_name!r} and {other_name!r} disagree"
                )

    records = {
        key_id: _gold_record(key_id, chosen_rows[key_id])
        for key_id in event_key_ids
    }
    return GoldDataset(chosen_name, len(rows), records)


def load_event_meta(result_dir: str | Path) -> tuple[dict[str, dict[tuple[str, int], EventMeta]], dict[str, Any]]:
    result = Path(result_dir)
    run_manifest_path = result / "run_manifest.json"
    selection_path = result / "selection_manifest.json"
    event_rows_path = result / "event_rows.jsonl"
    for path in (run_manifest_path, selection_path, event_rows_path):
        if not path.exists():
            raise FileNotFoundError(path)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("mode") != "all":
        raise ValueError("P0c requires the all-mode 01b result directory")
    if run_manifest.get("selection", {}).get("mode") != "all":
        raise ValueError("run_manifest selection mode is not all")
    if bool(run_manifest.get("protocol", {}).get("raw_onset_moved")):
        raise ValueError("P0c requires raw_onset_moved=false")

    indexes: dict[str, dict[tuple[str, int], EventMeta]] = {"M0": {}, "M1": {}}
    prompt_models: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in iter_jsonl(event_rows_path):
        if "prevalence" not in row.get("selection_roles", []):
            continue
        model = str(row.get("model_tag"))
        if model not in MODEL_TAGS:
            raise ValueError(f"unexpected model_tag={model!r}")
        key_id = canonical_key(row.get("key"))
        seed = int(row.get("seed"))
        pair = (key_id, seed)
        if pair in indexes[model]:
            raise ValueError(f"duplicate event row model={model} key={row.get('key')!r} seed={seed}")
        block_index = row.get("block_index", [])
        if not isinstance(block_index, list):
            raise TypeError("event block_index must be a list")
        first_seed = row.get("first_nonempty_triple_reuse", {})
        meta = EventMeta(
            sample_id=str(row.get("sample_id")),
            stable_prompt_id=str(row.get("stable_prompt_id")),
            model_tag=model,
            key=row.get("key"),
            key_id=key_id,
            seed=seed,
            response_sha256=str(row.get("response_sha256")),
            n_blocks=int(row.get("n_blocks", 0)),
            n_identity_complete_blocks=int(row.get("n_identity_complete_blocks", 0)),
            block_triple_hashes=tuple(str(item.get("triple_hash")) for item in block_index),
            block_quad_hashes=tuple(str(item.get("quad_hash")) for item in block_index),
            block_segments=tuple(int(item.get("sequence_segment", -1)) for item in block_index),
            block_identity_complete=tuple(bool(item.get("identity_complete")) for item in block_index),
            first_seed_index_0based=(
                int(first_seed["block_index_0based"]) if first_seed.get("exists") else None
            ),
            capture=row.get("motif_capture_triple", {}),
            legacy=row.get("legacy_orbit", {}),
            alignment_type=str(row.get("alignment_type", "")),
            alignment_evidence=row.get("alignment_evidence", {}),
        )
        if meta.n_blocks != len(meta.block_triple_hashes):
            raise ValueError(f"n_blocks/hash count mismatch: {meta.sample_id}")
        indexes[model][pair] = meta
        prompt_models[meta.stable_prompt_id][model].add(seed)

    if not indexes["M0"] or not indexes["M1"]:
        raise ValueError("event_rows lacks M0/M1 prevalence rows")
    if set(indexes["M0"]) != set(indexes["M1"]):
        missing0 = sorted(set(indexes["M1"]) - set(indexes["M0"]))[:5]
        missing1 = sorted(set(indexes["M0"]) - set(indexes["M1"]))[:5]
        raise ValueError(f"M0/M1 event pair mismatch; missing0={missing0}, missing1={missing1}")
    for prompt_id, models in prompt_models.items():
        if set(models) != set(MODEL_TAGS) or models["M0"] != models["M1"]:
            raise ValueError(f"prompt-level seed mismatch: {prompt_id}")
    return indexes, run_manifest


def response_text(sample: Mapping[str, Any], target: str) -> str:
    answer = sample.get("response", "") or ""
    reasoning = sample.get("reasoning", "") or ""
    if not isinstance(answer, str) or not isinstance(reasoning, str):
        raise ValueError("response/reasoning must be strings")
    if target == "answer":
        return answer
    if target == "reasoning":
        return reasoning
    if target == "full":
        return reasoning + "\n" + answer if reasoning and answer else reasoning or answer
    raise ValueError(f"unsupported target={target!r}")


def progress_bin_index(progress: float) -> int:
    if progress < 0:
        raise ValueError("progress cannot be negative")
    for index, (_name, left, right) in enumerate(PROGRESS_BINS):
        if left <= progress < right:
            return index
    return len(PROGRESS_BINS) - 1


def _validate_reparse(text: str, parsed: ParseResult, meta: EventMeta) -> None:
    if sha256_text(text) != meta.response_sha256:
        raise ValueError(f"response SHA mismatch: {meta.sample_id}")
    blocks = parsed.blocks
    if len(blocks) != meta.n_blocks:
        raise ValueError(
            f"reparse n_blocks mismatch {meta.sample_id}: {len(blocks)} != {meta.n_blocks}"
        )
    if sum(block.identity_complete for block in blocks) != meta.n_identity_complete_blocks:
        raise ValueError(f"reparse identity-complete count mismatch: {meta.sample_id}")
    triple_hashes = tuple(_signature_hash(block.triple_signature) for block in blocks)
    quad_hashes = tuple(_signature_hash(block.canonical_signature) for block in blocks)
    segments = tuple(block.sequence_segment for block in blocks)
    identities = tuple(block.identity_complete for block in blocks)
    if triple_hashes != meta.block_triple_hashes:
        raise ValueError(f"reparse triple hashes mismatch: {meta.sample_id}")
    if quad_hashes != meta.block_quad_hashes:
        raise ValueError(f"reparse quad hashes mismatch: {meta.sample_id}")
    if segments != meta.block_segments:
        raise ValueError(f"reparse continuity segments mismatch: {meta.sample_id}")
    if identities != meta.block_identity_complete:
        raise ValueError(f"reparse identity flags mismatch: {meta.sample_id}")


def enumerate_nonempty_reuse_events(blocks: Sequence[ParsedBlock]) -> list[dict[str, int]]:
    """Enumerate every raw-exact nonempty triple reuse block chronologically."""
    seen: dict[tuple[str, str, str], int] = {}
    events: list[dict[str, int]] = []
    for index, block in enumerate(blocks):
        if not block.identity_complete:
            continue
        signature = block.triple_signature
        previous = seen.get(signature)
        if previous is not None:
            events.append(
                {
                    "ordinal": len(events) + 1,
                    "block_index_0based": index,
                    "previous_block_index_0based": previous,
                    "distance_blocks": index - previous,
                }
            )
        seen[signature] = index
    return events


def first_normalized_nonempty_reuse(blocks: Sequence[ParsedBlock]) -> int | None:
    seen: set[tuple[str, str, str]] = set()
    for index, block in enumerate(blocks):
        triple = normalized_triple(block.triple_signature)
        if not all(triple):
            continue
        if triple in seen:
            return index
        seen.add(triple)
    return None


def _coverage_prefixes(
    blocks: Sequence[ParsedBlock], gold: GoldRecord
) -> tuple[
    list[str],
    list[bool],
    list[bool],
    list[int],
    list[int],
    list[int],
    list[int],
    int | None,
    int | None,
]:
    """Classify valid blocks and return prefix coverage/uniqueness arrays.

    Prefix arrays have length ``n_blocks + 1`` and represent state before the
    block at the same zero-based index.
    """
    categories: list[str] = []
    adds_strict: list[bool] = []
    adds_relaxed: list[bool] = []
    strict_prefix = [0]
    relaxed_prefix = [0]
    unique_triple_prefix = [0]
    nonempty_block_prefix = [0]
    seen_triples: set[tuple[str, str, str]] = set()
    seen_pairs: set[tuple[str, str]] = set()
    covered_strict: set[tuple[str, str, str]] = set()
    covered_relaxed: set[tuple[str, str]] = set()
    last_new_strict: int | None = None
    last_new_relaxed: int | None = None

    for index, block in enumerate(blocks):
        triple = normalized_triple(block.triple_signature)
        pair = normalized_pair(block.triple_signature)
        nonempty = all(triple)
        triple_seen = triple in seen_triples if nonempty else False
        strict_added = False
        relaxed_added = False
        if not nonempty:
            category = "invalid_identity"
        elif triple_seen:
            category = "seen_triple_reuse"
        elif triple in gold.strict_triples:
            category = "new_strict_gold"
            strict_added = triple not in covered_strict
        elif pair in gold.relaxed_pairs and pair not in seen_pairs:
            category = "new_relaxed_pair_only"
        elif pair in gold.relaxed_pairs:
            category = "gold_pair_already_covered_variant"
        else:
            category = "new_unmatched"

        if nonempty:
            if triple in gold.strict_triples and triple not in covered_strict:
                covered_strict.add(triple)
                strict_added = True
                last_new_strict = index
            if pair in gold.relaxed_pairs and pair not in covered_relaxed:
                covered_relaxed.add(pair)
                relaxed_added = True
                last_new_relaxed = index
            seen_triples.add(triple)
            seen_pairs.add(pair)

        categories.append(category)
        adds_strict.append(strict_added)
        adds_relaxed.append(relaxed_added)
        strict_prefix.append(len(covered_strict))
        relaxed_prefix.append(len(covered_relaxed))
        unique_triple_prefix.append(len(seen_triples))
        nonempty_block_prefix.append(nonempty_block_prefix[-1] + int(nonempty))

    return (
        categories,
        adds_strict,
        adds_relaxed,
        strict_prefix,
        relaxed_prefix,
        unique_triple_prefix,
        nonempty_block_prefix,
        last_new_strict,
        last_new_relaxed,
    )


def _fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def _capture_case(
    *,
    kind: str,
    second_copy_start: int,
    reuse_ordinal_by_block: Mapping[int, int],
    strict_prefix: Sequence[int],
    relaxed_prefix: Sequence[int],
    final_relaxed: int,
    last_new_relaxed: int | None,
    gold: GoldRecord,
    meta: EventMeta,
) -> CaptureCase:
    ordinal = reuse_ordinal_by_block.get(second_copy_start)
    if ordinal is None:
        raise ValueError(
            f"capture second-copy start is not a nonempty reuse event: {meta.sample_id}, block={second_copy_start}"
        )
    progress = (second_copy_start + 1) / gold.gold_blocks
    relaxed = relaxed_prefix[second_copy_start]
    return CaptureCase(
        sample_id=meta.sample_id,
        stable_prompt_id=meta.stable_prompt_id,
        model_tag=meta.model_tag,
        seed=meta.seed,
        kind=kind,
        seed_ordinal=ordinal,
        prior_reuse_events=ordinal - 1,
        progress=progress,
        after_1g=progress >= 1.0,
        after_1_25g=progress >= 1.25,
        after_last_new_relaxed=(
            second_copy_start > last_new_relaxed if last_new_relaxed is not None else None
        ),
        relaxed_coverage=_fraction(relaxed, len(gold.relaxed_pairs)),
        relaxed_fraction_of_final=_fraction(relaxed, final_relaxed),
    )


def analyze_response(
    *,
    text: str,
    meta: EventMeta,
    gold: GoldRecord,
) -> tuple[ResponseAnalysis, list[CaptureCase]]:
    parsed = parse_relation_blocks(text, offsets=None, reject_extra_fields=True)
    _validate_reparse(text, parsed, meta)
    blocks = parsed.blocks
    (
        categories,
        _adds_strict,
        _adds_relaxed,
        strict_prefix,
        relaxed_prefix,
        unique_prefix,
        nonempty_prefix,
        last_new_strict,
        last_new_relaxed,
    ) = _coverage_prefixes(blocks, gold)

    opportunity = np.zeros((len(PROGRESS_BINS), len(EVENT_CATEGORIES)), dtype=np.int64)
    utility = np.zeros((len(PROGRESS_BINS), len(UTILITY_NAMES)), dtype=np.int64)
    for index, category in enumerate(categories):
        progress = (index + 1) / gold.gold_blocks
        bidx = progress_bin_index(progress)
        opportunity[bidx, EVENT_INDEX[category]] += 1
        if _adds_strict[index]:
            utility[bidx, UTILITY_INDEX["new_strict_gold"]] += 1
        if _adds_relaxed[index]:
            utility[bidx, UTILITY_INDEX["new_relaxed_gold_pair"]] += 1
    terminal_progress = (len(blocks) + 1) / gold.gold_blocks
    terminal = "admin_censor" if bool(meta.legacy.get("hit_max_tokens")) else "normal_stop"
    opportunity[progress_bin_index(terminal_progress), EVENT_INDEX[terminal]] += 1

    starts = [block.char_start for block in blocks]
    invalid_bins = np.zeros(len(PROGRESS_BINS), dtype=np.int64)
    invalid_objects = 0
    continuity_breaks = 0
    for diagnostic in parsed.diagnostics:
        if diagnostic.parse_status in INVALID_OBJECT_STATUSES:
            invalid_objects += 1
            opportunity_index = bisect.bisect_left(starts, diagnostic.char_start) + 1
            invalid_bins[progress_bin_index(opportunity_index / gold.gold_blocks)] += 1
        elif diagnostic.parse_status == "continuity_break":
            continuity_breaks += 1

    normalized_first = first_normalized_nonempty_reuse(blocks)
    frozen_first = meta.first_seed_index_0based
    if frozen_first is not None:
        raw_events = enumerate_nonempty_reuse_events(blocks)
        if not raw_events or raw_events[0]["block_index_0based"] != frozen_first:
            raise ValueError(f"frozen first seed does not match recomputed raw reuse: {meta.sample_id}")
    else:
        raw_events = enumerate_nonempty_reuse_events(blocks)
        if raw_events:
            raise ValueError(f"event row misses recomputed nonempty reuse: {meta.sample_id}")
    reuse_ordinal_by_block = {
        event["block_index_0based"]: event["ordinal"] for event in raw_events
    }

    final_strict = strict_prefix[-1]
    final_relaxed = relaxed_prefix[-1]
    at_g_index = min(gold.gold_blocks, len(blocks))
    strict_at_g = strict_prefix[at_g_index]
    relaxed_at_g = relaxed_prefix[at_g_index]
    final_nonempty = nonempty_prefix[-1]
    final_unique_ratio = _fraction(unique_prefix[-1], final_nonempty)

    seed_progress = seed_strict = seed_relaxed = None
    seed_strict_final = seed_relaxed_final = seed_unique_ratio = None
    no_strict_after = no_relaxed_after = None
    n_relaxed_after_seed = None
    seed_after_last_relaxed = seed_ge_1 = seed_ge_125 = None
    if frozen_first is not None:
        seed_progress = (frozen_first + 1) / gold.gold_blocks
        seed_strict = _fraction(strict_prefix[frozen_first], len(gold.strict_triples))
        seed_relaxed = _fraction(relaxed_prefix[frozen_first], len(gold.relaxed_pairs))
        seed_strict_final = _fraction(strict_prefix[frozen_first], final_strict)
        seed_relaxed_final = _fraction(relaxed_prefix[frozen_first], final_relaxed)
        seed_unique_ratio = _fraction(unique_prefix[frozen_first], nonempty_prefix[frozen_first])
        no_strict_after = strict_prefix[frozen_first] == final_strict
        no_relaxed_after = relaxed_prefix[frozen_first] == final_relaxed
        n_relaxed_after_seed = final_relaxed - relaxed_prefix[frozen_first]
        seed_after_last_relaxed = (
            frozen_first > last_new_relaxed if last_new_relaxed is not None else None
        )
        seed_ge_1 = seed_progress >= 1.0
        seed_ge_125 = seed_progress >= 1.25

    capture_cases: list[CaptureCase] = []
    capture = meta.capture
    semantic_capture = bool(
        capture.get("exists")
        and capture.get("motif_triples")
        and all(
            isinstance(triple, list)
            and len(triple) == 3
            and all(isinstance(value, str) and value.strip() for value in triple)
            for triple in capture.get("motif_triples", [])
        )
    )
    capture_case: CaptureCase | None = None
    if semantic_capture:
        second_start = int(capture["second_copy_start_0based"])
        capture_case = _capture_case(
            kind="primary_semantic_capture",
            second_copy_start=second_start,
            reuse_ordinal_by_block=reuse_ordinal_by_block,
            strict_prefix=strict_prefix,
            relaxed_prefix=relaxed_prefix,
            final_relaxed=final_relaxed,
            last_new_relaxed=last_new_relaxed,
            gold=gold,
            meta=meta,
        )
        capture_cases.append(capture_case)

    legacy_structured = bool(meta.legacy.get("exists")) and meta.alignment_type in STRUCTURED_ALIGNMENT_TYPES
    legacy_case: CaptureCase | None = None
    matching_run = meta.alignment_evidence.get("matching_quad_run")
    if legacy_structured and isinstance(matching_run, Mapping):
        second_start = int(matching_run["block_onset"]) + int(matching_run["block_period"])
        legacy_case = _capture_case(
            kind="legacy_aligned_capture",
            second_copy_start=second_start,
            reuse_ordinal_by_block=reuse_ordinal_by_block,
            strict_prefix=strict_prefix,
            relaxed_prefix=relaxed_prefix,
            final_relaxed=final_relaxed,
            last_new_relaxed=last_new_relaxed,
            gold=gold,
            meta=meta,
        )
        capture_cases.append(legacy_case)

    # Validate the frozen capture locally without rerunning the exhaustive
    # O(n^2) motif search over every all-mode response.  Per-block hashes have
    # already been matched to the frozen event row above.
    if semantic_capture:
        onset = int(capture["block_onset_0based"])
        period = int(capture["block_period"])
        confirmation = int(capture["confirmed_at_block_exclusive_0based"])
        if period <= 0 or confirmation != onset + 3 * period or confirmation > len(blocks):
            raise ValueError(f"invalid frozen capture coordinates: {meta.sample_id}")
        motif = [block.triple_signature for block in blocks[onset : onset + period]]
        if not motif or not all(all(value.strip() for value in triple) for triple in motif):
            raise ValueError(f"frozen semantic capture has empty identity: {meta.sample_id}")
        for repeat in (1, 2):
            candidate = [
                block.triple_signature
                for block in blocks[onset + repeat * period : onset + (repeat + 1) * period]
            ]
            if candidate != motif:
                raise ValueError(f"frozen capture does not contain three exact motif copies: {meta.sample_id}")

    return (
        ResponseAnalysis(
            sample_id=meta.sample_id,
            stable_prompt_id=meta.stable_prompt_id,
            model_tag=meta.model_tag,
            key=meta.key,
            seed=meta.seed,
            gold_blocks=gold.gold_blocks,
            n_blocks=len(blocks),
            output_gold_ratio=_fraction(len(blocks), gold.gold_blocks),
            hit_max=bool(meta.legacy.get("hit_max_tokens")),
            full_json_list_valid=parsed.full_json_list_valid,
            has_incomplete_tail=parsed.incomplete_tail_start is not None,
            parser_invalid_objects=invalid_objects,
            parser_continuity_breaks=continuity_breaks,
            normalized_first_reuse_index_0based=normalized_first,
            frozen_first_seed_index_0based=frozen_first,
            frozen_seed_matches_normalized_first=(
                frozen_first == normalized_first if frozen_first is not None or normalized_first is not None else None
            ),
            final_strict_coverage=_fraction(final_strict, len(gold.strict_triples)),
            final_relaxed_coverage=_fraction(final_relaxed, len(gold.relaxed_pairs)),
            strict_coverage_at_1g=_fraction(strict_at_g, len(gold.strict_triples)),
            relaxed_coverage_at_1g=_fraction(relaxed_at_g, len(gold.relaxed_pairs)),
            post_1g_strict_coverage_gain=_fraction(final_strict - strict_at_g, len(gold.strict_triples)),
            post_1g_relaxed_coverage_gain=_fraction(final_relaxed - relaxed_at_g, len(gold.relaxed_pairs)),
            final_unique_nonempty_triple_ratio=final_unique_ratio,
            first_seed_progress=seed_progress,
            first_seed_strict_coverage=seed_strict,
            first_seed_relaxed_coverage=seed_relaxed,
            first_seed_strict_fraction_of_final=seed_strict_final,
            first_seed_relaxed_fraction_of_final=seed_relaxed_final,
            first_seed_unique_nonempty_triple_ratio=seed_unique_ratio,
            no_new_strict_after_first_seed=no_strict_after,
            no_new_relaxed_after_first_seed=no_relaxed_after,
            n_new_relaxed_after_first_seed=n_relaxed_after_seed,
            first_seed_after_last_new_relaxed=seed_after_last_relaxed,
            first_seed_ge_1g=seed_ge_1,
            first_seed_ge_1_25g=seed_ge_125,
            semantic_capture=semantic_capture,
            capture_seed_ordinal=(capture_case.seed_ordinal if capture_case else None),
            capture_progress=(capture_case.progress if capture_case else None),
            capture_strict_coverage=(
                _fraction(strict_prefix[int(capture["second_copy_start_0based"])], len(gold.strict_triples))
                if capture_case
                else None
            ),
            capture_relaxed_coverage=(capture_case.relaxed_coverage if capture_case else None),
            capture_relaxed_fraction_of_final=(
                capture_case.relaxed_fraction_of_final if capture_case else None
            ),
            no_new_relaxed_after_capture_seed=(
                relaxed_prefix[int(capture["second_copy_start_0based"])] == final_relaxed
                if capture_case
                else None
            ),
            n_new_relaxed_after_capture_seed=(
                final_relaxed - relaxed_prefix[int(capture["second_copy_start_0based"])]
                if capture_case
                else None
            ),
            capture_after_last_new_relaxed=(
                capture_case.after_last_new_relaxed if capture_case else None
            ),
            capture_seed_ge_1g=(capture_case.after_1g if capture_case else None),
            capture_seed_ge_1_25g=(capture_case.after_1_25g if capture_case else None),
            stable_orbit=bool(meta.legacy.get("exists")),
            legacy_structured=legacy_structured,
            legacy_capture_seed_ordinal=(legacy_case.seed_ordinal if legacy_case else None),
            legacy_capture_progress=(legacy_case.progress if legacy_case else None),
            legacy_capture_after_last_new_relaxed=(
                legacy_case.after_last_new_relaxed if legacy_case else None
            ),
            legacy_capture_seed_ge_1g=(legacy_case.after_1g if legacy_case else None),
            n_nonempty_reuse_events=len(raw_events),
            last_new_strict_block_1based=(last_new_strict + 1 if last_new_strict is not None else None),
            last_new_relaxed_block_1based=(last_new_relaxed + 1 if last_new_relaxed is not None else None),
            opportunity_counts=tuple(tuple(int(value) for value in row) for row in opportunity),
            utility_counts=tuple(tuple(int(value) for value in row) for row in utility),
            invalid_object_bin_counts=tuple(int(value) for value in invalid_bins),
        ),
        capture_cases,
    )


def analyze_response_files(
    *,
    event_index: Mapping[str, Mapping[tuple[str, int], EventMeta]],
    gold: GoldDataset,
    responses_m0: str | Path,
    responses_m1: str | Path,
    target: str,
) -> tuple[list[ResponseAnalysis], list[CaptureCase]]:
    analyses: list[ResponseAnalysis] = []
    captures: list[CaptureCase] = []
    for model, response_path in (("M0", responses_m0), ("M1", responses_m1)):
        expected = event_index[model]
        expected_by_key: dict[str, set[int]] = defaultdict(set)
        for candidate_key, candidate_seed in expected:
            expected_by_key[candidate_key].add(candidate_seed)
        seen: set[tuple[str, int]] = set()
        for row in iter_jsonl(response_path):
            if "key" not in row:
                raise ValueError(f"response row lacks key: {response_path}")
            key_id = canonical_key(row["key"])
            relevant = expected_by_key.get(key_id, set())
            if not relevant:
                continue
            samples = row.get("responses")
            if not isinstance(samples, list):
                raise ValueError(f"response key={row['key']!r} lacks responses list")
            by_seed: dict[int, Mapping[str, Any]] = {}
            for sample in samples:
                if not isinstance(sample, Mapping):
                    raise ValueError(
                        f"response key={row['key']!r} contains a non-object sample"
                    )
                if sample.get("seed") is None:
                    raise ValueError(
                        f"response key={row['key']!r} contains a sample without seed"
                    )
                seed = int(sample.get("seed"))
                if seed in by_seed:
                    raise ValueError(f"duplicate response seed={seed}, key={row['key']!r}")
                by_seed[seed] = sample
            for seed in sorted(relevant):
                pair = (key_id, seed)
                if seed not in by_seed:
                    raise ValueError(f"missing selected response model={model}, key={row['key']!r}, seed={seed}")
                meta = expected[pair]
                gold_record = gold.records.get(key_id)
                if gold_record is None:
                    raise KeyError(f"gold mapping misses key={row['key']!r}")
                analysis, cases = analyze_response(
                    text=response_text(by_seed[seed], target),
                    meta=meta,
                    gold=gold_record,
                )
                analyses.append(analysis)
                captures.extend(cases)
                seen.add(pair)
        missing = sorted(set(expected) - seen)
        if missing:
            raise ValueError(f"response file misses {len(missing)} selected {model} rows; first={missing[:5]}")
    return analyses, captures


def prompt_ids_and_validate(analyses: Sequence[ResponseAnalysis]) -> list[str]:
    grouped: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    sample_ids: set[str] = set()
    for row in analyses:
        if row.sample_id in sample_ids:
            raise ValueError(f"duplicate response analysis: {row.sample_id}")
        sample_ids.add(row.sample_id)
        grouped[row.stable_prompt_id][row.model_tag].add(row.seed)
    for prompt_id, models in grouped.items():
        if set(models) != set(MODEL_TAGS) or models["M0"] != models["M1"]:
            raise ValueError(f"analysis prompt/seed mismatch: {prompt_id}")
    return sorted(grouped)


def build_opportunity_array(
    analyses: Sequence[ResponseAnalysis], prompt_ids: Sequence[str]
) -> np.ndarray:
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    counts = np.zeros(
        (len(prompt_ids), len(MODEL_TAGS), len(PROGRESS_BINS), len(EVENT_CATEGORIES)),
        dtype=np.int64,
    )
    for row in analyses:
        counts[prompt_index[row.stable_prompt_id], MODEL_INDEX[row.model_tag]] += np.asarray(
            row.opportunity_counts, dtype=np.int64
        )
    return counts


def build_utility_array(
    analyses: Sequence[ResponseAnalysis], prompt_ids: Sequence[str]
) -> np.ndarray:
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    counts = np.zeros(
        (len(prompt_ids), len(MODEL_TAGS), len(PROGRESS_BINS), len(UTILITY_NAMES)),
        dtype=np.int64,
    )
    for row in analyses:
        counts[prompt_index[row.stable_prompt_id], MODEL_INDEX[row.model_tag]] += np.asarray(
            row.utility_counts, dtype=np.int64
        )
    return counts


def _event_metric_counts(
    counts: np.ndarray, utility_counts: np.ndarray | None = None
) -> tuple[np.ndarray, list[str]]:
    """Return metric numerators; natural-event denominator is shared."""
    natural = counts[..., : len(NATURAL_EVENT_CATEGORIES)]
    strict = (
        utility_counts[..., UTILITY_INDEX["new_strict_gold"]]
        if utility_counts is not None
        else counts[..., EVENT_INDEX["new_strict_gold"]]
    )
    relaxed = (
        utility_counts[..., UTILITY_INDEX["new_relaxed_gold_pair"]]
        if utility_counts is not None
        else strict + counts[..., EVENT_INDEX["new_relaxed_pair_only"]]
    )
    metrics = np.stack(
        [
            strict,
            relaxed,
            counts[..., EVENT_INDEX["new_strict_gold"]]
            + counts[..., EVENT_INDEX["new_relaxed_pair_only"]],
            counts[..., EVENT_INDEX["seen_triple_reuse"]],
            counts[..., EVENT_INDEX["gold_pair_already_covered_variant"]],
            counts[..., EVENT_INDEX["new_unmatched"]],
            counts[..., EVENT_INDEX["invalid_identity"]],
            counts[..., EVENT_INDEX["normal_stop"]],
        ],
        axis=-1,
    )
    names = [
        "new_strict_gold",
        "new_relaxed_gold_pair",
        "new_valid_any",
        "seen_triple_reuse",
        "gold_pair_already_covered_variant",
        "new_unmatched",
        "invalid_identity",
        "normal_stop",
    ]
    denominator = np.sum(natural, axis=-1)
    return np.concatenate([metrics, denominator[..., None]], axis=-1), names + ["denominator"]


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _ci(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if finite.size else math.nan


def bootstrap_opportunity_rates(
    counts: np.ndarray,
    *,
    utility_counts: np.ndarray | None = None,
    n_bootstrap: int,
    random_seed: int,
    batch_size: int = 200,
) -> dict[str, Any]:
    """Prompt-cluster bootstrap for per-opportunity event probabilities."""
    metric_counts, names = _event_metric_counts(counts, utility_counts)
    metric_names = names[:-1]
    observed_counts = np.sum(metric_counts, axis=0, dtype=np.float64)
    observed_rates = _safe_ratio(observed_counts[..., :-1], observed_counts[..., -1, None])

    draws: list[np.ndarray] = []
    count_draws: list[np.ndarray] = []
    utility_draws: list[np.ndarray] = []
    n_prompts = counts.shape[0]
    rng = np.random.default_rng(random_seed)
    done = 0
    while done < n_bootstrap:
        batch = min(batch_size, n_bootstrap - done)
        weights = rng.multinomial(n_prompts, np.full(n_prompts, 1.0 / n_prompts), size=batch)
        sampled = np.einsum("qn,nmge->qmge", weights, counts, optimize=True)
        count_draws.append(sampled)
        sampled_utility = (
            np.einsum("qn,nmgu->qmgu", weights, utility_counts, optimize=True)
            if utility_counts is not None
            else None
        )
        if sampled_utility is not None:
            utility_draws.append(sampled_utility)
        sampled_metric, _ = _event_metric_counts(sampled, sampled_utility)
        rates = _safe_ratio(sampled_metric[..., :-1], sampled_metric[..., -1, None])
        draws.append(rates)
        done += batch
    draw_rates = np.concatenate(draws, axis=0)
    draw_counts = np.concatenate(count_draws, axis=0)
    draw_utility = np.concatenate(utility_draws, axis=0) if utility_draws else None

    rows: list[dict[str, Any]] = []
    for bidx, (label, left, right) in enumerate(PROGRESS_BINS):
        for midx, metric in enumerate(metric_names):
            diff_draw = draw_rates[:, 1, bidx, midx] - draw_rates[:, 0, bidx, midx]
            rows.append(
                {
                    "range_type": "base_bin",
                    "progress_range": label,
                    "progress_left": left,
                    "progress_right": right,
                    "metric": metric,
                    "m0": float(observed_rates[0, bidx, midx]),
                    "m1": float(observed_rates[1, bidx, midx]),
                    "diff_m1_minus_m0": float(observed_rates[1, bidx, midx] - observed_rates[0, bidx, midx]),
                    "ci_low": _ci(diff_draw, 0.025),
                    "ci_high": _ci(diff_draw, 0.975),
                    "m0_denominator": int(observed_counts[0, bidx, -1]),
                    "m1_denominator": int(observed_counts[1, bidx, -1]),
                }
            )

    aggregate_ranges = {
        "0.50-1.00G": [BIN_INDEX["0.50-0.75G"], BIN_INDEX["0.75-1.00G"]],
        ">=1.00G": [BIN_INDEX["1.00-1.25G"], BIN_INDEX["1.25-1.50G"], BIN_INDEX[">=1.50G"]],
        ">=1.25G": [BIN_INDEX["1.25-1.50G"], BIN_INDEX[">=1.50G"]],
    }
    aggregate: dict[str, dict[str, Any]] = {}
    for label, indices in aggregate_ranges.items():
        obs = np.sum(observed_counts[:, indices, :], axis=1)
        obs_rate = _safe_ratio(obs[:, :-1], obs[:, -1, None])
        # Recompute aggregate bootstrap counts from prompt counts to avoid
        # averaging bin rates with different denominators.
        agg_draw_count = np.sum(draw_counts[:, :, indices, :], axis=2)
        agg_draw_utility = (
            np.sum(draw_utility[:, :, indices, :], axis=2) if draw_utility is not None else None
        )
        agg_draw_metric, _ = _event_metric_counts(
            agg_draw_count[:, :, None, :],
            agg_draw_utility[:, :, None, :] if agg_draw_utility is not None else None,
        )
        agg_draw_metric = agg_draw_metric[:, :, 0, :]
        agg_rates_draw = _safe_ratio(
            agg_draw_metric[..., :-1], agg_draw_metric[..., -1, None]
        )
        metric_map: dict[str, Any] = {}
        for midx, metric in enumerate(metric_names):
            diff_draw = agg_rates_draw[:, 1, midx] - agg_rates_draw[:, 0, midx]
            result = {
                "m0": float(obs_rate[0, midx]),
                "m1": float(obs_rate[1, midx]),
                "diff": float(obs_rate[1, midx] - obs_rate[0, midx]),
                "ci_low": _ci(diff_draw, 0.025),
                "ci_high": _ci(diff_draw, 0.975),
                "m0_denominator": int(obs[0, -1]),
                "m1_denominator": int(obs[1, -1]),
                "m0_draws": agg_rates_draw[:, 0, midx],
                "m1_draws": agg_rates_draw[:, 1, midx],
            }
            metric_map[metric] = result
            rows.append(
                {
                    "range_type": "aggregate",
                    "progress_range": label,
                    "progress_left": math.nan,
                    "progress_right": math.nan,
                    "metric": metric,
                    "m0": result["m0"],
                    "m1": result["m1"],
                    "diff_m1_minus_m0": result["diff"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "m0_denominator": result["m0_denominator"],
                    "m1_denominator": result["m1_denominator"],
                }
            )
        aggregate[label] = metric_map

    return {
        "rows": rows,
        "observed_rates": observed_rates,
        "draw_rates": draw_rates,
        "metric_names": metric_names,
        "aggregate": aggregate,
    }


def opportunity_contrasts(opportunity: Mapping[str, Any]) -> list[dict[str, Any]]:
    aggregate = opportunity["aggregate"]
    pre = aggregate["0.50-1.00G"]
    post = aggregate[">=1.00G"]
    deep = aggregate[">=1.25G"]
    rows: list[dict[str, Any]] = []

    def add(name: str, observed: float, draws: np.ndarray) -> None:
        rows.append(
            {
                "contrast": name,
                "value": float(observed),
                "ci_low": _ci(draws, 0.025),
                "ci_high": _ci(draws, 0.975),
            }
        )

    for metric in ("new_valid_any", "seen_triple_reuse", "normal_stop"):
        add(
            f"M1_post_ge1_minus_pre_0.5_1.0::{metric}",
            post[metric]["m1"] - pre[metric]["m1"],
            post[metric]["m1_draws"] - pre[metric]["m1_draws"],
        )
        add(
            f"M1_deep_ge1.25_minus_pre_0.5_1.0::{metric}",
            deep[metric]["m1"] - pre[metric]["m1"],
            deep[metric]["m1_draws"] - pre[metric]["m1_draws"],
        )
    add(
        "M1_post_ge1::new_valid_minus_seen_reuse",
        post["new_valid_any"]["m1"] - post["seen_triple_reuse"]["m1"],
        post["new_valid_any"]["m1_draws"] - post["seen_triple_reuse"]["m1_draws"],
    )
    add(
        "M1_deep_ge1.25::new_valid_minus_seen_reuse",
        deep["new_valid_any"]["m1"] - deep["seen_triple_reuse"]["m1"],
        deep["new_valid_any"]["m1_draws"] - deep["seen_triple_reuse"]["m1_draws"],
    )
    return rows


def _metric_arrays(
    analyses: Sequence[ResponseAnalysis],
    prompt_ids: Sequence[str],
    definitions: Sequence[tuple[str, Any, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    sums = np.zeros((len(prompt_ids), len(MODEL_TAGS), len(definitions)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for row in analyses:
        pidx = prompt_index[row.stable_prompt_id]
        midx = MODEL_INDEX[row.model_tag]
        for didx, (_name, value_fn, eligible_fn) in enumerate(definitions):
            if not eligible_fn(row):
                continue
            value = value_fn(row)
            if value is None:
                continue
            number = float(value)
            if math.isnan(number):
                continue
            sums[pidx, midx, didx] += number
            counts[pidx, midx, didx] += 1.0
    return sums, counts, [item[0] for item in definitions]


def bootstrap_response_metrics(
    analyses: Sequence[ResponseAnalysis],
    prompt_ids: Sequence[str],
    *,
    n_bootstrap: int,
    random_seed: int,
    batch_size: int = 200,
) -> dict[str, dict[str, float]]:
    seed_exists = lambda row: row.frozen_first_seed_index_0based is not None
    capture_exists = lambda row: row.semantic_capture
    legacy_exists = lambda row: row.legacy_structured and row.legacy_capture_seed_ordinal is not None
    definitions = [
        ("output_gold_ratio", lambda r: r.output_gold_ratio, lambda r: True),
        ("final_strict_coverage", lambda r: r.final_strict_coverage, lambda r: True),
        ("final_relaxed_coverage", lambda r: r.final_relaxed_coverage, lambda r: True),
        ("strict_coverage_at_1g", lambda r: r.strict_coverage_at_1g, lambda r: True),
        ("relaxed_coverage_at_1g", lambda r: r.relaxed_coverage_at_1g, lambda r: True),
        ("post_1g_strict_coverage_gain", lambda r: r.post_1g_strict_coverage_gain, lambda r: True),
        ("post_1g_relaxed_coverage_gain", lambda r: r.post_1g_relaxed_coverage_gain, lambda r: True),
        ("seed_rate", lambda r: float(seed_exists(r)), lambda r: True),
        ("seed_progress", lambda r: r.first_seed_progress, seed_exists),
        ("seed_strict_coverage", lambda r: r.first_seed_strict_coverage, seed_exists),
        ("seed_relaxed_coverage", lambda r: r.first_seed_relaxed_coverage, seed_exists),
        ("seed_strict_fraction_of_final", lambda r: r.first_seed_strict_fraction_of_final, seed_exists),
        ("seed_relaxed_fraction_of_final", lambda r: r.first_seed_relaxed_fraction_of_final, seed_exists),
        ("no_new_relaxed_after_seed_rate", lambda r: float(bool(r.no_new_relaxed_after_first_seed)), seed_exists),
        (
            "seed_after_last_new_relaxed_rate",
            lambda r: float(bool(r.first_seed_after_last_new_relaxed)),
            lambda r: seed_exists(r) and r.first_seed_after_last_new_relaxed is not None,
        ),
        ("seed_ge_1g_rate", lambda r: float(bool(r.first_seed_ge_1g)), seed_exists),
        ("seed_ge_1_25g_rate", lambda r: float(bool(r.first_seed_ge_1_25g)), seed_exists),
        ("semantic_capture_rate", lambda r: float(r.semantic_capture), lambda r: True),
        ("capture_later_than_first_rate", lambda r: float((r.capture_seed_ordinal or 0) > 1), capture_exists),
        ("capture_seed_ordinal_mean", lambda r: r.capture_seed_ordinal, capture_exists),
        ("capture_progress", lambda r: r.capture_progress, capture_exists),
        ("capture_relaxed_coverage", lambda r: r.capture_relaxed_coverage, capture_exists),
        (
            "capture_relaxed_fraction_of_final",
            lambda r: r.capture_relaxed_fraction_of_final,
            capture_exists,
        ),
        (
            "no_new_relaxed_after_capture_rate",
            lambda r: float(bool(r.no_new_relaxed_after_capture_seed)),
            capture_exists,
        ),
        (
            "capture_after_last_new_relaxed_rate",
            lambda r: float(bool(r.capture_after_last_new_relaxed)),
            lambda r: capture_exists(r) and r.capture_after_last_new_relaxed is not None,
        ),
        ("capture_seed_ge_1g_rate", lambda r: float(bool(r.capture_seed_ge_1g)), capture_exists),
        ("stable_orbit_rate", lambda r: float(r.stable_orbit), lambda r: True),
        (
            "legacy_capture_later_than_first_rate",
            lambda r: float((r.legacy_capture_seed_ordinal or 0) > 1),
            legacy_exists,
        ),
        (
            "legacy_capture_seed_ordinal_mean",
            lambda r: r.legacy_capture_seed_ordinal,
            legacy_exists,
        ),
        (
            "legacy_capture_seed_ge_1g_rate",
            lambda r: float(bool(r.legacy_capture_seed_ge_1g)),
            legacy_exists,
        ),
        (
            "legacy_capture_after_last_new_relaxed_rate",
            lambda r: float(bool(r.legacy_capture_after_last_new_relaxed)),
            lambda r: legacy_exists(r) and r.legacy_capture_after_last_new_relaxed is not None,
        ),
        (
            "frozen_seed_equals_normalized_first_rate",
            lambda r: float(bool(r.frozen_seed_matches_normalized_first)),
            lambda r: r.frozen_first_seed_index_0based is not None or r.normalized_first_reuse_index_0based is not None,
        ),
    ]
    sums, counts, names = _metric_arrays(analyses, prompt_ids, definitions)
    observed_sum = np.sum(sums, axis=0)
    observed_count = np.sum(counts, axis=0)
    observed = _safe_ratio(observed_sum, observed_count)

    draws_m0: list[np.ndarray] = []
    draws_m1: list[np.ndarray] = []
    n_prompts = len(prompt_ids)
    rng = np.random.default_rng(random_seed)
    done = 0
    while done < n_bootstrap:
        batch = min(batch_size, n_bootstrap - done)
        weights = rng.multinomial(n_prompts, np.full(n_prompts, 1.0 / n_prompts), size=batch)
        sum_b = np.einsum("bn,nmk->bmk", weights, sums, optimize=True)
        count_b = np.einsum("bn,nmk->bmk", weights, counts, optimize=True)
        rate_b = _safe_ratio(sum_b, count_b)
        draws_m0.append(rate_b[:, 0, :])
        draws_m1.append(rate_b[:, 1, :])
        done += batch
    m0_draw = np.concatenate(draws_m0, axis=0)
    m1_draw = np.concatenate(draws_m1, axis=0)

    results: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        diff = m1_draw[:, index] - m0_draw[:, index]
        results[name] = {
            "m0": float(observed[0, index]),
            "m1": float(observed[1, index]),
            "diff": float(observed[1, index] - observed[0, index]),
            "ci_low": _ci(diff, 0.025),
            "ci_high": _ci(diff, 0.975),
            "m0_ci_low": _ci(m0_draw[:, index], 0.025),
            "m0_ci_high": _ci(m0_draw[:, index], 0.975),
            "m1_ci_low": _ci(m1_draw[:, index], 0.025),
            "m1_ci_high": _ci(m1_draw[:, index], 0.975),
            "n_m0": int(observed_count[0, index]),
            "n_m1": int(observed_count[1, index]),
        }
    return results


def descriptive_summary(analyses: Sequence[ResponseAnalysis]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in MODEL_TAGS:
        rows = [row for row in analyses if row.model_tag == model]
        seeded = [row for row in rows if row.frozen_first_seed_index_0based is not None]
        captures = [row for row in rows if row.semantic_capture]
        legacy = [
            row for row in rows
            if row.legacy_structured and row.legacy_capture_seed_ordinal is not None
        ]

        def median(values: Iterable[float | int | None]) -> float:
            array = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
            return float(np.median(array)) if array.size else math.nan

        def ordinal_bins(values: Iterable[int | None]) -> dict[str, int]:
            counts = Counter()
            for value in values:
                if value is None:
                    continue
                if value == 1:
                    label = "1"
                elif value == 2:
                    label = "2"
                elif value <= 5:
                    label = "3-5"
                elif value <= 10:
                    label = "6-10"
                else:
                    label = ">10"
                counts[label] += 1
            return {label: int(counts[label]) for label in ("1", "2", "3-5", "6-10", ">10")}

        output[model] = {
            "n_responses": len(rows),
            "output_gold_ratio_median": median(row.output_gold_ratio for row in rows),
            "seed_progress_median": median(row.first_seed_progress for row in seeded),
            "seed_relaxed_coverage_median": median(row.first_seed_relaxed_coverage for row in seeded),
            "seed_relaxed_fraction_of_final_median": median(
                row.first_seed_relaxed_fraction_of_final for row in seeded
            ),
            "capture_seed_ordinal_median": median(row.capture_seed_ordinal for row in captures),
            "capture_seed_ordinal_bins": ordinal_bins(row.capture_seed_ordinal for row in captures),
            "legacy_capture_seed_ordinal_median": median(
                row.legacy_capture_seed_ordinal for row in legacy
            ),
            "legacy_capture_seed_ordinal_bins": ordinal_bins(
                row.legacy_capture_seed_ordinal for row in legacy
            ),
            "parser_invalid_objects": sum(row.parser_invalid_objects for row in rows),
            "parser_continuity_breaks": sum(row.parser_continuity_breaks for row in rows),
            "incomplete_tail_responses": sum(row.has_incomplete_tail for row in rows),
        }
    return output


def decision_from_results(
    opportunity: Mapping[str, Any],
    contrasts: Sequence[Mapping[str, Any]],
    response_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, str]:
    contrast = {str(row["contrast"]): row for row in contrasts}
    stop_diff = opportunity["aggregate"][">=1.00G"]["normal_stop"]
    utility = contrast["M1_post_ge1_minus_pre_0.5_1.0::new_valid_any"]
    reuse = contrast["M1_post_ge1_minus_pre_0.5_1.0::seen_triple_reuse"]
    dominance = contrast["M1_post_ge1::new_valid_minus_seen_reuse"]

    stop_deficit = stop_diff["ci_high"] < 0
    utility_decline = utility["ci_high"] < 0
    reuse_increase = reuse["ci_low"] > 0
    reuse_dominates = dominance["ci_high"] < 0

    if stop_deficit and utility_decline and (reuse_increase or reuse_dominates):
        completion = "SET_COMPLETION_FAILURE_SUPPORTED"
    elif utility["ci_low"] > 0 and dominance["ci_low"] > 0:
        completion = "OVERRUN_REMAINS_PRODUCTIVE"
    else:
        completion = "SET_COMPLETION_UNRESOLVED"

    legacy_later = response_metrics["legacy_capture_later_than_first_rate"]
    primary_later = response_metrics["capture_later_than_first_rate"]
    chosen = legacy_later if legacy_later["n_m1"] >= 20 else primary_later
    if chosen["n_m1"] == 0:
        capture_origin = "CAPTURE_ORIGIN_UNAVAILABLE"
    elif chosen["m1_ci_low"] > 0.5:
        capture_origin = "LATER_SEED_DOMINANT"
    elif chosen["m1_ci_high"] < 0.5:
        capture_origin = "FIRST_SEED_DOMINANT"
    else:
        capture_origin = "CAPTURE_ORIGIN_MIXED"

    if completion == "SET_COMPLETION_FAILURE_SUPPORTED" and capture_origin == "LATER_SEED_DOMINANT":
        label = "P0C_COMPLETION_FAILURE_PLUS_MULTISEED_CAPTURE"
        next_step = "进入模型级 set-completion/close-list vs continue-dict 边界分析；capture 作为下游多次 seed 暴露结果"
    elif completion == "SET_COMPLETION_FAILURE_SUPPORTED":
        label = "P0C_COMPLETION_FAILURE_SUPPORTED"
        next_step = "进入模型级 set-completion/termination 边界分析，不先做 capture/SAE DRS"
    elif completion == "OVERRUN_REMAINS_PRODUCTIVE":
        label = "P0C_OVERRUN_REMAINS_PRODUCTIVE"
        next_step = "参考 G 不能代表真实完成位置；先重建任务级 completion target"
    else:
        label = "P0C_UNRESOLVED"
        next_step = "检查边际效用曲线和 exact/relaxed 差异，暂不进入 SAE/DRS"
    return {
        "label": label,
        "completion_component": completion,
        "capture_origin": capture_origin,
        "next_step": next_step,
    }
