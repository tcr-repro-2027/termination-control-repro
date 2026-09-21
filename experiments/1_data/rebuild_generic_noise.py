# coding: utf-8
"""Rebuild generic label noise using the frozen input-edit anchor manifest.

The manifest supplies the same anchor positions as the input-edit controls.
``apply_generic_noise`` maintains a running token balance within each record,
so each record is regenerated in full rather than patched one block at a time.
The optional anchor manifest must be supplied separately from the core data.

Usage from the repository root::

    python experiments/1_data/rebuild_generic_noise.py \
        --manifest datasets/controlled/edit_anchor_manifest.jsonl \
        --results datasets/build_reports --out-dir datasets/controlled
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcr.data.controlled.common import (  # noqa: E402
    TokenCounter,
    compact_json,
    read_jsonl,
    write_jsonl,
)
from tcr.data.controlled.noise import apply_generic_noise, build_clause_pool  # noqa: E402

ANCHOR_CONDITION = "isc/generic_noise/benign_input"


@dataclass
class Anchor:
    rec_id: str
    line: int
    block_index: int
    noise_relation: str


def tokenizer_dir(explicit, repo_root):
    """--tokenizer, else $MODEL_ROOT/Qwen3-4B, else <repo>/models/Qwen3/Qwen3-4B."""
    import os
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("MODEL_ROOT", str(repo_root / "models" / "Qwen3"))) / "Qwen3-4B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--tokenizer", default=None,
                        help="default: $MODEL_ROOT/Qwen3-4B, else <repo>/models/Qwen3/Qwen3-4B")
    parser.add_argument("--manifest", default=None,
                        help="pair manifest to read anchors from "
                             "(default: datasets/controlled/edit_anchor_manifest.jsonl)")
    parser.add_argument("--results", default=None,
                        help="where the GenericNoise manifest and report go "
                             "(default: datasets/build_reports/)")
    parser.add_argument("--out-dir", default=None,
                        help="where train_generic_noise.jsonl goes "
                             "(default: datasets/controlled)")
    parser.add_argument("--manifest-out", default=None,
                        help="default: <results>/generic_noise_manifest.jsonl")
    parser.add_argument("--report-out", default=None,
                        help="default: <results>/generic_noise_manifest.json")
    parser.add_argument("--compare-manifest", default=None,
                        help="an earlier GenericNoise manifest to diff against, "
                             "so the report can say which rows moved and why")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    datasets = repo_root / "datasets"
    results = Path(args.results) if args.results else datasets / "build_reports"
    data_out = Path(args.out_dir) if args.out_dir else datasets / "controlled"
    manifest_path = (Path(args.manifest) if args.manifest
                     else datasets / "controlled" / "edit_anchor_manifest.jsonl")
    manifest_out = (Path(args.manifest_out) if args.manifest_out
                    else results / "generic_noise_manifest.jsonl")
    report_out = (Path(args.report_out) if args.report_out
                  else results / "generic_noise_manifest.json")
    results.mkdir(parents=True, exist_ok=True)

    print(f"[noise] manifest={manifest_path}", file=sys.stderr, flush=True)
    print(f"[noise] out-dir={data_out} results={results}", file=sys.stderr, flush=True)
    print("[noise] loading cleanv2 train ...", file=sys.stderr, flush=True)
    records = read_jsonl(datasets / "cleanv2" / "train_supportclean_keep8.jsonl")
    counter = TokenCounter(tokenizer_dir(args.tokenizer, repo_root))

    print("[noise] building clause pool ...", file=sys.stderr, flush=True)
    pool = build_clause_pool(records, counter)
    print(f"[noise] clause pool: {sum(len(v) for v in pool.values())} clauses "
          f"across {len(pool)} token lengths", file=sys.stderr, flush=True)

    by_line: dict[int, list[Anchor]] = defaultdict(list)
    for row in read_jsonl(manifest_path):
        if row["condition"] != ANCHOR_CONDITION:
            continue
        by_line[row["line"]].append(Anchor(
            rec_id=row["rec_id"], line=row["line"],
            block_index=row["block_index"],
            # Reserve rows written by 20 may carry no v1 relation string; the
            # field is provenance only, `apply_generic_noise` reselects.
            noise_relation=row.get("noise_relation", "")))
    for anchors in by_line.values():
        anchors.sort(key=lambda a: a.block_index)
    total_anchors = sum(len(v) for v in by_line.values())
    print(f"[noise] anchors: {total_anchors} across {len(by_line)} records",
          file=sys.stderr, flush=True)

    stats: Counter = Counter()
    rows: list[dict] = []
    clean_tokens_total = 0
    signed_delta = 0
    absolute_delta = 0
    over_budget = 0
    max_ratio = 0.0

    def emit():
        nonlocal clean_tokens_total, signed_delta, absolute_delta, over_budget, max_ratio
        for line, record in enumerate(records):
            clean_tokens = counter.length(compact_json(record["output"]), cache=False)
            clean_tokens_total += clean_tokens
            anchors = by_line.get(line)
            if not anchors:
                yield record
                continue
            payload = dict(record)
            payload["output"] = apply_generic_noise(record, anchors, pool, counter,
                                                    stats, rows)
            mutated = counter.length(compact_json(payload["output"]), cache=False)
            signed_delta += mutated - clean_tokens
            absolute_delta += abs(mutated - clean_tokens)
            ratio = abs(mutated - clean_tokens) / max(clean_tokens, 1)
            max_ratio = max(max_ratio, ratio)
            over_budget += ratio > 0.01
            yield payload

    print(f"[noise] writing {data_out / 'train_generic_noise.jsonl'} ...",
          file=sys.stderr, flush=True)
    write_jsonl(data_out / "train_generic_noise.jsonl", emit())
    write_jsonl(manifest_out, rows)

    manifest = {
        "version": "generic-noise-2.1",
        "source_manifest": str(manifest_path),
        "mutation_fields": ["relation", "description"],
        "domain_scope": "same_cleanv2_record",
        "mechanisms": {
            "relation": "substitute a relation from the same source record, preferring "
                        "equal token length; retain the relation only when the record has "
                        "no alternative",
            "description": "only reorder clauses already present in the target block; "
                           "no clause is imported from another block",
        },
        "anchors": total_anchors,
        "records_touched": len(by_line),
        "mechanism_counts": dict(stats),
        "token_budget": {
            "clean_tokens_total": clean_tokens_total,
            "aggregate_signed_delta_ratio": round(signed_delta / max(clean_tokens_total, 1), 6),
            "aggregate_absolute_delta_ratio": round(absolute_delta / max(clean_tokens_total, 1), 6),
            "max_record_delta_ratio": round(max_ratio, 6),
            "records_over_1pct": over_budget,
        },
    }

    if args.compare_manifest:
        # A swapped anchor moves its record's running token balance, so rows the
        # manifest never mentions can still change.  Saying how many, and in
        # which records, is the only way that claim is checkable.
        before = {(str(row["rec_id"]), int(row["block_index"])): row
                  for row in read_jsonl(Path(args.compare_manifest))}
        after = {(str(row["rec_id"]), int(row["block_index"])): row for row in rows}
        removed = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        shared = set(before) & set(after)
        changed = [key for key in sorted(shared)
                   if (before[key].get("relation_after"),
                       before[key].get("description_after"))
                   != (after[key].get("relation_after"),
                       after[key].get("description_after"))]
        touched_records = {key[0] for key in removed} | {key[0] for key in added}
        collateral = [key for key in changed if key[0] in touched_records]
        manifest["diff_vs"] = {
            "manifest": str(args.compare_manifest),
            "rows_before": len(before),
            "rows_after": len(after),
            "anchors_removed": len(removed),
            "anchors_added": len(added),
            "symmetric_difference": len(removed) + len(added),
            "shared_rows_with_changed_content": len(changed),
            "changed_inside_a_touched_record": len(collateral),
            "changed_outside_any_touched_record": len(changed) - len(collateral),
            "records_touched": len(touched_records),
            "unchanged_records_identical": len(changed) - len(collateral) == 0,
            "examples": {"removed": removed[:10], "added": added[:10],
                         "changed": changed[:10]},
        }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
