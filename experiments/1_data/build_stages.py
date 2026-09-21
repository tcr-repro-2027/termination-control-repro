# coding: utf-8
"""Build the frozen base / dedup / filter / clean / cleanv2 datasets.

Usage:
    python scripts/build_stages.py --repo-root ../..
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcr.data.layout import stage_directory  # noqa: E402
from tcr.data.stage_build.stages import (  # noqa: E402
    STAGE_FILENAMES,
    Record,
    StageStats,
    cleanv2_descriptions,
    drop_cross_split_text,
    drop_near_duplicate_text,
    load_base,
    norm,
    stage_clean,
    stage_cleanv2,
    stage_dedup,
    stage_filter,
    to_payload,
    write_jsonl,
)


def emit(stage: str, split: str, records: list[Record], data_out: Path) -> Path:
    path = stage_directory(data_out, stage) / STAGE_FILENAMES[stage].format(split=split)
    write_jsonl(path, (to_payload(record) for record in records))
    write_jsonl(path.with_name(path.stem + "_rowmap.jsonl"),
                ({"line": i, "rec_id": r.rec_id, "blocks": len(r.blocks)}
                 for i, r in enumerate(records)))
    return path


def run_split(split: str, raw_path: Path, data_out: Path,
              eval_base: list[Record] | None,
              protected_descriptions: set[str],
              report: dict,
              near_duplicates: set[str] | None = None) -> tuple[list[Record], list[dict]]:
    stats: dict[str, StageStats] = {name: StageStats() for name in STAGE_FILENAMES}

    records = load_base(raw_path, split, stats["base"])
    if eval_base is not None:
        records = drop_cross_split_text(records, eval_base, stats["base"])
    if near_duplicates:
        records = drop_near_duplicate_text(records, near_duplicates, stats["base"])
    emit("base", split, records, data_out)

    records = stage_dedup(records, stats["dedup"])
    emit("dedup", split, records, data_out)

    records = stage_filter(records, stats["filter"])
    emit("filter", split, records, data_out)

    records, diff_rows = stage_clean(records, stats["clean"])
    emit("clean", split, records, data_out)

    records = stage_cleanv2(records, stats["cleanv2"], protected_descriptions)
    emit("cleanv2", split, records, data_out)

    report[split] = {name: stats[name].as_dict() for name in STAGE_FILENAMES}
    report[split]["input_file"] = str(raw_path)
    return records, diff_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--data-out", default=None,
                        help="Dataset output root: cleanv2/ and optional stages/<stage>/")
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--drop-near-duplicates", default=None,
                        help="near_duplicate_removals.json from "
                             "near_duplicate_scan.py --source raw")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    datasets = repo_root / "datasets"
    data_out = Path(args.data_out) if args.data_out else datasets
    report_out = Path(args.report_out) if args.report_out else datasets / "build_reports"
    report_out.mkdir(parents=True, exist_ok=True)

    report: dict = {"version": "e0-stages-1.1"}

    near_duplicates: set[str] = set()
    near_meta: dict = {}
    if args.drop_near_duplicates:
        payload = json.loads(Path(args.drop_near_duplicates).read_text(encoding="utf-8"))
        near_duplicates = set(payload["train_rec_ids"])
        near_meta = {k: v for k, v in payload.items() if k != "train_rec_ids"}
        print(f"[stages] near-duplicate removals: {len(near_duplicates)}",
              file=sys.stderr, flush=True)

    # Evaluation runs first: its cleanv2 descriptions protect the train split,
    # and its base texts define the cross-split leakage to remove from train.
    print("[stages] eval ...", file=sys.stderr, flush=True)
    eval_base = load_base(datasets / "eval_orig.jsonl", "eval", StageStats())
    eval_records, eval_diff = run_split("eval", datasets / "eval_orig.jsonl", data_out,
                                        None, set(), report)
    protected = cleanv2_descriptions(eval_records)
    del eval_records

    print("[stages] train ...", file=sys.stderr, flush=True)
    train_records, train_diff = run_split("train", datasets / "train_orig.jsonl", data_out,
                                          eval_base, protected, report, near_duplicates)
    del eval_base

    diff_path = report_out / "filter_clean_problem4_diff.jsonl"
    write_jsonl(diff_path, train_diff)
    write_jsonl(report_out / "filter_clean_problem4_diff_eval.jsonl", eval_diff)

    report["cross_split"] = {
        "near_duplicate_removals": len(near_duplicates),
        "near_duplicate_criterion": near_meta,
        "protected_eval_descriptions": len(protected),
        "problem4_train_drops": len(train_diff),
        "problem4_eval_drops": len(eval_diff),
    }
    report["outputs"] = {
        stage: {split: str(stage_directory(data_out, stage) / STAGE_FILENAMES[stage].format(split=split))
                for split in ("train", "eval")}
        for stage in STAGE_FILENAMES
    }
    (report_out / "stage_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["stage      | train recs | train blocks | eval recs | eval blocks"]
    for stage in STAGE_FILENAMES:
        t, e = report["train"][stage], report["eval"][stage]
        lines.append(f"{stage:<10} | {t['records_out']:>10} | {t['blocks_out']:>12} | "
                     f"{e['records_out']:>9} | {e['blocks_out']:>11}")
    summary = "\n".join(lines)
    (report_out / "stage_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
