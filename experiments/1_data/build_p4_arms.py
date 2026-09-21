# coding: utf-8
"""Build keep4 / randdrop / keep4_a / keep4_ae -- but only after proving the
rewritten cleaning pass is the frozen one.

Two reproductions run before anything is written, and a failure of either stops
the build:

1. `label_problem4` keeping only the "Valid" blocks must reproduce the frozen
   `clean` file, record for record and block for block.  That is the problem-4
   decision.
2. `apply_later_problems` run on those `clean` records must reproduce the frozen
   `cleanv2` file, byte for byte.  That is the problems 5-11 pass with the
   corrected problem-10 fallout (see `p4x/rules.py`).

If both hold, the new arms are the frozen cleaning applied to a corpus that
still carries problem 4, and nothing about the rules has drifted.  If either
fails, the arms would be some other cleaning and the run is worthless -- so it
does not start.

    python experiments/1_data/build_p4_arms.py --datasets-root ../../datasets
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (REPO_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tcr.data.stage_build.stages import (                                # noqa: E402
    StageStats, cleanv2_descriptions, to_payload, write_jsonl,
)

from tcr.data.raw_arms.arms import (                                          # noqa: E402
    ARMS, AXIS_ARMS, BUILD_SEED, QUEUE_ARMS, build_axis_arm, build_keep4,
    build_randdrop, load_stage, removal_counts,
)
from tcr.data.raw_arms.rules import apply_later_problems, label_problem4      # noqa: E402


def log(message: str) -> None:
    print(f"[p4x] {message}", flush=True)


def blocks_signature(records: Sequence[Any]) -> list[tuple[str, int]]:
    return [(r.rec_id, len(r.blocks)) for r in records]


# ------------------------------------------------------------ reproductions

def check_problem4(filter_records, clean_path: Path) -> list[str]:
    """Keeping only the "Valid" blocks must give the frozen `clean` file."""
    problems: list[str] = []
    frozen = {r.rec_id: r for r in load_stage(clean_path, "train")}
    reproduced: dict[str, list[dict[str, str]]] = {}
    for record in filter_records:
        kinds = label_problem4(record)
        keep = [b for b, k in zip(record.blocks, kinds) if k == "Valid"]
        if keep:
            reproduced[record.rec_id] = keep
    if set(reproduced) != set(frozen):
        only_new = sorted(set(reproduced) - set(frozen))[:3]
        only_old = sorted(set(frozen) - set(reproduced))[:3]
        problems.append(f"record sets differ: {len(reproduced)} vs "
                        f"{len(frozen)} (e.g. new-only {only_new}, "
                        f"frozen-only {only_old})")
    mismatched = [rec_id for rec_id, blocks in reproduced.items()
                  if rec_id in frozen and blocks != frozen[rec_id].blocks]
    if mismatched:
        problems.append(f"{len(mismatched)} record(s) differ in their kept "
                        f"blocks, e.g. {mismatched[:3]}")
    return problems


def check_later_problems(clean_path: Path, cleanv2_path: Path,
                         protected: set[str]) -> list[str]:
    """The rewritten problems 5-11 pass must give the frozen `cleanv2` file."""
    records = load_stage(clean_path, "train")
    produced = apply_later_problems(records, StageStats(), protected)
    frozen = load_stage(cleanv2_path, "train")
    problems: list[str] = []
    if blocks_signature(produced) != blocks_signature(frozen):
        made = {r.rec_id: len(r.blocks) for r in produced}
        want = {r.rec_id: len(r.blocks) for r in frozen}
        differing = [k for k in want if made.get(k) != want[k]][:3]
        problems.append(f"{len(produced)} records / "
                        f"{sum(len(r.blocks) for r in produced)} blocks vs frozen "
                        f"{len(frozen)} / {sum(len(r.blocks) for r in frozen)}"
                        + (f"; e.g. {differing}" if differing else ""))
        return problems
    for made, want in zip(produced, frozen):
        if made.blocks != want.blocks:
            problems.append(f"{made.rec_id}: block contents differ")
            break
        if made.entities_str != want.entities_str:
            problems.append(f"{made.rec_id}: entities_str differs")
            break
    return problems


# ------------------------------------------------------------------- build

def duplicate_of(records: Sequence[Any],
                 frozen: dict[str, list[tuple[str, int]]]) -> str | None:
    """The frozen stage this arm reproduces, if it reproduces one.

    Compared on (rec_id, block count) per record and then on block contents --
    the same signature the reproduction checks use.  An arm that equals an
    existing stage carries no new information and must not reach the queue.
    """
    signature = blocks_signature(records)
    for name, other in frozen.items():
        if signature == other:
            return name
    return None


def emit(records: Sequence[Any], out_dir: Path, filename: str) -> dict[str, Any]:
    path = out_dir / filename
    rows = write_jsonl(path, (to_payload(r) for r in records))
    write_jsonl(path.with_name(path.stem + "_rowmap.jsonl"),
                ({"line": i, "rec_id": r.rec_id, "blocks": len(r.blocks)}
                 for i, r in enumerate(records)))
    blocks = sum(len(r.blocks) for r in records)
    log(f"  wrote {rows} records / {blocks} blocks -> {path.name}")
    return {"file": str(path), "records": rows, "blocks": blocks,
            "blocks_per_record": round(blocks / rows, 2) if rows else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-root",
                        default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--out-subdir", default="raw",
                        help="written under <datasets-root>/<out-subdir>")
    parser.add_argument("--seed", type=int, default=BUILD_SEED)
    parser.add_argument("--only", default=None,
                        help=f"comma-separated subset of {sorted(ARMS)}; "
                             f"default is the queue arms {list(QUEUE_ARMS)}")
    parser.add_argument("--report-out", default=None)
    args = parser.parse_args()

    root = Path(args.datasets_root).resolve()
    stages = root / "stages"
    filter_path = stages / "filter" / "train_filter.jsonl"
    clean_path = stages / "clean" / "train_clean.jsonl"
    cleanv2_path = root / "cleanv2" / "train_supportclean_keep8.jsonl"
    eval_cleanv2 = root / "cleanv2" / "eval_supportclean_keep8.jsonl"
    for path in (filter_path, clean_path, cleanv2_path, eval_cleanv2):
        if not path.is_file():
            log(f"FATAL: missing {path}")
            return 1

    log(f"datasets root: {root}")
    # Problem 11 protects descriptions that occur in the EVAL split, exactly as
    # the frozen chain does; read from the frozen eval file, not recomputed.
    protected = cleanv2_descriptions(load_stage(eval_cleanv2, "eval"))
    log(f"protected descriptions from eval cleanv2: {len(protected)}")

    log("reproduction 1/2: problem-4 labelling vs the frozen clean file")
    problems = check_problem4(load_stage(filter_path, "train"), clean_path)
    if problems:
        for problem in problems:
            log(f"  MISMATCH: {problem}")
        log("FATAL: the problem-4 decision here is not the frozen one; the arms "
            "would carry a different definition of 'out of bounds'.")
        return 1
    log("  ok: keeping only Valid blocks reproduces clean exactly")

    log("reproduction 2/2: problems 5-11 pass vs the frozen cleanv2 file")
    problems = check_later_problems(clean_path, cleanv2_path, protected)
    if problems:
        for problem in problems:
            log(f"  MISMATCH: {problem}")
        log("FATAL: the rewritten cleaning does not reproduce cleanv2; the "
            "corrected problem-10 fallout changed the normal chain, which it "
            "must not.")
        return 1
    log("  ok: the rewritten pass reproduces cleanv2 exactly")

    wanted = ({v.strip() for v in args.only.split(",") if v.strip()}
              if args.only else set(QUEUE_ARMS))
    unknown = wanted - set(ARMS)
    if unknown:
        log(f"FATAL: unknown arm(s) {sorted(unknown)}")
        return 1

    frozen_signatures = {
        "clean": blocks_signature(load_stage(clean_path, "train")),
        "cleanv2": blocks_signature(load_stage(cleanv2_path, "train")),
        "filter": blocks_signature(load_stage(filter_path, "train")),
    }

    out_dir = root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "datasets_root": str(root), "out_dir": str(out_dir), "seed": args.seed,
        "source": {"filter": str(filter_path), "clean": str(clean_path),
                   "cleanv2": str(cleanv2_path)},
        "reproductions": {"problem4_vs_clean": "ok",
                          "later_problems_vs_cleanv2": "ok"},
        "arms": {},
    }

    if "keep4" in wanted:
        log("keep4: problems 5-11 on filter, problem 4 left alone")
        stats = StageStats()
        records = build_keep4(load_stage(filter_path, "train"), stats, protected)
        report["arms"]["keep4"] = {**emit(records, out_dir, ARMS["keep4"]),
                                   "stats": stats.as_dict()}

    if "randdrop" in wanted:
        log("randdrop: matched-size random removal from filter, nothing else")
        filter_records = load_stage(filter_path, "train")
        counts = removal_counts(filter_records, clean_path)
        stats = StageStats()
        records = build_randdrop(filter_records, counts, stats, seed=args.seed)
        report["arms"]["randdrop"] = {
            **emit(records, out_dir, ARMS["randdrop"]),
            "stats": stats.as_dict(),
            "matched_removal_total": sum(counts.values()),
        }

    for arm in ("keep4_a", "keep4_ae", "keep4_e"):
        if arm not in wanted:
            continue
        kinds = AXIS_ARMS[arm]
        log(f"{arm}: keep4 restricted to problem-4 kind(s) {list(kinds)}")
        stats = StageStats()
        axis: Counter = Counter()
        records = build_axis_arm(load_stage(filter_path, "train"), stats,
                                 protected, keep_kinds=kinds, axis_stats=axis)
        # An arm that reproduces a stage we already have is not an arm.  On this
        # corpus `keep4_e` is exactly that: there are no pure evidence
        # violations, so it comes out as cleanv2 and training it would burn a
        # GPU night to re-measure an arm that is already in the CSV.
        duplicate = duplicate_of(records, frozen_signatures)
        if duplicate:
            log(f"  DEGENERATE: identical to `{duplicate}` "
                f"({len(records)} records / "
                f"{sum(len(r.blocks) for r in records)} blocks) -- not written")
            report["arms"][arm] = {"skipped_degenerate": duplicate,
                                   "records": len(records),
                                   "blocks": sum(len(r.blocks) for r in records),
                                   "axis": dict(axis)}
            continue
        report["arms"][arm] = {**emit(records, out_dir, ARMS[arm]),
                               "stats": stats.as_dict(), "axis": dict(axis)}

    report_path = (Path(args.report_out) if args.report_out
                   else out_dir / "raw_build_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    log(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
