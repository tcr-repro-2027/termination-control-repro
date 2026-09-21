# coding=utf-8
"""Rebuild `e1_metrics.csv` from the per-task summaries, and audit it.

The orchestrator appends a row as each task finishes, which is what you want
while it runs and not what you want afterwards: rows appear in completion
order, a column added later is missing from earlier rows, and a task re-run by
hand appears twice.  This rebuilds the file from `summary/*.json` -- the source
of truth -- in the queue's priority order, one row per tag, with the current
column set.

It also runs the comparability checks E1 exists to make possible, because a
number that is not comparable is worse than a missing one:

* every task scored the same evaluation file (`eval_data_sha256`);
* every task used the same rendered prompt (`prompt_rendered_sha256`);
* every task used the same protocol version;
* every task saw the same denominator (`n_records`, `n_responses`);
* every checkpoint's chat prefix matched training.

Finally it writes `e1_stage_curve.csv`: the §E1 main comparisons
(dedup->filter->clean->cleanv2 per size, and each terminal-aware model against
the SAME data base it was trained on) as explicit before/after pairs, so the
paper's table is read off a file instead of assembled by hand.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcr.evaluation.aggregate import CSV_FIELDS, summary_to_row     # noqa: E402
from tcr.evaluation.io_utils import (                               # noqa: E402
    FileLock, read_csv_rows, read_json, write_csv,
)

#: §E1 主比较.  Each pair is (before, after) on one size.
STAGE_PAIRS = (("base", "dedup"), ("dedup", "filter"), ("filter", "clean"),
               ("clean", "cleanv2"), ("none", "base"))

#: Metrics the stage curve reports a delta for.
CURVE_METRICS = ("semantic_capture_rate", "stable_orbit_rate",
                 "legacy_loop_rate", "hit_max_rate", "first_triple_reuse_rate",
                 "episodes_per_response", "per_episode_capture_hazard",
                 "gap_stop_hazard", "strict_f1", "relaxed_f1",
                 "json_valid_rate", "out_of_candidate_block_rate",
                 "gen_tokens_mean")


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


#: Reading order inside one size, so the CSV reads like the paper's table
#: instead of like an alphabetical listing (`clean` before `dedup`).
VARIANT_ORDER = {name: index for index, name in enumerate(
    ("none", "base", "dedup", "filter", "clean", "cleanv2",
     "benign_input", "generic_noise",
     "obr_p10", "obr_p15", "obr",          # the OBR dose curve, ascending
     "isc_a", "isc_e", "isc_ae"))}
SIZE_ORDER = {"1.7B": 0, "4B": 1, "8B": 2}


def build_rows(summary_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(summary_dir.glob("*_summary.json")):
        summary = read_json(path)
        task = summary.get("task", {})
        row = summary_to_row(summary)
        row["_tier"] = task.get("tier", 99)
        row["_size"] = SIZE_ORDER.get(str(task.get("size")), 9)
        row["_variant"] = VARIANT_ORDER.get(str(task.get("data_variant")), 99)
        row["_terminal"] = 1 if task.get("terminal_mode") else 0
        row["_step"] = task.get("ckpt_step") or 0
        rows.append(row)
    rows.sort(key=lambda row: (row["_tier"], row["_size"], row["_variant"],
                               row["_terminal"], str(row.get("run_name")),
                               row["_step"]))
    for row in rows:
        for name in ("_tier", "_size", "_variant", "_terminal", "_step"):
            row.pop(name, None)
    return rows


def audit(rows: List[Dict[str, Any]]) -> List[str]:
    problems: List[str] = []
    for column, label in (("eval_data_sha256", "evaluation set"),
                          ("prompt_rendered_sha256", "prompt"),
                          ("protocol_version", "protocol version"),
                          ("n_records", "record denominator"),
                          ("n_responses", "response denominator")):
        seen = defaultdict(list)
        for row in rows:
            seen[str(row.get(column))].append(row["tag"])
        if len(seen) > 1:
            detail = "; ".join(
                f"{value or '<empty>'}: {len(tags)} task(s) e.g. {tags[0]}"
                for value, tags in sorted(seen.items(), key=lambda item: -len(item[1])))
            problems.append(f"{label} differs across tasks -- {detail}")
    for row in rows:
        if str(row.get("chat_prefix_ok")).lower() not in ("true", "1"):
            problems.append(f"{row['tag']}: chat prefix did not match training")
        if _float(row.get("episode_errors")) not in (0.0, None):
            problems.append(f"{row['tag']}: {row['episode_errors']} episode "
                            "row(s) failed P0d validation")
    return problems


#: The seed every §E1 comparison is defined on.  The matrix deliberately
#: contains a second seed for one cell (`4B-cleanv2-s123`), and a stage curve
#: that silently picked whichever row was read last would depend on filesystem
#: ordering.  Extra seeds belong in a seed-stability check, not in the curve.
CANONICAL_SEED = "42"


def stage_curve(rows: List[Dict[str, Any]],
                problems: List[str] | None = None) -> List[Dict[str, Any]]:
    """The §E1 main comparisons as explicit (before, after, delta) rows."""
    # Ambiguity is tracked in its own set rather than by poisoning `by_key`
    # with None: a third row matching the same cell would then dereference the
    # poison value.  Here every extra match just adds a tag to the list.
    candidates: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("is_final")).lower() not in ("true", "1"):
            continue
        if row.get("arm_family") not in ("stage", "reference"):
            continue
        if row.get("terminal_mode"):
            continue
        seed = str(row.get("seed") or "").strip()
        # Untrained reference models have no training seed; every trained cell
        # of the curve must be the canonical one.
        if row.get("arm_family") != "reference" and seed != CANONICAL_SEED:
            continue
        candidates[(row.get("size"), row.get("data_variant"))].append(row)

    by_key: Dict[tuple, Dict[str, Any]] = {}
    for key, matches in candidates.items():
        if len(matches) > 1:
            if problems is not None:
                tags = ", ".join(str(match.get("tag")) for match in matches)
                problems.append(
                    f"stage curve: {key[0]} / {key[1]} matches "
                    f"{len(matches)} rows ({tags}); the cell is ambiguous and "
                    "was left out")
            continue
        by_key[key] = matches[0]

    out: List[Dict[str, Any]] = []
    for size in sorted({row.get("size") for row in by_key.values()}):
        for before, after in STAGE_PAIRS:
            left = by_key.get((size, before))
            right = by_key.get((size, after))
            if left is None or right is None:
                continue
            entry: Dict[str, Any] = {
                "comparison": f"{before}->{after}", "size": size,
                "before_tag": left["tag"], "after_tag": right["tag"],
            }
            for metric in CURVE_METRICS:
                low, high = _float(left.get(metric)), _float(right.get(metric))
                entry[f"{metric}_before"] = low
                entry[f"{metric}_after"] = high
                entry[f"{metric}_delta"] = (
                    None if low is None or high is None else high - low)
            out.append(entry)

    # terminal-aware is only interpretable against its OWN data base (§E1).
    for row in rows:
        if not row.get("terminal_mode"):
            continue
        baseline = by_key.get((row.get("size"), row.get("data_variant")))
        if baseline is None:
            continue
        entry = {
            "comparison": f"{row.get('data_variant')}->"
                          f"{row.get('data_variant')}+terminal_aware",
            "size": row.get("size"), "before_tag": baseline["tag"],
            "after_tag": row["tag"],
        }
        for metric in CURVE_METRICS:
            low, high = _float(baseline.get(metric)), _float(row.get(metric))
            entry[f"{metric}_before"] = low
            entry[f"{metric}_after"] = high
            entry[f"{metric}_delta"] = (
                None if low is None or high is None else high - low)
        out.append(entry)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--csv", default=None,
                        help="Default: <result_root>/e1_metrics.csv")
    parser.add_argument("--curve_csv", default=None,
                        help="Default: <result_root>/e1_stage_curve.csv")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero when a comparability check fails.")
    args = parser.parse_args()

    root = Path(args.result_root)
    rows = build_rows(root / "summary")
    if not rows:
        print(f"no summaries under {root / 'summary'}")
        return 1

    target = Path(args.csv) if args.csv else root / "e1_metrics.csv"
    # The orchestrator appends to this file as tasks finish, and the README
    # allows rebuilding while it runs.  Take the SAME lock it appends under, and
    # merge rather than replace: a row appended between reading the summaries
    # and writing the file would otherwise be dropped.  Summaries win on
    # conflict -- they are the source of truth, the appended row is a copy.
    unverified: List[str] = []
    with FileLock(target):
        # Re-read the summaries INSIDE the lock.  The row we are trying not to
        # lose was appended by the orchestrator only after its summary was
        # written, so the second read almost always finds it and it comes back
        # as a first-class, auditable row rather than a copy of a CSV line.
        rows = build_rows(root / "summary")
        known = {str(row["tag"]) for row in rows}
        carried = [row for row in read_csv_rows(target)
                   if str(row.get("tag")) not in known]
        unverified = [str(row.get("tag")) for row in carried]
        ordered = rows + carried
        write_csv(target, ordered, CSV_FIELDS)
    print(f"wrote {len(ordered)} row(s) -> {target}")

    # Audit what is IN the file, not just what came from summaries: a carried
    # row can be a leftover from an earlier, incompatible run in this result
    # root, and reporting "comparability checks passed" over a file that
    # contains one would be the exact failure E1 exists to prevent.
    problems = audit(ordered)
    for tag in unverified:
        problems.append(
            f"{tag}: kept from the existing CSV because no summary/*.json "
            "backs it.  Either it was appended by a running orchestrator (it "
            "will verify on the next rebuild), or it is a leftover from an "
            "earlier run -- delete the row, or the whole CSV, and rebuild.")
    curve = stage_curve(rows, problems)
    curve_path = Path(args.curve_csv) if args.curve_csv else root / "e1_stage_curve.csv"
    if curve:
        write_csv(curve_path, curve)
        print(f"wrote {len(curve)} comparison(s) -> {curve_path}")
    else:
        print("no complete stage pair yet; skipping the stage-curve file")

    if problems:
        print("\n---- comparability problems ----")
        for problem in problems:
            print(f"  [!] {problem}")
        if args.strict:
            return 1
    else:
        print("comparability checks passed: one eval set, one prompt, one "
              "protocol, one denominator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
