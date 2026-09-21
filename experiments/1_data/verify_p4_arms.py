# coding: utf-8
"""Prove each new arm carries what its name claims, before any GPU time.

The failure this exists for is not a crash.  It is an arm that looks entirely
ordinary -- right record count, right schema, trains fine -- while carrying a
different corpus than the one the experiment is about.  The specific way that
happens here is documented in `p4x/rules.py`: a naive "skip the clean stage"
deletes every out-of-candidate block under the name `problem10`, and produces
something close to `clean` while the logs say `keep4`.

So the single most important line below is the one that fails when `keep4`'s
out-of-candidate share is near zero.

    python experiments/1_data/verify_p4_arms.py --datasets-root ../../datasets
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (REPO_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tcr.data.stage_build.stages import (                                # noqa: E402
    contains, is_low_quality_entity, norm, read_jsonl, relation_leak,
)

from tcr.data.raw_arms.arms import load_stage                                 # noqa: E402
from tcr.data.raw_arms.rules import label_problem4                            # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def profile(path: Path, split: str = "train") -> dict[str, Any]:
    """Everything the gates below need, in one pass over the file."""
    records = load_stage(path, split)
    counts: Counter = Counter()
    blocks = 0
    per_record: dict[str, int] = {}
    for record in records:
        per_record[record.rec_id] = len(record.blocks)
        blocks += len(record.blocks)
        entity_set = set(record.entities)
        for kind in label_problem4(record):
            counts[f"p4_{kind}"] += 1
        for block in record.blocks:
            if norm(block["source"]) == norm(block["target"]):
                counts["problem5_self_loop"] += 1
            reason, _swapped = relation_leak(block)
            if reason:
                counts["problem7_leak"] += 1
            if (not contains(block["description"], block["source"])
                    and not contains(block["description"], block["target"])):
                counts["problem9_weak_description"] += 1
        if any(is_low_quality_entity(e) for e in record.entities):
            counts["problem10_low_quality_entity_record"] += 1
        seen_pairs: Counter = Counter()
        seen_descriptions: Counter = Counter()
        for block in record.blocks:
            seen_pairs[frozenset((norm(block["source"]), norm(block["target"])))] += 1
            seen_descriptions[norm(block["description"])] += 1
        counts["problem6_reverse_duplicate"] += sum(
            n - 1 for n in seen_pairs.values() if n > 1)
        counts["problem11_duplicate_description"] += sum(
            n - 1 for n in seen_descriptions.values() if n > 1)
    violating = counts["p4_A"] + counts["p4_E"] + counts["p4_AE"]
    return {
        "path": str(path), "records": len(records), "blocks": blocks,
        "blocks_per_record": round(blocks / len(records), 2) if records else 0.0,
        "out_of_bounds_blocks": violating,
        "out_of_bounds_share": round(violating / blocks, 4) if blocks else 0.0,
        "p4_kinds": {k: counts[f"p4_{k}"] for k in ("A", "E", "AE")},
        "later_problems": {k: v for k, v in sorted(counts.items())
                           if not k.startswith("p4_")},
        "_per_record": per_record,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-root", default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--arms-subdir", default="raw")
    parser.add_argument("--report-out", default=None)
    args = parser.parse_args()

    root = Path(args.datasets_root).resolve()
    stages = root / "stages"
    arms = root / args.arms_subdir
    sources = {
        "filter": stages / "filter" / "train_filter.jsonl",
        "clean": stages / "clean" / "train_clean.jsonl",
        "cleanv2": root / "cleanv2" / "train_supportclean_keep8.jsonl",
        "keep4": arms / "train_keep4.jsonl",
        "randdrop": arms / "train_randdrop.jsonl",
        "keep4_a": arms / "train_keep4_a.jsonl",
        "keep4_ae": arms / "train_keep4_ae.jsonl",
    }
    profiles: dict[str, dict[str, Any]] = {}
    for name, path in sources.items():
        if not path.is_file():
            log(f"[skip] {name}: {path} not built")
            continue
        log(f"[read] {name}")
        profiles[name] = profile(path)

    ok = True
    failures: list[str] = []

    def check(condition: bool, label: str, detail: str) -> None:
        nonlocal ok
        if condition:
            log(f"  [ok]   {label}: {detail}")
        else:
            ok = False
            failures.append(f"{label}: {detail}")
            log(f"  [FAIL] {label}: {detail}")

    log("\n==== the table ====")
    header = f"{'arm':10s} {'records':>8s} {'blocks':>9s} {'b/rec':>7s} {'oob':>9s} {'oob%':>7s}"
    log(header)
    for name, p in profiles.items():
        log(f"{name:10s} {p['records']:8d} {p['blocks']:9d} "
            f"{p['blocks_per_record']:7.1f} {p['out_of_bounds_blocks']:9d} "
            f"{p['out_of_bounds_share'] * 100:6.2f}%")

    if "keep4" in profiles:
        log("\n==== keep4: must still carry problem 4 ====")
        k, f = profiles["keep4"], profiles.get("filter", {})
        # THE load-bearing check.  Near zero here means the problem-10 fallout
        # ate the out-of-candidate blocks and this arm is a relabelled `clean`.
        check(k["out_of_bounds_share"] > 0.15,
              "problem 4 present",
              f"{k['out_of_bounds_share'] * 100:.2f}% of blocks violate support "
              f"(filter is {f.get('out_of_bounds_share', 0) * 100:.2f}%); "
              "near zero would mean the clean step ran by accident")
        for problem in ("problem5_self_loop", "problem7_leak",
                        "problem9_weak_description", "problem6_reverse_duplicate",
                        "problem11_duplicate_description"):
            check(k["later_problems"].get(problem, 0) == 0,
                  f"{problem} cleaned",
                  f"{k['later_problems'].get(problem, 0)} residual")
        if "cleanv2" in profiles:
            check(k["records"] >= profiles["cleanv2"]["records"],
                  "record count",
                  f"{k['records']} vs cleanv2 {profiles['cleanv2']['records']}")

    if "randdrop" in profiles and "clean" in profiles:
        log("\n==== randdrop: must match clean record-for-record ====")
        r, c = profiles["randdrop"], profiles["clean"]
        mismatched = [rec for rec, n in c["_per_record"].items()
                      if r["_per_record"].get(rec) != n]
        # Not "the same average length" -- the same length in every record.
        check(not mismatched, "per-record block counts equal clean's",
              "identical" if not mismatched
              else f"{len(mismatched)} record(s) differ, e.g. {mismatched[:3]}")
        check(r["records"] == c["records"], "record count",
              f"{r['records']} vs clean {c['records']}")
        check(r["out_of_bounds_share"] > 0.15, "problem 4 left in proportion",
              f"{r['out_of_bounds_share'] * 100:.2f}% (a random removal must not "
              "change the composition)")
        residual = r["later_problems"].get("problem6_reverse_duplicate", 0)
        check(residual > 0, "problems 5-11 NOT cleaned",
              f"{residual} reverse duplicates remain, as in clean "
              f"({c['later_problems'].get('problem6_reverse_duplicate', 0)}); "
              "cleaning them would break the pairing with clean")

    if "keep4_a" in profiles and "keep4_ae" in profiles:
        log("\n==== axis arms ====")
        a, ae = profiles["keep4_a"], profiles["keep4_ae"]
        check(a["p4_kinds"]["AE"] == 0, "keep4_a is admissibility only",
              f"AE blocks: {a['p4_kinds']['AE']}")
        check(ae["p4_kinds"]["A"] == 0, "keep4_ae is both-axes only",
              f"A-only blocks: {ae['p4_kinds']['A']}")
        if "keep4" in profiles:
            check(profiles["keep4"]["p4_kinds"]["E"] == 0,
                  "no pure evidence violations exist",
                  "0 -- every candidate entity occurs in its own text, so a "
                  "keep4_e arm would be cleanv2; A vs AE is the only evidence "
                  "contrast this corpus supports")

    log("\n==== " + ("PASSED" if ok else "FAILED") + " ====")
    for failure in failures:
        log(f"  [!] {failure}")

    for p in profiles.values():
        p.pop("_per_record", None)
    report_path = (Path(args.report_out) if args.report_out
                   else arms / "raw_verify_report.json")
    report_path.write_text(
        json.dumps({"ok": ok, "failures": failures, "profiles": profiles},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report -> {report_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
