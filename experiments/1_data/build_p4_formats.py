# coding: utf-8
"""Render the new arms into prompt / chatml / swift, with the frozen renderer.

`build_training_formats.py` is imported and its `convert()` called, rather than
copied: the prompt template, the `<think>` handling and the output serialisation
are exactly what every other arm was built with, and a second implementation of
them is a second thing that can drift.  Its `discover()` only knows the five
frozen stage directories, so the job list is built here instead -- that is the
whole reason this file exists.

    python experiments/1_data/build_p4_formats.py --datasets-root ../../datasets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (REPO_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from build_training_formats import (                            # noqa: E402
    PROMPT_TEMPLATE_IS_TRUNCATED, convert, log,
)

from tcr.data.raw_arms.arms import ARMS, QUEUE_ARMS                           # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-root", default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--arms-subdir", default="raw")
    parser.add_argument("--out-root", default=None,
                        help="default: <datasets-root>/processed")
    parser.add_argument("--only", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-out", default=None)
    args = parser.parse_args()

    if PROMPT_TEMPLATE_IS_TRUNCATED:
        log("FATAL: prompt_template.py is the abridged copy; the rendered "
            "prompts would not match the evaluation pipeline.")
        return 1

    root = Path(args.datasets_root).resolve()
    arms_dir = root / args.arms_subdir
    out_root = (Path(args.out_root).resolve() if args.out_root
                else root / "processed")
    wanted = ({v.strip() for v in args.only.split(",") if v.strip()}
              if args.only else set(QUEUE_ARMS))

    jobs = []
    for arm in sorted(wanted):
        path = arms_dir / ARMS[arm]
        if not path.is_file():
            log(f"WARNING: {arm} not built ({path}); skipping")
            continue
        jobs.append({"group": args.arms_subdir, "split": "train",
                     "name": arm, "path": path})
    if not jobs:
        log("nothing to convert")
        return 1

    log(f"arms dir:    {arms_dir}")
    log(f"output root: {out_root / args.arms_subdir}")
    reports = {}
    for job in jobs:
        log(f"{job['group']} / {job['name']}")
        reports[f"{job['group']}/train_{job['name']}"] = convert(
            job, out_root / job["group"], args.force)

    report_path = (Path(args.report_out) if args.report_out
                   else out_root / args.arms_subdir / "raw_formats_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"arms_dir": str(arms_dir), "out_root": str(out_root),
                    "renderer": "experiments/1_data/build_training_formats.py",
                    "files": reports}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    log(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
