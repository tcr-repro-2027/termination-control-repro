# coding=utf-8
"""Is the anchor set on disk complete, current, and the one its report describes?

The launcher skips construction when the file exists.  Existence is not enough:
a build killed midway used to leave a short file that every later run reused in
silence.  Construction is atomic now, but this still guards a file copied in by
hand, truncated in transit, built under an older protocol, built without a
tokenizer, or left over from a different eval set.

The rules live in `tcr.support_probe.identity.anchor_set_problems` so that `--check`, the
launcher's skip decision and the preflight all apply the SAME ones; a guard that
exists in only one of the three paths is a guard the operator will step around
without knowing.

    python scripts/e2_check_anchors.py <anchors.jsonl> <report.json> <n_per_type>
        [--eval_data <eval.jsonl>] [--allow_char_fallback]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcr.support_probe.identity import anchor_set_problems                    # noqa: E402
from tcr.support_probe.io_utils import sha256_file                            # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("anchors")
    parser.add_argument("report")
    parser.add_argument("n_per_type", type=int)
    parser.add_argument("--eval_data", default=None,
                        help="If given, the report's eval_data_sha256 must be "
                             "this file's: anchors built from another eval set "
                             "are not the frozen E-Natural ones.")
    parser.add_argument("--max_length_delta", type=int, default=0,
                        help="The donor length tolerance this run means.  The "
                             "report's must match: anchors built with a looser "
                             "one are not the length-matched set.")
    parser.add_argument("--allow_char_fallback", action="store_true",
                        help="Accept a build made without a tokenizer.  That is "
                             "a pilot build: donor lengths are matched in "
                             "characters, not tokens.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    problems = anchor_set_problems(
        args.anchors, args.report, n_per_type=args.n_per_type,
        eval_data=args.eval_data,
        require_token_matched=not args.allow_char_fallback,
        max_length_delta=args.max_length_delta)
    if problems:
        for problem in problems:
            print(f"[anchors] {problem}")
        print("[anchors] rebuild with REBUILD=1")
        return 1
    print(f"[anchors] complete and current, sha256 "
          f"{sha256_file(args.anchors)[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
