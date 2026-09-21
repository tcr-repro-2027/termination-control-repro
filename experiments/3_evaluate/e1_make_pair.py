# coding=utf-8
"""Turn two E1 tasks into the M0/M1 pair the 01b P0 analysers expect.

`tcr/events`'s P0b / P0c / P0d are PAIRED analyses: they read
one `event_rows.jsonl` holding both arms, tagged `M0` and `M1`, and refuse
anything else (`validate_prevalence_rows`).  E1 scores one model at a time and
tags each row with its own model tag, which is the right thing on disk and the
wrong thing for those scripts.

This writes `<out_dir>/event_rows.jsonl` with

    model_tag  M0 / M1                (the original kept as `e1_model_tag`)
    sample_id  rebuilt as M0:<key>:<seed>, since P0b requires it unique

and a `run_manifest.json` beside it, so the P0 analysers can be pointed at the
directory directly::

    python scripts/analyze_p0d_episode_hazard.py \
        --result_dir  <out_dir> \
        --gold        .../cleanv2/eval_supportclean_keep8.jsonl

Rows whose prompt is missing from either side are dropped and counted -- the
analysers require a complete M0/M1 pairing on every prompt and seed, and a
truncated arm (a task still generating) would otherwise abort them with a
prompt-mismatch error a long way from its cause.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcr.evaluation.io_utils import (                                # noqa: E402
    JsonlWriter, read_json, read_jsonl, sha256_file, stable_json, write_json,
)


def _index(path: Path) -> Dict[tuple, Dict[str, Any]]:
    """`(prompt, seed) -> row`, refusing to silently drop a duplicate.

    A duplicated (prompt, seed) means the file holds two runs' rows.  Keeping
    the last one would produce a paired analysis over a mixture, so it stops
    here instead."""
    rows: Dict[tuple, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = (str(row.get("stable_prompt_id")), int(row.get("seed")))
        if key in rows:
            raise SystemExit(
                f"{path} has two rows for prompt={key[0]} seed={key[1]}; the "
                "file mixes more than one run and cannot be paired.  "
                "Re-run the analysis for this tag.")
        rows[key] = row
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def _summary_for(events_path: Path) -> Dict[str, Any] | None:
    """The task summary that belongs to an `events/<tag>_event_rows.jsonl`."""
    stem = events_path.name[:-len("_event_rows.jsonl")] \
        if events_path.name.endswith("_event_rows.jsonl") else events_path.stem
    candidate = events_path.parent.parent / "summary" / f"{stem}_summary.json"
    if not candidate.is_file():
        return None
    try:
        return read_json(candidate)
    except (OSError, ValueError):
        return None


def check_comparable(m0: Path, m1: Path, *, allow_unverified: bool = False) -> None:
    """Both arms must have been scored under the same protocol and eval set.

    Pairing two arms that saw different prompts or different gold produces a
    paired difference that is not a difference between the arms.  This check
    therefore fails CLOSED: with no summary beside an event file there is
    nothing to compare, and "nothing to compare" is not "compatible" -- the
    prompt keys would still line up and the pairing would look perfectly
    healthy.  `--allow_unverified` is for event files copied away from the run
    that produced them, and puts the burden on the caller."""
    left, right = _summary_for(m0), _summary_for(m1)
    if left is None or right is None:
        missing = [str(path) for path, summary in ((m0, left), (m1, right))
                   if summary is None]
        message = ("no summary/*.json found beside " + ", ".join(missing) +
                   "; the two arms cannot be shown to share an evaluation set, "
                   "a prompt and a protocol.")
        if not allow_unverified:
            raise SystemExit(
                message + "  Run e1_make_pair from the result root that "
                "produced these event files, or pass --allow_unverified if you "
                "have checked their provenance yourself.")
        print(f"[!] {message}  Continuing because --allow_unverified was given.")
        return
    for path, label in ((("protocol_version",), "protocol version"),
                        (("generation", "eval_data_sha256"), "evaluation set"),
                        (("generation", "prompt_rendered_sha256"), "prompt")):
        def dig(document: Dict[str, Any]) -> Any:
            value: Any = document
            for step in path:
                value = (value or {}).get(step) if isinstance(value, dict) else None
            return value

        if dig(left) != dig(right):
            raise SystemExit(
                f"the two arms do not share the same {label} "
                f"({dig(left)!r} vs {dig(right)!r}); pairing them would compare "
                "two different evaluations, not two models.")
    print("[ok] both arms share one protocol, one evaluation set and one prompt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0", required=True, help="event_rows.jsonl of the baseline arm")
    parser.add_argument("--m1", required=True, help="event_rows.jsonl of the treated arm")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--allow_unverified", action="store_true",
                        help="Pair event files that have no summary beside "
                             "them.  Only for files copied away from the run "
                             "that produced them; it disables the "
                             "eval-set/prompt/protocol comparison.")
    args = parser.parse_args()

    m0_path, m1_path = Path(args.m0), Path(args.m1)
    check_comparable(m0_path, m1_path, allow_unverified=args.allow_unverified)
    left, right = _index(m0_path), _index(m1_path)
    shared = sorted(set(left) & set(right), key=lambda item: (item[0], item[1]))
    dropped = len(set(left) ^ set(right))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "event_rows.jsonl"

    prompts: set = set()
    with JsonlWriter(out_path, mode="w") as handle:
        for tag, source in (("M0", left), ("M1", right)):
            for key in shared:
                row = dict(source[key])
                row["e1_model_tag"] = row.get("model_tag")
                row["model_tag"] = tag
                row["sample_id"] = f"{tag}:{stable_json(row.get('key'))}:{row['seed']}"
                roles = set(row.get("selection_roles") or [])
                roles.add("prevalence")
                row["selection_roles"] = sorted(roles)
                handle.write(row)
                prompts.add(key[0])

    manifest = {
        "pair": {"M0": str(m0_path), "M1": str(m1_path)},
        "m0_sha256": sha256_file(m0_path),
        "m1_sha256": sha256_file(m1_path),
        "n_prompts": len(prompts),
        "n_rows": 2 * len(shared),
        "n_dropped_unpaired_responses": dropped,
        "output": str(out_path),
        "note": "written by experiments/3_evaluate/e1_make_pair.py for the event "
                "P0b/P0c/P0d analysers; model_tag relabelled to M0/M1",
    }
    write_json(out_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if dropped:
        print(f"\n[!] {dropped} response(s) had no counterpart and were dropped; "
              "check that both tasks finished before trusting a paired rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
