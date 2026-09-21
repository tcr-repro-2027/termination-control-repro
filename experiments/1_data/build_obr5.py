# coding: utf-8
"""Cut the OBR-5% arm from the released 24.30% pairing.

The stratified ordering in ``build_obr_dose.py`` yields nested subsets:
5% within 10% within 15% within 24.30%. The 10% and 15% prefixes are
recomputed and compared against the supplied training files before writing
the 5% arm. Inputs and output locations are explicit command-line arguments.

Outputs are ``train_obr_p5.jsonl``, its SWIFT chat representation, and a
report of the realized dose and token-length differences. SWIFT rendering
uses the shared prompt and ``json.dumps(output, ensure_ascii=False)``.
The Qwen3 chat template supplies the empty non-thinking block.

Usage from the repository root::

    python experiments/1_data/build_obr5.py \
        --reference-data datasets/cleanv2/train_supportclean_keep8.jsonl \
        --obr-data datasets/controlled/train_obr.jsonl \
        --pairs datasets/controlled/obr_pair_manifest.jsonl \
        --out datasets/controlled/train_obr_p5.jsonl \
        --swift-out datasets/controlled/swift_train_obr_p5.jsonl \
        --trained-arm p15=datasets/controlled/train_obr_p15.jsonl \
        --trained-arm p10=datasets/controlled/train_obr_p10.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (REPO_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from build_obr_dose import (  # noqa: E402
    ORDER_SALT, Pair, assign_order, blocks_of, deviation, iter_records,
    load_manifest, profile, write_dose,
)
from tcr.prompt_template import (  # noqa: E402
    PROMPT_TEMPLATE_IS_TRUNCATED, build_extraction_relation_prompt,
)

#: Doses whose prefixes must contain the new one.  These are the two arms that
#: were already trained, so they are what "nested" has to mean here.
TRAINED_DOSES = (("p15", 0.15), ("p10", 0.10))


def log(message: str) -> None:
    print(f"[obr5 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def serialize_output(output: Any) -> str:
    """The training target string; identical to build_training_formats.py."""
    return json.dumps(output, ensure_ascii=False)


def replaced_positions(base_path: Path, arm_path: Path, label: str
                       ) -> tuple[set[tuple[int, int]], int, int]:
    """Every `(line, block_index)` where `arm` differs from the cleanv2 base.

    Also the record and block totals, so the realized dose has a denominator
    that came from the data rather than from a remembered number.  A row whose
    input or block count moved is a hard stop: OBR replaces target blocks and
    nothing else, and a dose cut from a file that does something else would be
    describing the wrong experiment.
    """
    positions: set[tuple[int, int]] = set()
    records = blocks = 0
    for line, (base, arm) in enumerate(zip(iter_records(base_path),
                                           iter_records(arm_path))):
        records += 1
        if base.get("text") != arm.get("text") or \
                base.get("entities_str") != arm.get("entities_str"):
            raise SystemExit(f"{label} line {line}: the INPUT differs from the "
                             "reference; OBR must only replace target blocks")
        left, right = blocks_of(base), blocks_of(arm)
        if len(left) != len(right):
            raise SystemExit(f"{label} line {line}: block count "
                             f"{len(left)} != {len(right)}")
        blocks += len(left)
        positions.update((line, index) for index, (a, b)
                         in enumerate(zip(left, right)) if a != b)
        if records % 2000 == 0:
            log(f"  {label}: {records} records")
    return positions, records, blocks


def prefix(ordered: Sequence[Pair], blocks: int, dose: float
           ) -> tuple[list[Pair], set[tuple[int, int]]]:
    target = round(blocks * dose)
    if target > len(ordered):
        raise SystemExit(f"dose {dose:.4f} needs {target} pairs but the "
                         f"pairing holds {len(ordered)} (24.30% is the ceiling)")
    chosen = list(ordered[:target])
    return chosen, {pair.at for pair in chosen}


def token_delta_summary(pairs: Sequence[Pair]) -> dict[str, Any]:
    """Block-level token movement of the replacement, from the pairing itself.

    `token_delta` is `len(replacement) - len(host)` in Qwen3 tokens, measured
    per block when the 24.30% pairing was solved.  Summing it is exact for the
    blocks and a close approximation for the record, which is why the
    record-level assistant length is measured separately by
    `dataset_statistics.py` rather than inferred from this table.
    """
    deltas = sorted(pair.token_delta for pair in pairs)
    n = max(len(deltas), 1)

    def at(fraction: float) -> int:
        return deltas[min(int(fraction * (len(deltas) - 1)), len(deltas) - 1)] \
            if deltas else 0

    return {
        "pairs": len(deltas),
        "signed_sum": sum(deltas),
        "absolute_sum": sum(abs(value) for value in deltas),
        "mean": round(sum(deltas) / n, 4),
        "p05": at(0.05), "p50": at(0.50), "p95": at(0.95),
        "min": deltas[0] if deltas else 0,
        "max": deltas[-1] if deltas else 0,
        "within_5_tokens_share": round(
            sum(abs(value) <= 5 for value in deltas) / n, 4),
    }


def write_swift(raw_path: Path, swift_path: Path) -> tuple[int, int]:
    """Render the record form into the ms-swift training form."""
    swift_path.parent.mkdir(parents=True, exist_ok=True)
    temp = swift_path.with_suffix(swift_path.suffix + ".partial")
    rows = blocks = 0
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in iter_records(raw_path):
            prompt = build_extraction_relation_prompt(
                text=str(record.get("text", "")),
                entities_str=str(record.get("entities_str", "")))
            output = record.get("output", [])
            blocks += len(output) if isinstance(output, list) else 0
            handle.write(json.dumps(
                {"messages": [{"role": "user", "content": prompt},
                              {"role": "assistant",
                               "content": serialize_output(output)}]},
                ensure_ascii=False) + "\n")
            rows += 1
            if rows % 2000 == 0:
                log(f"  swift: {rows} rows")
    temp.replace(swift_path)
    return rows, blocks


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-data", required=True,
                        help="cleanv2 record file the dose is measured against")
    parser.add_argument("--obr-data", required=True,
                        help="the shipped 24.30%% OBR record file")
    parser.add_argument("--pairs", required=True,
                        help="obr_pair_manifest.jsonl from the 24.30%% build")
    parser.add_argument("--out", required=True,
                        help="output record file, e.g. train_obr_p5.jsonl")
    parser.add_argument("--swift-out", default=None,
                        help="output ms-swift file; omit to write records only")
    parser.add_argument("--report", default=None,
                        help="default: <out dir>/obr_p5_build_report.json")
    parser.add_argument("--dose", type=float, default=0.05)
    parser.add_argument("--trained-arm", action="append", default=[],
                        metavar="LABEL=PATH",
                        help="an already-trained lower-dose arm to confirm the "
                             "new one nests inside, e.g. p10=<path>. Repeatable.")
    args = parser.parse_args()

    if PROMPT_TEMPLATE_IS_TRUNCATED:
        raise SystemExit("prompt_template.py is the abridged copy; the Swift "
                         "prompts would not match the evaluation pipeline")

    base_path = Path(args.reference_data).resolve()
    obr_path = Path(args.obr_data).resolve()
    manifest_path = Path(args.pairs).resolve()
    out_path = Path(args.out).resolve()
    swift_path = Path(args.swift_out).resolve() if args.swift_out else None
    report_path = (Path(args.report).resolve() if args.report
                   else out_path.parent / f"obr_p{int(round(args.dose * 100))}"
                        "_build_report.json")
    arms: dict[str, Path] = {}
    for entry in args.trained_arm:
        label, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--trained-arm wants LABEL=PATH, got {entry!r}")
        arms[label.strip()] = Path(path).resolve()

    for path in (base_path, obr_path, manifest_path, *arms.values()):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    log(f"manifest {manifest_path}")
    pairs = load_manifest(manifest_path)
    log(f"{len(pairs)} OBR pairs; reading the shipped 24.30% arm")
    full_positions, records, blocks = replaced_positions(
        base_path, obr_path, "obr")
    expected = {pair.at for pair in pairs}
    if full_positions != expected:
        raise SystemExit(
            f"the manifest does not describe {obr_path.name}: "
            f"{len(full_positions - expected)} replaced blocks are absent from "
            f"the manifest, {len(expected - full_positions)} manifest pairs are "
            "not replaced in the data")
    log(f"validated: {records} records, {blocks} blocks, "
        f"rho={len(pairs) / blocks:.6f}")

    ordered = assign_order(pairs)
    full_profile = profile(ordered)

    # The prefixes of the two arms that already have E1 results.  Re-deriving
    # them is what turns "5% is nested" into something checked against the data
    # the models were actually trained on.
    nesting: dict[str, Any] = {}
    outer: set[tuple[int, int]] | None = None
    outer_label = "obr (24.30%)"
    for label, dose in TRAINED_DOSES:
        chosen, keep = prefix(ordered, blocks, dose)
        entry = {"pairs": len(chosen),
                 "realized_rho": round(len(chosen) / blocks, 6),
                 "checked_against_file": None}
        if outer is not None and not keep.issubset(outer):
            raise SystemExit(f"nesting broken: {label} is not inside {outer_label}")
        arm_path = arms.get(label)
        if arm_path is not None:
            log(f"re-deriving {label} and comparing against {arm_path.name}")
            arm_positions, arm_records, arm_blocks = replaced_positions(
                base_path, arm_path, label)
            if (arm_records, arm_blocks) != (records, blocks):
                raise SystemExit(f"{label}: {arm_records} records / {arm_blocks} "
                                 f"blocks, expected {records} / {blocks}")
            if arm_positions != keep:
                raise SystemExit(
                    f"{label}: the ordering re-derived {len(keep)} positions but "
                    f"{arm_path.name} replaces {len(arm_positions)}; "
                    f"{len(arm_positions ^ keep)} disagree. The new dose cannot "
                    "be claimed to nest inside the trained arm.")
            entry["checked_against_file"] = str(arm_path)
            log(f"  {label}: {len(keep)} positions, identical to the trained arm")
        nesting[label] = entry
        outer, outer_label = keep, label

    chosen, keep = prefix(ordered, blocks, args.dose)
    label = f"p{int(round(args.dose * 100))}"
    if outer is not None and not keep.issubset(outer):
        raise SystemExit(f"nesting broken: {label} is not inside {outer_label}")
    observed = profile(chosen)
    worst = max((abs(value) for values in deviation(observed, full_profile).values()
                 for value in values.values()), default=0.0)
    log(f"{label}: {len(chosen)} pairs, rho={len(chosen) / blocks:.6f}, "
        f"records touched={len({pair.line for pair in chosen})}, "
        f"nested in {outer_label}, max profile deviation={worst:.4f}")

    rows = write_dose(base_path, obr_path, keep, out_path)
    log(f"{label}: wrote {rows} rows -> {out_path}")
    swift_rows = swift_blocks = None
    if swift_path is not None:
        swift_rows, swift_blocks = write_swift(out_path, swift_path)
        log(f"{label}: wrote {swift_rows} rows, {swift_blocks} blocks "
            f"-> {swift_path}")

    report = {
        "version": "paper-closeout-obr5-1.0",
        "order_salt": ORDER_SALT,
        "source": {
            "reference": str(base_path), "obr": str(obr_path),
            "manifest": str(manifest_path), "manifest_pairs": len(pairs),
            "records": records, "blocks": blocks,
            "obr_realized_rho": round(len(pairs) / blocks, 6),
        },
        "dose": {
            "label": label, "requested_rho": args.dose,
            "pairs": len(chosen), "realized_rho": round(len(chosen) / blocks, 6),
            "records_touched": len({pair.line for pair in chosen}),
            "records": rows, "blocks": blocks,
            "blocks_per_record_mean": round(blocks / max(rows, 1), 4),
        },
        "nesting": {
            "rule": "prefixes of one stratified interleaved order",
            "inside": nesting,
        },
        "profile": observed,
        "profile_deviation_vs_full": deviation(observed, full_profile),
        "max_profile_deviation": worst,
        "full_profile": full_profile,
        "block_token_delta": token_delta_summary(chosen),
        "outputs": {
            "records": str(out_path), "rows": rows,
            "swift": str(swift_path) if swift_path else None,
            "swift_rows": swift_rows, "swift_blocks": swift_blocks,
        },
        "note": "Record-level assistant token length is measured separately by "
                "experiments/1_data/dataset_statistics.py; the table "
                "here is the block-level movement carried by the pairing.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    log(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
