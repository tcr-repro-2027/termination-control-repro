# coding: utf-8
"""Cut nested lower-dose OBR arms out of the shipped 24.30% pairing.

Why subset instead of rebuilding
--------------------------------
`rebuild_obr.py` solved a constrained assignment: same cleanv2 record, same
early/middle/late bucket, exact A/AE and position profile, then minimal degree,
relative-position and token-length error.  Re-running that solver at a lower
dose would produce a *different* set of host blocks, and any difference between
OBR-15% and OBR-24.3% could then be a difference in matching quality rather
than in dose.  Reverting a subset of the pairs that solver already chose keeps
every one of its guarantees and makes dose the only thing that moves.

    C_10% subset of C_15% subset of C_24.30%

Nesting is not approximate here: the doses are prefixes of ONE ordering, so a
pair kept at 10% is kept at every higher dose, and a curve over
0 / 10 / 15 / 24.3 is a curve over one nested corruption family -- the same
discipline the plan already imposes on the ISC doses (§6.1).

How the ordering keeps the profile at every prefix
--------------------------------------------------
Pairs are grouped into strata (joint type x position bucket x endpoint role x
nesting), ordered inside a stratum by a stable hash, and given the fractional
position ``(i + 0.5) / n_s``.  Sorting globally on that fraction interleaves the
strata proportionally, so ANY prefix holds each stratum in its full-set share to
within one pair.  The §4.3 profile is therefore preserved at 15%, at 10%, and at
any other dose cut from the same order later, without re-running the solver.

What this script does NOT do
----------------------------
It cannot raise the dose above 24.30%, and it cannot change which cleanv2 host
blocks were chosen.  Both are properties of the shipped pairing.

    python scripts/build_obr_dose.py --check          # validate only
    python scripts/build_obr_dose.py                  # write 15% and 10%
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcr.data.controlled.common import compact_json, stable_hash  # noqa: E402

#: Salt for the within-stratum ordering.  Frozen: changing it reshuffles which
#: pairs survive at a given dose and silently breaks comparability with any
#: arm already trained from this script.
ORDER_SALT = "obr-dose-nested-v1"

#: Marginals the 24.30% build matched against the real filter->clean profile
#: (`obr_rebuild_report.json` -> `real_profile`).  Reported per dose so a
#: deviation is visible rather than assumed.
PROFILE_KEYS = ("joint_type", "host_bucket", "role",
                "source_nested", "target_nested")

BOOL_FIELDS = ("source_only", "target_only", "both_endpoints",
               "source_nested", "target_nested")


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def role_of(row: Mapping[str, Any]) -> str:
    """Which endpoint carries the conflict.  The three manifest flags do not
    partition the pairs (about 2% are neither), so `neither` is a real level."""
    if as_bool(row["both_endpoints"]):
        return "both"
    if as_bool(row["source_only"]):
        return "source"
    if as_bool(row["target_only"]):
        return "target"
    return "neither"


class Pair:
    __slots__ = ("line", "block_index", "rec_id", "stratum", "token_delta",
                 "order_key")

    def __init__(self, row: Mapping[str, Any]) -> None:
        self.line = int(row["line"])
        self.block_index = int(row["block_index"])
        self.rec_id = str(row["rec_id"])
        self.token_delta = int(row["token_delta"])
        self.stratum = (str(row["joint_type"]), str(row["host_bucket"]),
                        role_of(row), as_bool(row["source_nested"]),
                        as_bool(row["target_nested"]))
        self.order_key = 0.0

    @property
    def at(self) -> tuple[int, int]:
        return (self.line, self.block_index)


# ------------------------------------------------------------------ loading

def load_manifest(path: Path) -> list[Pair]:
    pairs: list[Pair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("condition")) != "obr":
                continue
            pairs.append(Pair(row))
    if not pairs:
        raise SystemExit(f"{path} holds no OBR pairs")
    seen = Counter(pair.at for pair in pairs)
    duplicates = [at for at, n in seen.items() if n > 1]
    if duplicates:
        raise SystemExit(f"manifest has {len(duplicates)} duplicated "
                         f"(line, block_index), e.g. {duplicates[:3]}")
    return pairs


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def blocks_of(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = record.get("output", [])
    if isinstance(out, str):
        out = json.loads(out)
    return [dict(block) for block in out if isinstance(block, dict)]


def validate(base_path: Path, obr_path: Path,
             pairs: Sequence[Pair]) -> dict[str, Any]:
    """The manifest must describe EXACTLY the shipped 24.30% file.

    If it does not, every dose cut from it is fiction, so this is a hard stop
    rather than a warning."""
    expected: dict[int, set[int]] = defaultdict(set)
    for pair in pairs:
        expected[pair.line].add(pair.block_index)

    n_records = n_blocks = 0
    observed_total = 0
    mismatch: list[str] = []
    for line, (base, obr) in enumerate(zip(iter_records(base_path),
                                           iter_records(obr_path))):
        n_records += 1
        if base.get("text") != obr.get("text") or \
                base.get("entities_str") != obr.get("entities_str"):
            mismatch.append(f"line {line}: OBR changed the INPUT; it must only "
                            "replace target blocks")
            continue
        left, right = blocks_of(base), blocks_of(obr)
        if len(left) != len(right):
            mismatch.append(f"line {line}: block count {len(left)} != {len(right)}")
            continue
        n_blocks += len(left)
        diff = {i for i, (a, b) in enumerate(zip(left, right)) if a != b}
        observed_total += len(diff)
        if diff != expected.get(line, set()):
            only_data = sorted(diff - expected.get(line, set()))[:3]
            only_man = sorted(expected.get(line, set()) - diff)[:3]
            mismatch.append(f"line {line}: manifest/data disagree "
                            f"(data-only={only_data}, manifest-only={only_man})")
        if len(mismatch) >= 5:
            break
    if mismatch:
        raise SystemExit("manifest does not describe train_obr.jsonl:\n  "
                         + "\n  ".join(mismatch))
    return {"records": n_records, "blocks": n_blocks,
            "replaced_blocks": observed_total,
            "realized_rho": round(observed_total / n_blocks, 6)}


# ------------------------------------------------------- nested dose ordering

def assign_order(pairs: Sequence[Pair]) -> list[Pair]:
    """One ordering whose every prefix is a stratified sample of the whole."""
    by_stratum: dict[tuple, list[Pair]] = defaultdict(list)
    for pair in pairs:
        by_stratum[pair.stratum].append(pair)
    for stratum, members in by_stratum.items():
        members.sort(key=lambda p: (stable_hash(ORDER_SALT, p.rec_id,
                                                p.block_index), p.at))
        n = len(members)
        for index, pair in enumerate(members):
            pair.order_key = (index + 0.5) / n
    return sorted(pairs, key=lambda p: (p.order_key, p.stratum, p.at))


def profile(pairs: Sequence[Pair]) -> dict[str, dict[str, float]]:
    total = max(len(pairs), 1)
    out: dict[str, dict[str, float]] = {}
    for position, key in enumerate(PROFILE_KEYS):
        counts = Counter(str(pair.stratum[position]) for pair in pairs)
        out[key] = {name: round(value / total, 4)
                    for name, value in sorted(counts.items())}
    return out


def deviation(observed: Mapping[str, dict[str, float]],
              reference: Mapping[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {key: {name: round(value - reference.get(key, {}).get(name, 0.0), 4)
                  for name, value in sorted(values.items())}
            for key, values in observed.items()}


# ------------------------------------------------------------------ writing

def write_dose(base_path: Path, obr_path: Path, keep: set[tuple[int, int]],
               out_path: Path) -> int:
    """Start from the OBR record and revert every pair that is not kept."""
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_suffix(out_path.suffix + ".partial")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for line, (base, obr) in enumerate(zip(iter_records(base_path),
                                               iter_records(obr_path))):
            left, right = blocks_of(base), blocks_of(obr)
            blocks = [right[i] if (line, i) in keep else left[i]
                      for i in range(len(right))]
            record = dict(obr)
            record["output"] = blocks
            handle.write(compact_json(record) + "\n")
            written += 1
    temp.replace(out_path)
    return written


def main() -> int:
    repo = REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=str(repo))
    parser.add_argument("--manifest", default=None,
                        help="default: <repo>/datasets/controlled/obr_pair_manifest.jsonl")
    parser.add_argument("--doses", default="0.15,0.10",
                        help="Comma-separated edge-level conflict rates.")
    parser.add_argument("--out-dir", default=None,
                        help="default: <repo>/datasets/controlled")
    parser.add_argument("--report", default=None,
                        help="default: <repo>/datasets/build_reports/obr_dose_report.json")
    parser.add_argument("--check", action="store_true",
                        help="Validate and report only; write nothing.")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest) if args.manifest else \
        repo / "datasets/controlled/obr_pair_manifest.jsonl"
    base_path = repo / "datasets/cleanv2/train_supportclean_keep8.jsonl"
    obr_path = repo / "datasets/controlled/train_obr.jsonl"
    out_dir = Path(args.out_dir) if args.out_dir else repo / "datasets/controlled"
    report_path = Path(args.report) if args.report else \
        repo / "datasets/build_reports/obr_dose_report.json"

    for path in (manifest_path, base_path, obr_path):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    print(f"[dose] manifest {manifest_path}", flush=True)
    pairs = load_manifest(manifest_path)
    print(f"[dose] {len(pairs)} OBR pairs; validating against the shipped data "
          "(streams both 240 MB files) ...", flush=True)
    facts = validate(base_path, obr_path, pairs)
    if facts["replaced_blocks"] != len(pairs):
        raise SystemExit(f"manifest has {len(pairs)} pairs but the data shows "
                         f"{facts['replaced_blocks']} replaced blocks")
    print(f"[dose] validated: {facts['records']} records, {facts['blocks']} blocks, "
          f"rho={facts['realized_rho']:.6f}", flush=True)

    ordered = assign_order(pairs)
    full_profile = profile(ordered)
    full_delta = sum(pair.token_delta for pair in ordered)

    doses = [float(value) for value in args.doses.split(",") if value.strip()]
    report: dict[str, Any] = {
        "version": "e0-obr-dose-1.0",
        "order_salt": ORDER_SALT,
        "source": {"manifest": str(manifest_path), "obr": str(obr_path),
                   "base": str(base_path), **facts},
        "full_profile": full_profile,
        "full_signed_token_delta": full_delta,
        "nesting": "prefixes of one stratified interleaved order",
        "doses": {},
    }

    previous: set[tuple[int, int]] | None = None
    previous_label = None
    for dose in sorted(doses, reverse=True):
        target = round(facts["blocks"] * dose)
        if target > len(ordered):
            raise SystemExit(f"dose {dose:.4f} needs {target} pairs but only "
                             f"{len(ordered)} exist (24.30% is the ceiling)")
        chosen = ordered[:target]
        keep = {pair.at for pair in chosen}
        label = f"p{int(round(dose * 100))}"

        if previous is not None and not keep.issubset(previous):
            raise SystemExit(f"nesting broken: {label} is not a subset of "
                             f"{previous_label}")

        observed = profile(chosen)
        delta = sum(pair.token_delta for pair in chosen)
        entry = {
            "requested_rho": dose,
            "pairs": len(chosen),
            "realized_rho": round(len(chosen) / facts["blocks"], 6),
            "records_touched": len({pair.line for pair in chosen}),
            "profile": observed,
            "profile_deviation_vs_full": deviation(observed, full_profile),
            "signed_token_delta": delta,
            "signed_token_delta_share_of_full": round(delta / full_delta, 4)
            if full_delta else None,
            "nested_in": previous_label,
        }
        worst = max((abs(v) for values in entry["profile_deviation_vs_full"].values()
                     for v in values.values()), default=0.0)
        entry["max_profile_deviation"] = worst
        report["doses"][label] = entry

        print(f"[dose] {label}: {len(chosen)} pairs, rho={entry['realized_rho']:.6f}, "
              f"records={entry['records_touched']}, "
              f"max profile deviation={worst:.4f}", flush=True)

        if not args.check:
            out_path = out_dir / f"train_obr_{label}.jsonl"
            rows = write_dose(base_path, obr_path, keep, out_path)
            entry["output"] = str(out_path)
            entry["rows"] = rows
            print(f"[dose] {label}: wrote {rows} rows -> {out_path}", flush=True)

        previous, previous_label = keep, label

    if not args.check:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"[dose] report -> {report_path}", flush=True)
    else:
        print(json.dumps(report["doses"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
