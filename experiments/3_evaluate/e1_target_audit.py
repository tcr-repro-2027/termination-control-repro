# coding=utf-8
"""E1's target-length / history-gap audit of the TRAINING stage datasets.

§E1 asks for one thing the evaluation itself cannot answer: how much of the
`filter -> clean` (and neighbouring) contrast is a change in what the target
*is*, rather than in what the model does.  So, per record and per stage:

* target block count;
* assistant target token count (the actual SFT loss length);
* "bad" blocks -- blocks whose endpoints are not licensed by the input, using
  the SAME two support axes as `tcr.evaluation.quality`: both endpoints in the
  candidate list (admissibility) and both endpoints in the document
  (evidence);
* the terminal-quartile bad-block ratio, i.e. whether the unlicensed material
  sits where termination has to happen.

Records are paired across stages by `rec_id` from the `*_rowmap.jsonl` files,
so a paired difference is a difference on the same source record and not a
difference between two record populations.  Differences get a record-clustered
bootstrap CI and a quantile profile.

This describes covariation between support cleaning and target length.  It is
NOT a causal estimate of a length channel, and §E1 says so explicitly; the
numbers exist to size the effects E3/E5 later have to account for.

    python scripts/e1_target_audit.py \
        --datasets_root ./datasets \
        --out_dir       ./outputs/eval/target_audit \
        --tokenizer     ./models/Qwen3/Qwen3-4B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcr.data.layout import stage_directory                           # noqa: E402
from tcr.evaluation.io_utils import read_jsonl, write_csv, write_json   # noqa: E402
from tcr.evaluation.quality import candidate_entities, norm_field       # noqa: E402

#: stage -> the dataset file stem it ships under.
STAGE_FILES = {
    "base": "train_base.jsonl",
    "dedup": "train_dedup.jsonl",
    "filter": "train_filter.jsonl",
    "clean": "train_clean.jsonl",
    "cleanv2": "train_supportclean_keep8.jsonl",
}
#: §E1 主比较, plus base->dedup for completeness.
PAIRS = (("base", "dedup"), ("dedup", "filter"), ("filter", "clean"),
         ("clean", "cleanv2"))
METRICS = ("n_blocks", "assistant_tokens", "bad_blocks", "bad_block_ratio",
           "terminal_quartile_bad_blocks", "terminal_quartile_bad_ratio")


def rowmap_ids(stage_dir: Path, stem: str) -> List[str]:
    """`rec_id` per line, so records can be paired across stages."""
    path = stage_dir / f"{stem}_rowmap.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; without it stages cannot be paired by record")
    return [str(row["rec_id"]) for row in read_jsonl(path)]


def audit_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Block/support profile of one training record's TARGET."""
    output = record.get("output", [])
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            output = []
    blocks = [item for item in output if isinstance(item, dict)]
    candidates = {norm_field(name)
                  for name in candidate_entities(record.get("entities_str", ""))}
    norm_text = norm_field(record.get("text", "") or "")

    flags: List[bool] = []
    for block in blocks:
        source = norm_field(block.get("source", ""))
        target = norm_field(block.get("target", ""))
        admissible = source in candidates and target in candidates
        evidenced = (bool(source) and bool(target)
                     and source in norm_text and target in norm_text)
        flags.append(not (admissible and evidenced))

    n = len(flags)
    # Terminal quartile = the last ceil(n/4) blocks; that is where a model has
    # to decide to stop, so unlicensed material there is the interesting kind.
    tail_start = n - -(-n // 4) if n else 0
    tail = flags[tail_start:]
    return {
        "n_blocks": n,
        "bad_blocks": sum(flags),
        "bad_block_ratio": (sum(flags) / n) if n else None,
        "terminal_quartile_blocks": len(tail),
        "terminal_quartile_bad_blocks": sum(tail),
        "terminal_quartile_bad_ratio": (sum(tail) / len(tail)) if tail else None,
        "target_chars": len(json.dumps(output, ensure_ascii=False)),
    }


def token_counts(records: Iterable[str], tokenizer) -> List[int]:
    counts: List[int] = []
    batch: List[str] = []
    for text in records:
        batch.append(text)
        if len(batch) >= 64:
            counts += [len(ids) for ids in
                       tokenizer(batch, add_special_tokens=False)["input_ids"]]
            batch = []
    if batch:
        counts += [len(ids) for ids in
                   tokenizer(batch, add_special_tokens=False)["input_ids"]]
    return counts


def quantiles(values: Sequence[float]) -> Dict[str, float | None]:
    numeric = np.array([value for value in values if value is not None],
                       dtype=float)
    if numeric.size == 0:
        return {name: None for name in
                ("mean", "p05", "p25", "median", "p75", "p95", "max")}
    return {
        "mean": float(numeric.mean()),
        "p05": float(np.percentile(numeric, 5)),
        "p25": float(np.percentile(numeric, 25)),
        "median": float(np.percentile(numeric, 50)),
        "p75": float(np.percentile(numeric, 75)),
        "p95": float(np.percentile(numeric, 95)),
        "max": float(numeric.max()),
    }


def bootstrap_mean(values: Sequence[float], *, n_boot: int, seed: int
                   ) -> tuple[float, float, float]:
    numeric = np.array([value for value in values if value is not None],
                       dtype=float)
    if numeric.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, numeric.size, size=(n_boot, numeric.size))
    means = numeric[draws].mean(axis=1)
    return (float(numeric.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets_root", required=True,
                        help="Directory containing stages/<stage>/...")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tokenizer", default=None,
                        help="Qwen3 tokenizer dir; omit to skip token counts "
                             "(all Qwen3 sizes share one vocabulary, so any "
                             "of them gives the same numbers).")
    parser.add_argument("--stages", default=",".join(STAGE_FILES))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    root = Path(args.datasets_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = [name.strip() for name in args.stages.split(",") if name.strip()]

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    per_record: Dict[str, Dict[str, Dict[str, Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for stage in stages:
        stem = STAGE_FILES[stage][: -len(".jsonl")]
        stage_dir = stage_directory(root, stage)
        path = stage_dir / STAGE_FILES[stage]
        if not path.is_file():
            print(f"[skip] {stage}: {path} not found")
            continue
        ids = rowmap_ids(stage_dir, stem)
        print(f"[{stage}] auditing {path.name} ...", flush=True)

        profiles: List[Dict[str, Any]] = []
        targets: List[str] = []
        for record in read_jsonl(path):
            profiles.append(audit_record(record))
            if tokenizer is not None:
                output = record.get("output", [])
                targets.append(json.dumps(output, ensure_ascii=False)
                               if not isinstance(output, str) else output)
        if len(profiles) != len(ids):
            raise ValueError(f"{stage}: {len(profiles)} rows but "
                             f"{len(ids)} rowmap entries")
        if tokenizer is not None:
            for profile, count in zip(profiles, token_counts(targets, tokenizer)):
                profile["assistant_tokens"] = count
        else:
            for profile in profiles:
                profile["assistant_tokens"] = None

        for rec_id, profile in zip(ids, profiles):
            per_record.setdefault(rec_id, {})[stage] = profile

        row: Dict[str, Any] = {"stage": stage, "n_records": len(profiles)}
        for metric in METRICS:
            for name, value in quantiles([p[metric] for p in profiles]).items():
                row[f"{metric}_{name}"] = value
        row["total_blocks"] = sum(p["n_blocks"] for p in profiles)
        row["total_bad_blocks"] = sum(p["bad_blocks"] for p in profiles)
        summary_rows.append(row)

    write_csv(out_dir / "target_audit_summary.csv", summary_rows)
    print(f"wrote {out_dir / 'target_audit_summary.csv'}")

    pair_rows: List[Dict[str, Any]] = []
    for before, after in PAIRS:
        shared = [rec_id for rec_id, stages_of in per_record.items()
                  if before in stages_of and after in stages_of]
        if not shared:
            continue
        shared.sort()
        entry: Dict[str, Any] = {"comparison": f"{before}->{after}",
                                 "n_paired_records": len(shared)}
        for index, metric in enumerate(METRICS):
            deltas = []
            for rec_id in shared:
                low = per_record[rec_id][before][metric]
                high = per_record[rec_id][after][metric]
                deltas.append(None if low is None or high is None else high - low)
            mean, ci_low, ci_high = bootstrap_mean(
                deltas, n_boot=args.bootstrap, seed=args.seed + index)
            entry[f"{metric}_delta_mean"] = mean
            entry[f"{metric}_delta_ci_low"] = ci_low
            entry[f"{metric}_delta_ci_high"] = ci_high
            for name, value in quantiles(deltas).items():
                entry[f"{metric}_delta_{name}"] = value
        pair_rows.append(entry)
    write_csv(out_dir / "target_audit_pairs.csv", pair_rows)
    print(f"wrote {out_dir / 'target_audit_pairs.csv'}")

    write_json(out_dir / "target_audit_manifest.json", {
        "datasets_root": str(root), "stages": stages,
        "tokenizer": args.tokenizer, "bootstrap": args.bootstrap,
        "seed": args.seed, "n_records_seen": len(per_record),
        "support_definition": "admissibility = both endpoints in the candidate "
                              "list; evidence = both endpoints in the document; "
                              "bad = fails either.  Same normalisation as "
                              "tcr.evaluation.quality (strip, lowercase, collapse "
                              "whitespace).",
        "terminal_quartile": "the last ceil(n_blocks/4) target blocks",
        "interpretation": "descriptive covariation of support cleaning with "
                          "target length; NOT a causal length-channel estimate "
                          "(§E1).",
    })
    print(f"wrote {out_dir / 'target_audit_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
