# coding: utf-8
"""The data-description table behind Table 2: what the OBR doses actually fix.

Table 2 reports repetition and quality per dose.  Its footing is the claim that
the doses differ in one thing only -- how many target blocks were replaced --
while the input, the record count, the block count and the single list close /
EOS per record stay put.  This script writes the numbers that claim rests on,
for 0% (cleanv2) and every dose cut from the shipped 24.30% pairing:

    replacements, realized rho, records, blocks, blocks per record,
    assistant-token mean and its distance from cleanv2,
    and the block-level token movement the replacement carries.

Two different token measurements appear here on purpose:

* `assistant_tokens_mean` is the record-level training length, measured on the
  rendered conversation by `experiments/1_data/dataset_statistics.py`.
  It is the honest answer to "did the target get longer";
* `block_token_delta_*` is per replaced block, `len(replacement) - len(host)`
  in Qwen3 tokens, recorded when the 24.30% pairing was solved.  It says how
  closely each individual swap matched, which the record-level mean hides:
  the pairing balanced the signed record delta toward zero, so a small mean can
  sit on top of individually large swaps.

Neither is a claim that the token sequence is identical.  A one-for-one
replacement changes entities, relation and description, and only approximates
block length; the fixed quantities are the input, the record and block counts,
and the number of list terminators.

    python experiments/1_data/obr_dose_table.py \
        --pairs datasets/controlled/obr_pair_manifest.jsonl \
        --stats datasets/dataset_statistics/dataset_statistics.csv \
        --out-dir datasets/controlled
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (REPO_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from build_obr_dose import assign_order, load_manifest  # noqa: E402
from build_obr5 import token_delta_summary  # noqa: E402

#: (paper label, dataset_statistics name, dose).  0% is cleanv2 itself, which
#: is why it has no dose: OBR-0 was never built or trained as a separate arm.
CONDITIONS: tuple[tuple[str, str, float | None], ...] = (
    ("OBR 0% (cleanv2)", "cleanv2_train", None),
    ("OBR 5%", "train_obr_p5", 0.05),
    ("OBR 10%", "train_obr_p10", 0.10),
    ("OBR 15%", "train_obr_p15", 0.15),
    ("OBR 24.3% (appendix)", "train_obr", 0.243),
)

COLUMNS = (
    "condition", "dataset", "replacements", "realized_rho", "records",
    "records_touched", "blocks", "blocks_per_record_mean",
    "assistant_tokens_mean", "assistant_tokens_delta_vs_cleanv2",
    "chat_tokens_mean", "block_token_delta_signed_sum",
    "block_token_delta_mean", "block_token_delta_p05",
    "block_token_delta_p50", "block_token_delta_p95",
    "block_token_delta_within5_share",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--stats", required=True,
                        help="dataset_statistics.csv holding every arm's row")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stem", default="obr_dose_data_table")
    args = parser.parse_args()

    with Path(args.stats).open(encoding="utf-8", newline="") as handle:
        stats = {row["dataset"]: row for row in csv.DictReader(handle)}
    missing = [name for _, name, _ in CONDITIONS if name not in stats]
    if missing:
        raise SystemExit(f"{args.stats} has no row for: {', '.join(missing)}")

    ordered = assign_order(load_manifest(Path(args.pairs)))
    blocks = int(stats["cleanv2_train"]["relation_dicts"])
    reference = float(stats["cleanv2_train"]["assistant_tokens_mean"])

    rows: list[dict[str, Any]] = []
    for label, name, dose in CONDITIONS:
        stat = stats[name]
        row: dict[str, Any] = {
            "condition": label, "dataset": name,
            "records": int(stat["records"]),
            "blocks": int(stat["relation_dicts"]),
            "blocks_per_record_mean": stat["dicts_per_record_mean"],
            "assistant_tokens_mean": stat["assistant_tokens_mean"],
            "assistant_tokens_delta_vs_cleanv2": round(
                float(stat["assistant_tokens_mean"]) - reference, 2),
            "chat_tokens_mean": stat["chat_tokens_mean"],
        }
        if dose is None:
            row.update({"replacements": 0, "realized_rho": 0.0,
                        "records_touched": 0})
            row.update({key: "" for key in COLUMNS
                        if key.startswith("block_token_delta")})
        else:
            chosen = ordered[:round(blocks * dose)]
            delta = token_delta_summary(chosen)
            row.update({
                "replacements": len(chosen),
                "realized_rho": round(len(chosen) / blocks, 6),
                "records_touched": len({pair.line for pair in chosen}),
                "block_token_delta_signed_sum": delta["signed_sum"],
                "block_token_delta_mean": delta["mean"],
                "block_token_delta_p05": delta["p05"],
                "block_token_delta_p50": delta["p50"],
                "block_token_delta_p95": delta["p95"],
                "block_token_delta_within5_share": delta["within_5_tokens_share"],
            })
        rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    header = ("| 条件 | 替换块数 | 实际比例 | 记录数 | 触及记录 | 目标块数 | "
              "块/记录 | assistant token 均值 | 相对 cleanv2 | "
              "块 token 差 中位数 | 块 token 差 p05–p95 | |Δ|≤5 占比 |")
    lines = [header, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        window = ("—" if row["block_token_delta_p05"] == ""
                  else f"{row['block_token_delta_p05']}–{row['block_token_delta_p95']}")
        lines.append(
            f"| {row['condition']} | {row['replacements']} | "
            f"{row['realized_rho']:.4%} | {row['records']} | "
            f"{row['records_touched']} | {row['blocks']} | "
            f"{row['blocks_per_record_mean']} | {row['assistant_tokens_mean']} | "
            f"{row['assistant_tokens_delta_vs_cleanv2']:+.2f} | "
            f"{row['block_token_delta_p50'] if row['block_token_delta_p50'] != '' else '—'} | "
            f"{window} | "
            f"{row['block_token_delta_within5_share'] if row['block_token_delta_within5_share'] != '' else '—'} |")
    note = (
        "\n固定量：原始文本、候选实体列表、记录数、每条记录的关系块数，以及每条"
        "回答一次的列表收尾与 EOS。替换会改变端点、关系与描述，块 token 长度只是"
        "近似匹配，不是逐 token 相同。5% ⊂ 10% ⊂ 15% ⊂ 24.3% 取自同一套分层嵌套"
        "顺序，剂量是唯一变化的量；24.3% 只在附录使用。\n")
    md_path = out_dir / f"{args.stem}.md"
    md_path.write_text("\n".join(lines) + "\n" + note, encoding="utf-8")

    print("\n".join(lines) + "\n" + note)
    print(f"CSV -> {csv_path}")
    print(f"MD  -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
