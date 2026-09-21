# coding: utf-8
"""Near-duplicate scan over the frozen cleanv2 documents.

The E0 gate asks for "近重复阈值样本全部复核" — exact duplicates alone are not
enough, because two documents that differ by a header line would still leak
evaluation content into training.

Method: deterministic mod-sampled character shingles (a standard unbiased
sketch), an inverted index to generate candidate pairs, then exact scores on
the full shingle sets for every candidate that clears the screen.  No
approximate score is reported: the screen only decides what gets measured.

Two scores, because they answer different questions:

  jaccard      |A n B| / |A u B| -- symmetric similarity, deflated when one
               document is much longer than the other.
  containment  |A n B| / |B|     -- how much of B already appears in A.  For a
               train/eval pair this is the leakage measure that matters: an
               evaluation document 90% contained in a training document is
               memorisable no matter how much extra text that training
               document carries.  The gate runs on containment of the *eval*
               side.

    python scripts/near_duplicate_scan.py --source raw    # before build_stages
    python scripts/near_duplicate_scan.py                 # after, to verify zero
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher
from zlib import crc32

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcr.data.layout import stage_directory  # noqa: E402
from tcr.data.stage_build.stages import STAGE_FILENAMES, norm, read_jsonl  # noqa: E402

SHINGLE = 8
# Keep 1/SAMPLE_MOD of the shingle space.  Sampling by hash value keeps the
# estimate unbiased and is reproducible across runs.
SAMPLE_MOD = 32
# A shingle shared by this many documents is boilerplate (headers, licence
# blocks); it only inflates the candidate list.
MAX_POSTINGS = 300
REPORT_THRESHOLD = 0.30
GATE_THRESHOLD = 0.80
SAMPLE_LIMIT = 200


def shingles(text: str) -> set[int]:
    if len(text) < SHINGLE:
        return {crc32(text.encode("utf-8"))} if text else set()
    return {value for value in
            (crc32(text[i:i + SHINGLE].encode("utf-8"))
             for i in range(len(text) - SHINGLE + 1))}


def sampled(values: set[int]) -> set[int]:
    return {value for value in values if value % SAMPLE_MOD == 0}


def jaccard(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def containment(whole: set[int], part: set[int]) -> float:
    """Fraction of `part` that also occurs in `whole`."""
    if not part:
        return 0.0
    return len(whole & part) / len(part)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--data-out", default=None, help="Dataset root containing cleanv2/")
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--report-threshold", type=float, default=REPORT_THRESHOLD)
    parser.add_argument("--gate-threshold", type=float, default=GATE_THRESHOLD)
    parser.add_argument("--source", choices=("stages", "raw"), default="stages",
                        help="'raw' scans train_orig/eval_orig and writes the removal "
                             "list build_stages.py consumes; 'stages' verifies the "
                             "frozen cleanv2 files")
    args = parser.parse_args()
    report_threshold = args.report_threshold
    gate_threshold = args.gate_threshold

    repo_root = Path(args.repo_root).resolve()
    datasets = repo_root / "datasets"
    data_out = Path(args.data_out) if args.data_out else datasets
    report_out = Path(args.report_out) if args.report_out else datasets / "build_reports"
    sample_out = report_out / "audit_samples"
    sample_out.mkdir(parents=True, exist_ok=True)

    # Only the normalised texts and the 1/SAMPLE_MOD sketches stay resident;
    # full shingle sets for ~10k documents would be several GB, and they are
    # only needed for the handful of pairs that clear the screen.
    docs: list[dict[str, Any]] = []
    texts: list[str] = []
    sketches: list[set[int]] = []
    for split in ("train", "eval"):
        if args.source == "raw":
            path = datasets / f"{split}_orig.jsonl"
            # `rec_id` must match `load_base`, which numbers by file position.
            ids = None
        else:
            path = stage_directory(data_out, "cleanv2") / STAGE_FILENAMES["cleanv2"].format(split=split)
            ids = [str(row["rec_id"]) for row in
                   read_jsonl(path.with_name(path.stem + "_rowmap.jsonl"))]
        count = 0
        for line, record in enumerate(read_jsonl(path)):
            text = norm(str(record.get("text", "")))
            sketch = sampled(shingles(text))
            rec_id = ids[line] if ids is not None else f"{split}:row_{line + 1:06d}"
            docs.append({"rec_id": rec_id, "split": split, "chars": len(text)})
            texts.append(text)
            sketches.append(sketch)
            count += 1
        print(f"[nd] {split}: {count} documents from {path.name}",
              file=sys.stderr, flush=True)

    def full_shingles(doc_index: int) -> set[int]:
        return shingles(texts[doc_index])

    index: dict[int, list[int]] = defaultdict(list)
    for doc_index, sketch in enumerate(sketches):
        for value in sketch:
            index[value].append(doc_index)
    dropped = sum(1 for postings in index.values() if len(postings) > MAX_POSTINGS)
    print(f"[nd] shingle index: {len(index)} keys, {dropped} boilerplate keys dropped",
          file=sys.stderr, flush=True)

    shared: dict[tuple[int, int], int] = defaultdict(int)
    for postings in index.values():
        if len(postings) > MAX_POSTINGS:
            continue
        for i in range(len(postings)):
            for j in range(i + 1, len(postings)):
                shared[(postings[i], postings[j])] += 1
    print(f"[nd] candidate pairs: {len(shared)}", file=sys.stderr, flush=True)

    pairs: list[dict[str, Any]] = []
    screened = 0
    for (left, right), overlap in shared.items():
        # Screen on the sketch: a pair cannot reach the report threshold unless
        # its sampled overlap already approaches it.
        smaller = min(len(sketches[left]), len(sketches[right]))
        if smaller and overlap / smaller < report_threshold * 0.8:
            continue
        screened += 1
        left_set, right_set = full_shingles(left), full_shingles(right)
        score = jaccard(left_set, right_set)
        smaller_containment = max(containment(left_set, right_set),
                                  containment(right_set, left_set))
        if max(score, smaller_containment) < report_threshold:
            continue
        a, b = docs[left], docs[right]
        kind = (f"{a['split']}-{b['split']}" if a["split"] <= b["split"]
                else f"{b['split']}-{a['split']}")
        cross = kind == "eval-train"
        # For a cross-split pair, score the evaluation document specifically:
        # the question is how much of *it* the training set already contains.
        eval_side, train_side = ((left, right) if a["split"] == "eval" else (right, left))
        eval_containment = (containment(full_shingles(train_side),
                                        full_shingles(eval_side)) if cross else None)
        # Jaccard and containment are both over shingle *sets*, so they ignore
        # order and repetition; character similarity does not.
        ratio = (SequenceMatcher(None, texts[left], texts[right],
                                 autojunk=False).ratio() if cross else None)
        pairs.append({
            "jaccard": round(score, 6),
            "containment": round(smaller_containment, 6),
            "eval_containment": round(eval_containment, 6) if cross else None,
            "kind": kind,
            "char_ratio": round(ratio, 6) if cross else None,
            "identical_shingle_set": score == 1.0,
            "eval_rec_id": docs[eval_side]["rec_id"] if cross else None,
            "train_rec_id": docs[train_side]["rec_id"] if cross else None,
            "left_rec_id": a["rec_id"], "left_split": a["split"], "left_chars": a["chars"],
            "right_rec_id": b["rec_id"], "right_split": b["split"], "right_chars": b["chars"],
        })
    pairs.sort(key=lambda row: (-(row["eval_containment"] or row["jaccard"]),
                                row["left_rec_id"], row["right_rec_id"]))

    with (report_out / "near_duplicate_pairs.jsonl").open(
            "w", encoding="utf-8", newline="\n") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    def tally(predicate) -> dict[str, int]:
        subset = [row for row in pairs if predicate(row)]
        return {
            "pairs": len(subset),
            "identical_shingle_set": sum(1 for row in subset if row["identical_shingle_set"]),
            "near_only": sum(1 for row in subset if not row["identical_shingle_set"]),
        }

    cross_all = [row for row in pairs if row["kind"] == "eval-train"]
    cross = [row for row in cross_all if row["eval_containment"] >= gate_threshold]
    gate = [row for row in pairs
            if max(row["jaccard"], row["containment"]) >= gate_threshold]
    report = {
        "version": "e0-near-duplicate-1.0",
        "method": {
            "shingle": f"char {SHINGLE}-gram over the whitespace-normalised text",
            "screen": f"1/{SAMPLE_MOD} hash-sampled shingles, inverted index, "
                      f"boilerplate keys with >{MAX_POSTINGS} postings dropped",
            "score": "exact Jaccard and containment on the full shingle sets of "
                     "every screened pair",
            "gate_metric": "containment of the eval document in one train document",
            "report_threshold": report_threshold,
            "gate_threshold": gate_threshold,
        },
        "documents": {"train": sum(1 for d in docs if d["split"] == "train"),
                      "eval": sum(1 for d in docs if d["split"] == "eval")},
        "at_or_above_report_threshold": {
            "all": tally(lambda row: True),
            "train-train": tally(lambda row: row["kind"] == "train-train"),
            "eval-eval": tally(lambda row: row["kind"] == "eval-eval"),
            "eval-train": tally(lambda row: row["kind"] == "eval-train"),
        },
        "at_or_above_gate_threshold": {
            "all": len(gate),
            "cross_split": len(cross),
            "cross_split_records": sorted(
                {row["left_rec_id"] for row in cross} | {row["right_rec_id"] for row in cross}),
        },
        "cross_split_bands": [
            {"eval_containment_at_least": round(low, 2),
             "pairs": sum(1 for row in cross_all if row["eval_containment"] >= low),
             "train_records": len({row["train_rec_id"] for row in cross_all
                                   if row["eval_containment"] >= low}),
             "eval_records": len({row["eval_rec_id"] for row in cross_all
                                  if row["eval_containment"] >= low}),
             "median_char_ratio": round(statistics.median(
                 [row["char_ratio"] for row in cross_all
                  if low <= row["eval_containment"] < low + 0.05]), 4)
             if any(low <= row["eval_containment"] < low + 0.05 for row in cross_all)
             else None}
            for low in [round(0.3 + 0.05 * i, 2) for i in range(15)]
        ],
        "verdict": ("PASS: no cross-split near duplicate at or above the gate threshold"
                    if not cross else
                    f"FAIL: {len(cross)} cross-split pairs whose eval document is "
                    f"{gate_threshold:.0%}+ contained in a train document"),
    }
    report["source"] = args.source
    (report_out / "near_duplicate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.source == "raw":
        removals = sorted({row["train_rec_id"] for row in cross})
        (report_out / "near_duplicate_removals.json").write_text(json.dumps({
            "version": "e0-near-duplicate-removals-1.0",
            "metric": "containment of the eval document in one train document",
            "gate_threshold": gate_threshold,
            "shingle": f"char {SHINGLE}-gram over the whitespace-normalised text",
            "scanned": {"train": report["documents"]["train"],
                        "eval": report["documents"]["eval"]},
            "eval_documents_protected": len({row["eval_rec_id"] for row in cross}),
            "train_rec_ids": removals,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[nd] removal list: {len(removals)} train records "
              f"protecting {len({row['eval_rec_id'] for row in cross})} eval documents",
              file=sys.stderr, flush=True)

    lines = ["# 近重复阈值样本复核清单", "",
             f"报告阈值 Jaccard ≥ {report_threshold}，闸门阈值 ≥ {gate_threshold}。",
             f"字符 {SHINGLE}-gram，全量精确 Jaccard（筛选只决定算哪些对，不参与打分）。", "",
             f"- 阈值以上总对数：{len(pairs)}",
             f"- 其中 shingle 集完全相同：{sum(1 for row in pairs if row['identical_shingle_set'])}",
             f"- 跨集（train↔eval）且 eval 包含率 ≥ {gate_threshold}：**{len(cross)}**", "",
             "跨集对按 Jaccard 降序全部列出（闸门只看这一类）；同集对见 "
             "`near_duplicate_pairs.jsonl`。", "",
             "| # | eval 包含率 | Jaccard | 字符相似 | eval 记录 | train 记录 |",
             "|---:|---:|---:|---:|---|---|"]
    for index, row in enumerate(cross_all, 1):
        lines.append(f"| {index} | {row['eval_containment']:.4f} | {row['jaccard']:.4f} | "
                     f"{row['char_ratio']:.4f} | {row['eval_rec_id']} | "
                     f"{row['train_rec_id']} |")
    lines += ["", f"（阈值以上共 {len(pairs)} 对，其中跨集 {len(cross_all)} 对全部在上表；"
                  "完整清单见 `near_duplicate_pairs.jsonl`）"]
    (sample_out / "near_duplicate_audit_sample.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
