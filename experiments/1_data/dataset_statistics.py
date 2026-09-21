# coding: utf-8
"""Build a presentation-ready size and token-length table for all datasets.

The source rows are read from ``datasets`` rather than from already-rendered
files.  For every row the script rebuilds the frozen relation-extraction
prompt, serializes ``output`` exactly as the training renderer does, and
measures the complete two-turn conversation with the local Qwen3 chat
template using ``enable_thinking=False``.  This makes ``chat_tokens`` the
authoritative training-length column; input and assistant segment lengths are
reported as useful diagnostics.

Usage::

    python experiments/1_data/dataset_statistics.py --datasets-root datasets

Outputs (default ``<datasets-root>/dataset_statistics``)::

    dataset_statistics.json  # full distributions and provenance
    dataset_statistics.csv   # one compact row per dataset
    dataset_statistics.md    # meeting-ready Markdown table
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcr.prompt_template import (  # noqa: E402
    PROMPT_TEMPLATE_IS_TRUNCATED,
    build_extraction_relation_prompt,
)


MAX_LENGTH = 32768
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"


DATASET_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("train_orig", "original", "train", "train_orig.jsonl"),
    ("eval_orig", "original", "eval", "eval_orig.jsonl"),
    ("base_train", "stage", "train", "stages/base/train_base.jsonl"),
    ("base_eval", "stage", "eval", "stages/base/eval_base.jsonl"),
    ("dedup_train", "stage", "train", "stages/dedup/train_dedup.jsonl"),
    ("dedup_eval", "stage", "eval", "stages/dedup/eval_dedup.jsonl"),
    ("filter_train", "stage", "train", "stages/filter/train_filter.jsonl"),
    ("filter_eval", "stage", "eval", "stages/filter/eval_filter.jsonl"),
    ("clean_train", "stage", "train", "stages/clean/train_clean.jsonl"),
    ("clean_eval", "stage", "eval", "stages/clean/eval_clean.jsonl"),
    ("cleanv2_train", "stage", "train",
     "cleanv2/train_supportclean_keep8.jsonl"),
    ("cleanv2_eval", "stage", "eval",
     "cleanv2/eval_supportclean_keep8.jsonl"),
    ("isc_a", "controlled", "train", "controlled/train_isc_a.jsonl"),
    ("isc_e", "controlled", "train", "controlled/train_isc_e.jsonl"),
    ("isc_ae", "controlled", "train", "controlled/train_isc_ae.jsonl"),
    ("benign_input", "controlled", "train",
     "controlled/train_benign_input.jsonl"),
    ("generic_noise", "controlled", "train",
     "controlled/train_generic_noise.jsonl"),
    ("train_obr", "controlled", "train", "controlled/train_obr.jsonl"),
    ("train_obr_p10", "controlled", "train", "controlled/train_obr_p10.jsonl"),
    ("train_obr_p15", "controlled", "train", "controlled/train_obr_p15.jsonl"),
    ("train_obr_p5", "controlled", "train", "controlled/train_obr_p5.jsonl"),
    ("keep4", "keep4", "train", "raw/train_keep4.jsonl"),
    ("randdrop", "keep4", "train", "raw/train_randdrop.jsonl"),
    ("keep4_a", "keep4", "train", "raw/train_keep4_a.jsonl"),
    ("keep4_ae", "keep4", "train", "raw/train_keep4_ae.jsonl"),
)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield value


def serialize_output(output: Any) -> str:
    """Match build_training_formats.py's compact, non-ASCII serialization."""
    return json.dumps(output, ensure_ascii=False)


def output_items(record: Mapping[str, Any], path: Path, row_index: int
                 ) -> list[Any]:
    output = record.get("output", [])
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{row_index}: output string is not JSON") from exc
    if output is None:
        return []
    if not isinstance(output, list):
        raise ValueError(f"{path}:{row_index}: output is not a list")
    return output


def chatml_input(prompt: str) -> str:
    return f"{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"


def chatml_output(serialized: str) -> str:
    return f"{EMPTY_THINK}{serialized}{IM_END}\n"


def quantile(sorted_values: list[int], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def distribution(values: Iterable[int]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    if not ordered:
        return {name: None for name in
                ("min", "mean", "p50", "p95", "p99", "max")}
    return {
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 2),
        "p50": round(quantile(ordered, 0.50) or 0.0, 2),
        "p95": round(quantile(ordered, 0.95) or 0.0, 2),
        "p99": round(quantile(ordered, 0.99) or 0.0, 2),
        "max": ordered[-1],
    }


def tokenizer_lengths(tokenizer, texts: list[str]) -> list[int]:
    if not texts:
        return []
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    return [len(ids) for ids in encoded]


def measure_dataset(name: str, group: str, split: str, path: Path,
                    tokenizer, limit: int | None, batch_size: int
                    ) -> dict[str, Any]:
    records = relation_dicts = non_dict_items = empty_outputs = 0
    blocks_per_record: list[int] = []
    input_lengths: list[int] = []
    assistant_lengths: list[int] = []
    chat_lengths: list[int] = []
    pending_input: list[str] = []
    pending_assistant: list[str] = []
    pending_chat: list[str] = []

    def flush() -> None:
        input_lengths.extend(tokenizer_lengths(tokenizer, pending_input))
        assistant_lengths.extend(tokenizer_lengths(tokenizer, pending_assistant))
        chat_lengths.extend(tokenizer_lengths(tokenizer, pending_chat))
        pending_input.clear()
        pending_assistant.clear()
        pending_chat.clear()

    for row_index, record in enumerate(read_jsonl(path)):
        if limit is not None and records >= limit:
            break
        items = output_items(record, path, row_index + 1)
        n_dicts = sum(isinstance(item, dict) for item in items)
        records += 1
        relation_dicts += n_dicts
        non_dict_items += len(items) - n_dicts
        blocks_per_record.append(len(items))
        if not items:
            empty_outputs += 1

        prompt = build_extraction_relation_prompt(
            text=str(record.get("text", "")),
            entities_str=str(record.get("entities_str", "")),
        )
        serialized = serialize_output(record.get("output", []))
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": serialized},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        pending_input.append(chatml_input(prompt))
        pending_assistant.append(chatml_output(serialized))
        pending_chat.append(rendered)
        if len(pending_chat) >= batch_size:
            flush()
        if records % 1000 == 0:
            print(f"  {name}: {records} rows", flush=True)
    flush()

    over = sum(length > MAX_LENGTH for length in chat_lengths)
    return {
        "name": name,
        "group": group,
        "split": split,
        "file": str(path),
        "records": records,
        "relation_dicts": relation_dicts,
        "non_dict_output_items": non_dict_items,
        "empty_output_records": empty_outputs,
        "blocks_per_record": distribution(blocks_per_record),
        "input_tokens": distribution(input_lengths),
        "assistant_tokens": distribution(assistant_lengths),
        "chat_tokens": distribution(chat_lengths),
        "over_max_length": over,
        "over_max_length_share": round(over / records, 6) if records else 0.0,
        "kept_at_max_length": records - over,
    }


def csv_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        blocks = result["blocks_per_record"]
        prompt = result["input_tokens"]
        assistant = result["assistant_tokens"]
        chat = result["chat_tokens"]
        rows.append({
            "dataset": result["name"],
            "group": result["group"],
            "split": result["split"],
            "records": result["records"],
            "relation_dicts": result["relation_dicts"],
            "dicts_per_record_mean": blocks["mean"],
            "input_tokens_mean": prompt["mean"],
            "assistant_tokens_mean": assistant["mean"],
            "chat_tokens_mean": chat["mean"],
            "chat_tokens_p50": chat["p50"],
            "chat_tokens_p95": chat["p95"],
            "chat_tokens_p99": chat["p99"],
            "chat_tokens_max": chat["max"],
            "over_32768": result["over_max_length"],
            "over_32768_share": result["over_max_length_share"],
        })
    return rows


CSV_FIELDS = (
    "dataset", "group", "split", "records", "relation_dicts",
    "dicts_per_record_mean", "input_tokens_mean", "assistant_tokens_mean",
    "chat_tokens_mean", "chat_tokens_p50", "chat_tokens_p95",
    "chat_tokens_p99", "chat_tokens_max", "over_32768", "over_32768_share",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Group | Split | Records | Relation dicts | Dicts/record | "
        "Input tok mean | Assistant tok mean | Chat tok mean | Chat p50 | "
        "Chat p95 | Chat max | >32768 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {group} | {split} | {records} | {relation_dicts} | "
            "{dicts_per_record_mean} | {input_tokens_mean} | "
            "{assistant_tokens_mean} | {chat_tokens_mean} | {chat_tokens_p50} | "
            "{chat_tokens_p95} | {chat_tokens_max} | {over_32768} ({over_32768_share:.2%}) |"
            .format(**row))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-root", default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--tokenizer", default=None,
                        help="default: <datasets-root>/Qwen3-4B")
    parser.add_argument("--out-dir", default=None,
                        help="default: <datasets-root>/dataset_statistics")
    parser.add_argument("--only", default=None,
                        help="comma-separated dataset names; default: all specs")
    parser.add_argument("--limit", type=int, default=None,
                        help="measure only the first N rows per dataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-missing", action="store_true",
                        help="skip missing files instead of failing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    root = Path(args.datasets_root).resolve()
    tokenizer_dir = (Path(args.tokenizer).resolve() if args.tokenizer
                     else root / "Qwen3-4B")
    if not (tokenizer_dir / "tokenizer.json").is_file():
        print(f"FATAL: tokenizer.json not found at {tokenizer_dir}", file=sys.stderr)
        return 1
    if PROMPT_TEMPLATE_IS_TRUNCATED:
        print("FATAL: prompt_template.py is marked truncated", file=sys.stderr)
        return 1

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), use_fast=True, local_files_only=True,
        trust_remote_code=False)
    print(f"tokenizer: {tokenizer_dir}")
    print("chat template: Qwen3 apply_chat_template(enable_thinking=False)")

    wanted = ({value.strip() for value in args.only.split(",") if value.strip()}
              if args.only else None)
    unknown = wanted - {spec[0] for spec in DATASET_SPECS} if wanted else set()
    if unknown:
        raise SystemExit(f"unknown dataset name(s): {', '.join(sorted(unknown))}")

    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for name, group, split, relative in DATASET_SPECS:
        if wanted is not None and name not in wanted:
            continue
        path = root / relative
        if not path.is_file():
            missing.append(str(path))
            if not args.allow_missing:
                print(f"FATAL: missing dataset: {path}", file=sys.stderr)
                return 1
            print(f"WARNING: skipping missing dataset: {path}", file=sys.stderr)
            continue
        print(f"[{name}] {path}")
        results.append(measure_dataset(name, group, split, path, tokenizer,
                                       args.limit, args.batch_size))

    out_dir = (Path(args.out_dir).resolve() if args.out_dir
               else root / "dataset_statistics")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = csv_rows(results)
    report = {
        "version": "dataset-statistics-1.0",
        "datasets_root": str(root),
        "tokenizer": str(tokenizer_dir),
        "max_length": MAX_LENGTH,
        "truncation_strategy": "delete (reported only; no rows removed here)",
        "prompt_template": str(TRAIN_SCRIPTS / "prompt_template.py"),
        "chat_template": {
            "method": "tokenizer.apply_chat_template",
            "add_generation_prompt": False,
            "enable_thinking": False,
            "messages": "one user turn + one assistant turn",
        },
        "length_fields": {
            "input_tokens": "rendered user turn plus assistant prefix",
            "assistant_tokens": "empty non-thinking block plus serialized output and assistant end marker",
            "chat_tokens": "complete tokenizer.apply_chat_template result; authoritative training length",
        },
        "serialization": "json.dumps(output, ensure_ascii=False)",
        "batch_size": args.batch_size,
        "limit": args.limit,
        "missing": missing,
        "datasets": results,
        "csv_rows": rows,
    }
    json_path = out_dir / "dataset_statistics.json"
    csv_path = out_dir / "dataset_statistics.csv"
    markdown_path = out_dir / "dataset_statistics.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    write_csv(csv_path, rows)
    markdown_path.write_text(markdown_table(rows), encoding="utf-8")

    print("\n" + markdown_table(rows))
    print(f"JSON -> {json_path}")
    print(f"CSV  -> {csv_path}")
    print(f"MD   -> {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
