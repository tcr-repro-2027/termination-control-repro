# -*- coding: utf-8 -*-
"""Render available stage and controlled datasets into prompt, ChatML and SWIFT forms.

Record-form inputs contain ``text``, ``entities_str`` and ``output``;
evaluation rows also contain ``key`` and ``source``. Outputs use the source
stem with ``prompt_``, ``chatml_`` or ``swift_`` prefixes. Prompt-form targets
remain lists; the other forms serialize targets with ``ensure_ascii=False``.

ChatML contains the empty non-thinking block. SWIFT targets omit that block
because the Qwen3 template adds it. Files are streamed to temporary outputs
and renamed after completion.

Usage from the repository root::

    python experiments/1_data/build_training_formats.py \
        --datasets-root datasets --out-root datasets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcr.data.layout import stage_directory  # noqa: E402
from tcr.prompt_template import (  # noqa: E402
    PROMPT_TEMPLATE_IS_TRUNCATED, build_extraction_relation_prompt,
)

STAGES = ("base", "dedup", "filter", "clean", "cleanv2")
CONTROLLED_DIR = "controlled"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"


def log(message: str) -> None:
    print(f"[formats {time.strftime('%H:%M:%S')}] {message}", flush=True)


def serialize_output(output: Any) -> str:
    """The training target string. Compact, Chinese kept literal."""
    return json.dumps(output, ensure_ascii=False)


def chatml_input(prompt: str) -> str:
    return f"{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"


def chatml_output(serialized: str) -> str:
    return f"{EMPTY_THINK}{serialized}{IM_END}\n"


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(datasets_root: Path) -> list[dict[str, Any]]:
    """Every source file to convert, with the name its outputs will carry."""
    jobs: list[dict[str, Any]] = []
    for stage in STAGES:
        stage_dir = stage_directory(datasets_root, stage)
        if not stage_dir.is_dir():
            log(f"WARNING: no such stage directory: {stage_dir}")
            continue
        for split in ("train", "eval"):
            for path in sorted(stage_dir.glob(f"{split}_*.jsonl")):
                if path.stem.endswith("_rowmap"):
                    continue          # row provenance, not training data
                jobs.append({"group": stage_dir.relative_to(datasets_root).as_posix(), "split": split,
                             "name": path.stem[len(split) + 1:], "path": path})
    controlled = datasets_root / CONTROLLED_DIR
    if controlled.is_dir():
        for path in sorted(controlled.glob("train_*.jsonl")):
            jobs.append({"group": CONTROLLED_DIR, "split": "train",
                         "name": path.stem[len("train_"):], "path": path})
    else:
        log(f"WARNING: no controlled directory: {controlled}")
    return jobs


def convert(job: dict[str, Any], out_dir: Path, force: bool) -> dict[str, Any]:
    split, name, src = job["split"], job["name"], job["path"]
    is_eval = split == "eval"
    targets = {form: out_dir / f"{form}_{split}_{name}.jsonl"
               for form in ("prompt", "chatml", "swift")}
    if not force and all(path.is_file() for path in targets.values()):
        log(f"  {name} ({split}): already built, skipping (--force to redo)")
        return {"skipped": True,
                "outputs": {form: str(path) for form, path in targets.items()}}

    out_dir.mkdir(parents=True, exist_ok=True)
    temps = {form: path.with_suffix(path.suffix + ".partial")
             for form, path in targets.items()}
    rows = blocks = 0
    non_list_output = 0
    non_dict_blocks = 0
    missing_source = 0

    handles = {form: path.open("w", encoding="utf-8", newline="\n")
               for form, path in temps.items()}
    try:
        for index, record in enumerate(iter_jsonl(src)):
            prompt = build_extraction_relation_prompt(
                text=record.get("text", ""),
                entities_str=record.get("entities_str", ""))
            output = record.get("output", [])
            if isinstance(output, list):
                n_output_dicts = len(output)
                non_dict_blocks += sum(1 for item in output
                                       if not isinstance(item, dict))
            else:
                # Kept verbatim rather than coerced: a row that is not a list is
                # a data problem to look at, not one to paper over here.
                non_list_output += 1
                n_output_dicts = 0
            blocks += n_output_dicts
            serialized = serialize_output(output)

            head: dict[str, Any] = {"index": index}
            if is_eval:
                if "source" not in record:
                    missing_source += 1
                head["source"] = record.get("source", "")
            head["n_output_dicts"] = n_output_dicts

            handles["prompt"].write(json.dumps(
                {**head, "prompt": prompt, "output": output},
                ensure_ascii=False) + "\n")
            handles["chatml"].write(json.dumps(
                {**head, "input": chatml_input(prompt),
                 "output": chatml_output(serialized)},
                ensure_ascii=False) + "\n")
            handles["swift"].write(json.dumps(
                {"messages": [{"role": "user", "content": prompt},
                              {"role": "assistant", "content": serialized}]},
                ensure_ascii=False) + "\n")
            rows = index + 1
            if rows % 2000 == 0:
                log(f"    {name} ({split}): {rows} rows")
    except BaseException:
        for handle in handles.values():
            handle.close()
        for path in temps.values():
            path.unlink(missing_ok=True)
        raise
    for handle in handles.values():
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
    for form, path in temps.items():
        os.replace(path, targets[form])

    report = {
        "skipped": False,
        "source": str(src),
        "source_sha256": sha256_of(src),
        "rows": rows,
        "output_blocks": blocks,
        "rows_with_non_list_output": non_list_output,
        "non_dict_blocks": non_dict_blocks,
        "eval_rows_missing_source": missing_source,
        "outputs": {form: str(path) for form, path in targets.items()},
        "output_sha256": {form: sha256_of(path)
                          for form, path in targets.items()},
        "output_bytes": {form: targets[form].stat().st_size
                         for form in targets},
    }
    flags = ""
    if non_list_output:
        flags += f"  [{non_list_output} rows whose output is not a list]"
    if non_dict_blocks:
        flags += f"  [{non_dict_blocks} non-dict blocks]"
    if missing_source:
        flags += f"  [{missing_source} eval rows with no source field]"
    log(f"  {name} ({split}): {rows} rows, {blocks} blocks{flags}")
    return report


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-root", default=str(here.parents[1] / "datasets"))
    parser.add_argument("--out-root", default=None,
                        help="default: <datasets-root>/processed")
    parser.add_argument("--only", default=None,
                        help="comma-separated dataset names to build "
                             "(e.g. isc_a,base); default is all of them")
    parser.add_argument("--force", action="store_true",
                        help="rebuild files that already exist")
    parser.add_argument("--report-out", default=None,
                        help="default: <out-root>/build_formats_report.json")
    args = parser.parse_args()

    if PROMPT_TEMPLATE_IS_TRUNCATED:
        log("FATAL: prompt_template.py is the abridged template; the rendered "
            "prompts would not match the evaluation pipeline.")
        return 1

    datasets_root = Path(args.datasets_root).resolve()
    out_root = (Path(args.out_root).resolve() if args.out_root
                else datasets_root / "processed")
    jobs = discover(datasets_root)
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        jobs = [job for job in jobs if job["name"] in wanted]
        log(f"restricted to {len(jobs)} files matching {sorted(wanted)}")
    if not jobs:
        log("nothing to convert")
        return 1

    log(f"datasets root: {datasets_root}")
    log(f"output root:   {out_root}")
    log(f"{len(jobs)} source files -> {len(jobs) * 3} output files")

    reports: dict[str, Any] = {}
    for job in jobs:
        log(f"{job['group']} / {job['name']} ({job['split']})")
        reports[f"{job['group']}/{job['split']}_{job['name']}"] = convert(
            job, out_root / job["group"], args.force)

    built = [row for row in reports.values() if not row.get("skipped")]
    report = {
        "datasets_root": str(datasets_root),
        "out_root": str(out_root),
        "prompt_template": "tcr/prompt_template.py",
        "prompt_template_is_truncated": PROMPT_TEMPLATE_IS_TRUNCATED,
        "serialization": "json.dumps(output, ensure_ascii=False)  # compact",
        "chatml_input": "<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n",
        "chatml_output": "<think>\\n\\n</think>\\n\\n{serialized}<|im_end|>\\n",
        "swift_assistant": "serialized output only; the qwen3 template adds the "
                           "empty think block itself",
        "files": reports,
        "totals": {
            "source_files": len(jobs),
            "built": len(built),
            "skipped": len(jobs) - len(built),
            "rows": sum(int(row["rows"]) for row in built),
            "output_blocks": sum(int(row["output_blocks"]) for row in built),
            "rows_with_non_list_output": sum(
                int(row["rows_with_non_list_output"]) for row in built),
            "output_bytes": sum(sum(row["output_bytes"].values())
                                for row in built),
        },
    }
    report_path = (Path(args.report_out) if args.report_out
                   else out_root / "build_formats_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    totals = report["totals"]
    log(f"done: {totals['built']} built, {totals['skipped']} skipped, "
        f"{totals['rows']} rows, "
        f"{totals['output_bytes'] / (1 << 30):.2f} GiB written")
    log(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
