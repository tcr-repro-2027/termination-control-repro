# coding=utf-8
"""Refuse to start an overnight E1 run that is going to fail at 03:00.

Checks, in the order a failure would actually bite:

1. the sibling projects import (tcr.token_orbit / tcr.extraction / tcr.events);
2. torch + vLLM + transformers are importable and the requested GPUs exist;
3. the evaluation set is the frozen E-Natural file, parses, and has unique keys;
4. the vendored prompt renders identically to the one training used;
5. the training matrix parses and the queue is non-empty, with a per-priority
   breakdown of how much of it is trained yet;
6. every queued checkpoint is loadable, and its tokenizer is available;
7. free disk is enough for the responses + event rows the queue will write;
8. no GPU is already busy (training on the same box would fight for memory).

Exit code 0 = safe to launch, 1 = something must be fixed first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcr.evaluation import protocol                                    # noqa: E402
from tcr.evaluation.io_utils import read_jsonl                         # noqa: E402
from tcr.evaluation.prompts import prompt_fingerprint, verify_against  # noqa: E402
from tcr.evaluation.registry import (                                  # noqa: E402
    build_queue, expected_task_count, has_tokenizer, is_loadable_model_dir,
    parse_run_matrix,
)

#: Rough per-task disk cost of what E1 writes, in GB: responses jsonl plus the
#: 01b event rows.  Measured on the historical 8B-dedup run (high loop rate);
#: a well-behaved model writes appreciably less.
GB_PER_TASK = 0.5


def validate_eval_set(path: Path) -> List[str]:
    """Check EVERY record, not just the first one.

    A record missing `output` used to be scored against an empty gold set --
    every prediction a false positive, every recall denominator short -- and the
    result looked perfectly valid.  A record missing `text` or `entities_str`
    would render a prompt with a hole in it.  Both are cheap to catch here and
    expensive to notice afterwards.
    """
    problems: List[str] = []
    seen: Dict[Any, int] = {}
    total = 0
    for line, record in enumerate(read_jsonl(path), start=1):
        total += 1
        key = record.get("key")
        if key is None:
            problems.append(f"line {line}: no 'key'")
        elif key in seen:
            problems.append(f"line {line}: duplicate key {key!r} "
                            f"(first seen on line {seen[key]})")
        else:
            seen[key] = line
        for field in ("text", "entities_str"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"line {line} (key={key!r}): "
                                f"'{field}' is missing or empty")
        output = record.get("output")
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                problems.append(f"line {line} (key={key!r}): 'output' is an "
                                "unparseable JSON string")
                continue
        if output is None:
            problems.append(f"line {line} (key={key!r}): no 'output'")
        elif not isinstance(output, list):
            problems.append(f"line {line} (key={key!r}): 'output' is "
                            f"{type(output).__name__}, expected a list")
        elif any(not isinstance(item, dict) for item in output):
            problems.append(f"line {line} (key={key!r}): 'output' holds a "
                            "non-object item")
    if total == 0:
        problems.append("the file is empty")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--train_script", required=True)
    parser.add_argument("--train_output_root", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--priority_max", type=int, default=99)
    parser.add_argument("--finals_only", action="store_true")
    parser.add_argument("--no_reference", action="store_true")
    parser.add_argument("--only", default=None,
                        help="Comma-separated tags or run names; must match "
                             "what the orchestrator will be given.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke subset size; scales the disk estimate.")
    parser.add_argument("--no_check_tokenizers", action="store_true",
                        help="Skip actually loading each checkpoint's fast "
                             "tokenizer (the analysis stage requires one).")
    parser.add_argument("--training_prompt_module", default=None,
                        help="tcr/prompt_template.py, when "
                             "it is on this machine.")
    parser.add_argument("--skip_gpu_check", action="store_true")
    args = parser.parse_args()

    ok = True

    def fail(message: str) -> None:
        nonlocal ok
        ok = False
        print(f"[FAIL] {message}")

    def warn(message: str) -> None:
        print(f"[WARN] {message}")

    print("==== E1 preflight ====")
    print(f"protocol: {protocol.PROTOCOL_VERSION}  mode={protocol.MODE}  "
          f"K={protocol.K}  max_model_len={protocol.MAX_MODEL_LEN}")

    # 1 -- the detector / scorer subpackages import
    try:
        import tcr.token_orbit                       # noqa: F401
        import tcr.events.audit             # noqa: F401
        import tcr.extraction.evaluation.metrics    # noqa: F401
        print("[ok]   detector and scorer subpackages import")
    except ImportError as exc:
        fail(f"package import failed: {exc}")

    # 2 -- runtime
    for module in ("numpy", "transformers", "vllm", "torch"):
        try:
            __import__(module)
        except ImportError:
            fail(f"{module} is not importable")
    # GPU ids, not just how many.  `GPUS=7` on a one-card box would pass a
    # count check and then fail per task, hours later, on CUDA_VISIBLE_DEVICES.
    raw = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not raw:
        fail("--gpus is empty")
    non_numeric = [value for value in raw if not value.isdigit()]
    if non_numeric:
        fail(f"--gpus must be integer device ids; got {non_numeric}")
    wanted = [int(value) for value in raw if value.isdigit()]
    duplicates = sorted({value for value in wanted if wanted.count(value) > 1})
    if duplicates:
        fail(f"--gpus repeats device id(s) {duplicates}; two vLLM workers would "
             "share one card and OOM")
    try:
        import torch
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        out_of_range = [value for value in wanted if not 0 <= value < n_gpu]
        if out_of_range:
            fail(f"GPU id(s) {out_of_range} do not exist; torch sees {n_gpu} "
                 f"device(s) (valid ids 0..{max(n_gpu - 1, 0)})")
        else:
            print(f"[ok]   {n_gpu} GPU(s) visible; requested {wanted}")
    except ImportError:
        pass

    # 3 -- evaluation set
    eval_path = Path(args.eval_data)
    if not eval_path.is_file():
        fail(f"evaluation set missing: {eval_path}")
    else:
        if eval_path.name != "eval_supportclean_keep8.jsonl":
            warn(f"E-Natural is frozen as eval_supportclean_keep8.jsonl (§8.1); "
                 f"you passed {eval_path.name}")
        try:
            problems = validate_eval_set(eval_path)
            if problems:
                for problem in problems[:10]:
                    fail(f"evaluation set: {problem}")
                if len(problems) > 10:
                    fail(f"evaluation set: ... and {len(problems) - 10} more")
            else:
                n = sum(1 for _ in read_jsonl(eval_path))
                print(f"[ok]   evaluation set: {n} records, every one complete, "
                      "keys unique")
        except (OSError, ValueError) as exc:
            fail(f"evaluation set unreadable: {exc}")

    # 4 -- prompt identity
    fingerprint = prompt_fingerprint()
    if fingerprint["prompt_is_truncated"]:
        fail("the vendored prompt template is flagged as truncated")
    print(f"[ok]   prompt digest {fingerprint['prompt_rendered_sha256'][:16]} "
          f"({fingerprint['prompt_rendered_chars']} chars rendered)")
    if args.training_prompt_module:
        report = verify_against(args.training_prompt_module)
        if report["available"] and not report["match"]:
            fail("the vendored prompt does NOT match "
                 f"{args.training_prompt_module}: evaluation would use a "
                 "different prompt than training did")
        elif report["available"]:
            print("[ok]   prompt matches the training-side module")
        else:
            warn(f"training prompt module not on this machine: "
                 f"{args.training_prompt_module}")

    # 5/6 -- the queue
    try:
        runs = parse_run_matrix(args.train_script)
    except (OSError, ValueError) as exc:
        fail(f"cannot parse the training matrix: {exc}")
        return 1
    only = ([value.strip() for value in args.only.split(",") if value.strip()]
            if args.only else None)
    tasks = build_queue(runs, output_root=args.train_output_root,
                        model_root=args.model_root,
                        include_reference=not args.no_reference,
                        include_intermediate=not args.finals_only,
                        priority_max=args.priority_max, only=only)
    expected = expected_task_count(runs, priority_max=args.priority_max,
                                   include_reference=not args.no_reference)
    print(f"[ok]   training matrix: {len(runs)} run(s)")
    if not tasks:
        fail(f"no evaluable checkpoint under {args.train_output_root} "
             "(is any run finished? does it carry TRAIN_SUCCESS?)")
    finals = sum(1 for task in tasks if task.is_final)
    print(f"       queue: {len(tasks)} task(s) of an expected ~{expected} "
          f"({finals} final checkpoint(s), "
          f"{len(tasks) - finals} intermediate)")
    # Every priority in the matrix, including the ones with nothing trained
    # yet: "what is still missing" is the reason to read this section.
    for priority in sorted({run.priority for run in runs
                            if run.priority <= args.priority_max}):
        runs_at = [run.run_name for run in runs if run.priority == priority]
        done_at = {task.run_name for task in tasks
                   if task.priority == priority and task.kind == "trained"}
        print(f"       P{priority}: {len(done_at)}/{len(runs_at)} run(s) trained")
        for name in runs_at:
            if name not in done_at:
                print(f"          not yet trained: {name}")

    bad = [task for task in tasks if not is_loadable_model_dir(task.model_path)]
    if bad:
        fail(f"{len(bad)} queued checkpoint(s) are not loadable, e.g. "
             f"{bad[0].model_path}")
    fallback = [task for task in tasks if task.tokenizer_is_fallback]
    if fallback:
        warn(f"{len(fallback)} checkpoint(s) carry no fast tokenizer; falling "
             f"back to the base model's, e.g. {fallback[0].tag}")
        for task in fallback:
            if not has_tokenizer(task.tokenizer_path):
                fail(f"fallback tokenizer missing too: {task.tokenizer_path}")

    if not args.no_check_tokenizers and tasks:
        # `tokenizer_config.json` on disk does not make a tokenizer loadable,
        # let alone fast -- and `structured.load_tokenizer` hard-requires fast,
        # so without this the failure lands AFTER a checkpoint has generated.
        try:
            from transformers import AutoTokenizer
        except ImportError:
            warn("transformers unavailable; skipping the fast-tokenizer check")
        else:
            paths = sorted({task.tokenizer_path for task in tasks})
            print(f"       loading {len(paths)} tokenizer(s) ...")
            for path in paths:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
                except Exception as exc:                      # noqa: BLE001
                    fail(f"tokenizer will not load: {path} ({exc})")
                    continue
                if not getattr(tokenizer, "is_fast", False):
                    fail(f"tokenizer at {path} is not a fast tokenizer; the "
                         "analysis stage needs offset mapping")
            print("[ok]   every queued tokenizer loads and is fast")

    # 7 -- disk
    root = Path(args.result_root)
    root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root).free / (1 << 30)
    # A smoke subset writes proportionally less.
    scale = min(1.0, args.limit / 1106.0) if args.limit else 1.0
    need_gb = len(tasks) * GB_PER_TASK * scale
    if free_gb < need_gb:
        fail(f"{root} has {free_gb:.0f} GB free; the queue writes about "
             f"{need_gb:.0f} GB of responses and event rows")
    else:
        print(f"[ok]   disk: {free_gb:.0f} GB free, about {need_gb:.0f} GB needed")

    # 8 -- cards already busy
    if not args.skip_gpu_check:
        try:
            import subprocess
            used = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30)
            for line in used.stdout.strip().splitlines():
                index, memory = (value.strip() for value in line.split(","))
                if int(memory) > 2048:
                    warn(f"GPU {index} already holds {memory} MiB -- is "
                         "training still running on this box?")
        except (OSError, ValueError, subprocess.SubprocessError):
            warn("nvidia-smi unavailable; skipping the busy-GPU check")

    print("==== preflight " + ("PASSED" if ok else "FAILED") + " ====")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
