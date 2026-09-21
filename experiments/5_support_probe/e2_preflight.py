# coding=utf-8
"""Refuse to start an E2 run that is going to fail after the first checkpoint.

Checks, in the order a failure would bite:

1. the sibling projects import (`tcr.evaluation`, `tcr.extraction`) -- E2 must score
   support with E1's definition, not a second copy of it;
2. torch / transformers / vLLM are importable and the requested GPU ids exist;
3. the evaluation set is the frozen E-Natural file;
4. the anchor set exists, is the one its report describes, was built under this
   protocol from THIS eval file with a real tokenizer, has the promised counts,
   and EVERY anchor still passes its invariants (§E2 criterion 4 is re-checked
   here, not trusted from build time -- an anchor file can be edited or
   truncated in transit);
5. the vendored prompt renders identically to the one training used;
6. every queued checkpoint is loadable and has a FAST tokenizer;
7. free disk covers the readouts.

Exit 0 = safe to launch, 1 = fix something first.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcr.support_probe import protocol                                        # noqa: E402
from tcr.support_probe.anchors import anchor_from_dict, validate_anchor       # noqa: E402
from tcr.support_probe.identity import anchor_set_problems                    # noqa: E402
from tcr.support_probe.io_utils import (                                      # noqa: E402
    read_jsonl,
)
from tcr.support_probe.prompts import prompt_fingerprint, verify_against      # noqa: E402

#: Readouts are small (one line per cell); hazard adds a little more.
GB_PER_MODEL = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--train_script", required=True)
    parser.add_argument("--train_output_root", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--priority_max", type=int, default=99)
    parser.add_argument("--only", default=None)
    parser.add_argument("--no_reference", action="store_true")
    parser.add_argument("--training_prompt_module", default=None)
    parser.add_argument("--no_check_tokenizers", action="store_true")
    parser.add_argument("--no_hazard", action="store_true",
                        help="Skip the vLLM check: only the hazard stage needs it.")
    parser.add_argument("--skip_gpu_check", action="store_true")
    parser.add_argument("--anchor_report", default=None,
                        help="default: <anchors>.report.json")
    parser.add_argument("--n_per_type", type=int,
                        default=protocol.N_ANCHORS_PER_TYPE)
    parser.add_argument("--max_length_delta", type=int, default=0,
                        help="Donor length tolerance this run means; the "
                             "anchor report's must match.")
    parser.add_argument("--allow_char_fallback", action="store_true",
                        help="Accept anchors built without a tokenizer "
                             "(donor lengths matched in characters: a pilot "
                             "build, not the protocol's).")
    args = parser.parse_args()

    ok = True

    def fail(message: str) -> None:
        nonlocal ok
        ok = False
        print(f"[FAIL] {message}")

    def warn(message: str) -> None:
        print(f"[WARN] {message}")

    print("==== E2 preflight ====")
    print(f"protocol: {protocol.PROTOCOL_VERSION}  mode={protocol.MODE}  "
          f"levels={list(protocol.LEVELS)}  tau={protocol.PC_TAU}")

    # 1 -- the shared support definition and model registry import
    try:
        import tcr.evaluation.quality                      # noqa: F401
        import tcr.evaluation.registry                     # noqa: F401
        print("[ok]   evaluation subpackage imports (quality, registry)")
    except ImportError as exc:
        fail(f"package import failed: {exc}")

    # 2 -- runtime + GPU ids
    required = ["numpy", "torch", "transformers"]
    if not args.no_hazard:
        required.append("vllm")          # only the hazard stage generates
    for module in required:
        try:
            __import__(module)
        except ImportError:
            fail(f"{module} is not importable")
    raw = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not raw:
        fail("--gpus is empty")
    if any(not value.isdigit() for value in raw):
        fail(f"--gpus must be integer ids; got {raw}")
    wanted = [int(value) for value in raw if value.isdigit()]
    if len(set(wanted)) != len(wanted):
        fail(f"--gpus repeats a device id: {wanted}")
    try:
        import torch
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        bad = [value for value in wanted if not 0 <= value < n_gpu]
        if bad:
            fail(f"GPU id(s) {bad} do not exist; torch sees {n_gpu}")
        else:
            print(f"[ok]   {n_gpu} GPU(s) visible; requested {wanted}")
    except ImportError:
        pass

    # 3 -- evaluation set
    eval_path = Path(args.eval_data)
    if not eval_path.is_file():
        fail(f"evaluation set missing: {eval_path}")
    elif eval_path.name != "eval_supportclean_keep8.jsonl":
        warn(f"anchors should come from the frozen E-Natural file; "
             f"you passed {eval_path.name}")

    # 4 -- anchors: first their PROVENANCE (is this the anchor set this run
    # means?), then their contents (does every anchor still hold?).  The
    # provenance rules are shared with the launcher's skip decision and with
    # `scripts/e2_check_anchors.py`, so `--check` cannot pass a file the
    # launcher would refuse, or the other way round.
    anchor_path = Path(args.anchors)
    report_path = Path(args.anchor_report) if args.anchor_report else \
        anchor_path.with_suffix(anchor_path.suffix + ".report.json")
    provenance = anchor_set_problems(
        anchor_path, report_path, n_per_type=args.n_per_type,
        eval_data=args.eval_data,
        require_token_matched=not args.allow_char_fallback,
        max_length_delta=args.max_length_delta)
    for problem in provenance:
        fail(f"anchors: {problem}")
    if not provenance:
        print(f"[ok]   anchors match {report_path.name}: this protocol, this "
              "eval set, every type full"
              + ("" if args.allow_char_fallback else ", token-matched donors"))
    if args.allow_char_fallback:
        warn("--allow_char_fallback: donor lengths may be matched in characters "
             "rather than tokens; this is a pilot build")

    if anchor_path.is_file():
        # Counts and provenance are already settled above; what is left is the
        # per-anchor invariants (§E2 criterion 4), re-derived here rather than
        # trusted from build time.
        from tcr.evaluation.quality import candidate_entities, norm_field
        counts: Counter = Counter()
        problems: List[str] = []
        n_variants = 0
        for row in read_jsonl(anchor_path):
            anchor = anchor_from_dict(row)
            counts[anchor.anchor_type] += 1
            n_variants += len(anchor.variants)
            issues = validate_anchor(anchor, norm=norm_field,
                                     parse_candidates=candidate_entities)
            if issues:
                problems.append(f"{anchor.anchor_id}: {issues[0]}")
        if problems:
            for problem in problems[:5]:
                fail(f"anchor invariant: {problem}")
            if len(problems) > 5:
                fail(f"... and {len(problems) - 5} more anchors fail invariants")
        else:
            print(f"[ok]   anchors: {dict(counts)}, {n_variants} variants, "
                  "every invariant holds")
        print(f"[note] ECI-add is UNMEASURED in this run: "
              f"{protocol.describe()['unbuilt_anchor_types']} not built")

    # 5 -- prompt identity
    fingerprint = prompt_fingerprint()
    if fingerprint["prompt_is_truncated"]:
        fail("the vendored prompt template is flagged as truncated")
    print(f"[ok]   prompt digest {fingerprint['prompt_rendered_sha256'][:16]}")
    if args.training_prompt_module:
        report = verify_against(args.training_prompt_module)
        if report["available"] and not report["match"]:
            fail("the vendored prompt does NOT match "
                 f"{args.training_prompt_module}")
        elif report["available"]:
            print("[ok]   prompt matches the training-side module")
        else:
            warn(f"training prompt module not here: {args.training_prompt_module}")

    # 6 -- the model queue
    try:
        from tcr.evaluation.registry import build_queue, has_tokenizer, \
            is_loadable_model_dir, parse_run_matrix
        runs = parse_run_matrix(args.train_script)
        only = ([v.strip() for v in args.only.split(",")] if args.only else None)
        tasks = build_queue(runs, output_root=args.train_output_root,
                            model_root=args.model_root,
                            include_reference=not args.no_reference,
                            include_intermediate=False,
                            priority_max=args.priority_max, only=only)
        print(f"[ok]   {len(runs)} run(s) -> {len(tasks)} final checkpoint(s)")
        if not tasks:
            fail(f"no scoreable checkpoint under {args.train_output_root}")
        bad = [t for t in tasks if not is_loadable_model_dir(t.model_path)]
        if bad:
            fail(f"{len(bad)} checkpoint(s) not loadable, e.g. {bad[0].model_path}")
        if not args.no_check_tokenizers and tasks:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                warn("transformers unavailable; skipping the tokenizer check")
            else:
                for path in sorted({t.tokenizer_path for t in tasks}):
                    if not has_tokenizer(path):
                        fail(f"no fast tokenizer files at {path}")
                        continue
                    try:
                        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
                    except Exception as exc:                  # noqa: BLE001
                        fail(f"tokenizer will not load: {path} ({exc})")
                        continue
                    if not getattr(tok, "is_fast", False):
                        fail(f"tokenizer at {path} is not fast; offsets and the "
                             "decision point both need one")
                print("[ok]   every queued tokenizer loads and is fast")
    except (OSError, ValueError, ImportError) as exc:
        fail(f"cannot build the model queue: {exc}")
        tasks = []

    # 7 -- disk
    root = Path(args.result_root)
    root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root).free / (1 << 30)
    need_gb = max(1.0, len(tasks) * GB_PER_MODEL)
    if free_gb < need_gb:
        fail(f"{root} has {free_gb:.0f} GB free; about {need_gb:.1f} GB needed")
    else:
        print(f"[ok]   disk: {free_gb:.0f} GB free, about {need_gb:.1f} GB needed")

    if not args.skip_gpu_check:
        try:
            import subprocess
            used = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"], capture_output=True,
                text=True, timeout=30)
            for line in used.stdout.strip().splitlines():
                index, memory = (v.strip() for v in line.split(","))
                if int(memory) > 2048:
                    warn(f"GPU {index} already holds {memory} MiB")
        except (OSError, ValueError, subprocess.SubprocessError):
            warn("nvidia-smi unavailable; skipping the busy-GPU check")

    print("==== preflight " + ("PASSED" if ok else "FAILED") + " ====")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
