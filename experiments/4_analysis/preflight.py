# coding: utf-8
"""What can run right now, what is waiting on an evaluation, and what is broken.

Those three are different states and the exit code treats them differently.  An
evaluation that has not finished yet is not an error -- most of the closeout
does not depend on the newly trained models at all -- so it is reported and the
check still passes.  A missing gold file or a checkpoint that will not load is
an error, because no amount of waiting fixes it.

Model states
------------
``ready``       the summary exists.  That is the only artefact whose presence
                means the evaluation finished: `responses` is append-only and
                resumable, so a partial one sits on disk for the whole run.
``generating``  responses on disk, no summary yet: still being evaluated.
``absent``      nothing on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (REPO_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tcr.paper import registry                                            # noqa: E402
from tcr.paper.io import read_csv                                          # noqa: E402
from tcr.paper.layout import Layout, model_path_for                        # noqa: E402


def model_state(layout: Layout, tag: str) -> str:
    """Readiness is the summary and nothing else.

    `responses` is append-only and resumed by key, so a partial file is on disk
    for the whole of generation and proves nothing; `events` is replaced into
    place at the end of the analysis stage and `summary` is written after that
    stage returns.  Summary present implies the other two.
    """
    if layout.summary(tag).is_file():
        return "ready"
    if layout.responses(tag).is_file():
        return "generating"
    return "absent"


def gated(name: str, ready: int, total: int, blockers: Iterable[str],
          prerequisites: Sequence[tuple[str, bool]] = ()) -> str:
    """One stage's line, with the artefacts it needs as well as the models.

    "Every model is evaluated" is not the same as "this stage can run": R1
    needs the prefix pool, R2 needs the pool as well, and a stage reported
    READY that then fails on a missing file is worse than no report at all.
    """
    missing_models = sorted(set(blockers))
    if missing_models:
        return line(name, ready, total, missing_models)
    unmet = [label for label, present in prerequisites if not present]
    if unmet:
        return f"  {name:<26s} {'NEEDS':<8s} {ready}/{total}  {unmet[0]}"
    return line(name, ready, total, [])


def line(name: str, ready: int, total: int, blockers: Iterable[str]) -> str:
    missing = sorted(set(blockers))
    if not missing:
        status = "READY"
    elif ready:
        status = "PARTIAL"
    else:
        status = "BLOCKED"
    detail = ("waiting on " + ", ".join(registry.label(tag) for tag in missing)
              if missing else "")
    return f"  {name:<26s} {status:<8s} {ready}/{total}  {detail}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    layout = Layout.from_env()
    problems: list[str] = []
    warnings: list[str] = []

    print("paths")
    for name, value in layout.describe().items():
        print(f"  {name:<16s} {value}")

    if not layout.gold.is_file():
        problems.append(f"missing evaluation set {layout.gold}")
    metrics: dict[str, dict[str, Any]] = {}
    if layout.metrics_csv.is_file():
        metrics = {row["tag"]: row for row in read_csv(layout.metrics_csv)}
    else:
        warnings.append(f"no {layout.metrics_csv} yet; the tables need it, the "
                        "mechanism stages do not")

    states = {tag: model_state(layout, tag) for tag in registry.all_tags()}
    ready = {tag for tag, state in states.items() if state == "ready"}
    print("\nmodels")
    for state in ("generating", "absent"):
        tags = [tag for tag in registry.all_tags() if states[tag] == state]
        if tags:
            print(f"  {state:<12s} {', '.join(registry.label(tag) for tag in tags)}")
    print(f"  ready        {len(ready)}/{len(states)}")

    def blockers(tags: Sequence[str]) -> list[str]:
        return [tag for tag in tags if tag not in ready]

    print("\nstages")
    r0_ok = [pair for pair in registry.R0_PAIRS
             if not blockers([pair.m0, pair.m1])]
    r0_blocked = [tag for pair in registry.R0_PAIRS
                  for tag in blockers([pair.m0, pair.m1])]
    print(line("R0 (model pairs)", len(r0_ok), len(registry.R0_PAIRS), r0_blocked))
    for scale, sources in registry.R1_POOL_SOURCES.items():
        missing = blockers(list(sources))
        candidates = all((layout.r0 / f"R0_prefix_candidates_{tag}.jsonl").is_file()
                         for tag in sources)
        print(gated(f"{scale} prefix pool", len(sources) - len(missing),
                    len(sources), missing,
                    [("run run_r0.py first; it writes the per-block candidate "
                      "table the pool selects from", candidates)]))
    for scale, tags in registry.R1_SCORED.items():
        pool_missing = blockers(list(registry.R1_POOL_SOURCES[scale]))
        missing = blockers(list(tags))
        pool = (layout.r1 / f"prefix_pool_{scale}.jsonl").is_file()
        print(gated(f"{scale} R1 (scored models)", len(tags) - len(missing),
                    len(tags), set(missing) | set(pool_missing),
                    [("run build_prefix_pool.py first", pool)]))
    for scale, config in registry.R2_MODELS.items():
        tags = [config["clean"], config["raw"], *config.get("transfer", ())]
        missing = set(blockers(tags)) | set(blockers(list(registry.R1_POOL_SOURCES[scale])))
        pool = (layout.r1 / f"prefix_pool_{scale}.jsonl").is_file()
        readouts = (layout.r1 / f"R1_anchor_readouts_{scale}.jsonl").is_file()
        print(gated(f"{scale} R2 (intervention)", len(tags) - len(blockers(tags)),
                    len(tags), missing,
                    [("run build_prefix_pool.py first", pool),
                     ("run R1 first; its margins choose the layer", readouts)]))
    print(line("X1 (motif gain)",
               len(registry.X1_MODELS) - len(blockers(list(registry.X1_MODELS))),
               len(registry.X1_MODELS), blockers(list(registry.X1_MODELS))))
    t1 = [tag for tag in registry.T1_ROWS if tag in metrics]
    t2 = [tag for _label, tag, _dose in registry.T2_ROWS if tag in metrics]
    print(line("Table 1 rows", len(t1), len(registry.T1_ROWS),
               [tag for tag in registry.T1_ROWS if tag not in metrics]))
    print(line("Table 2 rows", len(t2), len(registry.T2_ROWS),
               [tag for _l, tag, _d in registry.T2_ROWS if tag not in metrics]))

    # A finished evaluation whose data is gone is not something to wait for:
    # the files were moved or deleted afterwards, and R0 reads both of them.
    for tag in sorted(ready):
        gone = [name for name, path in (("responses", layout.responses(tag)),
                                        ("events", layout.events(tag)))
                if not path.is_file()]
        if gone:
            problems.append(f"{tag}: the evaluation finished but its "
                            + " and ".join(gone) + " file is missing")

    # Only a model that has finished evaluating is expected to have a loadable
    # checkpoint; one still training legitimately does not.
    loaded = registry.required_tags(
        *registry.R1_SCORED.values(),
        [config[key] for config in registry.R2_MODELS.values() for key in ("clean", "raw")],
        [tag for config in registry.R2_MODELS.values() for tag in config.get("transfer", ())],
        registry.X1_MODELS)
    for tag in loaded:
        if tag not in ready:
            continue
        try:
            path = model_path_for(layout, tag, metrics_row=metrics.get(tag))
        except FileNotFoundError as exc:
            problems.append(f"{tag}: {exc}")
            continue
        if not (path / "config.json").is_file():
            problems.append(f"{tag}: {path} has no config.json")

    for scale in registry.R1_POOL_SOURCES:
        base = layout.base_model(scale)
        if not (base / "tokenizer.json").is_file():
            problems.append(f"{scale}: no fast tokenizer at {base}")
    if any(config.get("sae") for config in registry.R2_MODELS.values()):
        if not layout.sae_root.is_dir():
            warnings.append(f"no SAE directory at {layout.sae_root}; the feature "
                            "views are skipped and nothing else changes")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  cuda                       {torch.cuda.device_count()} device(s)")
        else:
            (problems if args.require_gpu else warnings).append(
                "torch reports no CUDA device; only the CPU stages can run")
    except ImportError:
        (problems if args.require_gpu else warnings).append("torch is not importable")

    print()
    for message in warnings:
        print(f"  [warn] {message}")
    for message in problems:
        print(f"  [FAIL] {message}")
    waiting = len(states) - len(ready)
    if problems:
        print(f"\npreflight FAILED: {len(problems)} problem(s) that waiting will not fix")
        return 1
    print(f"\npreflight passed"
          + (f"; {waiting} model(s) still evaluating -- the stages marked READY "
             "above can start now" if waiting else "")
          + (f"; {len(warnings)} warning(s)" if warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
