# coding=utf-8
"""What to evaluate: the training matrix, plus the checkpoints on disk.

The training matrix is READ FROM `experiments/2_train/run_train_all.sh` rather
than duplicated here.  That file is the thing the user edits when the plan
changes, and a second hand-maintained copy of a 24-row matrix is a guaranteed
source of silent mismatches between "what was trained" and "what was scored".

What is actually evaluated is then discovered on disk: for every run that
carries a `TRAIN_SUCCESS` sentinel, every `checkpoint-<step>/` directory that
looks like a loadable model.  Two consequences worth stating:

* a run still training is invisible until it finishes, so a `--follow` eval
  can be started before training ends without ever reading a half-written
  checkpoint (ms-swift writes `TRAIN_SUCCESS` only after `swift sft` returns);
* `trajectory` runs contribute ~10 tasks each and `final_only` runs exactly 1,
  with no assumption about the step numbers -- if `save_steps 42` over 417
  steps did not produce the 10 checkpoints the training script predicted, the
  registry reports what is really there.

Ordering (see :func:`sort_key`)
-------------------------------
1. the three untrained Qwen3 checkpoints, smallest first.  They need no
   training to finish, they are the M0 anchor of every comparison in the
   paper, and their historical numbers are known (`loop_results.csv`), so they
   double as an end-to-end validation of the harness before it spends a night
   on new checkpoints;
2. every FINAL checkpoint, in the priority order declared by `run_train_all.sh`
   (P1 controlled pilot first -- it is the run that decides the hypothesis);
3. every intermediate trajectory checkpoint, **step-major**: all runs' step-42
   point, then all runs' step-84 point, and so on.  Nothing in E1 depends on
   these (they are E7's time-evolution curve), and step-major means an
   interrupted queue leaves four partial curves at matched steps rather than
   one complete curve and three empty ones.
"""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Stage datasets, in the pipeline order the paper reports them in.
STAGE_ORDER = ("base", "dedup", "filter", "clean", "cleanv2")
#: `cleanv2` is the stage name; `supportclean_keep8` is the file name it ships
#: under.  Both appear in the plan documents, so both must resolve here.
DATASET_ALIASES = {"supportclean_keep8": "cleanv2"}

#: `obr_p15` / `obr_p10` / `obr_p5` are nested lower-dose cuts of the shipped
#: 24.30% OBR pairing, added after the first E1 round to separate "dose" from
#: "kind of corruption" (OBR-15% carries exactly the ISC dose of 87,754
#: conflict blocks).  `obr_p5` completes the paper's 0/5/10/15% main series;
#: 24.30% stays here because its result is still reported, in the appendix.
CONTROLLED_ARMS = ("isc_a", "isc_e", "isc_ae", "generic_noise",
                   "benign_input", "obr", "obr_p15", "obr_p10", "obr_p5")

SIZES = ("1.7B", "4B", "8B")
#: Untrained instruct checkpoints.  In `loop_results.csv` these are the rows
#: named `Qwen3-<size>_nothink_n8`; here they get an explicit `-notrain` tag so
#: they can never be confused with the `base`-STAGE SFT models, which are a
#: different thing entirely.
REFERENCE_TAGS = {size: f"qwen3-{size.lower()}-notrain" for size in SIZES}

_RUNS_BLOCK = re.compile(r"^RUNS=\(\s*$", re.MULTILINE)
_RUN_ROW = re.compile(r'^\s*"([^"]*)"\s*$')
_CHECKPOINT_DIR = re.compile(r"^checkpoint-(\d+)$")

_WEIGHT_PATTERNS = ("model.safetensors", "model-*.safetensors",
                    "pytorch_model.bin", "pytorch_model-*.bin")
#: A FAST tokenizer is required (the analysis stage needs offset mapping), and
#: `tokenizer.json` is what makes one loadable without a slow->fast conversion.
#: A checkpoint carrying only `tokenizer_config.json` therefore does NOT count:
#: falling back to the base model's tokenizer is both safe (identical Qwen3
#: vocabulary) and better than discovering the problem after generation.
_TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json")


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """One row of `run_train_all.sh`'s RUNS array."""
    order: int                  # position in the file; ties broken by it
    priority: int
    run_name: str
    size: str
    dataset: str                # as written, e.g. "$S/clean/swift_train_clean.jsonl"
    seed: int
    save_mode: str              # trajectory | final_only
    terminal_mode: str          # "" when plain CE
    terminal_rho: str

    @property
    def dataset_file(self) -> str:
        return self.dataset.rsplit("/", 1)[-1]

    @property
    def data_variant(self) -> str:
        """`base` / `dedup` / ... / `cleanv2` / `isc_a` / ... from the file name."""
        stem = self.dataset_file
        for prefix in ("swift_train_", "chatml_train_", "prompt_train_", "train_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        stem = stem[:-len(".jsonl")] if stem.endswith(".jsonl") else stem
        return DATASET_ALIASES.get(stem, stem)

    @property
    def arm_family(self) -> str:
        variant = self.data_variant
        if variant in STAGE_ORDER:
            return "stage"
        if variant in CONTROLLED_ARMS:
            return "controlled"
        return "other"


@dataclasses.dataclass(frozen=True)
class EvalTask:
    """One (model, checkpoint) to generate and score."""
    tag: str                    # unique; names every artefact of this task
    run_name: str
    kind: str                   # reference | trained
    tier: int                   # 0 reference, 1..6 final, 90 intermediate
    priority: int               # the training priority the task inherits
    size: str
    data_variant: str
    arm_family: str
    seed: Optional[int]
    save_mode: str
    terminal_mode: str
    ckpt_step: Optional[int]
    is_final: bool
    model_path: str
    tokenizer_path: str
    tokenizer_is_fallback: bool
    order: int                  # position of the run in the training matrix

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ------------------------------------------------------------------- matrix

def parse_run_matrix(path: str | os.PathLike[str]) -> List[RunSpec]:
    """Parse the `RUNS=( ... )` array of `run_train_all.sh`.

    Only the array is read; the surrounding bash is not executed and `$S` /
    `$C` are left unexpanded, because the registry needs the dataset's
    identity (the supervision condition), independent of the storage path.
    """
    text = Path(path).read_text(encoding="utf-8")
    match = _RUNS_BLOCK.search(text)
    if match is None:
        raise ValueError(f"{path}: no `RUNS=(` array found")
    tail = text[match.end():]
    end = tail.find("\n)")
    if end == -1:
        raise ValueError(f"{path}: RUNS array is not terminated by a line `)`")

    runs: List[RunSpec] = []
    for line in tail[:end].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        row = _RUN_ROW.match(line)
        if row is None:
            raise ValueError(f"{path}: cannot parse RUNS entry: {line!r}")
        fields = row.group(1).split("|")
        if len(fields) != 8:
            raise ValueError(
                f"{path}: RUNS entry must have 8 `|` fields "
                f"(priority|name|size|dataset|seed|save_mode|terminal_mode|"
                f"terminal_rho), got {len(fields)}: {row.group(1)!r}")
        priority, name, size, dataset, seed, save_mode, tmode, trho = fields
        if size not in SIZES:
            raise ValueError(f"{path}: unknown size {size!r} in run {name!r}")
        if save_mode not in ("trajectory", "final_only"):
            raise ValueError(f"{path}: unknown save_mode {save_mode!r} "
                             f"in run {name!r}")
        runs.append(RunSpec(order=len(runs), priority=int(priority),
                            run_name=name, size=size, dataset=dataset,
                            seed=int(seed), save_mode=save_mode,
                            terminal_mode=tmode, terminal_rho=trho))
    if not runs:
        raise ValueError(f"{path}: RUNS array is empty")

    duplicates = [name for name in {run.run_name for run in runs}
                  if sum(run.run_name == name for run in runs) > 1]
    if duplicates:
        raise ValueError(f"{path}: duplicate run_name(s): {sorted(duplicates)}")
    return runs


# --------------------------------------------------------------- filesystem

def _has_weights(directory: Path) -> bool:
    for pattern in _WEIGHT_PATTERNS:
        if any(directory.glob(pattern)):
            return True
    return False


def is_loadable_model_dir(directory: str | os.PathLike[str]) -> bool:
    """A directory vLLM has a chance of loading: config + weight shards."""
    path = Path(directory)
    return path.is_dir() and (path / "config.json").is_file() and _has_weights(path)


def has_tokenizer(directory: str | os.PathLike[str]) -> bool:
    path = Path(directory)
    return all((path / name).is_file() for name in _TOKENIZER_FILES)


def find_checkpoints(run_dir: str | os.PathLike[str]) -> List[tuple[int, Path]]:
    """`(step, dir)` for every loadable `checkpoint-<step>/`, ascending."""
    base = Path(run_dir)
    if not base.is_dir():
        return []
    found: List[tuple[int, Path]] = []
    for child in sorted(base.iterdir()):
        match = _CHECKPOINT_DIR.match(child.name)
        if match is None or not is_loadable_model_dir(child):
            continue
        found.append((int(match.group(1)), child))
    found.sort(key=lambda item: item[0])
    return found


def reference_tasks(model_root: str | os.PathLike[str],
                    sizes: Sequence[str] = SIZES) -> List[EvalTask]:
    """The untrained Qwen3 instruct checkpoints, smallest first."""
    root = Path(model_root)
    tasks: List[EvalTask] = []
    for order, size in enumerate(sizes):
        path = root / f"Qwen3-{size}"
        if not is_loadable_model_dir(path):
            continue
        tasks.append(EvalTask(
            tag=REFERENCE_TAGS[size], run_name=REFERENCE_TAGS[size],
            kind="reference", tier=0, priority=0, size=size,
            data_variant="none", arm_family="reference", seed=None,
            save_mode="none", terminal_mode="", ckpt_step=None, is_final=True,
            model_path=str(path), tokenizer_path=str(path),
            tokenizer_is_fallback=False, order=order,
        ))
    return tasks


def trained_tasks(runs: Iterable[RunSpec], *, output_root: str | os.PathLike[str],
                  model_root: str | os.PathLike[str],
                  require_train_success: bool = True,
                  include_intermediate: bool = True) -> List[EvalTask]:
    """Every evaluable checkpoint of every finished run."""
    out_root = Path(output_root)
    tasks: List[EvalTask] = []
    for run in runs:
        run_dir = out_root / run.run_name
        if require_train_success and not (run_dir / "TRAIN_SUCCESS").is_file():
            continue
        checkpoints = find_checkpoints(run_dir)
        if not checkpoints:
            continue
        final_step = checkpoints[-1][0]
        base_model = Path(model_root) / f"Qwen3-{run.size}"
        for step, path in checkpoints:
            is_final = step == final_step
            if not is_final and not include_intermediate:
                continue
            fallback = not has_tokenizer(path)
            tasks.append(EvalTask(
                tag=run.run_name if is_final else f"{run.run_name}__step{step}",
                run_name=run.run_name,
                kind="trained",
                tier=run.priority if is_final else 90,
                priority=run.priority,
                size=run.size,
                data_variant=run.data_variant,
                arm_family=run.arm_family,
                seed=run.seed,
                save_mode=run.save_mode,
                terminal_mode=run.terminal_mode,
                ckpt_step=step,
                is_final=is_final,
                model_path=str(path),
                tokenizer_path=str(base_model if fallback else path),
                tokenizer_is_fallback=fallback,
                order=run.order,
            ))
    return tasks


def sort_key(task: EvalTask) -> tuple:
    """Reference first, then finals by training priority, then intermediates
    step-major (see the module docstring for why)."""
    if task.tier == 90:
        return (90, task.ckpt_step or 0, task.priority, task.order, task.tag)
    return (task.tier, task.priority, task.order, task.ckpt_step or 0, task.tag)


def build_queue(runs: Sequence[RunSpec], *, output_root: str | os.PathLike[str],
                model_root: str | os.PathLike[str],
                include_reference: bool = True,
                include_intermediate: bool = True,
                require_train_success: bool = True,
                priority_max: int = 99,
                only: Optional[Sequence[str]] = None,
                sizes: Sequence[str] = SIZES) -> List[EvalTask]:
    """The ordered task queue.

    ``priority_max`` filters on the TRAINING priority (so `PRIORITY_MAX=1`
    means the same set of models here as it does in `run_train_all.sh`);
    intermediates keep their run's priority for this purpose and are only
    demoted for ordering.  ``only`` matches a task tag or its run name exactly.
    """
    tasks: List[EvalTask] = []
    if include_reference:
        tasks += reference_tasks(model_root, sizes=sizes)
    tasks += trained_tasks(runs, output_root=output_root, model_root=model_root,
                           require_train_success=require_train_success,
                           include_intermediate=include_intermediate)
    tasks = [task for task in tasks if task.priority <= priority_max]
    if only:
        wanted = {value.strip() for value in only if value.strip()}
        tasks = [task for task in tasks
                 if task.tag in wanted or task.run_name in wanted]
    tasks.sort(key=sort_key)

    seen: Dict[str, EvalTask] = {}
    for task in tasks:
        if task.tag in seen:
            raise ValueError(f"duplicate eval tag {task.tag!r}: "
                             f"{seen[task.tag].model_path} vs {task.model_path}")
        seen[task.tag] = task
    return tasks


def expected_task_count(runs: Sequence[RunSpec], *, trajectory_points: int = 10,
                        priority_max: int = 99,
                        include_reference: bool = True) -> int:
    """How many tasks a fully trained matrix should yield.

    Only used by the preflight to say "disk shows 23 of an expected 63", so a
    half-finished training run is visible before the queue starts rather than
    after."""
    total = len(SIZES) if include_reference else 0
    for run in runs:
        if run.priority > priority_max:
            continue
        total += trajectory_points if run.save_mode == "trajectory" else 1
    return total
