# coding=utf-8
"""Schedule generation across GPU slots and scoring across CPU workers.

A free GPU takes the next model task. Completed generations are passed to
the analysis worker pool, allowing generation to continue independently.

Progress is recorded in these files under the evaluation output root::

    responses/<tag>_nothink_n8.jsonl
    responses/<tag>_nothink_n8.done.json
    summary/<tag>_summary.json

Restarting skips finished tasks, resumes partially written response files,
and submits completed generations directly to analysis when needed.
``--follow`` rescans training outputs for ``TRAIN_SUCCESS`` sentinels.
Concurrent training and evaluation require separate GPU allocations.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import aggregate, protocol
from .identity import (
    build_identity, describe_mismatch, identity_mismatches, load_identity,
)
from .io_utils import (
    append_csv_row, ensure_dir, read_csv_rows, read_json, sanitize_filename,
    sha256_file, write_json,
)
from .registry import EvalTask, build_queue, expected_task_count, parse_run_matrix

LOG = logging.getLogger("e1.orchestrator")

LEDGER_FIELDS = ["tag", "stage", "attempt", "gpu", "started", "ended",
                 "minutes", "exit_code", "status", "model_path", "note"]


# --------------------------------------------------------------- task state

@dataclasses.dataclass
class TaskPaths:
    responses: Path
    sentinel: Path
    run_manifest: Path
    summary: Path
    task_json: Path

    @classmethod
    def build(cls, root: Path, tag: str) -> "TaskPaths":
        safe = sanitize_filename(tag)
        responses = root / "responses" / f"{safe}_{protocol.MODE}_n{protocol.K}.jsonl"
        return cls(responses=responses,
                   sentinel=responses.with_suffix(".done.json"),
                   run_manifest=responses.with_suffix(".run.json"),
                   summary=root / "summary" / f"{safe}_summary.json",
                   task_json=root / "tasks" / f"{safe}.json")


def task_stage(paths: TaskPaths) -> str:
    """`done` / `analyze` / `generate`, read off the artefacts."""
    if paths.summary.is_file():
        return "done"
    if paths.sentinel.is_file():
        return "analyze"
    return "generate"


#: Windows has no SIGKILL, and the process-group path is POSIX-only anyway;
#: resolving the numbers once keeps the code importable (and testable) there.
_SIGTERM = getattr(signal, "SIGTERM", None)
_SIGKILL = getattr(signal, "SIGKILL", _SIGTERM)


def _group_is_alive(pgid: Optional[int]) -> bool:
    """Does anything still exist in this process group?  (POSIX only.)"""
    if not pgid or os.name == "nt" or not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(pgid, 0)          # signal 0 = existence check only
        return True
    except ProcessLookupError:
        return False
    except PermissionError:         # exists, owned by someone else
        return True
    except OSError:
        return False


def _signal_group(pgid: Optional[int], sig: int) -> bool:
    if not pgid or os.name == "nt" or not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(pgid, sig)
        return True
    except (OSError, ProcessLookupError):
        return False


def sweep_group(pgid: Optional[int], *, label: str = "",
                grace: float = 5.0) -> bool:
    """Clean up a worker's process group after the wrapper itself is gone.

    This is the case `terminate_tree` cannot cover: the wrapper died on its own
    -- OOM-killed, SIGKILLed, or crashed -- while the vLLM workers it spawned
    are still running and still holding the card.  Nothing would ever signal
    them again, and every later task assigned to that GPU would OOM.

    Reaping the wrapper does not make the saved pgid unsafe to use: a pid stays
    reserved while it is still the group id of living processes, so `killpg`
    here either reaches that worker's own group or fails with ESRCH.
    Returns True when orphans were actually found.
    """
    if not _group_is_alive(pgid):
        return False
    LOG.warning("%s exited but its process group %s is still alive; "
                "terminating the orphaned worker(s)", label or "worker", pgid)
    _signal_group(pgid, _SIGTERM)
    deadline = time.time() + grace
    while time.time() < deadline and _group_is_alive(pgid):
        time.sleep(0.2)
    if _group_is_alive(pgid):
        _signal_group(pgid, _SIGKILL)
        time.sleep(0.5)
    if _group_is_alive(pgid):
        LOG.error("process group %s survived SIGKILL; check nvidia-smi -- a "
                  "stuck worker still holds its GPU", pgid)
    return True


def terminate_tree(process: subprocess.Popen, *, pgid: Optional[int] = None,
                   label: str = "", grace: float = 20.0) -> None:
    """Kill a worker AND its children, then make sure they are really gone.

    A vLLM worker spawns its own processes, and each of them holds a share of a
    card.  `Popen.terminate()` signals only the wrapper, so a stopped
    orchestrator would leave the GPU occupied and every later task on that card
    would OOM.  On POSIX the children are in their own session
    (`start_new_session=True`), so the whole group can be signalled at once;
    SIGTERM first so vLLM can release memory, SIGKILL if it does not.

    ``pgid`` is captured at spawn time.  It is what makes the group reachable
    even when the wrapper is already gone, which is exactly when orphans exist.
    """
    if process.poll() is not None:
        sweep_group(pgid, label=label)
        return
    if not _signal_group(pgid, _SIGTERM):
        try:
            process.terminate()
        except OSError:
            return
    try:
        process.wait(timeout=grace)
        sweep_group(pgid, label=label)
        return
    except subprocess.TimeoutExpired:
        LOG.warning("%s did not exit within %.0fs; sending SIGKILL",
                    label or process.pid, grace)
    if not _signal_group(pgid, _SIGKILL):
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        LOG.error("%s survived SIGKILL; check nvidia-smi for a stuck worker",
                  label or process.pid)
    sweep_group(pgid, label=label)


# ------------------------------------------------------------------- slots

@dataclasses.dataclass
class Running:
    task: EvalTask
    stage: str
    process: subprocess.Popen
    gpu: Optional[str]
    started: float
    attempt: int
    log_path: Path
    #: Captured at spawn, while the wrapper is certainly alive.  Reading it
    #: later (`os.getpgid`) would fail exactly in the case that matters: the
    #: wrapper died and left its vLLM workers holding the card.
    pgid: Optional[int] = None


class Orchestrator:
    def __init__(self, args: argparse.Namespace, tasks: Sequence[EvalTask]) -> None:
        self.args = args
        self.root = Path(args.result_root)
        self.tasks = {task.tag: task for task in tasks}
        self.order = [task.tag for task in tasks]
        self.pending_gen: List[str] = []
        self.pending_analyze: List[str] = []
        self.running: List[Running] = []
        self.attempts: Dict[tuple, int] = {}
        self.failed: Dict[str, str] = {}
        self.completed: List[str] = []
        self.stopping = False
        self.ledger = self.root / "e1_eval_ledger.csv"
        self.csv = Path(args.csv) if args.csv else self.root / "e1_metrics.csv"
        self._identity_checked: Dict[str, str | None] = {}
        self._eval_sha: str | None = None
        for name in ("responses", "summary", "events", "audit", "logs", "tasks"):
            ensure_dir(self.root / name)
        self._seed_attempts_from_ledger()

    # -- identity ---------------------------------------------------------

    def eval_sha(self) -> str:
        """Hash of the evaluation file, computed once (it is 33 MB)."""
        if self._eval_sha is None:
            self._eval_sha = sha256_file(self.args.eval_data)
        return self._eval_sha

    def expected_identity(self, task: EvalTask) -> Dict[str, Any]:
        return build_identity(eval_data_sha256=self.eval_sha(),
                              model_path=task.model_path,
                              tokenizer_path=task.tokenizer_path,
                              max_model_len=self.args.max_model_len,
                              limit=self.args.limit)

    def identity_error(self, task: EvalTask) -> str | None:
        """Why an existing artefact of this task may NOT be reused, if so.

        Cached per tag: the artefacts do not change under us, and hashing the
        evaluation file on every poll would be absurd.  The cache is cleared
        when this process launches the task itself -- what we write is
        compatible by construction.
        """
        if task.tag in self._identity_checked:
            return self._identity_checked[task.tag]
        paths = TaskPaths.build(self.root, task.tag)
        present = [path for path in
                   (paths.summary, paths.sentinel, paths.run_manifest,
                    paths.responses)
                   if path.is_file() and path.stat().st_size > 0]
        verdict: str | None = None
        if present:
            source = next((path for path in present
                           if load_identity(path) is not None), None)
            if source is None:
                verdict = (
                    f"[{task.tag}] artefacts exist ({', '.join(p.name for p in present)}) "
                    "but none records which run produced them, so they cannot be "
                    "shown to belong to this one.  Delete this tag's entries under "
                    "responses/, events/ and summary/ to regenerate.")
            else:
                problems = identity_mismatches(self.expected_identity(task),
                                               load_identity(source) or {})
                if problems:
                    verdict = describe_mismatch(task.tag, source, problems)
        self._identity_checked[task.tag] = verdict
        return verdict

    def _seed_attempts_from_ledger(self) -> None:
        """Carry failed-attempt counts across restarts.

        Without this, `--max_attempts` only bounds retries inside one process
        and a deterministically failing task is retried forever by anyone who
        restarts the orchestrator.
        """
        for row in read_csv_rows(self.ledger):
            if row.get("status") != "failed":
                continue
            key = (row.get("tag", ""), row.get("stage", ""))
            self.attempts[key] = self.attempts.get(key, 0) + 1
        exhausted = {tag for (tag, _stage), count in self.attempts.items()
                     if count >= self.args.max_attempts}
        for tag in sorted(exhausted):
            self.failed[tag] = (f"already failed {self.args.max_attempts} time(s) "
                                "in e1_eval_ledger.csv")
        if exhausted:
            LOG.warning("%d task(s) already used up their attempts in a previous "
                        "run and will be skipped: %s.  Fix them and delete their "
                        "ledger rows (or raise --max_attempts) to retry.",
                        len(exhausted), ", ".join(sorted(exhausted)))

    # -- queueing ---------------------------------------------------------

    def refresh_queue(self, tasks: Sequence[EvalTask] | None = None) -> None:
        """(Re)compute what still needs doing, preserving priority order."""
        if tasks is not None:
            for task in tasks:
                if task.tag not in self.tasks:
                    self.tasks[task.tag] = task
                    self.order.append(task.tag)
                    LOG.info("new task discovered: %s", task.tag)
        busy = {(item.task.tag, item.stage) for item in self.running}
        gen: List[str] = []
        analyze: List[str] = []
        for tag in self.order:
            if tag in self.failed:
                continue
            paths = TaskPaths.build(self.root, tag)
            if (tag, "generate") not in busy and (tag, "analyze") not in busy:
                problem = self.identity_error(self.tasks[tag])
                if problem:
                    LOG.error("%s", problem)
                    self.failed[tag] = "artefact from an incompatible run"
                    continue
            stage = task_stage(paths)
            if stage == "done":
                if tag not in self.completed:
                    self.completed.append(tag)
                continue
            if (tag, stage) in busy or (tag, "generate") in busy:
                continue
            if stage == "analyze" and not self.args.gen_only:
                analyze.append(tag)
            elif stage == "generate" and not self.args.analyze_only:
                gen.append(tag)
        self.pending_gen = gen
        self.pending_analyze = analyze

    # -- launching --------------------------------------------------------

    def _child_env(self) -> Dict[str, str]:
        """Child environment.  CUDA_VISIBLE_DEVICES is deliberately NOT set
        here: the generation worker sets it itself from `--gpu`, before vLLM is
        imported, and masking twice would make the worker select device 3 of an
        already one-device view."""
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _spawn(self, task: EvalTask, stage: str, gpu: Optional[str]) -> None:
        paths = TaskPaths.build(self.root, task.tag)
        paths.task_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(paths.task_json, task.as_dict())
        attempt = self.attempts.get((task.tag, stage), 0) + 1
        self.attempts[(task.tag, stage)] = attempt
        self._identity_checked[task.tag] = None
        log_path = self.root / "logs" / f"{stage}_{sanitize_filename(task.tag)}.out"

        if stage == "generate":
            command = [
                sys.executable, "-m", "tcr.evaluation.generate",
                "--eval_data", self.args.eval_data,
                "--output_dir", str(self.root / "responses"),
                "--model_path", task.model_path,
                "--tokenizer_path", task.tokenizer_path,
                "--tag", task.tag,
                "--gpu", str(gpu),
                "--max_model_len", str(self.args.max_model_len),
                "--gpu_memory_utilization", str(self.args.gpu_memory_utilization),
                "--batch_records", str(self.args.batch_records),
                "--log_file", str(self.root / "logs" /
                                  f"gen_{sanitize_filename(task.tag)}.log"),
            ]
            if self.args.limit:
                command += ["--limit", str(self.args.limit)]
            if self.args.max_num_seqs:
                command += ["--max_num_seqs", str(self.args.max_num_seqs)]
        else:
            command = [
                sys.executable, "-m", "tcr.evaluation.analyze",
                "--responses", str(paths.responses),
                "--eval_data", self.args.eval_data,
                "--tokenizer_path", task.tokenizer_path,
                "--output_root", str(self.root),
                "--tag", task.tag,
                "--task_json", str(paths.task_json),
                "--audit_samples", str(self.args.audit_samples),
                "--bootstrap", str(self.args.bootstrap),
                "--log_file", str(self.root / "logs" /
                                  f"analyze_{sanitize_filename(task.tag)}.log"),
            ]
            if self.args.keep_diagnostics:
                command.append("--keep_diagnostics")

        LOG.info("launch %-8s %-42s %s", stage, task.tag,
                 f"gpu={gpu}" if gpu else "cpu")
        handle = open(log_path, "a", encoding="utf-8")
        handle.write(f"\n===== {stage} attempt {attempt} "
                     f"{time.strftime('%F %T')} =====\n")
        handle.flush()
        process = subprocess.Popen(
            command, cwd=str(Path(__file__).resolve().parents[2]),
            env=self._child_env(),
            stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"))
        process._e1_log_handle = handle          # type: ignore[attr-defined]
        pgid: Optional[int] = None
        if os.name != "nt" and hasattr(os, "getpgid"):
            try:
                pgid = os.getpgid(process.pid)
            except OSError:                      # already gone; nothing to sweep
                pgid = None
        self.running.append(Running(task=task, stage=stage, process=process,
                                    gpu=gpu, started=time.time(),
                                    attempt=attempt, log_path=log_path,
                                    pgid=pgid))

    def free_gpus(self) -> List[str]:
        used = {item.gpu for item in self.running if item.stage == "generate"}
        return [gpu for gpu in self.args.gpu_list if gpu not in used]

    def n_analysis_running(self) -> int:
        return sum(1 for item in self.running if item.stage == "analyze")

    def schedule(self) -> None:
        if self.stopping:
            return
        # Analysis first: it frees nothing but it is what turns a finished
        # generation into a CSV row, and it must not fall behind the GPUs.
        while (self.pending_analyze
               and self.n_analysis_running() < self.args.analysis_workers):
            self._spawn(self.tasks[self.pending_analyze.pop(0)], "analyze", None)
        now = time.time()
        for gpu in self.free_gpus():
            if not self.pending_gen:
                break
            if now - getattr(self, "_last_launch", 0.0) < self.args.launch_stagger:
                break
            self._spawn(self.tasks[self.pending_gen.pop(0)], "generate", gpu)
            self._last_launch = time.time()
            now = self._last_launch

    # -- reaping ----------------------------------------------------------

    def reap(self) -> bool:
        """Collect finished children; True when at least one finished."""
        finished = [item for item in self.running
                    if item.process.poll() is not None]
        for item in finished:
            self.running.remove(item)
            code = item.process.returncode
            handle = getattr(item.process, "_e1_log_handle", None)
            if handle is not None:
                handle.close()
            # The wrapper is gone; anything left in its group is an orphan
            # holding a GPU, whether it exited cleanly or was OOM-killed.
            orphaned = sweep_group(item.pgid,
                                   label=f"{item.stage}:{item.task.tag}")
            minutes = (time.time() - item.started) / 60.0
            paths = TaskPaths.build(self.root, item.task.tag)
            # The sentinel, not the exit code, is the proof of completion: a
            # worker that exited 0 without writing it did not do the job.
            produced = (paths.sentinel.is_file() if item.stage == "generate"
                        else paths.summary.is_file())
            status = "success" if code == 0 and produced else "failed"
            note = "" if produced else "expected artefact missing"
            if orphaned:
                note = (note + "; " if note else "") + "orphaned workers swept"
            self._ledger(item, status, code, minutes, note)

            if status == "success":
                LOG.info("done  %-8s %-42s %6.1f min", item.stage, item.task.tag,
                         minutes)
                if item.stage == "analyze":
                    self._append_csv(item.task, paths)
            else:
                LOG.error("FAIL  %-8s %-42s exit=%s after %.1f min (%s) -- see %s",
                          item.stage, item.task.tag, code, minutes, note,
                          item.log_path)
                if self.attempts[(item.task.tag, item.stage)] >= self.args.max_attempts:
                    self.failed[item.task.tag] = f"{item.stage} exit={code}"
                    LOG.error("giving up on %s after %d attempt(s)",
                              item.task.tag, self.args.max_attempts)
        return bool(finished)

    def _append_csv(self, task: EvalTask, paths: TaskPaths) -> None:
        try:
            summary = read_json(paths.summary)
            row = aggregate.summary_to_row(summary)
            append_csv_row(self.csv, row, aggregate.CSV_FIELDS)
            events = summary.get("events", {})
            LOG.info("  %-42s capture=%.4f orbit=%.4f loop=%.4f strictF1=%.4f",
                     task.tag, events.get("semantic_capture_rate", float("nan")),
                     events.get("stable_orbit_rate", float("nan")),
                     events.get("legacy_loop_rate", float("nan")),
                     summary.get("quality", {}).get("strict_f1", float("nan")))
        except Exception as exc:                      # noqa: BLE001
            LOG.exception("could not append CSV row for %s: %s", task.tag, exc)

    def _ledger(self, item: Running, status: str, code: int, minutes: float,
                note: str) -> None:
        append_csv_row(self.ledger, {
            "tag": item.task.tag, "stage": item.stage, "attempt": item.attempt,
            "gpu": item.gpu or "", "ended": time.strftime("%F %T"),
            "started": time.strftime("%F %T", time.localtime(item.started)),
            "minutes": round(minutes, 1), "exit_code": code, "status": status,
            "model_path": item.task.model_path, "note": note,
        }, LEDGER_FIELDS)

    # -- lifecycle --------------------------------------------------------

    def stop(self, *_signal: Any) -> None:
        if self.stopping:
            return
        self.stopping = True
        LOG.warning("stop requested: terminating %d child process(es); "
                    "finished work is on disk and a restart resumes",
                    len(self.running))
        for item in self.running:
            terminate_tree(item.process, pgid=item.pgid,
                           label=f"{item.stage}:{item.task.tag}")

    def run(self) -> int:
        self.refresh_queue()
        total = len(self.order)
        LOG.info("queue: %d task(s), %d already complete, gpus=%s, "
                 "analysis_workers=%d", total, len(self.completed),
                 ",".join(self.args.gpu_list), self.args.analysis_workers)
        last_beat = 0.0
        last_scan = time.time()

        while not self.stopping:
            self.schedule()
            if not self.running and not self.pending_gen and not self.pending_analyze:
                if not self.args.follow:
                    break
                if time.time() - last_scan >= self.args.follow_interval:
                    self._rescan()
                    last_scan = time.time()
                    continue
            time.sleep(self.args.poll)
            self.reap()
            self.refresh_queue()
            if (self.args.follow
                    and time.time() - last_scan >= self.args.follow_interval):
                self._rescan()
                last_scan = time.time()
            if time.time() - last_beat >= self.args.heartbeat:
                self._heartbeat(total)
                last_beat = time.time()

        while self.running:                       # drain after a stop request
            self.reap()
            time.sleep(1.0)

        LOG.info("finished: %d/%d complete, %d failed", len(self.completed),
                 total, len(self.failed))
        for tag, reason in self.failed.items():
            LOG.error("  failed: %-42s %s", tag, reason)
        return 1 if self.failed else 0

    def _rescan(self) -> None:
        try:
            runs = parse_run_matrix(self.args.train_script)
            self.refresh_queue(build_queue(
                runs, output_root=self.args.train_output_root,
                model_root=self.args.model_root,
                include_reference=not self.args.no_reference,
                include_intermediate=not self.args.finals_only,
                priority_max=self.args.priority_max,
                only=self.args.only))
        except Exception as exc:                      # noqa: BLE001
            LOG.warning("registry rescan failed (keeping current queue): %s", exc)

    def _heartbeat(self, total: int) -> None:
        active = ", ".join(
            f"{item.stage[0]}:{item.task.tag}"
            f"@{(time.time() - item.started) / 60.0:.0f}m"
            for item in self.running) or "idle"
        LOG.info("[status] done=%d/%d failed=%d queued(gen=%d,analyze=%d) | %s",
                 len(self.completed), total, len(self.failed),
                 len(self.pending_gen), len(self.pending_analyze), active)


# ---------------------------------------------------------------------- CLI

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the whole E1 evaluation matrix unattended: 8 GPU "
                    "generation slots + a CPU analysis pool + one CSV.")
    parser.add_argument("--eval_data", required=True,
                        help="E-Natural: cleanv2/eval_supportclean_keep8.jsonl")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--train_script", required=True,
                        help="run_train_all.sh; the training matrix is read from it.")
    parser.add_argument("--train_output_root", required=True,
                        help="Where run_train_all.sh wrote its runs.")
    parser.add_argument("--model_root", required=True,
                        help="Directory holding Qwen3-1.7B / Qwen3-4B / Qwen3-8B.")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--analysis_workers", type=int, default=4)
    parser.add_argument("--priority_max", type=int, default=99)
    parser.add_argument("--only", default=None,
                        help="Comma-separated tags or run names to restrict to.")
    parser.add_argument("--finals_only", action="store_true",
                        help="Skip intermediate trajectory checkpoints (E1 "
                             "itself needs only the final ones).")
    parser.add_argument("--no_reference", action="store_true",
                        help="Skip the three untrained Qwen3 models.")
    parser.add_argument("--allow_unfinished_runs", action="store_true",
                        help="Evaluate checkpoints of runs with no "
                             "TRAIN_SUCCESS sentinel (unsafe while training).")
    parser.add_argument("--csv", default=None,
                        help="Unified metrics CSV (default result_root/e1_metrics.csv).")
    parser.add_argument("--max_model_len", type=int, default=protocol.MAX_MODEL_LEN)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--batch_records", type=int, default=64)
    parser.add_argument("--audit_samples", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=protocol.BOOTSTRAP)
    parser.add_argument("--keep_diagnostics", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test: only the first N eval records.")
    parser.add_argument("--max_attempts", type=int, default=2)
    parser.add_argument("--poll", type=float, default=10.0)
    parser.add_argument("--heartbeat", type=float, default=600.0)
    parser.add_argument("--launch_stagger", type=float, default=20.0,
                        help="Seconds between two vLLM launches; loading eight "
                             "models at once thrashes page cache and CPU.")
    parser.add_argument("--follow", action="store_true",
                        help="Keep re-scanning for runs that finish training.")
    parser.add_argument("--follow_interval", type=float, default=300.0)
    parser.add_argument("--gen_only", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--list", action="store_true",
                        help="Print the queue and exit.")
    parser.add_argument("--log_file", default=None)
    args = parser.parse_args(argv)
    args.gpu_list = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not args.gpu_list and not args.analyze_only:
        parser.error("--gpus is empty; nothing can generate")
    args.analysis_workers = max(1, int(args.analysis_workers))
    args.only = ([value.strip() for value in args.only.split(",")]
                 if args.only else None)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log_file = args.log_file or str(Path(args.result_root) / "logs" /
                                    "orchestrator.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_file, encoding="utf-8")],
        force=True)

    runs = parse_run_matrix(args.train_script)
    tasks = build_queue(runs, output_root=args.train_output_root,
                        model_root=args.model_root,
                        include_reference=not args.no_reference,
                        include_intermediate=not args.finals_only,
                        require_train_success=not args.allow_unfinished_runs,
                        priority_max=args.priority_max, only=args.only)
    expected = expected_task_count(runs, priority_max=args.priority_max,
                                   include_reference=not args.no_reference)
    LOG.info("training matrix: %d run(s); evaluable now: %d task(s) "
             "(a fully trained matrix yields about %d)",
             len(runs), len(tasks), expected)

    if args.list:
        print(f"{'tier':>4} {'pri':>3} {'tag':<44} {'size':<5} "
              f"{'variant':<20} {'step':>6} {'final':<5} model_path")
        for task in tasks:
            print(f"{task.tier:>4} {task.priority:>3} {task.tag:<44} "
                  f"{task.size:<5} {task.data_variant:<20} "
                  f"{task.ckpt_step if task.ckpt_step is not None else '-':>6} "
                  f"{str(task.is_final):<5} {task.model_path}")
        return 0

    if not tasks:
        LOG.error("nothing to evaluate: no reference model and no run with a "
                  "TRAIN_SUCCESS sentinel under %s", args.train_output_root)
        return 1

    orchestrator = Orchestrator(args, tasks)
    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), orchestrator.stop)
    try:
        return orchestrator.run()
    finally:
        # An orphaned vLLM worker holds a whole card hostage, so never leave
        # one behind on an unexpected exit.  (A SIGKILL of this process is the
        # one case nothing can cover: then `nvidia-smi` and `pkill -f
        # tcr.evaluation.generate` before restarting.)
        for item in list(orchestrator.running):
            terminate_tree(item.process, pgid=item.pgid,
                           label=f"{item.stage}:{item.task.tag}")


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
