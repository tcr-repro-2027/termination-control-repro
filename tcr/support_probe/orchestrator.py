# coding=utf-8
"""Run the whole E2 matrix unattended: 8 GPU slots + a CPU analysis pool.

Same shape as `tcr/evaluation/tcr/evaluation/orchestrator.py`, and for the same reason:
the tasks have very unequal cost, so a "launch everything, then wait" script
leaves most cards idle for most of the night.  Each GPU pulls the next stage the
moment it frees up; analysis is CPU-only and never blocks a card.

The model list is READ FROM E1's registry, not maintained here.  E2 must score
exactly the checkpoints E1 scored or the two experiments stop being comparable,
and a second hand-kept model list is the obvious way for that to rot.  Only
FINAL checkpoints are used -- §E2 is about models, not trajectories.

Stages per model:
    score    GPU, HuggingFace  -> readouts/<tag>_readouts.jsonl (+ .done.json)
    hazard   GPU, vLLM         -> hazard/<tag>_hazard.jsonl     (optional)
    analyze  CPU               -> analysis/<tag>_e2_summary.json

State lives on disk, so a restart resumes: a stage whose sentinel PROVES it
already answered this run's question is skipped, and `analyze` becomes
available as soon as its model's `score` lands.

"Proves" is the load-bearing word.  The workers check identity when they start,
but a skipped stage never starts one -- so rebuilt anchors, a changed
`--hazard_arms`, a moved checkpoint or a bumped protocol version would have
been silently satisfied by last week's sentinel.  :meth:`Orchestrator.survey`
therefore re-derives the identity of every existing sentinel BEFORE any GPU
time is spent.

A stale sentinel gets one of two treatments, and the difference matters:

* score and hazard OWN a data file that resume appends to, so re-running them
  needs the operator to remove that file too; the run REFUSES to start and says
  which files;
* analyze only DERIVES a summary from those files and rewrites it wholesale, so
  a summary that no longer describes its inputs is simply recomputed.  That is
  also why an analyze summary is never trusted while its own score/hazard stage
  is pending: deleting a score sentinel to force a re-score used to leave the
  old summary standing, and nothing recomputed it afterwards.
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

from .identity import mismatches, run_identity, sentinel_mismatches
from .io_utils import (
    append_csv_row, ensure_dir, read_json,
    sanitize_filename, sha256_file, write_json,
)
from . import protocol

from tcr.evaluation.registry import build_queue, parse_run_matrix  # noqa: E402

LOG = logging.getLogger("tcr.support_probe.orchestrator")

LEDGER_FIELDS = ["tag", "stage", "attempt", "gpu", "started", "ended",
                 "minutes", "exit_code", "status", "model_path", "note"]

GPU_STAGES = ("score", "hazard")


def sentinel_for(root: Path, tag: str, stage: str) -> Path:
    safe = sanitize_filename(tag)
    if stage == "score":
        return root / "readouts" / f"{safe}_readouts.done.json"
    if stage == "hazard":
        return root / "hazard" / f"{safe}_hazard.done.json"
    return root / "analysis" / f"{safe}_e2_summary.json"


@dataclasses.dataclass
class Running:
    tag: str
    stage: str
    task: Any
    process: subprocess.Popen
    gpu: Optional[str]
    started: float
    attempt: int
    log_path: Path
    pgid: Optional[int] = None


def _group_alive(pgid: Optional[int]) -> bool:
    if not pgid or os.name == "nt" or not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def sweep_group(pgid: Optional[int], label: str = "") -> bool:
    """Kill anything the wrapper left behind.

    vLLM spawns its own workers; if the wrapper dies on its own they keep the
    card and every later task on it OOMs."""
    if not _group_alive(pgid):
        return False
    LOG.warning("%s exited but its process group %s is alive; terminating",
                label, pgid)
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGKILL", None)):
        if sig is None:
            continue
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, OSError):
            return True
        deadline = time.time() + 5.0
        while time.time() < deadline and _group_alive(pgid):
            time.sleep(0.2)
        if not _group_alive(pgid):
            break
    return True


class Orchestrator:
    def __init__(self, args: argparse.Namespace, tasks: Sequence[Any]) -> None:
        self.args = args
        self.root = Path(args.result_root)
        self.tasks = {task.tag: task for task in tasks}
        self.order = [task.tag for task in tasks]
        self.running: List[Running] = []
        self.attempts: Dict[tuple, int] = {}
        self.failed: Dict[tuple, str] = {}
        self.done: set = set()
        self.stopping = False
        self.ledger = self.root / "e2_run_ledger.csv"
        self.csv = Path(args.csv) if args.csv else self.root / "e2_metrics.csv"
        # Hashed once: the anchor file is tens of megabytes and the identity of
        # every (model, stage) pair contains its digest.
        if not Path(args.anchors).is_file():
            raise SystemExit(f"anchor set missing: {args.anchors}\n"
                             "  build it first: bash run_e2_all.sh --build")
        self.anchors_sha256 = sha256_file(args.anchors)
        self._verified_done: set = set()
        #: Stages a check has already rejected.  Once rejected, a stage stays
        #: not-done until THIS process runs it: re-checking would let it come
        #: back the moment its prerequisite finished, which is exactly the
        #: forced-recompute hole (delete the score sentinel, keep the summary).
        self._invalidated: set = set()
        self._completed: set = set()
        self.will_recompute: List[str] = []
        for name in ("readouts", "hazard", "analysis", "logs", "tasks"):
            ensure_dir(self.root / name)

    # -- what "already done" is allowed to mean ---------------------------

    def identity_for(self, tag: str, stage: str) -> Dict[str, Any]:
        """The identity a fresh run of this stage would stamp on its sentinel."""
        task = self.tasks[tag]
        hazard = stage == "hazard"
        return run_identity(
            anchors_path=self.args.anchors,
            model_path=task.model_path,
            tokenizer_path=task.tokenizer_path,
            limit=None,
            arms=self.args.hazard_arms if hazard else None,
            # the worker's own default; the orchestrator does not override it,
            # so a hand-run hazard with a different ceiling is a mismatch
            max_model_len=protocol.MAX_MODEL_LEN if hazard else None,
            anchors_sha256=self.anchors_sha256)

    def upstream_of(self, stage: str) -> List[str]:
        """Stages whose output this one is derived from."""
        if stage != "analyze":
            return []
        return ["score"] if self.args.no_hazard else ["score", "hazard"]

    def _analysis_mismatches(self, tag: str) -> List[str]:
        """An existing summary is only reusable if it summarises THIS run.

        Its sentinel is the summary itself, which carries no identity block, so
        what is checked instead is everything that decides whether it describes
        this measurement:

        * the anchor set and protocol it was measured under;
        * whether it holds the hazard column this run is producing -- a
          `NO_HAZARD=1` pass followed by a full one would otherwise keep the
          margin-only summary and silently drop gate 7;
        * which ANALYSIS produced it.  `summary_schema` moves independently of
          the measurement protocol: the anchor-clustered hazard interval and
          the occurrence-matched sensitivity changed what a summary means
          without touching a single readout, so an older one has a perfectly
          matching measurement identity and incomparable numbers;
        * whether its own score/hazard stage is COMPLETE for this run.  A
          summary is derived data; while the stage it derives from is pending,
          it necessarily describes an earlier one.  Deleting a score sentinel
          to force a re-score used to leave the old summary marked done, so the
          re-scored readouts were never re-summarised;
        * WHICH score/hazard run it read.  Required, not optional: a summary
          that does not say is a summary that cannot be shown to be current,
          and recomputing one is seconds of CPU.
        """
        sentinel = sentinel_for(self.root, tag, "analyze")
        try:
            summary = read_json(sentinel)
        except (OSError, ValueError) as exc:
            return [f"summary is unreadable: {exc}"]
        problems: List[str] = []
        found = summary.get("anchors_sha256")
        if found != self.anchors_sha256:
            problems.append(f"anchors_sha256: existing {str(found)[:12]!r} != "
                            f"current {self.anchors_sha256[:12]!r}")
        version = (summary.get("protocol") or {}).get("protocol_version")
        if version != protocol.PROTOCOL_VERSION:
            problems.append(f"protocol_version: existing {version!r} != "
                            f"current {protocol.PROTOCOL_VERSION!r}")
        schema = summary.get("summary_schema")
        if schema != protocol.SUMMARY_SCHEMA_VERSION:
            problems.append(
                f"summary_schema: existing {schema!r} != current "
                f"{protocol.SUMMARY_SCHEMA_VERSION!r} (produced by different "
                "analysis code)")
        if not self.args.no_hazard and not summary.get("has_hazard"):
            problems.append("summary has no hazard column but this run measures "
                            "one (gate 7 would stay missing)")
        for stage in self.upstream_of("analyze"):
            if not self.stage_is_done(tag, stage):
                problems.append(f"its {stage} stage is not complete for this "
                                "run, so this summary is of an earlier one")
                continue
            recorded = summary.get(f"{stage}_identity")
            if not recorded:
                problems.append(f"does not record which {stage} run it read")
            else:
                problems += [f"{stage}.{detail}" for detail in
                             mismatches(self.identity_for(tag, stage), recorded)]
        return problems

    def stage_mismatches(self, tag: str, stage: str) -> List[str]:
        """Why the finished stage on disk is not this run's, if it is not."""
        if stage == "analyze":
            return self._analysis_mismatches(tag)
        return sentinel_mismatches(sentinel_for(self.root, tag, stage),
                                   self.identity_for(tag, stage))

    def stage_is_done(self, tag: str, stage: str) -> bool:
        """True only when a finished stage answered THIS run's question.

        Each verdict is remembered.  A rejection is remembered too, and that is
        deliberate: an analyze summary rejected because its score was pending
        must not become "done" again the moment that score finishes -- the
        whole point is that the re-scored readouts need re-summarising."""
        key = (tag, stage)
        if key in self._completed:
            return True
        if key in self._invalidated:
            return False
        if key in self._verified_done:
            return True
        if not sentinel_for(self.root, tag, stage).is_file():
            return False
        if self.stage_mismatches(tag, stage):
            self._invalidated.add(key)
            return False
        self._verified_done.add(key)
        self.done.add(key)
        return True

    def survey(self) -> List[str]:
        """Sentinels that claim a stage is done but describe another run.

        Returns only the BLOCKING ones -- score and hazard.  Those own an
        append-only data file that a re-run would resume into, so `guard_partial`
        would abort the worker anyway; the operator has to remove the file as
        well, and has to be told before the queue starts rather than one failure
        at a time.

        A stale analyze summary is not blocking: it is derived data, rewritten
        whole, so it is simply queued for recomputation and listed in
        :attr:`will_recompute`.
        """
        blocking: List[str] = []
        self.will_recompute = []
        # Prerequisites first: an analyze summary's validity depends on them.
        for stage in ("score", "hazard", "analyze"):
            for tag in self.order:
                if stage not in self.stages_for(tag):
                    continue
                if not sentinel_for(self.root, tag, stage).is_file():
                    continue
                if self.stage_is_done(tag, stage):
                    continue
                problems = self.stage_mismatches(tag, stage)
                entry = (f"{tag} [{stage}] {sentinel_for(self.root, tag, stage)}"
                         "\n      " + "\n      ".join(problems))
                (self.will_recompute if stage == "analyze"
                 else blocking).append(entry)
        return blocking

    # -- queue ------------------------------------------------------------

    def stages_for(self, tag: str) -> List[str]:
        stages = ["score"]
        if not self.args.no_hazard:
            stages.append("hazard")
        stages.append("analyze")
        return stages

    def pending(self) -> Dict[str, List[str]]:
        busy = {(item.tag, item.stage) for item in self.running}
        gpu: List[str] = []
        cpu: List[str] = []
        for tag in self.order:
            for stage in self.stages_for(tag):
                key = (tag, stage)
                if key in busy or key in self.failed:
                    continue
                if self.stage_is_done(tag, stage):
                    continue
                # Analysis needs its model's readouts -- and, when the hazard
                # is enabled, the hazard too: running earlier would silently
                # summarise the margins with an empty behaviour column.
                if stage == "analyze":
                    required = (["score"] if self.args.no_hazard
                                else ["score", "hazard"])
                    if any(not self.stage_is_done(tag, name)
                           for name in required):
                        continue
                (gpu if stage in GPU_STAGES else cpu).append(f"{tag}|{stage}")
        return {"gpu": gpu, "cpu": cpu}

    # -- launching --------------------------------------------------------

    def _command(self, tag: str, stage: str, gpu: Optional[str]) -> List[str]:
        task = self.tasks[tag]
        common = ["--anchors", self.args.anchors, "--tag", tag]
        if stage == "score":
            return [sys.executable, "-m", "tcr.support_probe.score", *common,
                    "--model_path", task.model_path,
                    "--tokenizer_path", task.tokenizer_path,
                    "--output_root", str(self.root), "--gpu", str(gpu),
                    "--batch_size", str(self.args.batch_size),
                    "--log_file", str(self.root / "logs" /
                                      f"score_{sanitize_filename(tag)}.log")]
        if stage == "hazard":
            return [sys.executable, "-m", "tcr.support_probe.hazard", *common,
                    "--model_path", task.model_path,
                    "--tokenizer_path", task.tokenizer_path,
                    "--output_root", str(self.root), "--gpu", str(gpu),
                    "--gpu_memory_utilization",
                    str(self.args.gpu_memory_utilization),
                    "--arms", self.args.hazard_arms,
                    "--log_file", str(self.root / "logs" /
                                      f"hazard_{sanitize_filename(tag)}.log")]
        safe = sanitize_filename(tag)
        task_json = self.root / "tasks" / f"{safe}.json"
        task_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(task_json, task.as_dict())
        command = [sys.executable, "-m", "tcr.support_probe.analyze",
                   "--readouts",
                   str(self.root / "readouts" / f"{safe}_readouts.jsonl"),
                   "--tag", tag, "--output_root", str(self.root)]
        if not self.args.no_hazard:
            command += ["--hazard",
                        str(self.root / "hazard" / f"{safe}_hazard.jsonl"),
                        "--hazard_manifest",
                        str(sentinel_for(self.root, tag, "hazard"))]
        return command + [
            "--anchor_report", self.args.anchor_report,
            "--score_manifest", str(sentinel_for(self.root, tag, "score")),
            "--task_json", str(task_json), "--csv", str(self.csv)]

    def _spawn(self, tag: str, stage: str, gpu: Optional[str]) -> None:
        attempt = self.attempts.get((tag, stage), 0) + 1
        self.attempts[(tag, stage)] = attempt
        log_path = self.root / "logs" / f"{stage}_{sanitize_filename(tag)}.out"
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["PYTHONUNBUFFERED"] = "1"
        handle = open(log_path, "a", encoding="utf-8")
        handle.write(f"\n===== {stage} attempt {attempt} {time.strftime('%F %T')} =====\n")
        handle.flush()
        LOG.info("launch %-8s %-42s %s", stage, tag, f"gpu={gpu}" if gpu else "cpu")
        process = subprocess.Popen(
            self._command(tag, stage, gpu),
            cwd=str(Path(__file__).resolve().parents[2]), env=env,
            stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"))
        process._e2_log = handle                    # type: ignore[attr-defined]
        pgid = None
        if os.name != "nt" and hasattr(os, "getpgid"):
            try:
                pgid = os.getpgid(process.pid)
            except OSError:
                pgid = None
        self.running.append(Running(tag=tag, stage=stage, task=self.tasks[tag],
                                    process=process, gpu=gpu, started=time.time(),
                                    attempt=attempt, log_path=log_path, pgid=pgid))

    def free_gpus(self) -> List[str]:
        used = {item.gpu for item in self.running if item.stage in GPU_STAGES}
        return [gpu for gpu in self.args.gpu_list if gpu not in used]

    def schedule(self) -> None:
        if self.stopping:
            return
        queues = self.pending()
        running_cpu = sum(1 for item in self.running if item.stage == "analyze")
        while queues["cpu"] and running_cpu < self.args.analysis_workers:
            tag, stage = queues["cpu"].pop(0).split("|")
            self._spawn(tag, stage, None)
            running_cpu += 1
        now = time.time()
        for gpu in self.free_gpus():
            if not queues["gpu"]:
                break
            if now - getattr(self, "_last_launch", 0.0) < self.args.launch_stagger:
                break
            tag, stage = queues["gpu"].pop(0).split("|")
            self._spawn(tag, stage, gpu)
            self._last_launch = now = time.time()

    # -- reaping ----------------------------------------------------------

    def reap(self) -> None:
        for item in [r for r in self.running if r.process.poll() is not None]:
            self.running.remove(item)
            code = item.process.returncode
            handle = getattr(item.process, "_e2_log", None)
            if handle is not None:
                handle.close()
            orphaned = sweep_group(item.pgid, f"{item.stage}:{item.tag}")
            minutes = (time.time() - item.started) / 60.0
            produced = sentinel_for(self.root, item.tag, item.stage).is_file()
            # analyze exits 2 when a gate fails: the artefact IS produced and
            # the verdict is the point, so that is a success for the queue.
            ok = produced and code in ((0, 2) if item.stage == "analyze" else (0,))
            status = "success" if ok else "failed"
            append_csv_row(self.ledger, {
                "tag": item.tag, "stage": item.stage, "attempt": item.attempt,
                "gpu": item.gpu or "", "minutes": round(minutes, 1),
                "started": time.strftime("%F %T", time.localtime(item.started)),
                "ended": time.strftime("%F %T"), "exit_code": code,
                "status": status, "model_path": item.task.model_path,
                "note": ("orphans swept; " if orphaned else "") +
                        ("" if produced else "expected artefact missing"),
            }, LEDGER_FIELDS)
            if ok:
                # Completion by THIS process outranks any earlier verdict: a
                # stage invalidated at survey time is done once we have run it.
                self._completed.add((item.tag, item.stage))
                self._invalidated.discard((item.tag, item.stage))
                self.done.add((item.tag, item.stage))
                LOG.info("done  %-8s %-42s %6.1f min%s", item.stage, item.tag,
                         minutes, "  (gate FAILED)" if code == 2 else "")
            else:
                LOG.error("FAIL  %-8s %-42s exit=%s -- see %s", item.stage,
                          item.tag, code, item.log_path)
                if item.attempt >= self.args.max_attempts:
                    self.failed[(item.tag, item.stage)] = f"exit={code}"

    def stop(self, *_signal: Any) -> None:
        if self.stopping:
            return
        self.stopping = True
        LOG.warning("stop requested: terminating %d child(ren)", len(self.running))
        for item in self.running:
            try:
                if item.pgid and os.name != "nt":
                    os.killpg(item.pgid, signal.SIGTERM)
                else:
                    item.process.terminate()
            except (OSError, ProcessLookupError):
                pass

    def run(self) -> int:
        total = len(self.order)
        LOG.info("E2 queue: %d model(s) x %d stage(s), gpus=%s, analysis=%d",
                 total, len(self.stages_for(self.order[0])) if self.order else 0,
                 ",".join(self.args.gpu_list), self.args.analysis_workers)
        blocking = self.survey()
        if blocking:
            LOG.error("%d finished GPU stage(s) on disk describe a DIFFERENT "
                      "measurement than this run:", len(blocking))
            for entry in blocking:
                LOG.error("  %s", entry)
            LOG.error(
                "Refusing to start.  Reusing them would put the models on "
                "different scales; re-running them without clearing their "
                "readouts would abort in the worker anyway.  Remove the stage "
                "artefacts listed above (sentinel, its .jsonl and the .jsonl"
                ".identity.json beside it), or point --result_root at a fresh "
                "directory, and start again.")
            return 1
        for entry in self.will_recompute:
            LOG.info("will recompute: %s", entry)
        LOG.info("%d stage(s) already complete for this exact measurement, "
                 "%d summary(ies) to recompute",
                 len(self._verified_done), len(self.will_recompute))
        last_beat = 0.0
        try:
            while not self.stopping:
                self.schedule()
                queues = self.pending()
                if not self.running and not queues["gpu"] and not queues["cpu"]:
                    break
                time.sleep(self.args.poll)
                self.reap()
                if time.time() - last_beat >= self.args.heartbeat:
                    LOG.info("[status] done=%d failed=%d running=%s",
                             len(self.done), len(self.failed),
                             ", ".join(f"{r.stage}:{r.tag}" for r in self.running)
                             or "idle")
                    last_beat = time.time()
            while self.running:
                self.reap()
                time.sleep(1.0)
        finally:
            for item in list(self.running):
                sweep_group(item.pgid, f"{item.stage}:{item.tag}")
        LOG.info("finished: %d stage(s) complete, %d failed", len(self.done),
                 len(self.failed))
        for (tag, stage), reason in self.failed.items():
            LOG.error("  failed: %-42s %-8s %s", tag, stage, reason)
        return 1 if self.failed else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--anchor_report", default="")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--train_script", required=True)
    parser.add_argument("--train_output_root", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--analysis_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--priority_max", type=int, default=99)
    parser.add_argument("--only", default=None)
    parser.add_argument("--no_reference", action="store_true")
    parser.add_argument("--no_hazard", action="store_true",
                        help="Skip the 16-sample continuations.  The rank and "
                             "sampler readouts already satisfy §9.5's "
                             "'at least one rank/sampler readout'.")
    parser.add_argument("--hazard_arms", default="base,manip,neutral")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--max_attempts", type=int, default=2)
    parser.add_argument("--poll", type=float, default=10.0)
    parser.add_argument("--heartbeat", type=float, default=600.0)
    parser.add_argument("--launch_stagger", type=float, default=20.0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--log_file", default=None)
    args = parser.parse_args(argv)
    args.gpu_list = [v.strip() for v in args.gpus.split(",") if v.strip()]
    args.only = ([v.strip() for v in args.only.split(",")] if args.only else None)
    args.analysis_workers = max(1, args.analysis_workers)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log_file = args.log_file or str(Path(args.result_root) / "logs" / "orchestrator.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_file, encoding="utf-8")],
                        force=True)

    runs = parse_run_matrix(args.train_script)
    tasks = build_queue(runs, output_root=args.train_output_root,
                        model_root=args.model_root,
                        include_reference=not args.no_reference,
                        include_intermediate=False,      # §E2 scores models
                        priority_max=args.priority_max, only=args.only)
    LOG.info("E1 registry: %d run(s) -> %d final checkpoint(s) to score",
             len(runs), len(tasks))
    if args.list:
        for task in tasks:
            print(f"{task.tier:>4} {task.tag:<44} {task.size:<5} "
                  f"{task.data_variant:<16} {task.model_path}")
        return 0
    if not tasks:
        LOG.error("nothing to score under %s", args.train_output_root)
        return 1

    orchestrator = Orchestrator(args, tasks)
    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), orchestrator.stop)
    return orchestrator.run()


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
