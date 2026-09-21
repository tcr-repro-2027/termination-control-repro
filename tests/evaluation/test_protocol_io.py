# coding=utf-8
"""Protocol identity, durable IO, and the orchestrator's on-disk state."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

from tcr.evaluation import protocol
from tcr.evaluation.io_utils import (
    FileLock, JsonlWriter, append_csv_row, load_done_keys, read_csv_rows,
    read_jsonl, write_json,
)
from tcr.evaluation.prompts import prompt_fingerprint, rendered_digest, verify_against

TRAINING_PROMPT = (Path(__file__).resolve().parents[2]
                   / "tcr" / "prompt_template.py")


# ------------------------------------------------------------------ protocol

def test_the_vendored_prompt_is_the_training_prompt():
    """If this ever fails, every E1 number is being produced under a prompt the
    models were not trained on -- which is the exact confound E1 removes."""
    if not TRAINING_PROMPT.is_file():
        pytest.skip("training prompt module not present in this checkout")
    report = verify_against(TRAINING_PROMPT)
    assert report["available"]
    assert report["match"] is True, report




def test_prompt_fingerprint_shape():
    fingerprint = prompt_fingerprint()
    assert fingerprint["prompt_is_truncated"] is False
    assert len(fingerprint["prompt_rendered_sha256"]) == 64
    assert fingerprint["prompt_rendered_chars"] > 2000
    assert rendered_digest() == fingerprint["prompt_rendered_sha256"]


def test_protocol_is_the_frozen_nothink_profile():
    described = protocol.describe()
    assert described["mode"] == "nothink"
    assert described["k"] == 8 and described["seeds"] == list(range(8))
    assert described["max_tokens"] is None, \
        "an explicit None is what gives each prompt its remaining context"
    assert described["sampling"] == {
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 1.5, "repetition_penalty": 1.0}
    assert described["legacy_detector"]["min_repeats"] == 50, \
        "must match the 01 run_detection.sh that produced loop_results.csv"
    assert described["structured_detector"]["max_motif_period_blocks"] == 0, \
        "0 = exhaustive motif period; a cap would silently miss long motifs"


# ------------------------------------------------------------------------ IO

def test_jsonl_writer_and_reader(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    with JsonlWriter(path, mode="w") as handle:
        for index in range(3):
            handle.write({"key": index, "text": "中文"})
    assert [row["key"] for row in read_jsonl(path)] == [0, 1, 2]
    assert "中文" in path.read_text(encoding="utf-8")


def test_load_done_keys_survives_a_torn_last_line(tmp_path: Path):
    """A worker killed mid-write leaves half a line; resume must drop it and
    regenerate that record rather than refuse to start."""
    path = tmp_path / "responses.jsonl"
    path.write_text('{"key": 1}\n{"key": 2}\n{"key": 3, "resp', encoding="utf-8")
    assert load_done_keys(path) == {1, 2}


def test_append_csv_writes_the_header_once(tmp_path: Path):
    path = tmp_path / "metrics.csv"
    fields = ["tag", "rate", "note"]
    append_csv_row(path, {"tag": "a", "rate": 0.5}, fields)
    append_csv_row(path, {"tag": "b", "rate": 0.25, "extra": "dropped"}, fields)

    rows = read_csv_rows(path)
    assert [row["tag"] for row in rows] == ["a", "b"]
    assert rows[0]["note"] == ""           # missing column, not a shifted row
    assert "extra" not in rows[0]


def test_file_lock_is_exclusive_and_released(tmp_path: Path):
    target = tmp_path / "x.csv"
    with FileLock(target, timeout=0.2):
        with pytest.raises(TimeoutError):
            with FileLock(target, timeout=0.2):
                pass
    with FileLock(target, timeout=0.2):        # released on exit
        pass


def test_write_json_is_atomic(tmp_path: Path):
    path = tmp_path / "summary.json"
    write_json(path, {"a": 1})
    write_json(path, {"a": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2}
    assert not list(tmp_path.glob("*.partial*"))


# -------------------------------------------------------------- orchestrator

def test_task_stage_is_read_off_the_artefacts(tmp_path: Path):
    from tcr.evaluation.orchestrator import TaskPaths, task_stage

    paths = TaskPaths.build(tmp_path, "qwen3-4b-clean")
    assert task_stage(paths) == "generate"

    paths.sentinel.parent.mkdir(parents=True, exist_ok=True)
    paths.sentinel.write_text("{}", encoding="utf-8")
    assert task_stage(paths) == "analyze"

    paths.summary.parent.mkdir(parents=True, exist_ok=True)
    paths.summary.write_text("{}", encoding="utf-8")
    assert task_stage(paths) == "done"


def _task(tag: str, *, model_path: str = "/m"):
    from tcr.evaluation.registry import EvalTask
    return EvalTask(tag=tag, run_name=tag, kind="trained", tier=1,
                    priority=1, size="4B", data_variant="clean",
                    arm_family="stage", seed=42, save_mode="final_only",
                    terminal_mode="", ckpt_step=417, is_final=True,
                    model_path=model_path, tokenizer_path=model_path,
                    tokenizer_is_fallback=False, order=0)


def _orchestrator(tmp_path: Path, tasks, *, limit=None, max_attempts=2):
    from tcr.evaluation.orchestrator import Orchestrator

    eval_data = tmp_path / "eval.jsonl"
    if not eval_data.is_file():
        eval_data.write_text('{"key": 1}\n', encoding="utf-8")
    args = argparse.Namespace(result_root=str(tmp_path), csv=None,
                              gen_only=False, analyze_only=False,
                              eval_data=str(eval_data), max_model_len=32768,
                              limit=limit, max_attempts=max_attempts)
    return Orchestrator(args, tasks)


def test_queue_skips_finished_and_resumes_the_rest(tmp_path: Path):
    from tcr.evaluation.orchestrator import TaskPaths

    tasks = [_task("done-one"), _task("mid-one"), _task("fresh-one")]
    orchestrator = _orchestrator(tmp_path, tasks)
    identity = orchestrator.expected_identity(tasks[0])

    write_json(TaskPaths.build(tmp_path, "done-one").summary,
               {"generation": {"identity": identity}})
    write_json(TaskPaths.build(tmp_path, "mid-one").sentinel,
               {"identity": identity})

    orchestrator.refresh_queue()
    assert orchestrator.completed == ["done-one"]
    assert orchestrator.pending_analyze == ["mid-one"]
    assert orchestrator.pending_gen == ["fresh-one"]
    assert orchestrator.failed == {}


def test_an_artefact_from_another_run_is_never_reused(tmp_path: Path):
    """A `LIMIT=8` smoke run, an edited eval file or a replaced checkpoint all
    leave a `.done.json` that would otherwise be read as "this model is done"."""
    from tcr.evaluation.orchestrator import TaskPaths

    task = _task("qwen3-4b-clean")
    orchestrator = _orchestrator(tmp_path, [task])
    smoke = dict(orchestrator.expected_identity(task), limit=8)
    write_json(TaskPaths.build(tmp_path, task.tag).sentinel, {"identity": smoke})

    orchestrator.refresh_queue()
    assert task.tag in orchestrator.failed
    assert orchestrator.pending_gen == [] and orchestrator.pending_analyze == []


def test_an_artefact_with_no_recorded_identity_is_refused(tmp_path: Path):
    from tcr.evaluation.orchestrator import TaskPaths

    task = _task("qwen3-4b-clean")
    orchestrator = _orchestrator(tmp_path, [task])
    TaskPaths.build(tmp_path, task.tag).sentinel.write_text("{}", encoding="utf-8")

    orchestrator.refresh_queue()
    assert task.tag in orchestrator.failed


def test_a_replaced_checkpoint_invalidates_the_old_responses(tmp_path: Path):
    from tcr.evaluation.orchestrator import TaskPaths

    old = _task("qwen3-4b-clean", model_path="/ckpt-417")
    orchestrator = _orchestrator(tmp_path, [old])
    write_json(TaskPaths.build(tmp_path, old.tag).run_manifest,
               {"identity": orchestrator.expected_identity(old)})

    retrained = _task("qwen3-4b-clean", model_path="/ckpt-459")
    fresh = _orchestrator(tmp_path, [retrained])
    problem = fresh.identity_error(retrained)
    assert problem and "model_path" in problem


def test_attempt_counts_survive_a_restart(tmp_path: Path):
    """`--max_attempts` must bound retries across restarts, not per process."""
    from tcr.evaluation.io_utils import append_csv_row
    from tcr.evaluation.orchestrator import LEDGER_FIELDS

    ledger = tmp_path / "e1_eval_ledger.csv"
    for _ in range(2):
        append_csv_row(ledger, {"tag": "broken", "stage": "generate",
                                "status": "failed", "exit_code": 1},
                       LEDGER_FIELDS)
    append_csv_row(ledger, {"tag": "fine", "stage": "generate",
                            "status": "success", "exit_code": 0}, LEDGER_FIELDS)

    orchestrator = _orchestrator(tmp_path, [_task("broken"), _task("fine")],
                                 max_attempts=2)
    assert "broken" in orchestrator.failed
    assert "fine" not in orchestrator.failed
    assert orchestrator.attempts[("broken", "generate")] == 2


# ------------------------------------------------- orphaned worker cleanup

def test_terminate_sweeps_the_group_even_when_the_wrapper_is_already_gone(
        monkeypatch):
    """The gap this closes: a wrapper OOM-killed or crashed on its own leaves
    vLLM workers holding the card, and `poll() is not None` used to mean
    'nothing to do'."""
    from tcr.evaluation import orchestrator

    calls = []
    alive = {"value": True}
    monkeypatch.setattr(orchestrator, "_group_is_alive",
                        lambda pgid: alive["value"])

    def fake_signal(pgid, sig):
        calls.append((pgid, sig))
        alive["value"] = False
        return True

    monkeypatch.setattr(orchestrator, "_signal_group", fake_signal)

    class DeadWrapper:
        pid = 4242

        def poll(self):
            return -9        # SIGKILLed

    orchestrator.terminate_tree(DeadWrapper(), pgid=777, label="gen:x")
    assert calls == [(777, orchestrator._SIGTERM)]


def test_sweep_escalates_to_sigkill_when_term_is_ignored(monkeypatch):
    from tcr.evaluation import orchestrator

    signals = []
    monkeypatch.setattr(orchestrator, "_group_is_alive", lambda pgid: True)
    monkeypatch.setattr(orchestrator, "_signal_group",
                        lambda pgid, sig: signals.append(sig) or True)

    assert orchestrator.sweep_group(99, label="gen:x", grace=0.1) is True
    assert signals == [orchestrator._SIGTERM, orchestrator._SIGKILL]


def test_sweep_is_a_no_op_for_a_clean_exit(monkeypatch):
    from tcr.evaluation import orchestrator

    monkeypatch.setattr(orchestrator, "_group_is_alive", lambda pgid: False)
    monkeypatch.setattr(orchestrator, "_signal_group",
                        lambda pgid, sig: pytest.fail("must not signal"))
    assert orchestrator.sweep_group(123, label="gen:x") is False
    assert orchestrator.sweep_group(None) is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_a_real_orphan_is_killed(tmp_path: Path):
    """End to end on POSIX: a wrapper that spawns a worker and is then killed
    outright must not leave that worker running."""
    import subprocess
    import time

    from tcr.evaluation.orchestrator import sweep_group

    marker = tmp_path / "worker.pid"
    script = (
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c',"
        " 'import time; time.sleep(120)'])\n"
        f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(120)\n"
    )
    wrapper = subprocess.Popen([sys.executable, "-c", script],
                               start_new_session=True)
    pgid = os.getpgid(wrapper.pid)
    deadline = time.time() + 20
    while time.time() < deadline and not marker.is_file():
        time.sleep(0.05)
    worker_pid = int(marker.read_text())

    wrapper.kill()                      # the wrapper dies; the worker does not
    wrapper.wait(timeout=10)
    os.kill(worker_pid, 0)              # still there -- this is the orphan

    assert sweep_group(pgid, label="test", grace=2.0) is True
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(worker_pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail("the orphaned worker survived the sweep")
