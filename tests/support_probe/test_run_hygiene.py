# coding=utf-8
"""The ways a run can look finished without being the run you asked for.

None of these are about the physics of the measurement; all of them are about a
number reaching the CSV that nobody could reproduce.  Each test corresponds to a
way that used to be possible:

* a `--limit` smoke test marking the real work done;
* a hazard sentinel from a different model/anchors being reused unchecked;
* resume appending to a partial file measured against a different anchor set;
* a build killed halfway leaving a short anchors.jsonl that every later run
  skipped construction for;
* an under-filled anchor set producing a perfectly ordinary summary;
* donor entities chosen with `hash()`, which is salted per process;
* the ORCHESTRATOR skipping a stage on the strength of a sentinel it never
  checked -- the identity code in the workers is dead code for a stage that is
  never launched, so `REBUILD=1` alone would have kept every stale result;
* an anchor file passing `--check` while its report is missing, stale, from
  another eval set, or from a character-fallback build.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tcr.support_probe import protocol
from tcr.support_probe.build import _stable_index
from tcr.support_probe.identity import (
    anchor_set_problems, guard_partial, mismatches, run_identity,
    sentinel_mismatches, sentinel_says_done, stamp_path,
)
from tcr.support_probe.io_utils import sha256_file, write_json


@pytest.fixture
def anchors_file(tmp_path: Path) -> Path:
    path = tmp_path / "anchors.jsonl"
    path.write_text('{"anchor_id": "a"}\n', encoding="utf-8")
    return path


def identity_for(anchors: Path, **kwargs):
    return run_identity(anchors_path=str(anchors), model_path="/models/x",
                        tokenizer_path="/models/x", **kwargs)


# ------------------------------------------------------------- the sentinel

def test_a_matching_sentinel_means_done(tmp_path, anchors_file):
    identity = identity_for(anchors_file)
    sentinel = tmp_path / "x.done.json"
    write_json(sentinel, {"identity": identity})
    assert sentinel_says_done(sentinel, identity, "x", "score") is True


def test_a_sentinel_from_another_model_is_refused_not_reused(tmp_path, anchors_file):
    write_json(tmp_path / "x.done.json",
               {"identity": identity_for(anchors_file)})
    other = run_identity(anchors_path=str(anchors_file),
                         model_path="/models/OTHER",
                         tokenizer_path="/models/OTHER")
    with pytest.raises(SystemExit) as excinfo:
        sentinel_says_done(tmp_path / "x.done.json", other, "x", "score")
    assert "model_path" in str(excinfo.value)


def test_a_sentinel_from_rebuilt_anchors_is_refused(tmp_path, anchors_file):
    write_json(tmp_path / "x.done.json",
               {"identity": identity_for(anchors_file)})
    anchors_file.write_text('{"anchor_id": "DIFFERENT"}\n', encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        sentinel_says_done(tmp_path / "x.done.json",
                           identity_for(anchors_file), "x", "score")
    assert "anchors_sha256" in str(excinfo.value)


def test_a_limited_run_does_not_satisfy_a_full_one(anchors_file):
    """`--limit` is a smoke test; its identity must not match the real run."""
    assert mismatches(identity_for(anchors_file, limit=None),
                      identity_for(anchors_file, limit=4))


def test_hazard_arms_are_part_of_the_identity(anchors_file):
    """A hazard run over three arms has not measured the positive controls."""
    assert mismatches(identity_for(anchors_file, arms="all"),
                      identity_for(anchors_file, arms="base,manip,neutral"))


# --------------------------------------------------------- the partial file

def test_resume_stamps_a_fresh_partial(tmp_path, anchors_file):
    out = tmp_path / "readouts.jsonl"
    guard_partial(out, identity_for(anchors_file), "x")
    assert stamp_path(out).is_file()


def test_resume_refuses_a_partial_from_a_different_anchor_set(tmp_path, anchors_file):
    out = tmp_path / "readouts.jsonl"
    guard_partial(out, identity_for(anchors_file), "x")
    out.write_text('{"cell_id": "a|b"}\n', encoding="utf-8")
    anchors_file.write_text('{"anchor_id": "REBUILT"}\n', encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        guard_partial(out, identity_for(anchors_file), "x")
    assert "refusing to resume" in str(excinfo.value)


def test_an_unidentifiable_partial_is_refused(tmp_path, anchors_file):
    """A readout file with no stamp cannot be shown to be this measurement."""
    out = tmp_path / "readouts.jsonl"
    out.write_text('{"cell_id": "a|b"}\n', encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        guard_partial(out, identity_for(anchors_file), "x")
    assert "cannot be identified" in str(excinfo.value)


# ------------------------------------------------------ reproducible donors

def test_the_donor_index_does_not_move_between_processes():
    """`hash()` is salted per interpreter, so the same --seed used to pick
    different donors on every run and the report could not show it."""
    assert _stable_index("anchor-7:manip:0.5", 12, 200) == 39
    script = ("import sys; sys.path.insert(0, %r);"
              "from tcr.support_probe.build import _stable_index;"
              "print(_stable_index('anchor-7:manip:0.5', 12, 200))"
              % str(Path(__file__).resolve().parents[2]))
    seen = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run([sys.executable, "-c", script], text=True,
                                capture_output=True,
                                env={"PYTHONHASHSEED": seed, "PATH": ""})
        assert result.returncode == 0, result.stderr
        seen.add(result.stdout.strip())
    assert seen == {"39"}


def test_the_stable_index_stays_in_range():
    for modulus in (1, 2, 7, 1000):
        for salt in ("a", "bb", "ccc"):
            assert 0 <= _stable_index(salt, 3, modulus) < modulus


# ------------------------------------------------------- the anchor set itself

def anchor_file(tmp_path: Path, *, n_per_type: int = 2, eval_data: Path | None = None,
                token_matched: bool = True, version: str | None = None,
                write_report: bool = True) -> tuple:
    """An anchor file plus the report a finished build would leave beside it."""
    anchors = tmp_path / "anchors.jsonl"
    rows = []
    for name in protocol.BUILT_ANCHOR_TYPES:
        for index in range(n_per_type):
            rows.append({"anchor_id": f"{name}:{index}", "anchor_type": name})
    anchors.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                       encoding="utf-8")
    report = anchors.with_suffix(anchors.suffix + ".report.json")
    if write_report:
        body = {
            "protocol": {"protocol_version": version or protocol.PROTOCOL_VERSION},
            "anchors_sha256": sha256_file(anchors),
            "token_matched": token_matched,
            "max_length_delta_allowed": 0,
        }
        if eval_data is not None:
            body["eval_data"] = str(eval_data)
            body["eval_data_sha256"] = sha256_file(eval_data)
        write_json(report, body)
    return anchors, report


def test_a_complete_anchor_set_passes(tmp_path):
    anchors, report = anchor_file(tmp_path)
    assert anchor_set_problems(anchors, report, n_per_type=2) == []


def test_a_report_without_a_hash_is_not_a_pass(tmp_path):
    """It used to be: `not in (None, digest)` let a missing key through, so a
    report from anywhere at all satisfied the check."""
    anchors, report = anchor_file(tmp_path)
    write_json(report, {"protocol": {"protocol_version": protocol.PROTOCOL_VERSION},
                        "token_matched": True})
    problems = anchor_set_problems(anchors, report, n_per_type=2)
    assert any("records no anchors_sha256" in problem for problem in problems)


def test_an_edited_anchor_file_no_longer_matches_its_report(tmp_path):
    anchors, report = anchor_file(tmp_path)
    anchors.write_text(anchors.read_text(encoding="utf-8")
                       + '{"anchor_id": "extra", "anchor_type": "evidence"}\n',
                       encoding="utf-8")
    assert any("does not match its report"
               in problem for problem in
               anchor_set_problems(anchors, report, n_per_type=2))


def test_anchors_from_another_eval_set_are_refused(tmp_path):
    mine = tmp_path / "mine.jsonl"
    mine.write_text('{"key": 1}\n', encoding="utf-8")
    theirs = tmp_path / "theirs.jsonl"
    theirs.write_text('{"key": 2}\n', encoding="utf-8")
    anchors, report = anchor_file(tmp_path, eval_data=theirs)
    problems = anchor_set_problems(anchors, report, n_per_type=2, eval_data=mine)
    assert any("different eval set" in problem for problem in problems)


def test_a_character_fallback_build_is_refused_by_default(tmp_path):
    """Donor lengths matched in characters are a pilot build, not the protocol's
    token-matched substitution -- and nothing downstream would show it."""
    anchors, report = anchor_file(tmp_path, token_matched=False)
    assert any("WITHOUT a tokenizer" in problem for problem in
               anchor_set_problems(anchors, report, n_per_type=2))
    assert anchor_set_problems(anchors, report, n_per_type=2,
                               require_token_matched=False) == []


def test_anchors_from_an_older_protocol_are_refused(tmp_path):
    anchors, report = anchor_file(tmp_path, version="e2-v0.9")
    assert any("protocol" in problem for problem in
               anchor_set_problems(anchors, report, n_per_type=2))


def test_a_missing_report_means_the_build_did_not_finish(tmp_path):
    anchors, report = anchor_file(tmp_path, write_report=False)
    assert any("did not finish" in problem for problem in
               anchor_set_problems(anchors, report, n_per_type=2))


def test_an_under_filled_type_is_caught_here_too(tmp_path):
    anchors, report = anchor_file(tmp_path, n_per_type=2)
    assert any("under-filled" in problem for problem in
               anchor_set_problems(anchors, report, n_per_type=128))


# ----------------------------------------------- the orchestrator's own skip

class FakeTask:
    """The three registry fields the orchestrator's identity depends on."""

    def __init__(self, tag: str, model_path: str = "/models/m1") -> None:
        self.tag = tag
        self.model_path = model_path
        self.tokenizer_path = model_path

    def as_dict(self):
        return {"tag": self.tag, "model_path": self.model_path}


def orchestrator(tmp_path: Path, anchors: Path, *, no_hazard: bool = True,
                 hazard_arms: str = "base,manip,neutral",
                 model_path: str = "/models/m1"):
    from tcr.support_probe.orchestrator import Orchestrator
    args = argparse.Namespace(
        result_root=str(tmp_path / "results"), anchors=str(anchors),
        anchor_report="", csv=None, no_hazard=no_hazard,
        hazard_arms=hazard_arms, batch_size=2, gpu_memory_utilization=0.9,
        gpu_list=["0"], analysis_workers=1, max_attempts=2, poll=0.01,
        heartbeat=600.0, launch_stagger=0.0)
    return Orchestrator(args, [FakeTask("m1", model_path)])


def finish_score(runner, tag: str, identity) -> Path:
    from tcr.support_probe.orchestrator import sentinel_for
    sentinel = sentinel_for(runner.root, tag, "score")
    write_json(sentinel, {"tag": tag, "identity": identity})
    return sentinel


def test_the_orchestrator_skips_a_stage_it_can_prove_is_this_run(tmp_path, anchors_file):
    runner = orchestrator(tmp_path, anchors_file)
    finish_score(runner, "m1", runner.identity_for("m1", "score"))
    assert runner.survey() == []
    assert runner.stage_is_done("m1", "score") is True
    assert "m1|score" not in runner.pending()["gpu"]


def test_rebuilt_anchors_do_not_leave_the_old_score_sentinel_standing(tmp_path,
                                                                     anchors_file):
    """The bug this pins: the worker checks identity, but a SKIPPED worker
    never runs.  `REBUILD=1` alone used to keep every stale result."""
    runner = orchestrator(tmp_path, anchors_file)
    finish_score(runner, "m1", runner.identity_for("m1", "score"))
    anchors_file.write_text('{"anchor_id": "REBUILT"}\n', encoding="utf-8")

    rebuilt = orchestrator(tmp_path, anchors_file)
    stale = rebuilt.survey()
    assert len(stale) == 1 and "anchors_sha256" in stale[0]
    assert rebuilt.stage_is_done("m1", "score") is False
    assert "m1|score" in rebuilt.pending()["gpu"]


def test_a_moved_checkpoint_invalidates_the_sentinel(tmp_path, anchors_file):
    runner = orchestrator(tmp_path, anchors_file, model_path="/models/OLD")
    finish_score(runner, "m1", runner.identity_for("m1", "score"))
    moved = orchestrator(tmp_path, anchors_file, model_path="/models/NEW")
    assert any("model_path" in entry for entry in moved.survey())


def test_changing_hazard_arms_invalidates_the_hazard_sentinel(tmp_path, anchors_file):
    from tcr.support_probe.orchestrator import sentinel_for
    runner = orchestrator(tmp_path, anchors_file, no_hazard=False)
    write_json(sentinel_for(runner.root, "m1", "hazard"),
               {"tag": "m1", "identity": runner.identity_for("m1", "hazard")})
    assert runner.survey() == []
    wider = orchestrator(tmp_path, anchors_file, no_hazard=False, hazard_arms="all")
    assert any("arms" in entry for entry in wider.survey())


def test_a_run_refuses_to_start_while_a_stale_sentinel_stands(tmp_path, anchors_file):
    runner = orchestrator(tmp_path, anchors_file)
    finish_score(runner, "m1", runner.identity_for("m1", "score"))
    anchors_file.write_text('{"anchor_id": "REBUILT"}\n', encoding="utf-8")
    assert orchestrator(tmp_path, anchors_file).run() == 1


def test_a_margin_only_summary_does_not_satisfy_a_hazard_run(tmp_path, anchors_file):
    """A NO_HAZARD=1 pass followed by a full one must re-analyse, or gate 7
    stays missing while every other gate reads green."""
    from tcr.support_probe.orchestrator import sentinel_for
    runner = orchestrator(tmp_path, anchors_file, no_hazard=False)
    # a complete margin-only pass: score done, summary written, no hazard
    finish_score(runner, "m1", runner.identity_for("m1", "score"))
    write_json(sentinel_for(runner.root, "m1", "analyze"), {
        "tag": "m1", "anchors_sha256": runner.anchors_sha256,
        "protocol": {"protocol_version": protocol.PROTOCOL_VERSION},
        "summary_schema": protocol.SUMMARY_SCHEMA_VERSION,
        "has_hazard": False,
        "score_identity": runner.identity_for("m1", "score")})
    # A stale SUMMARY is derived data: recomputed, not a reason to refuse the
    # run.  It must not, however, count as done.
    assert runner.survey() == []
    assert any("no hazard column" in entry for entry in runner.will_recompute)
    assert runner.stage_is_done("m1", "analyze") is False

    margins_only = orchestrator(tmp_path, anchors_file, no_hazard=True)
    assert margins_only.survey() == [] and margins_only.will_recompute == []
    assert margins_only.stage_is_done("m1", "analyze") is True


def test_a_sentinel_from_before_identity_checking_is_not_trusted(tmp_path,
                                                                 anchors_file):
    runner = orchestrator(tmp_path, anchors_file)
    from tcr.support_probe.orchestrator import sentinel_for
    write_json(sentinel_for(runner.root, "m1", "score"), {"tag": "m1"})
    assert any("no identity block" in entry for entry in runner.survey())


def test_sentinel_mismatches_never_raises(tmp_path, anchors_file):
    """The orchestrator surveys every stage before reporting; a raise on the
    first one would hide the rest."""
    assert sentinel_mismatches(tmp_path / "absent.json",
                               identity_for(anchors_file)) == ["sentinel missing"]


def test_a_summary_is_not_done_while_its_own_score_is_pending(tmp_path,
                                                              anchors_file):
    """Delete the score sentinel to force a re-score and the old summary used
    to stay marked done: score re-ran, nothing re-summarised the new readouts.
    A summary is derived data -- while its input is pending it necessarily
    describes an earlier run."""
    from tcr.support_probe.orchestrator import sentinel_for
    runner = orchestrator(tmp_path, anchors_file)
    finish_score(runner, "m1", runner.identity_for("m1", "score"))
    write_json(sentinel_for(runner.root, "m1", "analyze"), {
        "tag": "m1", "anchors_sha256": runner.anchors_sha256,
        "protocol": {"protocol_version": protocol.PROTOCOL_VERSION},
        "summary_schema": protocol.SUMMARY_SCHEMA_VERSION,
        "has_hazard": False,
        "score_identity": runner.identity_for("m1", "score")})
    assert runner.stage_is_done("m1", "analyze") is True    # intact: reusable

    sentinel_for(runner.root, "m1", "score").unlink()
    after = orchestrator(tmp_path, anchors_file)
    assert after.survey() == []                     # not a reason to refuse
    assert any("score stage is not complete" in entry
               for entry in after.will_recompute)
    assert after.stage_is_done("m1", "analyze") is False
    assert "m1|score" in after.pending()["gpu"]


def test_a_recomputed_summary_stays_queued_after_its_score_lands(tmp_path,
                                                                 anchors_file):
    """The invalidation has to STICK.  Re-checking on the next poll would let
    the stale summary come back as `done` the moment its score finished --
    which is the whole bug, one scheduling tick later."""
    from tcr.support_probe.orchestrator import sentinel_for
    runner = orchestrator(tmp_path, anchors_file)
    write_json(sentinel_for(runner.root, "m1", "analyze"), {
        "tag": "m1", "anchors_sha256": runner.anchors_sha256,
        "protocol": {"protocol_version": protocol.PROTOCOL_VERSION},
        "summary_schema": protocol.SUMMARY_SCHEMA_VERSION,
        "has_hazard": False})
    runner.survey()
    assert runner.stage_is_done("m1", "analyze") is False

    # score now completes, exactly as it would mid-run
    finish_score(runner, "m1", runner.identity_for("m1", "score"))
    runner._completed.add(("m1", "score"))
    assert runner.stage_is_done("m1", "analyze") is False
    assert "m1|analyze" in runner.pending()["cpu"]


def test_a_summary_of_a_different_score_run_is_recomputed(tmp_path, anchors_file):
    from tcr.support_probe.orchestrator import sentinel_for
    runner = orchestrator(tmp_path, anchors_file)
    finish_score(runner, "m1", runner.identity_for("m1", "score"))
    stale = dict(runner.identity_for("m1", "score"), model_path="/models/OLD")
    write_json(sentinel_for(runner.root, "m1", "analyze"), {
        "tag": "m1", "anchors_sha256": runner.anchors_sha256,
        "protocol": {"protocol_version": protocol.PROTOCOL_VERSION},
        "summary_schema": protocol.SUMMARY_SCHEMA_VERSION,
        "has_hazard": False, "score_identity": stale})
    assert runner.survey() == []
    assert any("score.model_path" in entry for entry in runner.will_recompute)
    assert runner.stage_is_done("m1", "analyze") is False


def test_the_hazard_context_ceiling_is_part_of_its_identity(anchors_file):
    """The hazard GENERATES, so vLLM's context ceiling is part of what it
    measured; scoring does not take one, so it carries no such field."""
    wide = identity_for(anchors_file, arms="base", max_model_len=32768)
    narrow = identity_for(anchors_file, arms="base", max_model_len=8192)
    assert mismatches(wide, narrow)
    assert "max_model_len" not in identity_for(anchors_file)


def test_a_summary_from_older_analysis_code_is_recomputed(tmp_path, anchors_file):
    """The measurement identity can match perfectly while the NUMBERS are from
    different analysis code -- the anchor-clustered interval, the threaded
    resample count and the matched-subset sensitivity all changed what a
    summary means without touching a readout."""
    from tcr.support_probe.orchestrator import sentinel_for
    runner = orchestrator(tmp_path, anchors_file)
    finish_score(runner, "m1", runner.identity_for("m1", "score"))
    write_json(sentinel_for(runner.root, "m1", "analyze"), {
        "tag": "m1", "anchors_sha256": runner.anchors_sha256,
        "protocol": {"protocol_version": protocol.PROTOCOL_VERSION},
        "summary_schema": "e2-summary-v1",          # an earlier analysis
        "has_hazard": False,
        "score_identity": runner.identity_for("m1", "score")})
    assert runner.survey() == []
    assert any("summary_schema" in entry for entry in runner.will_recompute)
    assert runner.stage_is_done("m1", "analyze") is False


def test_a_summary_that_does_not_say_what_it_read_is_recomputed(tmp_path,
                                                                anchors_file):
    """Absence is not a pass: a summary that does not record its score run
    cannot be shown to be current, and recomputing one costs seconds of CPU."""
    from tcr.support_probe.orchestrator import sentinel_for
    runner = orchestrator(tmp_path, anchors_file)
    finish_score(runner, "m1", runner.identity_for("m1", "score"))
    write_json(sentinel_for(runner.root, "m1", "analyze"), {
        "tag": "m1", "anchors_sha256": runner.anchors_sha256,
        "protocol": {"protocol_version": protocol.PROTOCOL_VERSION},
        "summary_schema": protocol.SUMMARY_SCHEMA_VERSION,
        "has_hazard": False})                        # no score_identity
    assert runner.survey() == []
    assert any("does not record which score run" in entry
               for entry in runner.will_recompute)
    assert runner.stage_is_done("m1", "analyze") is False


def test_anchors_built_with_a_looser_length_tolerance_are_refused(tmp_path):
    """`MAX_LENGTH_DELTA=2` builds a set that is not the §8.3 length-matched
    one; reusing it under the default 0 relaxes an invariant the run believes
    it is enforcing."""
    anchors, report = anchor_file(tmp_path)
    body = json.loads(report.read_text(encoding="utf-8"))
    write_json(report, {**body, "max_length_delta_allowed": 2})
    assert any("length tolerance" in problem for problem in
               anchor_set_problems(anchors, report, n_per_type=2))
    assert anchor_set_problems(anchors, report, n_per_type=2,
                               max_length_delta=2) == []


def test_a_report_predating_the_length_invariant_is_refused(tmp_path):
    anchors, report = anchor_file(tmp_path)
    body = json.loads(report.read_text(encoding="utf-8"))
    body.pop("max_length_delta_allowed")
    write_json(report, body)
    assert any("does not say what donor length tolerance" in problem
               for problem in anchor_set_problems(anchors, report, n_per_type=2))
