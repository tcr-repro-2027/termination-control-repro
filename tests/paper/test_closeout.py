# coding: utf-8
"""Regression checks for the closeout's own bookkeeping.

These checks cover the bookkeeping needed to interpret the reported results:

* a caller must not be able to overwrite a measured value with a label;
* a paired test must use every replicate, not the last one it happened to see;
* an interrupted run must not leave a half-finished cell looking finished;
* a contrast must hold the training seed fixed;
* "done" must mean every analyser finished, not just the first one;
* a completeness check must compare against the plan, not against the output.

Run these checks without GPUs, model weights, or datasets::

    python -m pytest -q tests/paper/test_closeout.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PACKAGE = REPO_ROOT / "experiments" / "4_analysis"
for _path in (REPO_ROOT, PACKAGE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tcr.paper import registry, stats                                      # noqa: E402
from tcr.paper.io import read_jsonl, write_jsonl                           # noqa: E402
import run_r0                                                       # noqa: E402
import run_r2                                                       # noqa: E402


def an_effect() -> stats.Effect:
    return stats.Effect(metric="m", m0=0.25, m1=0.75, diff=0.5, ci_low=0.4,
                        ci_high=0.6, n_prompts=10, n_m0=20, n_m1=20,
                        direction="m1_higher")


def test_as_dict_refuses_to_overwrite_measured_fields() -> None:
    effect = an_effect()
    try:
        effect.as_dict(m0="qwen3-4b-cleanv2-s42")
    except ValueError as exc:
        assert "m0" in str(exc)
    else:
        raise AssertionError("as_dict let a tag overwrite the measured level")
    row = effect.as_dict(m0_tag="qwen3-4b-cleanv2-s42")
    assert row["m0"] == 0.25 and row["m0_tag"].startswith("qwen3")


def draw_rows(value_by_condition: dict[str, float], n_anchors: int = 4,
              n_draws: int = 4) -> list[dict]:
    return [{"anchor_id": f"a{i}", "prompt_id": f"p{i}", "condition": condition,
             "draw": draw, "v": value}
            for i in range(n_anchors) for draw in range(n_draws)
            for condition, value in value_by_condition.items()]


def test_repeated_anchors_are_refused() -> None:
    rows = draw_rows({"baseline": 0.0, "treated": 1.0})
    try:
        stats.paired_condition_effect(rows, metric="v", value_field="v",
                                      condition_field="condition",
                                      baseline="baseline", treatment="treated")
    except ValueError as exc:
        assert "more than once" in str(exc)
    else:
        raise AssertionError("draw-level rows were silently de-duplicated")


def test_draws_are_averaged_not_overwritten() -> None:
    # The last draw differs from the first three; keeping only it would give 1.0.
    mixed = [{"anchor_id": "a", "prompt_id": "p", "condition": "c",
              "draw": i, "v": v} for i, v in enumerate((0.0, 0.0, 0.0, 1.0))]
    averaged = stats.mean_per_anchor(mixed, value_field="v")
    assert len(averaged) == 1
    assert abs(averaged[0]["v"] - 0.25) < 1e-9
    assert averaged[0]["n_draws"] == 4


def test_collapsed_effect_uses_every_anchor() -> None:
    rows = draw_rows({"baseline": 0.0, "treated": 1.0})
    collapsed = run_r2.collapse_draws(rows, "v")
    assert len(collapsed) == 8            # 4 anchors x 2 conditions
    effect = stats.paired_condition_effect(collapsed, metric="v", value_field="v",
                                           condition_field="condition",
                                           baseline="baseline", treatment="treated")
    assert effect.n_m0 == 4
    assert abs(effect.diff - 1.0) < 1e-9


def test_interrupted_long_cell_is_redone(tmp: Path) -> None:
    path = tmp / "long.jsonl"
    write_jsonl(path, [
        *[{"anchor_id": "a1", "condition": "baseline", "draw": d} for d in range(4)],
        *[{"anchor_id": "a1", "condition": "direction_a1", "draw": d} for d in range(2)],
    ])
    done = run_r2.complete_long_conditions(path, 4)
    assert ("a1", "baseline") in done
    assert ("a1", "direction_a1") not in done
    # The partial rows are removed, so the redo cannot append on top of them.
    assert len(read_jsonl(path)) == 4


def test_long_matrix_gaps_sees_all_three_failure_shapes() -> None:
    selected = [{"anchor_id": f"a{i}"} for i in range(3)]
    rows = []
    for condition in run_r2.LONG_CONDITIONS:            # a0: complete
        rows += [{"model_tag": "m", "scale": "4B", "anchor_id": "a0",
                  "condition": condition, "draw": d} for d in range(4)]
    for condition in run_r2.LONG_CONDITIONS[:3]:        # a1: one condition short
        rows += [{"model_tag": "m", "scale": "4B", "anchor_id": "a1",
                  "condition": condition, "draw": d} for d in range(4)]
    rows += [{"model_tag": "m", "scale": "4B", "anchor_id": "a1",
              "condition": run_r2.LONG_CONDITIONS[3], "draw": d}
             for d in range(2)]                        # ... and it is ragged too
    # a2 never ran at all.
    gaps = run_r2.long_matrix_gaps(rows, selected, ["m"])[0]
    assert gaps["missing_anchors"] == 1
    assert gaps["ragged_cells"] == 1
    assert gaps["measured_anchors"] == 2


def test_a_model_with_no_output_still_appears_in_the_report() -> None:
    """A model whose every anchor was skipped writes nothing; deriving the
    report's rows from the output would then hide the one case worth seeing."""
    selected = [{"anchor_id": f"a{i}"} for i in range(3)]
    rows = [{"model_tag": "ran", "scale": "4B", "anchor_id": "a0",
             "condition": condition, "draw": d}
            for condition in run_r2.LONG_CONDITIONS for d in range(4)]
    gaps = {row["model_tag"]: row
            for row in run_r2.long_matrix_gaps(rows, selected, ["ran", "silent"])}
    assert set(gaps) == {"ran", "silent"}
    assert gaps["silent"]["measured_anchors"] == 0
    assert gaps["silent"]["missing_anchors"] == len(selected)
    assert gaps["silent"]["missing_cells"] == len(selected) * len(run_r2.LONG_CONDITIONS)


def test_r1_contrasts_hold_the_training_seed_fixed() -> None:
    pairs = {(m0, m1) for _l, m0, m1, _k in registry.R1_CONTRASTS["4B"]}
    assert ("qwen3-4b-cleanv2-s123", "qwen3-4b-keep4-s123") in pairs
    assert ("qwen3-4b-cleanv2-s123", "qwen3-4b-obr-p15-s123") in pairs
    # No treatment contrast may cross training seeds.
    for _label, m0, m1, kind in registry.R1_CONTRASTS["4B"]:
        if kind != "treatment":
            continue
        assert registry.spec(m0).train_seed == registry.spec(m1).train_seed
    assert any(kind == "seed" for *_rest, kind in registry.R1_CONTRASTS["4B"])


def test_a_pair_is_done_only_when_every_analyser_finished(tmp: Path) -> None:
    root = tmp / "r0"
    shutil.rmtree(root, ignore_errors=True)
    with_p0c = next(p for p in registry.R0_PAIRS if p.with_p0c)
    (root / with_p0c.name / "p0d_episode_hazard").mkdir(parents=True)
    (root / with_p0c.name / "p0d_episode_hazard" / "P0D_MANIFEST.json").write_text("{}")
    assert not run_r0.p0_complete(root, with_p0c)
    (root / with_p0c.name / "p0c_set_completion").mkdir(parents=True)
    (root / with_p0c.name / "p0c_set_completion" / "P0C_MANIFEST.json").write_text("{}")
    assert run_r0.p0_complete(root, with_p0c)

    plain = next(p for p in registry.R0_PAIRS if not p.with_p0c)
    (root / plain.name / "p0d_episode_hazard").mkdir(parents=True)
    (root / plain.name / "p0d_episode_hazard" / "P0D_MANIFEST.json").write_text("{}")
    assert run_r0.p0_complete(root, plain)
    assert run_r0.p0_outputs(root, plain)["p0c"] is None


def test_a_failed_analyser_is_reported_as_a_failure(tmp: Path) -> None:
    """Waiting on an evaluation returns 0; an analyser that ran and did not
    finish must not, or the driver builds the prefix pool on a broken R0."""
    root = tmp / "r0fail"
    shutil.rmtree(root, ignore_errors=True)
    with_p0c = next(p for p in registry.R0_PAIRS if p.with_p0c)
    plain = next(p for p in registry.R0_PAIRS if not p.with_p0c)
    for pair in (with_p0c, plain):
        (root / pair.name / "p0d_episode_hazard").mkdir(parents=True)
        (root / pair.name / "p0d_episode_hazard" / "P0D_MANIFEST.json").write_text("{}")
    # P0d finished for both; P0c did not for the pair that needs it.
    reported = run_r0.analysis_failures(root, [with_p0c, plain])
    assert len(reported) == 1 and with_p0c.name in reported[0]
    assert "p0c" in reported[0]

    (root / with_p0c.name / "p0c_set_completion").mkdir(parents=True)
    (root / with_p0c.name / "p0c_set_completion" / "P0C_MANIFEST.json").write_text("{}")
    assert run_r0.analysis_failures(root, [with_p0c, plain]) == []


def test_a_stale_manifest_cannot_pass_for_a_fresh_analysis(tmp: Path) -> None:
    """An analyser that dies before clearing its own output directory leaves
    last run's manifest behind; "finished" must not be readable off it."""
    root = tmp / "r0stale"
    shutil.rmtree(root, ignore_errors=True)
    pair = next(p for p in registry.R0_PAIRS if p.with_p0c)
    for relative in ("p0d_episode_hazard/P0D_MANIFEST.json",
                     "p0c_set_completion/P0C_MANIFEST.json"):
        path = root / pair.name / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    assert run_r0.p0_complete(root, pair)          # looks finished, from last run
    assert sorted(run_r0.invalidate_p0_outputs(root, pair)) == ["p0c", "p0d"]
    assert not run_r0.p0_complete(root, pair)      # this run has not written yet
    # Nothing left to clear the second time round.
    assert run_r0.invalidate_p0_outputs(root, pair) == []


def test_a_nonzero_exit_is_a_failure_even_with_a_manifest(tmp: Path) -> None:
    """A process can write its manifest and then fail; the exit code is the
    second, independent signal."""
    root = tmp / "r0exit"
    shutil.rmtree(root, ignore_errors=True)
    pair = next(p for p in registry.R0_PAIRS if not p.with_p0c)
    path = root / pair.name / "p0d_episode_hazard" / "P0D_MANIFEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    assert run_r0.analysis_failures(root, [pair]) == []
    reported = run_r0.analysis_failures(root, [pair], {pair.name: {"p0d": False}})
    assert len(reported) == 1 and "exited non-zero" in reported[0]


def test_the_summary_reads_the_fields_the_direction_file_writes() -> None:
    """A field renamed on the writing side and not on the reading side turns a
    real number into `NA` in the summary the paper is written from, and nothing
    fails while it happens."""
    meta = {name: 0 for name in run_r2.DIRECTION_FIELDS}
    meta["layer"] = 26
    meta["test_auc_held_out"] = 0.912
    rendered = run_r2.direction_summary_line(meta)
    assert "NA" not in rendered, rendered
    assert "26" in rendered and "0.912" in rendered

    # ... and the same line must degrade honestly when the file is absent.
    assert "NA" in run_r2.direction_summary_line({})


def test_every_registered_tag_is_a_known_model() -> None:
    """Validate contrast and table-row identifiers before an experiment runs."""
    referenced = set()
    for pair in registry.R0_PAIRS:
        referenced |= {pair.m0, pair.m1}
    for group in registry.R1_SCORED.values():
        referenced |= set(group)
    for group in registry.R1_POOL_SOURCES.values():
        referenced |= set(group)
    for contrasts in registry.R1_CONTRASTS.values():
        for _label, m0, m1, _kind in contrasts:
            referenced |= {m0, m1}
    for config in registry.R2_MODELS.values():
        referenced |= {config["clean"], config["raw"], *config.get("transfer", ())}
    referenced |= set(registry.X1_MODELS) | set(registry.T1_ROWS)
    referenced |= {tag for _l, tag, _d in registry.T2_ROWS}
    referenced |= set(registry.APPENDIX_A_ROWS)
    unknown = sorted(tag for tag in referenced if tag not in registry.BY_TAG)
    assert not unknown, f"unknown model tag(s): {unknown}"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="closeout-test-"))
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = 0
    try:
        for name, test in tests:
            try:
                test(tmp) if test.__code__.co_argcount else test()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
            except Exception as exc:                     # noqa: BLE001
                failures += 1
                print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


# pytest passes a `tmp_path` fixture; the standalone runner passes its own.
try:
    import pytest

    @pytest.fixture(name="tmp")
    def _tmp(tmp_path):
        return tmp_path
except ImportError:
    pass


if __name__ == "__main__":
    raise SystemExit(main())
