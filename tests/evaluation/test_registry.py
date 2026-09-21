# coding=utf-8
"""Reading the training matrix, and turning it into an ordered task queue."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcr.evaluation.registry import (
    REFERENCE_TAGS, build_queue, expected_task_count, find_checkpoints,
    is_loadable_model_dir, parse_run_matrix, reference_tasks,
)

REAL_MATRIX = (Path(__file__).resolve().parents[2]
               / "experiments" / "2_train" / "run_train_all.sh")

MATRIX = '''#!/usr/bin/env bash
S=$DATA_ROOT/stages
C=$DATA_ROOT/scs
RUNS=(
  # a comment inside the array
  "1|qwen3-4b-cleanv2-s42|4B|$DATA_ROOT/cleanv2/swift_train_supportclean_keep8.jsonl|42|trajectory||"
  "1|qwen3-4b-isc-a-s42|4B|$C/swift_train_isc_a.jsonl|42|trajectory||"
  "3|qwen3-4b-base|4B|$S/base/swift_train_base.jsonl|42|final_only||"
  "6|qwen3-4b-clean-ta|4B|$S/clean/swift_train_clean.jsonl|42|final_only|close_eos|0.025"
)
echo done
'''


def write_model(directory: Path, *, tokenizer: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"\0")
    if tokenizer:
        # Both files: `tokenizer.json` is what makes the tokenizer FAST, and
        # the analysis stage cannot produce offsets without a fast one.
        (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    return directory


@pytest.fixture
def matrix_file(tmp_path: Path) -> Path:
    path = tmp_path / "run_train_all.sh"
    path.write_text(MATRIX, encoding="utf-8")
    return path


def test_parses_the_matrix(matrix_file: Path):
    runs = parse_run_matrix(matrix_file)
    assert [run.run_name for run in runs] == [
        "qwen3-4b-cleanv2-s42", "qwen3-4b-isc-a-s42", "qwen3-4b-base",
        "qwen3-4b-clean-ta"]
    assert [run.data_variant for run in runs] == [
        "cleanv2", "isc_a", "base", "clean"]
    assert [run.arm_family for run in runs] == [
        "stage", "controlled", "stage", "stage"]
    assert runs[-1].terminal_mode == "close_eos"
    assert runs[-1].terminal_rho == "0.025"


def test_rejects_a_malformed_row(tmp_path: Path):
    path = tmp_path / "bad.sh"
    path.write_text(MATRIX.replace("|42|final_only||\"", "|42|final_only\""),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="8 "):
        parse_run_matrix(path)


@pytest.mark.skipif(not REAL_MATRIX.is_file(), reason="training script absent")
def test_parses_the_real_training_matrix():
    """The registry must read the file the user actually edits, not a copy."""
    runs = parse_run_matrix(REAL_MATRIX)
    # the release matrix is exactly the 20 trained models of the paper
    assert len(runs) == 20
    assert {run.size for run in runs} == {"1.7B", "4B", "8B"}
    assert {run.data_variant for run in runs} == {
        "cleanv2", "keep4", "keep4_a", "keep4_ae", "obr", "obr_p5",
        "obr_p10", "obr_p15", "isc_a", "isc_e", "isc_ae", "benign_input",
        "generic_noise"}
    assert all(run.save_mode == "final_only" for run in runs)


def test_only_finished_runs_are_evaluable(matrix_file: Path, tmp_path: Path):
    runs = parse_run_matrix(matrix_file)
    models = tmp_path / "models"
    output = tmp_path / "out"
    for size in ("1.7B", "4B", "8B"):
        write_model(models / f"Qwen3-{size}")
    # one finished run, one still training (no TRAIN_SUCCESS)
    write_model(output / "qwen3-4b-base" / "checkpoint-417")
    write_model(output / "qwen3-4b-isc-a-s42" / "checkpoint-42")
    (output / "qwen3-4b-base" / "TRAIN_SUCCESS").write_text("", encoding="utf-8")

    tasks = build_queue(runs, output_root=output, model_root=models)
    trained = [task for task in tasks if task.kind == "trained"]
    assert [task.tag for task in trained] == ["qwen3-4b-base"]
    assert trained[0].is_final and trained[0].ckpt_step == 417


def test_trajectory_checkpoints_are_ordered_step_major(matrix_file: Path,
                                                       tmp_path: Path):
    runs = parse_run_matrix(matrix_file)
    models = tmp_path / "models"
    output = tmp_path / "out"
    for size in ("1.7B", "4B", "8B"):
        write_model(models / f"Qwen3-{size}")
    for name in ("qwen3-4b-cleanv2-s42", "qwen3-4b-isc-a-s42"):
        for step in (42, 84, 417):
            write_model(output / name / f"checkpoint-{step}")
        (output / name / "TRAIN_SUCCESS").write_text("", encoding="utf-8")

    tags = [task.tag for task in build_queue(runs, output_root=output,
                                             model_root=models)]
    # references first, then both finals, then the intermediates interleaved
    assert tags[:3] == [REFERENCE_TAGS["1.7B"], REFERENCE_TAGS["4B"],
                        REFERENCE_TAGS["8B"]]
    assert tags[3:5] == ["qwen3-4b-cleanv2-s42", "qwen3-4b-isc-a-s42"]
    assert tags[5:] == ["qwen3-4b-cleanv2-s42__step42",
                        "qwen3-4b-isc-a-s42__step42",
                        "qwen3-4b-cleanv2-s42__step84",
                        "qwen3-4b-isc-a-s42__step84"]


def test_finals_only_and_priority_filters(matrix_file: Path, tmp_path: Path):
    runs = parse_run_matrix(matrix_file)
    models = tmp_path / "models"
    output = tmp_path / "out"
    write_model(models / "Qwen3-4B")
    for name in ("qwen3-4b-cleanv2-s42", "qwen3-4b-base"):
        for step in (42, 417):
            write_model(output / name / f"checkpoint-{step}")
        (output / name / "TRAIN_SUCCESS").write_text("", encoding="utf-8")

    finals = build_queue(runs, output_root=output, model_root=models,
                         include_intermediate=False, include_reference=False)
    assert all(task.is_final for task in finals)
    assert len(finals) == 2

    p1 = build_queue(runs, output_root=output, model_root=models,
                     include_reference=False, priority_max=1)
    assert {task.run_name for task in p1} == {"qwen3-4b-cleanv2-s42"}


def test_missing_checkpoint_tokenizer_falls_back_to_the_base_model(
        matrix_file: Path, tmp_path: Path):
    runs = parse_run_matrix(matrix_file)
    models = tmp_path / "models"
    output = tmp_path / "out"
    base = write_model(models / "Qwen3-4B")
    write_model(output / "qwen3-4b-base" / "checkpoint-417", tokenizer=False)
    (output / "qwen3-4b-base" / "TRAIN_SUCCESS").write_text("", encoding="utf-8")

    task = build_queue(runs, output_root=output, model_root=models,
                       include_reference=False)[0]
    assert task.tokenizer_is_fallback
    assert Path(task.tokenizer_path) == base


def test_a_slow_only_tokenizer_also_falls_back(matrix_file: Path, tmp_path: Path):
    """`tokenizer_config.json` alone does not make a FAST tokenizer, and the
    analysis stage hard-requires one.  Falling back to the base model is safe
    (identical Qwen3 vocabulary) and beats failing after generation."""
    runs = parse_run_matrix(matrix_file)
    models = tmp_path / "models"
    output = tmp_path / "out"
    base = write_model(models / "Qwen3-4B")
    checkpoint = write_model(output / "qwen3-4b-base" / "checkpoint-417",
                             tokenizer=False)
    (checkpoint / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (output / "qwen3-4b-base" / "TRAIN_SUCCESS").write_text("", encoding="utf-8")

    task = build_queue(runs, output_root=output, model_root=models,
                       include_reference=False)[0]
    assert task.tokenizer_is_fallback
    assert Path(task.tokenizer_path) == base


def test_a_complete_checkpoint_uses_its_own_tokenizer(matrix_file: Path,
                                                      tmp_path: Path):
    runs = parse_run_matrix(matrix_file)
    models = tmp_path / "models"
    output = tmp_path / "out"
    write_model(models / "Qwen3-4B")
    checkpoint = write_model(output / "qwen3-4b-base" / "checkpoint-417")
    (output / "qwen3-4b-base" / "TRAIN_SUCCESS").write_text("", encoding="utf-8")

    task = build_queue(runs, output_root=output, model_root=models,
                       include_reference=False)[0]
    assert not task.tokenizer_is_fallback
    assert Path(task.tokenizer_path) == checkpoint


def test_half_written_checkpoints_are_invisible(tmp_path: Path):
    partial = tmp_path / "checkpoint-42"
    partial.mkdir()
    (partial / "config.json").write_text("{}", encoding="utf-8")
    assert not is_loadable_model_dir(partial)     # no weights yet
    assert find_checkpoints(tmp_path) == []


def test_expected_task_count(matrix_file: Path):
    runs = parse_run_matrix(matrix_file)
    # 2 trajectory runs x 10 + 2 final_only + 3 reference models
    assert expected_task_count(runs) == 2 * 10 + 2 + 3
    assert expected_task_count(runs, include_reference=False,
                               priority_max=3) == 2 * 10 + 1


def test_reference_models_need_no_training(tmp_path: Path):
    models = tmp_path / "models"
    write_model(models / "Qwen3-4B")
    tasks = reference_tasks(models)
    assert [task.tag for task in tasks] == [REFERENCE_TAGS["4B"]]
    assert tasks[0].tier == 0 and tasks[0].data_variant == "none"
