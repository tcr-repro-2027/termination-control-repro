"""Integration checks for cleaned-data discovery and format-preview protection."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tcr.data.layout import require_full_data
from tcr.data.stage_build.stages import Record
from tcr.prompt_template import build_extraction_relation_prompt

REPO = Path(__file__).resolve().parents[2]


def entrypoint(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "experiments" / "1_data" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_output_is_discovered_and_rendered_in_flat_cleanv2(tmp_path):
    builder = entrypoint("build_stages")
    formats = entrypoint("build_training_formats")
    root = tmp_path / "datasets"
    record = Record(rec_id="train:row_000001", split="train", row_index=0,
                    extra={}, text="甲包含乙。", entities=["甲", "乙"],
                    entities_str="甲, 乙", blocks=[{
                        "source": "甲", "target": "乙", "relation": "包含",
                        "description": "甲包含乙。"}])
    cleaned = builder.emit("cleanv2", "train", [record], root)
    upstream = builder.emit("clean", "train", [record], root)
    assert cleaned == root / "cleanv2" / "train_supportclean_keep8.jsonl"
    assert upstream == root / "stages" / "clean" / "train_clean.jsonl"
    assert cleaned.with_name("train_supportclean_keep8_rowmap.jsonl").is_file()

    jobs = formats.discover(root)
    assert len(jobs) == 2  # row maps must not become training examples
    job = next(job for job in jobs if job["path"] == cleaned)
    formats.convert(job, root / job["group"], force=False)
    output = root / "cleanv2" / "swift_train_supportclean_keep8.jsonl"
    messages = json.loads(output.read_text(encoding="utf-8"))["messages"]
    assert messages[0]["content"] == build_extraction_relation_prompt(record.text, record.entities_str)
    assert json.loads(messages[1]["content"]) == record.blocks


def test_preview_marker_blocks_directory_and_nested_file(tmp_path):
    preview = tmp_path / "datasets"
    preview.mkdir()
    (preview / "PREVIEW_ONLY.json").write_text("{}", encoding="utf-8")
    for path in (preview, preview / "cleanv2" / "train.jsonl"):
        with pytest.raises(ValueError, match="10 records per file"):
            require_full_data(path)


def test_full_download_next_to_previews_is_allowed(tmp_path):
    previews = tmp_path / "datasets"
    previews.mkdir()
    (previews / "PREVIEW_ONLY.json").write_text("{}", encoding="utf-8")
    full = tmp_path / "full_data" / "datasets" / "cleanv2"
    full.mkdir(parents=True)
    records = full / "train.jsonl"
    records.write_text('{"text": "example"}\n', encoding="utf-8")
    require_full_data(records)
