# coding=utf-8
"""The parts of the generation worker that do not need a GPU.

The vLLM call itself cannot be exercised here, but everything that decides
WHAT is sent to it can: the prompt, the chat prefix, the length guard and the
resume bookkeeping.  Those are exactly the places where a silent mistake would
produce a full, plausible, incomparable set of numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcr.evaluation import protocol
from tcr.evaluation.generate import (
    _skipped_record, check_chat_prefix, done_path, load_eval_records,
    render_prompt, response_path, split_reasoning,
)
from tcr.evaluation.io_utils import load_done_keys


class FakeTokenizer:
    """Stands in for a Qwen3 tokenizer's chat template."""

    def __init__(self, suffix: str = protocol.CHAT_PREFIX_SUFFIX) -> None:
        self.suffix = suffix

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        assert tokenize is False and add_generation_prompt is True
        assert enable_thinking is False, "E1 is nothink-only"
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n{self.suffix}"


def test_the_nothink_prefix_must_be_the_training_one():
    report = check_chat_prefix(FakeTokenizer())
    assert report["chat_prefix_ok"] is True

    # a checkpoint whose template lost the empty think block
    drifted = check_chat_prefix(FakeTokenizer("<|im_start|>assistant\n"))
    assert drifted["chat_prefix_ok"] is False
    assert "assistant" in drifted["chat_prefix_actual_tail"]


def test_rendered_prompt_carries_the_record(eval_record):
    prompt = render_prompt(eval_record)
    assert eval_record["text"] in prompt
    assert eval_record["entities_str"] in prompt
    assert len(prompt) > 2000, "the abridged template would be much shorter"


def test_eval_records_must_have_unique_keys(tmp_path: Path, eval_record):
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(
        json.dumps(eval_record, ensure_ascii=False) for _ in range(2)) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate keys"):
        load_eval_records(path)


def test_eval_records_must_carry_the_generation_fields(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps({"key": 1, "text": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="entities_str"):
        load_eval_records(path)


def test_artefact_names_are_stable_for_dotted_tags(tmp_path: Path):
    responses = response_path(tmp_path, "qwen3-1.7b-notrain")
    assert responses.name == "qwen3-1.7b-notrain_nothink_n8.jsonl"
    assert done_path(tmp_path, "qwen3-1.7b-notrain").name == \
        "qwen3-1.7b-notrain_nothink_n8.done.json"


def test_skipped_records_close_the_key_so_resume_does_not_retry(
        tmp_path: Path, eval_record):
    record = _skipped_record(eval_record, {eval_record["key"]: 40000})
    assert record["responses"] == []
    assert record["skipped_reason"] == "prompt_exceeds_context"

    path = tmp_path / "responses.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    assert load_done_keys(path) == {eval_record["key"]}


def test_split_reasoning_matches_the_frozen_rule():
    assert split_reasoning("ANSWER") == ("ANSWER", "")
    assert split_reasoning("<think>T</think>\nA") == ("A", "T")
    # cut off inside the think block: no answer at all, and that is the truth
    assert split_reasoning("<think>T only") == ("", "T only")


# --------------------------------------------------------- identity on resume

def _identity(**overrides):
    from tcr.evaluation.identity import build_identity
    base = build_identity(eval_data_sha256="EVALSHA", model_path="/ckpt-417",
                          tokenizer_path="/ckpt-417", max_model_len=32768,
                          limit=None)
    base.update(overrides)
    return base


def _started(tmp_path: Path, tag: str = "m"):
    """A responses file with one record already written."""
    from tcr.evaluation.generate import done_path, response_path, run_manifest_path
    output = response_path(tmp_path, tag)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"key": 1, "responses": []}\n', encoding="utf-8")
    return output, run_manifest_path(tmp_path, tag), done_path(tmp_path, tag)


def test_resume_is_allowed_when_the_identity_matches(tmp_path: Path):
    from tcr.evaluation.generate import verify_resume
    from tcr.evaluation.io_utils import write_json

    output, manifest, sentinel = _started(tmp_path)
    write_json(manifest, {"identity": _identity()})
    verify_resume(output, manifest, sentinel, _identity(), tag="m")


def test_resume_stops_when_the_evaluation_file_changed(tmp_path: Path):
    from tcr.evaluation.generate import verify_resume
    from tcr.evaluation.io_utils import write_json

    output, manifest, sentinel = _started(tmp_path)
    write_json(manifest, {"identity": _identity(eval_data_sha256="OTHER")})
    with pytest.raises(SystemExit, match="eval_data_sha256"):
        verify_resume(output, manifest, sentinel, _identity(), tag="m")


def test_resume_stops_when_a_smoke_subset_wrote_the_file(tmp_path: Path):
    """`LIMIT=8` leaves a manifest that would otherwise be read as a full run."""
    from tcr.evaluation.generate import verify_resume
    from tcr.evaluation.io_utils import write_json

    output, manifest, sentinel = _started(tmp_path)
    write_json(sentinel, {"identity": _identity(limit=8)})
    with pytest.raises(SystemExit, match="limit"):
        verify_resume(output, manifest, sentinel, _identity(), tag="m")


def test_resume_stops_when_the_checkpoint_was_replaced(tmp_path: Path):
    from tcr.evaluation.generate import verify_resume
    from tcr.evaluation.io_utils import write_json

    output, manifest, sentinel = _started(tmp_path)
    write_json(manifest, {"identity": _identity(model_path="/ckpt-459")})
    with pytest.raises(SystemExit, match="model_path"):
        verify_resume(output, manifest, sentinel, _identity(), tag="m")


def test_resume_stops_on_an_unidentified_file(tmp_path: Path):
    from tcr.evaluation.generate import verify_resume

    output, manifest, sentinel = _started(tmp_path)
    with pytest.raises(SystemExit, match="no run manifest"):
        verify_resume(output, manifest, sentinel, _identity(), tag="m")

    # ... unless the operator explicitly takes responsibility for it
    verify_resume(output, manifest, sentinel, _identity(), tag="m",
                  allow_unverified=True)


def test_nothing_to_verify_before_the_first_token(tmp_path: Path):
    from tcr.evaluation.generate import done_path, response_path, run_manifest_path
    from tcr.evaluation.generate import verify_resume

    verify_resume(response_path(tmp_path, "m"), run_manifest_path(tmp_path, "m"),
                  done_path(tmp_path, "m"), _identity(), tag="m")
