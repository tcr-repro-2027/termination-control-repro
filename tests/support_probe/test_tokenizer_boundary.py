# coding=utf-8
"""Pin the tokenizer facts the decision-point logic exists for.

These were measured, not assumed, and they are the reason `find_decision_point`
cannot be replaced by "cut after `}` and read the next token".  Keeping them as
a test means the claim is re-checked on the real tokenizer instead of living in
a comment -- and if a future model's BPE really does leave a clean boundary, the
test says so out loud rather than the code quietly doing needless work.

Skipped when `transformers` or the Qwen3 tokenizer is not on this machine, so
the suite still runs on a laptop.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tcr.support_probe.anchors import (
    CLOSE_CONTINUATION, CONTINUE_CONTINUATION_BLOCK, CONTINUE_CONTINUATION_EMPTY,
    assistant_prefix,
)
from tcr.support_probe.boundary import (
    DecisionPointFinder, common_prefix_length, find_decision_point,
)

def _find_tokenizer() -> Path:
    """Where the Qwen3 tokenizer is, across the layouts this has lived in.

    `E2_TOKENIZER_DIR` overrides; otherwise `$MODEL_ROOT/Qwen3-4B`, then
    `<repo>/models/Qwen3/Qwen3-4B`.  Without a tokenizer these tests skip.
    """
    override = os.environ.get("E2_TOKENIZER_DIR")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    model_root = os.environ.get("MODEL_ROOT")
    candidates = [
        *([Path(model_root) / "Qwen3-4B"] if model_root else []),
        here.parents[2] / "models" / "Qwen3" / "Qwen3-4B",
    ]
    for path in candidates:
        if (path / "tokenizer.json").is_file():
            return path
    return candidates[0]


TOKENIZER_DIR = _find_tokenizer()

pytest.importorskip("transformers", reason="tokenizer facts need transformers")
pytestmark = pytest.mark.skipif(
    not (TOKENIZER_DIR / "tokenizer.json").is_file(),
    reason=f"Qwen3 tokenizer not present at {TOKENIZER_DIR}; set "
           "E2_TOKENIZER_DIR to point at one")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)


@pytest.fixture(scope="module")
def encode(tok):
    return lambda text: tok(text, add_special_tokens=False)["input_ids"]


def blocks(n: int = 3):
    return [{"source": f"上海市食品药品监督管理局{i}", "target": f"化妆品生产企业{i}",
             "relation": "监管职能承担方",
             "description": "负责生产、流通、消费环节的监管职能。"}
            for i in range(n)]


def test_the_boundary_strings_really_are_single_tokens(encode):
    """The premise of the whole design: `}` merges forward in this vocabulary."""
    for piece in ("}]", "},", "[]"):
        assert len(encode(piece)) == 1, f"{piece!r} is not a single token"


def test_the_block_boundary_is_never_a_clean_token_boundary(encode):
    """Cutting after `}` would leave the model on `"}` while the trained
    sequence carries `"},` there."""
    prefix = assistant_prefix(blocks())
    assert prefix.endswith("}")
    context_ids = encode(prefix)
    close_ids = encode(prefix + CLOSE_CONTINUATION)
    cont_ids = encode(prefix + CONTINUE_CONTINUATION_BLOCK)
    position = common_prefix_length(close_ids, cont_ids)

    assert position < len(context_ids), (
        "the futures diverge before the end of the context's own tokenisation, "
        "which is exactly why the naive cut is wrong")
    assert close_ids[position] != cont_ids[position]


def test_the_empty_list_boundary_also_merges(encode, tok):
    """`[]` is one token, so the empty-answer decision resolves ON the `[`:
    the branches are `[]` against `[`, and nothing has been generated yet."""
    close_ids = encode("[" + CLOSE_CONTINUATION)
    cont_ids = encode("[" + CONTINUE_CONTINUATION_EMPTY)
    position = common_prefix_length(close_ids, cont_ids)
    assert position == 0
    assert tok.decode([close_ids[0]]) == "[]"
    assert tok.decode([cont_ids[0]]) == "["

    point = find_decision_point(encode, "[", CLOSE_CONTINUATION,
                                CONTINUE_CONTINUATION_EMPTY,
                                generation_prompt="")
    assert point.presence_ids == []         # legitimately empty, not a bug


def test_decision_point_recovers_both_branches(encode, tok):
    prefix = assistant_prefix(blocks())
    point = find_decision_point(encode, prefix, CLOSE_CONTINUATION,
                                CONTINUE_CONTINUATION_BLOCK)
    close = tok.decode([point.close_token])
    cont = tok.decode([point.continue_token])
    assert close.endswith("}")              # the closing branch commits to `}`
    assert cont.endswith(",")               # the continuing branch to `},`
    # and the shared context is a real tokenisation of a real string
    assert point.context_ids == encode(tok.decode(point.context_ids))


def test_the_chat_and_prefix_join_is_token_additive(encode):
    """What lets the finder tokenise an assistant prefix once per anchor."""
    prompt = "<|im_start|>user\n甲乙丙<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix = assistant_prefix(blocks())
    assert encode(prompt + prefix) == encode(prompt) + encode(prefix)


def test_the_fast_finder_equals_the_reference(encode, tok):
    """The optimisation must be invisible in the numbers, not just fast."""
    prompt = "<|im_start|>user\n甲乙丙<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    finder = DecisionPointFinder(encode, tok.decode, verify_first=0)
    for n in (1, 2, 5, 9):
        prefix = assistant_prefix(blocks(n))
        fast = finder.find(prompt_text=prompt, prefix_text=prefix,
                           close_continuation=CLOSE_CONTINUATION,
                           continue_continuation=CONTINUE_CONTINUATION_BLOCK)
        slow = find_decision_point(encode, prompt + prefix, CLOSE_CONTINUATION,
                                   CONTINUE_CONTINUATION_BLOCK,
                                   generation_prompt=prompt)
        assert fast.context_ids == slow.context_ids
        assert fast.close_token == slow.close_token
        assert fast.continue_token == slow.continue_token
        assert fast.generated_start == slow.generated_start
    assert not finder.fell_back


def test_the_finder_falls_back_instead_of_being_wrong(encode, tok):
    """A tokenizer where the window trick does not hold must cost speed, not
    correctness."""
    def broken_decode(ids):
        return "@@@" + tok.decode(ids)       # corrupts the tail window

    finder = DecisionPointFinder(encode, broken_decode, verify_first=1)
    prompt = "<|im_start|>user\n甲<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix = assistant_prefix(blocks(2))
    point = finder.find(prompt_text=prompt, prefix_text=prefix,
                        close_continuation=CLOSE_CONTINUATION,
                        continue_continuation=CONTINUE_CONTINUATION_BLOCK)
    assert finder.fell_back
    reference = find_decision_point(encode, prompt + prefix, CLOSE_CONTINUATION,
                                    CONTINUE_CONTINUATION_BLOCK,
                                    generation_prompt=prompt)
    assert point.context_ids == reference.context_ids
    assert finder.manifest()["decision_point_fell_back"] is True
