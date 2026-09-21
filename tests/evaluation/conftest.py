# coding=utf-8
"""Shared fixtures.  No GPU, no vLLM, no tokenizer download.

`build_event_row` takes token ids and offsets as arguments rather than a
tokenizer, so the whole structured/legacy pipeline is testable with a
deterministic character-level fake: one token per character, offsets
``(i, i+1)``.  That is a legitimate tokenisation for the detector's purposes
(it only ever compares ids and maps token indices to char positions), and it
keeps the tests runnable on a laptop with no model files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def fake_encode(text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """One token per character."""
    return [ord(char) for char in text], [(index, index + 1)
                                          for index in range(len(text))]


def block_text(source: str, target: str, relation: str = "关系",
               description: str = "这是一个用于测试的描述，长度足够让块超过若干 token。") -> str:
    return json.dumps({"source": source, "target": target,
                       "relation": relation, "description": description},
                      ensure_ascii=False)


def response_of(blocks: List[str], *, closed: bool = True) -> str:
    return "[" + ", ".join(blocks) + ("]" if closed else "")


@pytest.fixture
def looping_response() -> str:
    """One block repeated 60 times: a textbook stable orbit."""
    return response_of([block_text("甲", "乙")] * 60, closed=False)


@pytest.fixture
def clean_response() -> str:
    """Five distinct blocks, properly closed: no reuse at all."""
    return response_of([block_text(f"实体{index}", f"目标{index}")
                        for index in range(5)])


@pytest.fixture
def eval_record() -> Dict[str, Any]:
    return {
        "key": 1,
        "source": "unit_test",
        "text": "甲和乙之间存在关系。实体0 与 目标0 也相关。",
        "entities_str": "['甲', '乙', '实体0', '目标0']",
        "output": [{"source": "甲", "target": "乙", "relation": "关系",
                    "description": "d"}],
    }
