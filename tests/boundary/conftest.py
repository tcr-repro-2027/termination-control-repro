from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class CharTokenizer:
    """1 char = 1 token fake fast tokenizer (ids are codepoints)."""

    is_fast = True
    eos_token_id = 999999

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        ids = [ord(ch) for ch in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens: bool = True):
        return "".join(chr(i) for i in ids if i < 999999)
