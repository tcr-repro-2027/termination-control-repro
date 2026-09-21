# coding=utf-8
"""Tokenizer-backed ``encode`` factory.

The detector is tokenizer-agnostic and only needs an ``encode(text) ->
(token_ids, offset_mapping)`` callable.  This module provides the default
implementation backed by a HuggingFace *fast* tokenizer (which is what gives
us the ``offset_mapping`` used to map token indices back to char positions).

``transformers`` is imported lazily inside the factory so that importing the
rest of the library (and supplying your own ``encode``) does not require it.
"""

from typing import List, Sequence, Tuple


def build_hf_encode(tokenizer_path: str):
    """Load a HF fast tokenizer and return an ``encode(text)->(ids, offsets)``.

    ``tokenizer_path`` is a HF hub name or a local directory.  The tokenizer
    must be a *fast* tokenizer (it must support ``return_offsets_mapping``).
    Token counting, ``onset`` indices and the captured ``contextBefore`` all
    depend on this tokenizer matching the deployed model.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError(
            f"tokenizer at {tokenizer_path!r} is not a fast tokenizer; "
            "offset mapping (and hence onset positions) requires one."
        )

    def encode(text: str) -> Tuple[List[int], Sequence[Tuple[int, int]]]:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    return encode
