from __future__ import annotations

import json

import pytest

from conftest import CharTokenizer
from tcr.boundary.anchors import (
    build_all_anchors,
    c_boundaries,
    donor_from_event_row,
    extract_close_text,
    extract_opener_texts,
    modal_text,
    select_candidates,
    tokenize_anchor_paths,
    validate_response,
)
from tcr.boundary.io_utils import sha256_text


class MergingTokenizer:
    """Greedy longest-match tokenizer replicating Qwen's `"},`-style merges."""

    MERGES = ['"},\n', '"}]', '"},', '"}\n', '"}', ' {"', ',\n']

    is_fast = True
    eos_token_id = 999999

    def _tokenize(self, text: str):
        ids, offsets = [], []
        vocab = {tok: 100000 + i for i, tok in enumerate(self.MERGES)}
        i = 0
        while i < len(text):
            for merge in self.MERGES:
                if text.startswith(merge, i):
                    ids.append(vocab[merge])
                    offsets.append((i, i + len(merge)))
                    i += len(merge)
                    break
            else:
                ids.append(ord(text[i]))
                offsets.append((i, i + 1))
                i += 1
        return ids, offsets

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        ids, offsets = self._tokenize(text)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def encode(self, text: str, add_special_tokens: bool = False):
        return self._tokenize(text)[0]


def block_text(index: int) -> str:
    return '{"source":"s%d","target":"t%d"}' % (index, index)


def make_response(n_blocks: int, *, close: str = "]", sep: str = ", ") -> tuple[str, list[dict]]:
    text = "["
    spans = []
    for index in range(n_blocks):
        if index:
            text += sep
        start = len(text)
        body = block_text(index)
        text += body
        spans.append(
            {
                "char_start": start,
                "char_end": start + len(body),
                "token_start": start,
                "token_end": start + len(body),
                "sequence_segment": 0,
                "identity_complete": True,
            }
        )
    text += close
    return text, spans


def event_row(
    *,
    model: str,
    key: str,
    seed: int = 0,
    n_blocks: int = 8,
    close: str = "]",
    sep: str = ", ",
    hit_max: bool = False,
    orbit: bool = False,
    full_json: bool = True,
    capture_start: int | None = None,
) -> tuple[dict, str]:
    text, spans = make_response(n_blocks, close=close, sep=sep)
    row = {
        "sample_id": f"{model}:{key}:{seed}",
        "stable_prompt_id": json.dumps(key),
        "model_tag": model,
        "key": key,
        "seed": seed,
        "selection_roles": ["prevalence"],
        "n_blocks": n_blocks,
        "block_index": spans,
        "legacy_orbit": {"exists": orbit, "hit_max_tokens": hit_max},
        "full_json_list_valid": full_json,
        "motif_capture_triple": (
            {
                "exists": True,
                "second_copy_start_0based": capture_start,
                "motif_triples": [["s", "t", "r"]],
            }
            if capture_start is not None
            else {"exists": False}
        ),
        "response_sha256": sha256_text(text),
    }
    return row, text


def p0c_row(sample_id: str, *, gold_blocks: int = 10, last_new: int | None = None) -> dict:
    return {
        "sample_id": sample_id,
        "gold_blocks": gold_blocks,
        "last_new_relaxed_block_1based": last_new,
    }


def test_select_candidates_rules():
    rows = [
        event_row(model="M0", key="p1")[0],                                   # eligible A
        event_row(model="M0", key="p2", hit_max=True)[0],                     # excluded (hit max)
        event_row(model="M0", key="p3", n_blocks=3)[0],                       # excluded (short)
        event_row(model="M0", key="p4", capture_start=2)[0],                  # excluded (capture)
        event_row(model="M1", key="p1", hit_max=True)[0],                     # eligible B
        event_row(model="M1", key="p2", hit_max=False)[0],                    # excluded (not hitmax)
        event_row(model="M1", key="p5", hit_max=True)[0],                     # excluded (no last_new)
        event_row(model="M1", key="p6", hit_max=True, capture_start=4)[0],    # excluded (boundary >= capture)
        event_row(model="M1", key="p7", hit_max=True, capture_start=7)[0],    # eligible (boundary < capture)
    ]
    donors = [donor_from_event_row(r) for r in rows]
    p0c = {
        "M0:p1:0": p0c_row("M0:p1:0"),
        "M0:p2:0": p0c_row("M0:p2:0"),
        "M0:p3:0": p0c_row("M0:p3:0"),
        "M0:p4:0": p0c_row("M0:p4:0"),
        "M1:p1:0": p0c_row("M1:p1:0", last_new=5),
        "M1:p2:0": p0c_row("M1:p2:0", last_new=5),
        "M1:p5:0": p0c_row("M1:p5:0", last_new=None),
        "M1:p6:0": p0c_row("M1:p6:0", last_new=6),
        "M1:p7:0": p0c_row("M1:p7:0", last_new=6),
    }
    a_drafts, b_drafts, diag = select_candidates(donors, p0c)
    assert [d.donor.prompt_id for d in a_drafts] == [json.dumps("p1")]
    assert sorted(d.donor.key for d in b_drafts) == ["p1", "p7"]
    assert diag["b_boundary_not_before_capture"] == 1
    assert diag["a_state"] == 2
    assert a_drafts[0].boundary_block_1based == 8
    assert all(d.boundary_block_1based == d.last_new_relaxed_block_1based for d in b_drafts)


def test_char_extraction_and_validation():
    row, text = event_row(model="M0", key="p1")
    donor = donor_from_event_row(row)
    validate_response(donor, text)
    with pytest.raises(ValueError, match="SHA mismatch"):
        validate_response(donor, text + " ")

    char_end = donor.blocks[-1].char_end
    assert extract_close_text(text, char_end) == "]"
    assert extract_close_text(text[:char_end] + "\n]", char_end) == "\n]"
    assert extract_close_text(text[:char_end] + " 说明文字很长很长很长很长很长", char_end) is None

    openers = extract_opener_texts(donor, text, before_block_1based=donor.n_blocks)
    assert len(openers) == donor.n_blocks - 1
    assert modal_text(openers) == ', {"source'


def test_tokenize_anchor_paths_aligned_char_tokenizer():
    row, text = event_row(model="M0", key="p1")
    donor = donor_from_event_row(row)
    char_end = donor.blocks[-1].char_end
    payload, reason = tokenize_anchor_paths(
        CharTokenizer(), text=text, char_end=char_end, close_text="]", opener_text=', {"source'
    )
    assert payload is not None, reason
    assert payload["boundary_aligned"] is True
    assert payload["gap_text"] == ""
    assert payload["prefix_text"] == text[:char_end]
    assert payload["prefix_ids"] == [ord(c) for c in text[:char_end]]
    assert payload["close_text"] == "]"
    assert payload["close_ids"] == [ord("]")]
    assert payload["continue_text"] == ', {"source'
    assert payload["continue_ids"][0] == ord(",")


def test_tokenize_anchor_paths_with_qwen_style_merges():
    """Handle tokenizers that merge `"},` / `"}]` into single tokens."""
    row, text = event_row(model="M0", key="p1")
    donor = donor_from_event_row(row)
    char_end = donor.blocks[-1].char_end
    tok = MergingTokenizer()
    payload, reason = tokenize_anchor_paths(
        tok, text=text, char_end=char_end, close_text="]", opener_text=', {"source'
    )
    assert payload is not None, reason
    # close variant tail merges into `"}]`; continue variant into `"},` --
    # the divergence happens ON the block-closing token.
    assert payload["boundary_aligned"] is False
    assert payload["gap_text"] == '"}'
    vocab = {m: 100000 + i for i, m in enumerate(MergingTokenizer.MERGES)}
    assert payload["close_ids"][0] == vocab['"}]']
    assert payload["continue_ids"][0] == vocab['"},']
    assert payload["close_ids"] != payload["continue_ids"]
    # prefix stops right before the closing `"}`
    assert payload["prefix_text"] == text[: char_end - 2]
    assert payload["prefix_ids"] == tok.encode(text)[: len(payload["prefix_ids"])]
    # continue path covers through `"source`
    assert '"source' in payload["continue_text"]


def test_c_boundaries_dedup_and_clip():
    row, _text = event_row(model="M1", key="p1", n_blocks=10, hit_max=True)
    donor = donor_from_event_row(row)
    picks = c_boundaries(donor, 8)
    assert picks == [(0.5, 4), (0.75, 6), (1.0, 8), (1.25, 10)]
    picks = c_boundaries(donor, 100)
    assert picks == [(0.5, 10)]


@pytest.mark.parametrize("tokenizer_cls", [CharTokenizer, MergingTokenizer])
def test_build_all_anchors_end_to_end(tokenizer_cls):
    a_rows = [event_row(model="M0", key=f"a{i}") for i in range(4)]
    b_rows = [
        event_row(model="M1", key=f"b{i}", n_blocks=12, hit_max=True) for i in range(3)
    ]
    donors = [donor_from_event_row(r) for r, _t in a_rows + b_rows]
    texts = {d.sample_id: t for (r, t), d in zip(a_rows + b_rows, donors)}
    p0c = {}
    for donor in donors:
        last_new = 6 if donor.model_tag == "M1" else None
        p0c[donor.sample_id] = p0c_row(donor.sample_id, gold_blocks=8, last_new=last_new)
    a_drafts, b_drafts, _diag = select_candidates(donors, p0c)

    def lookup(donor):
        return {
            "response": texts[donor.sample_id],
            "finish_reason": "stop" if donor.model_tag == "M0" else "length",
            "text": "原文",
            "entities_str": '["e"]',
        }

    def prompt_builder(text: str, entities: str) -> str:
        return f"PROMPT<{text}|{entities}>"

    anchors, manifest = build_all_anchors(
        a_drafts=a_drafts,
        b_drafts=b_drafts,
        response_lookup=lookup,
        prompt_builder=prompt_builder,
        tokenizer=tokenizer_cls(),
        n_a=3,
        n_b=3,
        n_c_prompts=2,
    )
    a_anchors = [a for a in anchors if a.anchor_type == "A"]
    b_anchors = [a for a in anchors if a.anchor_type == "B"]
    c_anchors = [a for a in anchors if a.anchor_type == "C"]
    assert len(a_anchors) == 3 and len(b_anchors) == 3
    assert manifest["n_c"] == len(c_anchors) > 0
    assert manifest["modal_close_tail_text"] == "]"
    for anchor in anchors:
        assert anchor.prefix_token_count == len(anchor.prefix_response_ids)
        assert anchor.close_tail_ids[0] != anchor.continue_ids[0]
        assert anchor.prefix_response_text == texts[anchor.sample_id][: len(anchor.prefix_response_text)]
    for anchor in a_anchors:
        assert anchor.close_tail_source == "own"
        assert anchor.boundary_block_1based == anchor.n_blocks
        assert anchor.prompt_text.startswith("PROMPT<")
    for anchor in b_anchors:
        assert anchor.close_tail_source == "modal"
        assert anchor.boundary_block_1based == 6
        assert anchor.continue_source == "own_modal"
    fractions = {a.notes.get("c_fraction") for a in c_anchors}
    assert fractions <= {0.5, 0.75, 1.0, 1.25}
    blocks = sorted({a.boundary_block_1based for a in c_anchors})
    assert blocks == [4, 6, 8, 10]
    if tokenizer_cls is CharTokenizer:
        assert all(a.boundary_aligned for a in anchors)
        assert manifest["n_boundary_aligned"] == len(anchors)
    else:
        assert all(not a.boundary_aligned for a in anchors)
        assert set(manifest["gap_text_counts"]) == {'"}'}


def test_build_all_anchors_reports_diagnostics_on_total_failure():
    row, text = event_row(model="M0", key="a0")
    donor = donor_from_event_row(row)
    p0c = {donor.sample_id: p0c_row(donor.sample_id)}
    a_drafts, b_drafts, _ = select_candidates([donor], p0c)

    def lookup(_donor):
        return {"response": text, "finish_reason": "length", "text": "t", "entities_str": "e"}

    with pytest.raises(RuntimeError, match="a_finish_reason_not_stop"):
        build_all_anchors(
            a_drafts=a_drafts,
            b_drafts=b_drafts,
            response_lookup=lookup,
            prompt_builder=lambda t, e: "P",
            tokenizer=CharTokenizer(),
            n_a=1,
            n_b=1,
            n_c_prompts=1,
        )
