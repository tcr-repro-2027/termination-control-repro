# coding=utf-8
"""One response -> one 01b-schema event row (structured blocks + legacy orbit).

The primary repetition definition in E1 is structural, not lexical: a response
is a sequence of COMPLETE four-field relation objects, and the events of
interest are defined on that sequence --

* ``first_triple_reuse``      the earliest block whose (source,target,relation)
                              already appeared;
* ``motif_capture_triple``    the earliest block motif repeated three times
                              back to back (the "semantic capture");
* ``legacy_orbit``            the v1.0 raw token-period loop, kept for
                              continuity with `loop_results.csv`.

That ordering is the reason this project stopped scoring repetition with the
token-period detector alone: on a stream of JSON dicts its onset lands wherever
the token periodicity happens to start, which is routinely the middle of a
dict, so neither the onset nor the period corresponds to anything the model
"decided".  Block coordinates do.

Nothing here re-derives those definitions.  All of them come from
`tcr/events`, which is the frozen source of
truth, and the legacy record is produced by `tcr.token_orbit.score_response`
itself -- fed the tokenisation this module already computed, so the two views
of a response can never disagree about its tokens.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from . import protocol
from tcr.token_orbit import DetectorConfig, PeriodConfig, build_text, score_response
from tcr.events.audit import audit_response

Offsets = Sequence[Tuple[int, int]]

__all__ = [
    "build_event_row", "encode_batch", "legacy_detector", "load_tokenizer",
    "response_text", "score_legacy",
]


# ---------------------------------------------------------------- tokenizer

def load_tokenizer(path: str):
    """A fast tokenizer; offsets (hence every onset) depend on it."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(f"tokenizer at {path!r} is not a fast tokenizer; "
                         "exact offset mapping is required")
    return tokenizer


def encode_batch(tokenizer, texts: Sequence[str]) -> List[Tuple[List[int], List[Tuple[int, int]]]]:
    """Batch-encode with offsets, keeping empty texts exactly empty.

    An empty string is not sent to the tokenizer at all: some fast tokenizers
    return a BOS-like artefact for it, which would give an empty response a
    non-zero ``gen_len``.
    """
    filled = [index for index, text in enumerate(texts) if text]
    encoded: Dict[int, Tuple[List[int], List[Tuple[int, int]]]] = {}
    if filled:
        batch = tokenizer([texts[index] for index in filled],
                          add_special_tokens=False,
                          return_offsets_mapping=True)
        for position, index in enumerate(filled):
            ids = [int(value) for value in batch["input_ids"][position]]
            offsets = [(int(start), int(end))
                       for start, end in batch["offset_mapping"][position]]
            encoded[index] = (ids, offsets)
    return [encoded.get(index, ([], [])) for index in range(len(texts))]


# ------------------------------------------------------------------- legacy

def legacy_detector() -> DetectorConfig:
    """The frozen v1.0 token-period detector configuration."""
    return DetectorConfig(
        threshold=protocol.LEGACY_THRESHOLD,
        period=PeriodConfig(protocol.LEGACY_PERIOD_MIN,
                            protocol.LEGACY_PERIOD_MAX,
                            protocol.LEGACY_MIN_REPEATS),
        before_tokens=protocol.LEGACY_BEFORE_TOKENS,
    )


def response_text(sample: Dict[str, Any]) -> str:
    """The analysed text for one sample, per the frozen detection target."""
    return build_text(sample.get("response", ""), sample.get("reasoning", ""),
                      protocol.LEGACY_TARGET)


def score_legacy(text: str, token_ids: Sequence[int], offsets: Offsets,
                 finish_reason: str, config: DetectorConfig) -> Dict[str, Any]:
    """`tcr.token_orbit.score_response` on an ALREADY tokenised text.

    Passing a constant `encode` is what pins the legacy record and the block
    record to one tokenisation; `score_response` tokenises exactly once, so
    this is the frozen implementation, not a re-implementation of it.
    """
    return score_response(text, finish_reason, lambda _text: (token_ids, offsets),
                          config)


# ---------------------------------------------------------------- event row

def build_event_row(*, model_tag: str, key: Any, seed: int, text: str,
                    token_ids: Sequence[int], offsets: Offsets,
                    finish_reason: str, prompt_tokens: int | None = None,
                    gen_tokens_engine: int | None = None,
                    config: DetectorConfig | None = None,
                    keep_diagnostics: bool = False,
                    keep_snippets: bool = False,
                    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """`(event_row, audit_snippets)` for one response.

    The snippets are returned separately rather than embedded, because they are
    for human inspection of a bounded sample and are far too bulky to carry on
    all 8848 rows of every one of ~63 tasks.

    The extra top-level keys (`severity`, `finish_reason`, `prompt_tokens`,
    `gen_tokens_engine`) are additions, not changes: every field the 01b
    P0b/P0c/P0d analysers read keeps its frozen name, meaning and value, so
    `event_rows.jsonl` written here can be fed to them directly (see
    `scripts/e1_make_pair.py` for the M0/M1 relabelling they expect).

    ``parse_diagnostics`` and ``audit_snippets`` are dropped by default.  They
    carry the raw text of every malformed object and up to ~3 KB of context per
    response -- a large multiple of everything else, for fields no analyser
    reads.  `parse_diagnostic_counts`, which they do read, is kept.
    """
    detector = config or legacy_detector()
    legacy = score_legacy(text, token_ids, offsets, finish_reason, detector)
    row = audit_response(
        model_tag=model_tag,
        key=key,
        seed=int(seed),
        text=text,
        token_ids=token_ids,
        offsets=offsets,
        legacy_sample=legacy,
        selection_roles=("prevalence",),
        capture_min_repeats=protocol.CAPTURE_MIN_REPEATS,
        max_motif_period_blocks=protocol.MAX_MOTIF_PERIOD_BLOCKS,
        legacy_min_repeats=protocol.LEGACY_MIN_REPEATS,
        reject_extra_fields=protocol.REJECT_EXTRA_FIELDS,
    )
    snippets = row.pop("audit_snippets", None)
    if keep_snippets and snippets is not None:
        row["audit_snippets"] = snippets
    if not keep_diagnostics:
        row.pop("parse_diagnostics", None)

    row["finish_reason"] = finish_reason
    row["prompt_tokens"] = prompt_tokens
    row["gen_tokens_engine"] = gen_tokens_engine
    row["gen_tokens_mismatch"] = (
        None if gen_tokens_engine is None
        else int(gen_tokens_engine) != int(row["gen_tokens"]))
    row["severity"] = {
        "rep4": legacy["rep4"],
        "gzip_ratio": legacy["gzip_ratio"],
        "gen_len": legacy["gen_len"],
        "hit_max_tokens": legacy["hit_max_tokens"],
    }
    return row, snippets or {}
