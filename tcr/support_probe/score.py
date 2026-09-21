# coding=utf-8
"""Teacher-forced StopMargin for ONE model over the whole anchor set.

One forward pass per (anchor, variant): the context is run once and the full
final-position distribution is read, which is what makes the exact rank and the
top-k/top-p survival available.  vLLM cannot supply those -- its `logprobs` are
truncated to the top few -- so scoring uses HuggingFace directly and generation
(`e2.hazard`) uses vLLM, each for what it is good at.

Two implementation details that matter more than they look
---------------------------------------------------------
* **Only the last position's logits are materialised.**  A full-sequence logit
  tensor for a 12k-token context over a 150k vocabulary is several gigabytes
  per sequence; running the base model and applying `lm_head` to the final
  hidden state alone is exact and costs nothing.
* **Left padding.**  With left padding the last position of every row in a
  batch is the decision point, so a batch needs no per-row gather and cannot
  silently read the wrong position.
* **The assistant prefix is tokenised once per anchor**, not once per variant
  and future (`DecisionPointFinder`): the 8 variants of an anchor differ only
  in the prompt.  Measured 4.4x faster on the tokenisation pass, with the first
  cells checked token-for-token against the reference implementation.

Example::

    python -m tcr.support_probe.score --anchors anchors/anchors.jsonl \
      --model_path ./outputs/checkpoints/qwen3-4b-cleanv2-s42/checkpoint-417 \
      --tag qwen3-4b-cleanv2-s42 --output_root ./outputs/support_probe --gpu 0
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from . import protocol
from .anchors import anchor_from_dict, occurrence_gap
from .boundary import DecisionPointFinder, sampler_readouts
from .identity import guard_partial, run_identity, sentinel_says_done
from .io_utils import (
    JsonlWriter, load_done_keys, read_jsonl, sanitize_filename, sha256_file,
    write_json,
)
from .prompts import build_user_message, prompt_fingerprint

LOG = logging.getLogger("tcr.support_probe.score")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teacher-forced StopMargin readouts for one model.")
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--gpu", default=None,
                        help="Single GPU id; sets CUDA_VISIBLE_DEVICES before "
                             "torch is imported.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Contexts per forward.  Anchors carry the whole "
                             "emitted prefix, so they are long; 2 is safe for "
                             "8B at 24k tokens on an 80 GB card.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test: only the first N anchors.")
    parser.add_argument("--log_file", default=None)
    return parser.parse_args(argv)


def readout_path(root: str | os.PathLike[str], tag: str) -> Path:
    return Path(root) / "readouts" / f"{sanitize_filename(tag)}_readouts.jsonl"


def done_path(root: str | os.PathLike[str], tag: str) -> Path:
    return readout_path(root, tag).with_suffix(".done.json")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    log_file = args.log_file or f"logs/score_{sanitize_filename(args.tag)}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_file, encoding="utf-8")], force=True)

    started = time.time()
    tokenizer_path = args.tokenizer_path or args.model_path
    out_path = readout_path(args.output_root, args.tag)
    sentinel = done_path(args.output_root, args.tag)
    identity = run_identity(anchors_path=args.anchors,
                            model_path=args.model_path,
                            tokenizer_path=tokenizer_path, limit=args.limit)

    if sentinel_says_done(sentinel, identity, args.tag, "score"):
        LOG.info("[%s] already scored", args.tag)
        return 0

    anchors = list(read_jsonl(args.anchors))
    if args.limit:
        anchors = anchors[:args.limit]
        LOG.warning("[%s] --limit %d: this is a smoke test and will NOT write "
                    "the completion sentinel", args.tag, args.limit)
    guard_partial(out_path, identity, args.tag)
    done = load_done_keys(out_path, key_field="cell_id")
    LOG.info("[%s] %d anchors, %d cells already done", args.tag, len(anchors),
             len(done))

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit(f"tokenizer at {tokenizer_path} is not a fast tokenizer")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(text: str) -> List[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    # One tokenisation of the assistant prefix per ANCHOR (its 8 variants share
    # it) plus a short tail window per branch, checked against the reference on
    # the first few cells.  See DecisionPointFinder.
    finder = DecisionPointFinder(encode, tokenizer.decode)

    # `torch_dtype` was renamed to `dtype`; from_pretrained absorbs an unknown
    # kwarg into the config instead of raising, so guessing wrong would load
    # fp32 silently -- twice the memory and different numerics, no error.  Try
    # the current spelling, fall back to the old one, then VERIFY.
    def load_model(**extra):
        return AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=None, attn_implementation="sdpa",
            **extra)

    try:
        model = load_model(dtype=torch.bfloat16)
        if model.dtype != torch.bfloat16:
            raise TypeError("dtype kwarg ignored")
    except (TypeError, ValueError):
        model = load_model(torch_dtype=torch.bfloat16)
    if model.dtype != torch.bfloat16:
        LOG.warning("model loaded as %s, not bfloat16; casting", model.dtype)
        model = model.to(torch.bfloat16)
    LOG.info("[%s] model dtype=%s", args.tag, model.dtype)
    model.eval().cuda()
    base = getattr(model, "model", None)
    head = getattr(model, "lm_head", None)
    if base is None or head is None:               # pragma: no cover
        raise SystemExit("model does not expose .model/.lm_head; cannot read "
                         "only the final position's logits")

    # ---- build every context first: cheap, and it surfaces a broken chat
    # template or an over-long anchor before any GPU time is spent.
    # Checked BEFORE the resume filter, and on every run: gate 4 reads
    # `chat_prefix_ok` out of the manifest, and a resumed run with nothing left
    # to score would otherwise write it as false and fail a gate about the
    # instrument for a reason that has nothing to do with the instrument.
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "probe"}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    prefix_ok = probe.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    if not prefix_ok:
        raise SystemExit(
            "chat template does not end with the training-time nothink "
            "prefix; the margin would be conditioned on a different state "
            "than training produced")

    cells: List[Dict[str, Any]] = []
    for row in anchors:
        anchor = anchor_from_dict(row)
        # Carried onto every cell so the analysis can report the slope on the
        # subset where the manipulated and neutral arms rewrote exactly the
        # same amount of text (§9.5 sensitivity); computed here rather than
        # stored in the anchor file so it costs no rebuild.
        gap = occurrence_gap(anchor)
        for variant in anchor.variants:
            cell_id = f"{anchor.anchor_id}|{variant.variant_id}"
            if cell_id in done:
                continue
            user = build_user_message(variant.text, variant.entities_str,
                                      variant.instruction)
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            point = finder.find(
                prompt_text=chat, prefix_text=anchor.assistant_prefix,
                close_continuation=anchor.close_continuation,
                continue_continuation=anchor.continue_continuation)
            cells.append({"cell_id": cell_id, "anchor": anchor,
                          "variant": variant, "point": point,
                          "occurrence_gap": gap})
    LOG.info("[%s] %d cells to score; longest context %d tokens", args.tag,
             len(cells), max((c["point"].n_context_tokens for c in cells),
                             default=0))

    cells.sort(key=lambda cell: cell["point"].n_context_tokens)
    pad_id = tokenizer.pad_token_id
    written = 0
    with JsonlWriter(out_path, mode="a") as handle:
        for start in range(0, len(cells), args.batch_size):
            chunk = cells[start:start + args.batch_size]
            width = max(len(cell["point"].context_ids) for cell in chunk)
            input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            attention = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, cell in enumerate(chunk):
                ids = cell["point"].context_ids
                input_ids[row, width - len(ids):] = torch.tensor(ids)
                attention[row, width - len(ids):] = 1
            input_ids = input_ids.cuda()
            attention = attention.cuda()

            with torch.no_grad():
                hidden = base(input_ids=input_ids,
                              attention_mask=attention).last_hidden_state[:, -1, :]
                logits = head(hidden).float().cpu().numpy()

            for row, cell in enumerate(chunk):
                anchor, variant, point = cell["anchor"], cell["variant"], cell["point"]
                readouts = sampler_readouts(
                    np.asarray(logits[row]), close_token=point.close_token,
                    continue_token=point.continue_token,
                    presence_ids=point.presence_ids)
                handle.write({
                    "cell_id": cell["cell_id"],
                    "tag": args.tag,
                    "anchor_id": anchor.anchor_id,
                    "anchor_type": anchor.anchor_type,
                    "key": anchor.key,
                    "source": anchor.source,
                    "n_remainder": len(anchor.remainder),
                    "occurrence_gap": cell["occurrence_gap"],
                    "variant_id": variant.variant_id,
                    "arm": variant.arm,
                    "axis": variant.axis,
                    "level": variant.level,
                    "n_edits": variant.n_edits,
                    "n_remaining_supported": variant.n_remaining_supported,
                    "n_context_tokens": point.n_context_tokens,
                    "close_token": point.close_token,
                    "continue_token": point.continue_token,
                    "close_token_str": tokenizer.decode([point.close_token]),
                    "continue_token_str": tokenizer.decode([point.continue_token]),
                    **readouts,
                })
                written += 1
            if written % 200 < args.batch_size:
                LOG.info("[%s] %d/%d cells (%.1f min)", args.tag, written,
                         len(cells), (time.time() - started) / 60.0)

    elapsed = time.time() - started
    if args.limit:
        LOG.warning("[%s] scored %d cells of a LIMITED run in %.1f min; no "
                    "sentinel written, so a full run still has work to do",
                    args.tag, written, elapsed / 60.0)
        return 0
    write_json(sentinel, {
        "tag": args.tag, "identity": identity,
        "anchors": str(args.anchors),
        # the file's total, not this attempt's: on a resume `written` counts
        # only the cells this process added, and a manifest saying 40 cells
        # for a complete 2560-cell readout is worse than no count at all
        "n_cells": len(done) + written,
        "n_cells_this_run": written,
        "n_anchors": len(anchors), "readouts": str(out_path),
        # Binds the manifest to the FILE, not just to its path and row count.
        # Without it a readout file swapped in by hand satisfies every other
        # check -- same cell ids, same count, same manifest.
        "readouts_sha256": sha256_file(out_path),
        "score_minutes": round(elapsed / 60.0, 2),
        "chat_prefix_ok": bool(prefix_ok),
        **finder.manifest(),
        **prompt_fingerprint(), **protocol.describe(),
    })
    LOG.info("[%s] scored %d cells in %.1f min -> %s", args.tag, written,
             elapsed / 60.0, out_path)
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
