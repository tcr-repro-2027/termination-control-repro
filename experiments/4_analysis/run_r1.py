# coding: utf-8
"""R1: on ONE prefix, do the models still choose differently?

Scientific question (H3 / contribution C3).  R0 shows where the trajectories
diverge; it cannot show that the divergence is a decision, because the two
models were never in the same state.  Here every model scores the SAME pool of
natural prefixes, so the input, the generated text so far and the two candidate
continuations are identical and only the parameters differ.

Three readouts, deliberately not collapsed into one
---------------------------------------------------
1. **Scores.**  ``margin_first_policy`` is the processed-logit difference at the
   first token where the close path and the continue path diverge;
   ``margin_path_policy`` is S1's two short-path log-probability difference.
   They are not equivalent and neither is "the probability of stopping": a path
   score is one route to closing, not a sum over every way a response could end.
2. **Reachability.**  ``close_prob_pre_filter`` is the model's own probability
   for the close token; ``close_sampler_prob`` is what survives top-k then
   top-p and gets renormalised.  A score can move a long way without ever
   crossing the sampler's truncation, and that difference is the whole content
   of "the distribution changed but the behaviour did not".
3. **Behaviour.**  Sixteen short continuations under the deployed sampler, each
   classified by its first decisive character.  ``next_event_close_rate`` is the
   only readout here that is an action rather than a number about one.

Everything that is not a legal close or a legal continue is kept and named --
``eos_without_close`` (the model ended the sequence without closing the list)
and ``unresolved`` (32 tokens with no decisive character).  Folding those into
"stop" would count a malformed termination as a success.

A fourth, smaller readout separates CHOOSING to close from being ABLE to: given
the complete close branch already written, how much probability does the next
step put on EOS?  A model that closes rarely but executes a given close
correctly is a different failure from one that cannot write the ending at all.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (REPO_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tcr.paper import registry, stats                                     # noqa: E402
from tcr.paper.io import (append_jsonl, fmt, fmt_pct, iter_jsonl, log,     # noqa: E402
                   read_csv, write_compact, write_csv, write_json, write_jsonl)
from tcr.paper.layout import Layout, model_path_for                       # noqa: E402
from tcr.paper.sampler import (anchor_seed, boundary_metrics,             # noqa: E402
                        close_branch_readout, sample_next_event)
from tcr.boundary.constants import (K_RESAMPLE, MAX_NEW_TOKENS,    # noqa: E402
                                     NOTHINK_SAMPLING, SAMPLE_BATCH, S1_VERSION)
from tcr.boundary.measure import measure_anchor                    # noqa: E402
from tcr.boundary.policy import assert_frozen_sampling             # noqa: E402
from tcr.boundary.runtime import (audit_chat_template, build_chat_ids,  # noqa: E402
                                   load_model, load_tokenizer, stop_token_ids)


def measure_model(tag: str, anchors: Sequence[Mapping[str, Any]], *,
                  layout: Layout, out_path: Path, device: str, k_resample: int,
                  max_new_tokens: int, sample_batch: int, attn: str | None,
                  metrics_row: Mapping[str, Any] | None) -> int:
    done: set[str] = set()
    if out_path.exists():
        done = {str(row["anchor_id"]) for row in iter_jsonl(out_path)}
    todo = [anchor for anchor in anchors if str(anchor["anchor_id"]) not in done]
    log("r1", f"{tag}: {len(anchors)} anchors, {len(done)} done, {len(todo)} to do")
    if not todo:
        return 0

    model_path = model_path_for(layout, tag, metrics_row=metrics_row)
    tokenizer = load_tokenizer(model_path)
    audit_chat_template(tokenizer)
    stops = stop_token_ids(model_path, tokenizer)
    model, backend = load_model(model_path, device=device, attn_implementation=attn)
    log("r1", f"{tag}: loaded {model_path} ({backend}); stop_ids={stops}")

    chat_cache: dict[str, list[int]] = {}
    written = 0
    for index, anchor in enumerate(todo, start=1):
        started = time.time()
        prompt_sha = str(anchor["prompt_sha256"])
        chat_ids = chat_cache.get(prompt_sha)
        if chat_ids is None:
            chat_ids = build_chat_ids(tokenizer, anchor["prompt_text"])
            chat_cache[prompt_sha] = chat_ids
        prefix_ids = [int(v) for v in anchor["prefix_response_ids"]]
        close_ids = [int(v) for v in anchor["close_tail_ids"]]
        continue_ids = [int(v) for v in anchor["continue_ids"]]
        # S1's frozen two-path margins, unchanged.
        row = measure_anchor(model, tokenizer, anchor, chat_ids=chat_ids,
                             stop_ids=stops, device=device, with_hazard=False)
        # One teacher-forced forward gives both the boundary distribution the
        # reachability readouts need and the EOS-after-a-written-close diagnostic.
        branch = close_branch_readout(
            model, chat_ids=chat_ids, prefix_response_ids=prefix_ids,
            close_ids=close_ids, stop_ids=stops, device=device)
        row.update(boundary_metrics(
            branch.pop("_boundary_logits"), presence_ids=prefix_ids,
            close_first=close_ids[0], continue_first=continue_ids[0]))
        row.update(branch)
        behaviour = sample_next_event(
            model, tokenizer, chat_ids=chat_ids, prefix_response_ids=prefix_ids,
            stop_ids=stops, device=device, k_resample=k_resample,
            max_new_tokens=max_new_tokens, sample_batch=sample_batch,
            base_seed=anchor_seed(str(anchor["anchor_id"])))
        behaviour.pop("_boundary_logits", None)
        row.update(behaviour)
        row.update({
            "model_tag": tag,
            "model_path": str(model_path),
            "role": anchor.get("role"),
            "split": anchor.get("split"),
            "scale": anchor.get("scale"),
            "source_tag": anchor.get("source_tag"),
            "remaining_gold_pairs": anchor.get("remaining_gold_pairs"),
            "attn_backend": backend,
            "s1_protocol": S1_VERSION,
            "seconds": round(time.time() - started, 2),
        })
        append_jsonl(out_path, row)
        written += 1
        if index % 10 == 0 or index == len(todo):
            log("r1", f"{tag}: {index}/{len(todo)}")
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return written


# ------------------------------------------------------------------ aggregate

BEHAVIOUR_FIELDS = ("next_event_close_rate", "continue_rate",
                    "eos_without_close_rate", "unresolved_rate")
SCORE_FIELDS = ("margin_first_policy", "margin_path_policy", "margin_first_raw",
                "close_prob_pre_filter", "close_sampler_prob",
                "eos_after_forced_close_prob")


def aggregate(layout: Layout, scale: str) -> dict[str, Any]:
    out_dir = layout.r1
    rows_by_tag: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(out_dir.glob(f"R1_readouts_{scale}_*.jsonl")):
        for row in iter_jsonl(path):
            rows_by_tag.setdefault(str(row["model_tag"]), []).append(row)
    if not rows_by_tag:
        raise SystemExit(f"no R1 readouts for scale {scale} under {out_dir}")
    merged = [row for rows in rows_by_tag.values() for row in rows]
    write_jsonl(out_dir / f"R1_anchor_readouts_{scale}.jsonl", merged)

    scored = [tag for tag in registry.R1_SCORED.get(scale, ()) if tag in rows_by_tag]
    contrasts = [row for row in registry.R1_CONTRASTS.get(scale, ())
                 if row[1] in rows_by_tag and row[2] in rows_by_tag]
    skipped = [row[0] for row in registry.R1_CONTRASTS.get(scale, ())
               if row not in contrasts]
    if skipped:
        log("r1", f"{scale}: not yet measurable -- " + "; ".join(skipped))

    # Completeness is measured against the POOL, not against the union of what
    # was produced.  If the same shard is missing from every model, the union
    # shrinks with it and a comparison of outputs to outputs sees nothing wrong.
    pool_path = layout.r1 / f"prefix_pool_{scale}.jsonl"
    if pool_path.is_file():
        expected_ids = {str(row["anchor_id"]) for row in iter_jsonl(pool_path)}
        for tag, rows in sorted(rows_by_tag.items()):
            missing = expected_ids - {str(row["anchor_id"]) for row in rows}
            if missing:
                log("r1", f"[!] {tag}: {len(expected_ids) - len(missing)}/"
                          f"{len(expected_ids)} pool anchors measured "
                          f"({len(missing)} missing); a shard is missing or "
                          "failed, and every effect below is computed only on "
                          "the anchors both arms share")
    else:
        log("r1", f"[!] {pool_path.name} is gone; completeness cannot be checked")

    def subset(rows: Sequence[Mapping[str, Any]], **filters: Any) -> list[Mapping[str, Any]]:
        return [row for row in rows
                if all(str(row.get(key)) == str(value) for key, value in filters.items())]

    effects: list[dict[str, Any]] = []
    for label, m0, m1, kind in contrasts:
        for split in ("test", "dev", "all"):
            for role in ("all", "natural_stop", "early_remaining", "pre_first_reuse"):
                for source in ("all", *registry.R1_POOL_SOURCES[scale]):
                    filters: dict[str, Any] = {}
                    if split != "all":
                        filters["split"] = split
                    if role != "all":
                        filters["role"] = role
                    if source != "all":
                        filters["source_tag"] = source
                    left = subset(rows_by_tag[m0], **filters)
                    right = subset(rows_by_tag[m1], **filters)
                    if len(left) < 4 or len(right) < 4:
                        continue
                    for field in BEHAVIOUR_FIELDS + SCORE_FIELDS:
                        try:
                            effect = stats.paired_anchor_effect(
                                left, right, metric=field, value_field=field)
                        except ValueError:
                            continue
                        effects.append(effect.as_dict(
                            scale=scale, contrast=label, kind=kind,
                            m0_tag=m0, m1_tag=m1, split=split, role=role,
                            source=source, n_anchors=min(len(left), len(right))))
    write_csv(out_dir / f"R1_paired_effects_{scale}.csv", effects)

    execution: list[dict[str, Any]] = []
    for tag, rows in sorted(rows_by_tag.items()):
        for role in ("all", "natural_stop", "early_remaining", "pre_first_reuse"):
            group = rows if role == "all" else subset(rows, role=role)
            if not group:
                continue
            def mean(field: str) -> float:
                values = [float(row[field]) for row in group if row.get(field) is not None]
                return sum(values) / len(values) if values else math.nan

            execution.append({
                "scale": scale, "model_tag": tag, "role": role, "n_anchors": len(group),
                "close_reachable_share": sum(bool(row.get("close_reachable"))
                                             for row in group) / len(group),
                "close_prob_pre_filter_mean": mean("close_prob_pre_filter"),
                "close_sampler_prob_mean": mean("close_sampler_prob"),
                "eos_after_forced_close_prob_mean": mean("eos_after_forced_close_prob"),
                "next_event_close_rate_mean": mean("next_event_close_rate"),
                "eos_without_close_rate_mean": mean("eos_without_close_rate"),
                "unresolved_rate_mean": mean("unresolved_rate"),
                "margin_first_policy_mean": mean("margin_first_policy"),
                "margin_path_policy_mean": mean("margin_path_policy"),
            })
    write_csv(out_dir / f"R1_close_execution_{scale}.csv", execution)

    lines = [f"models={','.join(registry.label(tag) for tag in scored)}"]
    for label, m0, m1, kind in contrasts:
        lines.append(f"{label} [test]" + ("  (reference scale, not a treatment)"
                                          if kind == "seed" else ""))
        for role in ("natural_stop", "early_remaining", "pre_first_reuse"):
            picked = [row for row in effects
                      if row["contrast"] == label and row["split"] == "test"
                      and row["role"] == role and row["source"] == "all"]
            close = next((row for row in picked
                          if row["metric"] == "next_event_close_rate"), None)
            margin = next((row for row in picked
                           if row["metric"] == "margin_first_policy"), None)
            if close is None:
                lines.append(f"  {role}: no paired anchor")
                continue
            lines.append(
                f"  {role} n={close['n_anchors']}: close {fmt_pct(close['m0'])}"
                f"->{fmt_pct(close['m1'])} d={fmt_pct(close['diff'])}"
                f"[{fmt_pct(close['ci_low'])},{fmt_pct(close['ci_high'])}]"
                + (f"; margin d={fmt(margin['diff'])}"
                   f"[{fmt(margin['ci_low'])},{fmt(margin['ci_high'])}]" if margin else ""))
    write_compact(layout.compact / f"R1_{scale}.txt",
                  f"[R1 same-prefix termination, {scale}]", lines)
    summary = {"scale": scale, "models": scored, "n_effects": len(effects),
               "anchors_per_model": {tag: len(rows) for tag, rows in rows_by_tag.items()}}
    write_json(out_dir / f"R1_summary_{scale}.json", summary)
    log("r1", f"{scale}: {len(effects)} paired effects over {len(scored)} models")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", default="4B")
    parser.add_argument("--models", default="auto",
                        help="comma-separated E1 tags, or 'auto' for the registry")
    parser.add_argument("--pool", default=None,
                        help="default: <PAPER_ROOT>/R1/prefix_pool_<scale>.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--k-resample", type=int, default=K_RESAMPLE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--sample-batch", type=int, default=SAMPLE_BATCH)
    parser.add_argument("--attn", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    assert_frozen_sampling(NOTHINK_SAMPLING)
    layout = Layout.from_env()
    layout.ensure()
    if args.aggregate_only:
        aggregate(layout, args.scale)
        return 0

    pool_path = Path(args.pool) if args.pool else layout.r1 / f"prefix_pool_{args.scale}.jsonl"
    if not pool_path.is_file():
        raise SystemExit(f"{pool_path} is missing; run build_prefix_pool.py first")
    anchors = sorted(iter_jsonl(pool_path), key=lambda row: str(row["anchor_id"]))
    if not (0 <= args.shard < args.num_shards):
        raise SystemExit("shard index out of range")
    shard = anchors[args.shard::args.num_shards]
    if args.limit:
        shard = shard[: args.limit]

    tags = (list(registry.R1_SCORED[args.scale]) if args.models == "auto"
            else [value.strip() for value in args.models.split(",") if value.strip()])
    metrics: dict[str, dict[str, Any]] = {}
    if layout.metrics_csv.is_file():
        metrics = {row["tag"]: row for row in read_csv(layout.metrics_csv)}

    for tag in tags:
        out_path = layout.r1 / f"R1_readouts_{args.scale}_{tag}_shard{args.shard}.jsonl"
        measure_model(tag, shard, layout=layout, out_path=out_path,
                      device=args.device, k_resample=args.k_resample,
                      max_new_tokens=args.max_new_tokens,
                      sample_batch=args.sample_batch, attn=args.attn,
                      metrics_row=metrics.get(tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
