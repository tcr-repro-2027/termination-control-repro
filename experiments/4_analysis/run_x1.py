# coding: utf-8
"""X1: when a repetition is put in front of the model, does SFT make it worse?

Scientific question (H2 / contribution C2).  The free-generation numbers say the
raw-data model repeats far more often.  The convenient reading is that training
strengthened a local copy-me-again preference.  This entry tests the specific
prediction that reading makes: shown a reference prefix in which one complete
motif has just been repeated, does the model prefer repeating it AGAIN more than
it did before?

Construction (08e's frozen surfaces, nothing new)
-------------------------------------------------
    C1  reference prefix ending with one complete copy of an m-block motif
    C2  C1 followed by a second complete copy
    C3  C2 followed by a third
    R   the motif's FIRST block -- the candidate that continues the repetition
    N   the block the reference actually has next -- the candidate that recovers

The injected unit is the whole motif; the scored candidate is its first block,
not a whole extra cycle.  ``s(X | C)`` is the mean log-probability of the
identity tokens (source, target, relation) of candidate X under context C, in
nats per token.

    G = [s(R|C2) - s(N|C2)] - [s(R|C1) - s(N|C1)]
    E = s(R|C2) - s(R|C1)          the repeat becoming more attractive
    P = s(N|C1) - s(N|C2)          the recovery becoming less attractive
    G = E + P                      exactly, and it is asserted

G > 0 means an extra repetition in context increased the relative pull of
repeating; G < 0 means it decreased it.

What a negative G would and would not show
------------------------------------------
Three model contrasts are computed separately -- cleaned minus untrained, raw
minus cleaned, OBR minus cleaned -- because they are different questions and
folding them into "after SFT" would hide a sign flip.  A negative
cleaned-minus-untrained gain alongside a HIGHER answer-level repetition rate
would show that global degeneration does not require this local gain to grow.
It would not show that no copy mechanism exists, and it says nothing about
whether the raw-data contrast behaves the same way.

This score is not R1's path margin and not a survival probability under the
deployed sampler: it is a teacher-forced mean log-probability over identity
tokens, and it is only compared with itself across models and contexts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (REPO_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tcr.paper import gold as goldlib                                      # noqa: E402
from tcr.paper import registry, stats                                      # noqa: E402
from tcr.paper.io import (append_jsonl, fmt, iter_jsonl, log, read_csv,     # noqa: E402
                   write_compact, write_csv, write_json, write_jsonl)
from tcr.paper.layout import Layout, model_path_for                        # noqa: E402
from tcr.motif.detector_v2.block_parser import parse_relation_blocks    # noqa: E402
from tcr.motif.probes.motif_bank import build_probe_bank                # noqa: E402
from tcr.motif.schemas import as_record                                 # noqa: E402
from tcr.motif.scoring.candidate_score import score_branch              # noqa: E402
from tcr.motif.scoring.loop_gain import (assert_decomposition,          # noqa: E402
                                     compute_loop_gain, feedback_curve)
from tcr.boundary.runtime import (frozen_nothink_prefix_text,      # noqa: E402
                                   load_model, load_tokenizer)
from tcr.prompt_template import build_extraction_relation_prompt  # noqa: E402

PROTOCOL = "paper-closeout-x1-v1"
M_VALUES = (1, 2, 4, 8)
N_PROMPTS = 128
MAX_PROBES = 512
POOL_SEED = 20260911
BRANCHES = ("C1_R", "C1_N", "C2_R", "C2_N", "C3_R", "C3_N")


def build_pool(gold: goldlib.GoldSet, tokenizer, *, n_prompts: int,
               max_probes: int, m_values: Sequence[int]) -> tuple[list[dict[str, Any]],
                                                                  dict[str, Any]]:
    """One probe per (record, motif length), regions balanced across the pool.

    ``build_probe_bank`` returns at most one probe per (m, early/mid/late)
    region; taking one per m and steering the choice toward whichever region is
    currently under-represented keeps the pool balanced without letting any
    model's score influence which positions were chosen.
    """
    records = sorted(gold.records, key=lambda record: record.key_id)
    rng = random.Random(POOL_SEED)
    rng.shuffle(records)

    probes: list[dict[str, Any]] = []
    rejections: Counter = Counter()
    region_counts: Counter = Counter()
    used_prompts = 0
    for record in records:
        if used_prompts >= n_prompts or len(probes) >= max_probes:
            break
        # The reference answer, serialised exactly as training and evaluation
        # render it -- the model has to see the surface it was trained on.
        if not record.blocks:
            rejections["reference_has_no_block"] += 1
            continue
        output_text = json.dumps([dict(block) for block in record.blocks],
                                 ensure_ascii=False)
        parsed = parse_relation_blocks(output_text)
        if len(parsed.blocks) < max(m_values) + 2:
            rejections["too_few_reference_blocks"] += 1
            continue
        prompt = build_extraction_relation_prompt(
            text=record.text, entities_str=record.entities_str)
        chatml_input = frozen_nothink_prefix_text(prompt)
        try:
            bank, why = build_probe_bank(
                stable_prompt_id=record.key_id, split="x1",
                chatml_input=chatml_input, output_text=output_text, parsed=parsed,
                m_values=m_values, tokenizer=tokenizer, protocol_hash=PROTOCOL,
                source_hashes={"gold": record.key_id})
        except (AssertionError, ValueError) as exc:
            rejections[f"bank_error:{type(exc).__name__}"] += 1
            continue
        for row in why:
            rejections[str(row.get("reason"))] += 1
        if not bank:
            continue
        used_prompts += 1
        by_m: dict[int, list[Any]] = defaultdict(list)
        for probe in bank:
            by_m[int(probe.m)].append(probe)
        for m in sorted(by_m):
            if len(probes) >= max_probes:
                break
            choice = min(by_m[m], key=lambda p: (region_counts[p.anchor_region],
                                                 p.anchor_region, p.probe_id))
            region_counts[choice.anchor_region] += 1
            record_row = as_record(choice)
            record_row["key"] = record.key
            record_row["gold_blocks"] = record.n_blocks
            probes.append(record_row)

    manifest = {
        "protocol": PROTOCOL,
        "n_prompts_used": used_prompts,
        "n_probes": len(probes),
        "m_values": list(m_values),
        "by_m": dict(Counter(int(row["m"]) for row in probes)),
        "by_region": dict(region_counts),
        "rejections": dict(rejections),
    }
    return probes, manifest


def score_probes(tag: str, probes: Sequence[Mapping[str, Any]], *, layout: Layout,
                 out_path: Path, device: str, attn: str | None,
                 metrics_row: Mapping[str, Any] | None) -> int:
    import torch

    done = ({str(row["probe_id"]) for row in iter_jsonl(out_path)}
            if out_path.exists() else set())
    todo = [probe for probe in probes if str(probe["probe_id"]) not in done]
    log("x1", f"{tag}: {len(probes)} probes, {len(done)} done, {len(todo)} to do")
    if not todo:
        return 0
    model_path = model_path_for(layout, tag, metrics_row=metrics_row)
    model, backend = load_model(model_path, device=device, attn_implementation=attn)
    log("x1", f"{tag}: loaded {model_path} ({backend})")

    written = 0
    for index, probe in enumerate(todo, start=1):
        started = time.time()
        branches = probe["token_data"]["branches"]
        scores: dict[str, float] = {}
        for key in BRANCHES:
            branch = branches.get(key)
            if branch is None:
                break
            with torch.no_grad():
                result = score_branch(model, branch, device=device)
            scores[key] = float(result.mean_log_probability)
        if len(scores) != len(BRANCHES):
            log("x1", f"{tag}: probe {probe['probe_id']} lacks a branch; skipped")
            continue
        renamed = {"R_C1": scores["C1_R"], "N_C1": scores["C1_N"],
                   "R_C2": scores["C2_R"], "N_C2": scores["C2_N"],
                   "R_C3": scores["C3_R"], "N_C3": scores["C3_N"]}
        values = compute_loop_gain(renamed)
        assert_decomposition(values)
        curve = feedback_curve(renamed)
        append_jsonl(out_path, {
            "model_tag": tag, "model_path": str(model_path),
            "probe_id": probe["probe_id"],
            "stable_prompt_id": probe["stable_prompt_id"],
            "m": int(probe["m"]), "anchor_region": probe["anchor_region"],
            "k_1based": int(probe["k_1based"]), "gold_blocks": probe.get("gold_blocks"),
            **{f"s_{name}": value for name, value in renamed.items()},
            "gain_motif": float(values.loop_gain),
            "gain_copy": float(values.repeat_attraction),
            "gain_recovery": float(values.recovery_deficit),
            "margin_c1": float(values.margin_c1),
            "margin_c2": float(values.margin_c2),
            "gain_second": float(curve["gain_second"]),
            "seconds": round(time.time() - started, 2),
        })
        written += 1
        if index % 25 == 0 or index == len(todo):
            log("x1", f"{tag}: {index}/{len(todo)}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return written


CONTRASTS = (
    ("qwen3-4b-notrain", "qwen3-4b-cleanv2-s42", "SFT on cleaned data"),
    ("qwen3-4b-cleanv2-s42", "qwen3-4b-keep4-s42", "raw minus cleaned"),
    ("qwen3-4b-cleanv2-s42", "qwen3-4b-obr-p15-s42", "OBR 15% minus cleaned"),
)
GAIN_FIELDS = ("gain_motif", "gain_copy", "gain_recovery", "margin_c1",
               "margin_c2", "gain_second")


def aggregate(layout: Layout) -> int:
    rows: list[dict[str, Any]] = []
    for path in sorted(layout.x1.glob("X1_scores_*.jsonl")):
        rows.extend(iter_jsonl(path))
    if not rows:
        raise SystemExit(f"no X1 scores under {layout.x1}")
    write_jsonl(layout.x1 / "X1_probe_scores.jsonl", rows)
    write_csv(layout.x1 / "X1_probe_scores.csv", rows)

    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tag[str(row["model_tag"])].append(row)

    by_m: list[dict[str, Any]] = []
    for tag, group in sorted(by_tag.items()):
        for m in sorted({int(row["m"]) for row in group}) + ["all"]:
            bucket = group if m == "all" else [r for r in group if int(r["m"]) == m]
            entry: dict[str, Any] = {"model_tag": tag, "m": m, "n_probes": len(bucket)}
            for field in GAIN_FIELDS:
                values = [float(r[field]) for r in bucket]
                entry[field] = sum(values) / len(values) if values else math.nan
                entry[f"{field}_positive_share"] = (
                    sum(1 for v in values if v > 0) / len(values) if values else math.nan)
            by_m.append(entry)
    write_csv(layout.x1 / "X1_gain_by_m.csv", by_m)

    effects: list[dict[str, Any]] = []
    for m0, m1, purpose in CONTRASTS:
        if m0 not in by_tag or m1 not in by_tag:
            continue
        for scope in ("all", *[str(v) for v in sorted({int(r["m"]) for r in by_tag[m0]})]):
            left = (by_tag[m0] if scope == "all"
                    else [r for r in by_tag[m0] if str(int(r["m"])) == scope])
            right = (by_tag[m1] if scope == "all"
                     else [r for r in by_tag[m1] if str(int(r["m"])) == scope])
            for field in GAIN_FIELDS:
                try:
                    effect = stats.paired_anchor_effect(
                        left, right, metric=field, value_field=field,
                        anchor_key="probe_id", prompt_key="stable_prompt_id")
                except ValueError:
                    continue
                effects.append(effect.as_dict(m0_tag=m0, m1_tag=m1,
                                              purpose=purpose,
                                              motif_length=scope))
    write_csv(layout.x1 / "X1_paired_effects.csv", effects)

    lines = ["G = motif gain, E = repeat attraction, P = recovery deficit; "
             "nat/token, G = E + P"]
    for m0, m1, purpose in CONTRASTS:
        picked = [row for row in effects
                  if row["m0_tag"] == m0 and row["m1_tag"] == m1
                  and row["motif_length"] == "all"]
        if not picked:
            lines.append(f"{purpose}: not measured")
            continue
        def grab(field: str):
            return next((row for row in picked if row["metric"] == field), None)

        gain, copy, recovery = grab("gain_motif"), grab("gain_copy"), grab("gain_recovery")
        lines.append(
            f"{purpose} (n={gain['n_prompts']} prompts): "
            f"G {fmt(gain['m0'])}->{fmt(gain['m1'])} d={fmt(gain['diff'])}"
            f"[{fmt(gain['ci_low'])},{fmt(gain['ci_high'])}] {gain['direction']}"
            + (f"; E d={fmt(copy['diff'])}" if copy else "")
            + (f"; P d={fmt(recovery['diff'])}" if recovery else ""))
    write_compact(layout.compact / "X1_01.txt",
                  "[X1 artificial complete-motif gain]", lines)
    log("x1", f"aggregate: {len(rows)} scored probes, {len(effects)} contrasts")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("pool", "score", "aggregate", "all"),
                        default="all")
    parser.add_argument("--models", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn", default=None)
    parser.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--max-probes", type=int, default=MAX_PROBES)
    parser.add_argument("--tokenizer", default=None,
                        help="default: the 4B base model (one Qwen3 tokenizer)")
    args = parser.parse_args()

    layout = Layout.from_env()
    layout.ensure()
    pool_path = layout.x1 / "X1_probes.jsonl"

    if args.stage in ("pool", "all"):
        gold = goldlib.load_gold(layout.gold)
        tokenizer_path = Path(args.tokenizer) if args.tokenizer else layout.base_model("4B")
        tokenizer = load_tokenizer(tokenizer_path)
        probes, manifest = build_pool(gold, tokenizer, n_prompts=args.n_prompts,
                                      max_probes=args.max_probes, m_values=M_VALUES)
        write_jsonl(pool_path, probes)
        manifest["tokenizer"] = str(tokenizer_path)
        manifest["output"] = str(pool_path)
        write_json(layout.x1 / "X1_probes_manifest.json", manifest)
        log("x1", f"pool: {manifest['n_probes']} probes from "
                  f"{manifest['n_prompts_used']} records; by m={manifest['by_m']}, "
                  f"by region={manifest['by_region']}")

    if args.stage in ("score", "all"):
        if not pool_path.is_file():
            raise SystemExit(f"{pool_path} is missing; run --stage pool first")
        probes = list(iter_jsonl(pool_path))
        tags = (list(registry.X1_MODELS) if args.models == "auto"
                else [v.strip() for v in args.models.split(",") if v.strip()])
        metrics = ({row["tag"]: row for row in read_csv(layout.metrics_csv)}
                   if layout.metrics_csv.is_file() else {})
        for tag in tags:
            score_probes(tag, probes, layout=layout,
                         out_path=layout.x1 / f"X1_scores_{tag}.jsonl",
                         device=args.device, attn=args.attn,
                         metrics_row=metrics.get(tag))

    if args.stage in ("aggregate", "all"):
        aggregate(layout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
