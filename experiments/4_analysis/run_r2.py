# coding: utf-8
"""R2: does the data-related internal change actually do anything?

Scientific question (H3/H4, contributions C3/C4).  R1 can show that two models
choose differently in the same state.  It cannot show that the difference in
their internal states is what does the choosing -- a readable difference is not
a causal one, and a high classifier AUC least of all.

The claim this entry can support, and the one it cannot
-------------------------------------------------------
What is tested is narrow on purpose: **in the residual change produced by the
entity-constraint contrast, is there a component that participates in stopping
and in what follows?**  Not "can anything make the model stop" -- a logit bias
can, and it is included here precisely so the internal direction has to beat it
on something other than the stop rate.

Five stages, each resumable
---------------------------
``collect``    boundary residuals at three candidate layers, for every model
               that will be steered, on the shared prefix pool.
``direction``  a layer chosen on DEV only, and ``d`` = mean(h_clean - h_raw) at
               that layer.  Positive means "toward the cleaned model".  The
               choice is frozen before any test anchor is touched.
``short``      the single-pulse intervention on held-out anchors: restore the
               raw model, reverse the cleaned one, transfer to the OBR model,
               against eight equal-norm random directions and a calibrated
               close-token bias.
``long``       full continuations to EOS or the remaining context, so the
               question "did repetition fall, and what did it cost?" has an
               answer that is not a 32-token proxy.
``sae``        (8B only) which dictionary features the direction runs along.
               Feature cards explain; they never substitute for the causal test.

The pulse is one forward
------------------------
The vector enters the single forward that predicts the close/continue token and
is then removed.  Its effect on later tokens travels through the KV cache on its
own.  Holding the vector on every generated position would answer a different
question -- whether a model can be held in a state -- and would make "the
intervention only truncated the output" almost impossible to rule out.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (REPO_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tcr.paper import gold as goldlib                                      # noqa: E402
from tcr.paper import registry, stats                                      # noqa: E402
from tcr.paper.io import (append_jsonl, fmt, fmt_pct, iter_jsonl, log,      # noqa: E402
                   read_csv, write_compact, write_csv, write_json, write_jsonl)
from tcr.paper.layout import Layout, model_path_for                        # noqa: E402
from tcr.paper.sampler import (anchor_seed, boundary_metrics,              # noqa: E402
                        generate_long, sample_next_event)
from tcr.boundary.constants import NOTHINK_SAMPLING                # noqa: E402
from tcr.boundary.policy import assert_frozen_sampling             # noqa: E402
from tcr.boundary.runtime import (audit_chat_template, build_chat_ids,  # noqa: E402
                                   load_model, load_tokenizer, stop_token_ids)
from tcr.steering.collect import capture_boundary                       # noqa: E402
from tcr.steering.direction import layer_directions, paired_auc         # noqa: E402
from tcr.steering.steering import SteeringHook                          # noqa: E402

#: Alpha grid.  1.0 -- one dev-set mean gap -- is the headline; the other two
#: are short-readout context, never the number the quality claim rests on.
ALPHA_GRID = (0.5, 1.0, 2.0)
MAIN_ALPHA = 1.0
N_RANDOM = 8
RANDOM_SEED = 20260910
BIAS_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
#: Dev anchors that must carry a first-token margin in BOTH arms before the
#: layer can be chosen by the margin correlation rather than by separability.
MIN_DEV_MARGIN_ANCHORS = 8

#: Every field `stage_direction` writes into `R2_direction_<scale>.json`.
#: Declared once because the reporting stage reads this file back: with the
#: names written in two places, renaming one of them silently turns a number
#: into `NA` in the summary the paper is written from.
DIRECTION_FIELDS = (
    "scale", "layer", "candidate_layers", "selection_rule", "direction_norm",
    "sign_convention", "dev", "test_auc_held_out", "auc_note", "n_dev",
    "n_test", "n_dev_with_margin_both_arms", "dev_anchor_ids",
)


def direction_summary_line(meta: Mapping[str, Any]) -> str:
    """The one line of the R2 summary that comes from the direction file."""
    return (f"layer={meta.get('layer', 'NA')}; "
            f"held-out AUC={fmt(meta.get('test_auc_held_out'))} (descriptive; "
            "the dev AUC is in-sample and is not reported here)")

#: Roles the direction is estimated from: places where the donor model was
#: making a live exit decision, not places chosen because content was missing.
DEV_DIRECTION_ROLES = ("natural_stop", "pre_first_reuse")

#: Long-form quota per role, 64 anchors in total.
LONG_QUOTA = {"early_remaining": 32, "natural_stop": 16, "pre_first_reuse": 16}
LONG_DRAWS = 4
#: The four conditions every selected anchor must be run under.  The bias
#: control is one of them: without it the content-cost comparison has nothing
#: to say about whether the direction did more than make the model stop.
LONG_CONDITIONS = ("baseline", f"direction_a{MAIN_ALPHA:g}", "random_matched",
                   "close_bias")
CONTEXT_LIMIT = 32768


# --------------------------------------------------------------------- shared

def load_pool(layout: Layout, scale: str) -> list[dict[str, Any]]:
    path = layout.r1 / f"prefix_pool_{scale}.jsonl"
    if not path.is_file():
        raise SystemExit(f"{path} is missing; run build_prefix_pool.py first")
    return sorted(iter_jsonl(path), key=lambda row: str(row["anchor_id"]))


def r2_models(scale: str) -> dict[str, Any]:
    if scale not in registry.R2_MODELS:
        raise SystemExit(f"no R2 configuration for scale {scale!r}")
    return registry.R2_MODELS[scale]


def steer_sign(tag: str, config: Mapping[str, Any]) -> float:
    """+1 restores toward the cleaned model, -1 moves away from it.

    ``d`` points from the raw model's mean state toward the cleaned model's, so
    the raw model and the transfer model are pushed ALONG it and the cleaned
    model is pushed against it.  Both directions have to work, or the vector is
    describing model identity rather than a control component.
    """
    return -1.0 if tag == config["clean"] else 1.0


def steered_tags(config: Mapping[str, Any]) -> list[str]:
    return [config["clean"], config["raw"], *config.get("transfer", ())]


def candidate_layers(model) -> list[int]:
    """Three pre-registered depths, read from the model's own config.

    Scanning every layer would turn the layer into a fitted parameter; three
    fixed depths keep it a choice made once, on dev.
    """
    n = int(model.config.num_hidden_layers)
    layers = sorted({max(n // 2 - 1, 0), max((3 * n) // 4 - 1, 0), max(n - 4, 0)})
    return layers


def acts_path(layout: Layout, scale: str, tag: str, shard: int) -> Path:
    return layout.r2 / f"R2_acts_{scale}_{tag}_shard{shard}.npz"


def load_acts(layout: Layout, scale: str, tag: str) -> tuple[dict[str, np.ndarray], list[int]]:
    """anchor_id -> (n_layers, d_model) float32, merged over shards."""
    merged: dict[str, np.ndarray] = {}
    layers: list[int] = []
    paths = sorted(layout.r2.glob(f"R2_acts_{scale}_{tag}_shard*.npz"))
    if not paths:
        raise SystemExit(f"no collected activations for {tag} ({scale}); run "
                         "`run_r2.py collect` first")
    for path in paths:
        with np.load(path, allow_pickle=False) as bundle:
            ids = [str(value) for value in bundle["anchor_ids"]]
            block = bundle["acts"].astype(np.float32)
            layers = [int(value) for value in bundle["layers"]]
        for index, anchor_id in enumerate(ids):
            merged[anchor_id] = block[index]
    return merged, layers


# -------------------------------------------------------------------- collect

def stage_collect(layout: Layout, args) -> int:
    scale = args.scale
    config = r2_models(scale)
    anchors = load_pool(layout, scale)
    shard = anchors[args.shard::args.num_shards]
    metrics = ({row["tag"]: row for row in read_csv(layout.metrics_csv)}
               if layout.metrics_csv.is_file() else {})
    tags = ([value.strip() for value in args.models.split(",") if value.strip()]
            if args.models != "auto" else steered_tags(config))

    for tag in tags:
        out_path = acts_path(layout, scale, tag, args.shard)
        if out_path.exists() and not args.overwrite:
            log("r2", f"collect {tag}: {out_path.name} exists, skipping")
            continue
        model_path = model_path_for(layout, tag, metrics_row=metrics.get(tag))
        tokenizer = load_tokenizer(model_path)
        audit_chat_template(tokenizer)
        model, backend = load_model(model_path, device=args.device,
                                    attn_implementation=args.attn)
        layers = candidate_layers(model)
        log("r2", f"collect {tag}: {model_path} ({backend}), layers={layers}, "
                  f"{len(shard)} anchors")
        chat_cache: dict[str, list[int]] = {}
        ids: list[str] = []
        rows: list[np.ndarray] = []
        for index, anchor in enumerate(shard, start=1):
            sha = str(anchor["prompt_sha256"])
            chat_ids = chat_cache.get(sha)
            if chat_ids is None:
                chat_ids = build_chat_ids(tokenizer, anchor["prompt_text"])
                chat_cache[sha] = chat_ids
            block = capture_boundary(
                model, chat_ids=chat_ids,
                prefix_response_ids=[int(v) for v in anchor["prefix_response_ids"]],
                layers=layers, device=args.device)
            ids.append(str(anchor["anchor_id"]))
            rows.append(block.astype(np.float16))
            if index % 25 == 0 or index == len(shard):
                log("r2", f"collect {tag}: {index}/{len(shard)}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, anchor_ids=np.array(ids),
                            acts=np.stack(rows), layers=np.array(layers))
        log("r2", f"collect {tag}: wrote {out_path}")
        del model
        _empty_cache()
    return 0


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ------------------------------------------------------------------ direction

def r1_margins(layout: Layout, scale: str, tag: str) -> dict[str, float]:
    path = layout.r1 / f"R1_anchor_readouts_{scale}.jsonl"
    if not path.is_file():
        return {}
    return {str(row["anchor_id"]): float(row["margin_first_policy"])
            for row in iter_jsonl(path) if str(row.get("model_tag")) == tag}


def stage_direction(layout: Layout, args) -> int:
    scale = args.scale
    config = r2_models(scale)
    anchors = {str(row["anchor_id"]): row for row in load_pool(layout, scale)}
    clean_acts, layers = load_acts(layout, scale, config["clean"])
    raw_acts, layers_raw = load_acts(layout, scale, config["raw"])
    if layers != layers_raw:
        raise SystemExit(f"candidate layers differ between arms: {layers} vs {layers_raw}")

    dev_ids = sorted(
        anchor_id for anchor_id, anchor in anchors.items()
        if anchor.get("split") == "dev" and anchor.get("role") in DEV_DIRECTION_ROLES
        and anchor_id in clean_acts and anchor_id in raw_acts)
    test_ids = sorted(
        anchor_id for anchor_id, anchor in anchors.items()
        if anchor.get("split") == "test"
        and anchor_id in clean_acts and anchor_id in raw_acts)
    if len(dev_ids) < 8:
        raise SystemExit(f"only {len(dev_ids)} dev anchors carry both arms' "
                         "activations; the direction cannot be estimated")
    log("r2", f"direction {scale}: {len(dev_ids)} dev anchors "
              f"({', '.join(DEV_DIRECTION_ROLES)}), {len(test_ids)} test anchors")

    dev_clean = np.stack([clean_acts[a] for a in dev_ids])   # (n, L, d)
    dev_raw = np.stack([raw_acts[a] for a in dev_ids])
    directions = layer_directions(dev_raw, dev_clean)        # clean minus raw

    margins_clean = r1_margins(layout, scale, config["clean"])
    margins_raw = r1_margins(layout, scale, config["raw"])
    # The documented rule picks the layer whose projection gap tracks the
    # first-token margin gap most strongly.  That correlation needs enough DEV
    # anchors carrying a margin in BOTH arms -- a non-empty readout file is not
    # the same thing, and below the threshold every rho is NaN and the rule
    # quietly becomes "most separable layer".  Check the number that actually
    # governs the computation, not the presence of a file.
    with_margin = [anchor for anchor in dev_ids
                   if anchor in margins_clean and anchor in margins_raw]
    have_rule = len(with_margin) >= MIN_DEV_MARGIN_ANCHORS
    if not have_rule:
        if not args.force:
            raise SystemExit(
                f"only {len(with_margin)} of {len(dev_ids)} dev anchors carry a "
                f"first-token margin in both arms ({MIN_DEV_MARGIN_ANCHORS} are "
                f"needed).  Run `run_r1.py --scale {scale}` and its "
                "--aggregate-only over the whole pool first: the layer is chosen "
                "by how strongly the projection gap tracks the margin gap, and "
                "below this threshold that rule silently degrades to picking the "
                "most separable layer.  Pass --force to accept the fallback.")
        log("r2", f"[!] only {len(with_margin)} dev anchors carry a margin in "
                  "both arms; falling back to the AUC-only layer rule")
    rule_used = ("largest |Spearman(dev projection gap, dev first-token margin "
                 "gap)|; ties by the standardised dev projection gap, then the "
                 "shallower layer" if have_rule
                 else f"FALLBACK, not a supported basis for the paper's claim: "
                      f"the margin rule could not be applied ({len(with_margin)} "
                      f"dev anchors carried a margin in both arms, "
                      f"{MIN_DEV_MARGIN_ANCHORS} needed), so the layer was taken "
                      "by the standardised dev projection gap alone")

    rows: list[dict[str, Any]] = []
    for position, layer in enumerate(layers):
        d = directions[position]
        norm = float(np.linalg.norm(d))
        unit = d / norm if norm > 0 else d
        proj_clean = dev_clean[:, position, :].astype(np.float64) @ unit
        proj_raw = dev_raw[:, position, :].astype(np.float64) @ unit
        proj_gap = proj_clean - proj_raw
        margin_gap = [margins_clean.get(a, math.nan) - margins_raw.get(a, math.nan)
                      for a in dev_ids]
        usable = [(p, m) for p, m in zip(proj_gap, margin_gap) if math.isfinite(m)]
        rho = (stats.spearman([p for p, _ in usable], [m for _, m in usable])
               if len(usable) >= MIN_DEV_MARGIN_ANCHORS else math.nan)
        # Effect size of the gap in units of its own spread across anchors.
        # Unlike the AUC below it is not driven to a ceiling by the fit, so it
        # can break a tie between two layers without pretending to be evidence.
        spread = float(np.std(proj_gap, ddof=1)) if proj_gap.size > 1 else 0.0
        rows.append({
            "scale": scale, "layer": layer, "direction_norm": norm,
            # IN-SAMPLE: the direction is the mean gap of these same anchors, so
            # this is near 1.0 at every layer and is reported, not used.
            "dev_auc_in_sample": paired_auc(proj_raw, proj_clean),
            "dev_projection_gap_mean": float(proj_gap.mean()),
            "dev_projection_gap_effect": (float(proj_gap.mean() / spread)
                                          if spread > 0 else math.nan),
            "dev_margin_gap_spearman": rho,
            "n_dev": len(dev_ids), "n_dev_with_margin": len(usable),
            "residual_norm_clean_mean": float(
                np.linalg.norm(dev_clean[:, position, :], axis=1).mean()),
            "residual_norm_raw_mean": float(
                np.linalg.norm(dev_raw[:, position, :], axis=1).mean()),
        })
    write_csv(layout.r2 / f"R2_direction_dev_{scale}.csv", rows)

    def key(row: Mapping[str, Any]) -> tuple:
        rho = row["dev_margin_gap_spearman"]
        effect = row["dev_projection_gap_effect"]
        return (abs(rho) if math.isfinite(rho) else -1.0,
                abs(effect) if math.isfinite(effect) else -1.0,
                -int(row["layer"]))

    best = max(rows, key=key)
    position = layers.index(int(best["layer"]))
    d = directions[position]
    np.save(layout.r2 / f"R2_direction_{scale}.npy", d.astype(np.float32))
    log("r2", f"direction {scale}: layer {best['layer']} "
              f"(|rho|={fmt(best['dev_margin_gap_spearman'])}, "
              f"gap effect={fmt(best['dev_projection_gap_effect'])}, "
              f"||d||={fmt(best['direction_norm'])}, "
              f"typical residual norm {fmt(best['residual_norm_clean_mean'])})")

    unit = d / max(float(np.linalg.norm(d)), 1e-12)
    projection_rows: list[dict[str, Any]] = []
    for anchor_id in sorted(set(dev_ids) | set(test_ids)):
        anchor = anchors[anchor_id]
        h_clean = clean_acts[anchor_id][position].astype(np.float64)
        h_raw = raw_acts[anchor_id][position].astype(np.float64)
        projection_rows.append({
            "scale": scale, "anchor_id": anchor_id,
            "prompt_id": anchor["prompt_id"], "role": anchor["role"],
            "split": anchor["split"], "source_tag": anchor["source_tag"],
            "direction_projection_clean": float(h_clean @ unit),
            "direction_projection_raw": float(h_raw @ unit),
            "direction_projection_gap": float((h_clean - h_raw) @ unit),
            "margin_first_policy_clean": margins_clean.get(anchor_id),
            "margin_first_policy_raw": margins_raw.get(anchor_id),
        })
    for tag in config.get("transfer", ()):
        try:
            transfer_acts, transfer_layers = load_acts(layout, scale, tag)
        except SystemExit:
            log("r2", f"direction: no activations for transfer model {tag}; skipping")
            continue
        transfer_position = transfer_layers.index(int(best["layer"]))
        margins_transfer = r1_margins(layout, scale, tag)
        for row in projection_rows:
            block = transfer_acts.get(row["anchor_id"])
            if block is None:
                continue
            row[f"direction_projection_{tag}"] = float(
                block[transfer_position].astype(np.float64) @ unit)
            row[f"margin_first_policy_{tag}"] = margins_transfer.get(row["anchor_id"])
    write_csv(layout.r2 / f"R2_projection_test_{scale}.csv", projection_rows)

    test_clean = np.stack([clean_acts[a][position] for a in test_ids])
    test_raw = np.stack([raw_acts[a][position] for a in test_ids])
    chosen = {
        "scale": scale,
        "layer": int(best["layer"]),
        "candidate_layers": layers,
        "selection_rule": rule_used,
        "direction_norm": float(np.linalg.norm(d)),
        "sign_convention": "d = mean(h_clean) - mean(h_raw); positive is toward "
                           "the entity-constraint-cleaned model",
        "dev": {row["layer"]: row for row in rows}[int(best["layer"])],
        # Held-out: the direction was fitted on dev only, so this one means
        # something.  It is still descriptive; the causal test is the pulse.
        "test_auc_held_out": paired_auc(test_raw.astype(np.float64) @ unit,
                                        test_clean.astype(np.float64) @ unit),
        "auc_note": "dev_auc_in_sample is fitted and scored on the same anchors "
                    "and sits near 1.0 by construction; only the held-out value "
                    "separates states the direction has not seen.",
        "n_dev": len(dev_ids), "n_test": len(test_ids),
        "n_dev_with_margin_both_arms": len(with_margin),
        "dev_anchor_ids": dev_ids,
    }
    unexpected = sorted(set(chosen) ^ set(DIRECTION_FIELDS))
    if unexpected:
        raise AssertionError(
            f"the direction file's fields drifted from DIRECTION_FIELDS: "
            f"{unexpected}.  Update the constant and every reader with it -- "
            "`direction_summary_line` reads this file back.")
    write_json(layout.r2 / f"R2_direction_{scale}.json", chosen)
    log("r2", f"direction {scale}: held-out AUC={fmt(chosen['test_auc_held_out'])} "
              "(descriptive; the causal test is the intervention)")
    return 0


def load_direction(layout: Layout, scale: str) -> tuple[np.ndarray, dict[str, Any]]:
    vector_path = layout.r2 / f"R2_direction_{scale}.npy"
    meta_path = layout.r2 / f"R2_direction_{scale}.json"
    if not (vector_path.is_file() and meta_path.is_file()):
        raise SystemExit(f"no frozen direction for {scale}; run "
                         "`run_r2.py direction` first")
    return np.load(vector_path).astype(np.float64), json.loads(
        meta_path.read_text(encoding="utf-8"))


def random_directions(d: np.ndarray, n: int = N_RANDOM,
                      seed: int = RANDOM_SEED) -> list[np.ndarray]:
    """Equal-norm Gaussian directions.  A control smaller than the intervention
    it controls for is not a control."""
    rng = np.random.default_rng(seed)
    norm = float(np.linalg.norm(d))
    out = []
    for _ in range(n):
        vector = rng.normal(size=d.shape[0])
        out.append(vector / float(np.linalg.norm(vector)) * norm)
    return out


# ---------------------------------------------------------------------- short

def short_conditions(d: np.ndarray, sign: float, *, include_alpha: Sequence[float],
                     randoms: Sequence[np.ndarray], close_bias: float | None
                     ) -> list[dict[str, Any]]:
    """Every condition one anchor is measured under, baseline included.

    The baseline is re-measured here rather than reused from R1, and that is a
    deliberate ~8% of extra GPU time.  R1 reads its margins off a teacher-forced
    forward; the intervened conditions read theirs off the sampler's prefill.
    Subtracting an R1 baseline from an R2 condition would put a systematic
    difference between the two halves of every margin effect.  `stage_report`
    compares the two baselines against each other instead, so that they agree is
    a checked fact rather than an assumption.
    """
    conditions: list[dict[str, Any]] = [
        {"condition": "baseline", "family": "baseline", "alpha": 0.0,
         "vector": None, "close_bias": 0.0}]
    for alpha in include_alpha:
        conditions.append({"condition": f"direction_a{alpha:g}", "family": "direction",
                           "alpha": float(alpha), "vector": sign * alpha * d,
                           "close_bias": 0.0})
    for index, vector in enumerate(randoms):
        conditions.append({"condition": f"random{index}_a{MAIN_ALPHA:g}",
                           "family": "random", "alpha": MAIN_ALPHA,
                           "vector": sign * MAIN_ALPHA * vector, "close_bias": 0.0})
    if close_bias is not None:
        conditions.append({"condition": "close_bias", "family": "bias", "alpha": 0.0,
                           "vector": None, "close_bias": sign * float(close_bias)})
    return conditions


def run_conditions(model, tokenizer, anchor: Mapping[str, Any], *, chat_ids,
                   stop_ids, hook, device: str, conditions: Sequence[Mapping[str, Any]],
                   k_resample: int, max_new_tokens: int, sample_batch: int
                   ) -> list[dict[str, Any]]:
    prefix_ids = [int(v) for v in anchor["prefix_response_ids"]]
    close_first = int(anchor["close_tail_ids"][0])
    continue_first = int(anchor["continue_ids"][0])
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        result = sample_next_event(
            model, tokenizer, chat_ids=chat_ids, prefix_response_ids=prefix_ids,
            stop_ids=stop_ids, device=device, k_resample=k_resample,
            max_new_tokens=max_new_tokens, sample_batch=sample_batch,
            base_seed=anchor_seed(str(anchor["anchor_id"]), str(condition["condition"])),
            hook=hook, vector=condition["vector"],
            close_bias=float(condition["close_bias"]), close_token_id=close_first)
        logits = result.pop("_boundary_logits")
        row = dict(result)
        row.update(boundary_metrics(logits, presence_ids=prefix_ids,
                                    close_first=close_first,
                                    continue_first=continue_first,
                                    close_bias=float(condition["close_bias"])))
        row.update({
            "anchor_id": anchor["anchor_id"], "prompt_id": anchor["prompt_id"],
            "role": anchor["role"], "split": anchor["split"],
            "source_tag": anchor["source_tag"],
            "condition": condition["condition"], "family": condition["family"],
            "alpha": condition["alpha"], "close_bias_value": condition["close_bias"],
        })
        rows.append(row)
    return rows


def calibrate_bias(model, tokenizer, dev_anchors: Sequence[Mapping[str, Any]], *,
                   chat_builder, stop_ids, hook, device: str, d: np.ndarray,
                   sign: float, k_resample: int, max_new_tokens: int,
                   sample_batch: int, out_path: Path) -> tuple[float, list[dict[str, Any]]]:
    """Pick the bias whose mean close rate best matches the alpha=1 intervention.

    One value per model, fixed on dev and then applied unchanged to every test
    anchor.  Tuning it per test point would make the control a fitted competitor
    rather than a control.
    """
    conditions = [{"condition": f"direction_a{MAIN_ALPHA:g}", "family": "direction",
                   "alpha": MAIN_ALPHA, "vector": sign * MAIN_ALPHA * d,
                   "close_bias": 0.0}]
    conditions += [{"condition": f"bias{value:g}", "family": "bias", "alpha": 0.0,
                    "vector": None, "close_bias": sign * value} for value in BIAS_GRID]
    rows: list[dict[str, Any]] = []
    for index, anchor in enumerate(dev_anchors, start=1):
        chat_ids = chat_builder(anchor)
        rows.extend(run_conditions(
            model, tokenizer, anchor, chat_ids=chat_ids, stop_ids=stop_ids,
            hook=hook, device=device, conditions=conditions, k_resample=k_resample,
            max_new_tokens=max_new_tokens, sample_batch=sample_batch))
        if index % 10 == 0 or index == len(dev_anchors):
            log("r2", f"  bias calibration {index}/{len(dev_anchors)}")
    by_condition: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(float(row["next_event_close_rate"]))
    target = float(np.mean(by_condition[f"direction_a{MAIN_ALPHA:g}"]))
    best_value, best_gap = 0.0, math.inf
    for value in BIAS_GRID:
        rates = by_condition.get(f"bias{value:g}")
        if not rates:
            continue
        gap = abs(float(np.mean(rates)) - target)
        if gap < best_gap:
            best_value, best_gap = float(value), gap
    write_jsonl(out_path, rows)
    log("r2", f"  bias calibration: direction close rate {fmt_pct(target)}; "
              f"chose bias={best_value:g} (mean gap {fmt(best_gap)})")
    return best_value, rows


def stage_short(layout: Layout, args) -> int:
    scale = args.scale
    config = r2_models(scale)
    d, meta = load_direction(layout, scale)
    randoms = random_directions(d)
    anchors = load_pool(layout, scale)
    dev = [a for a in anchors if a.get("split") == "dev"][: args.calibration_anchors]
    test = [a for a in anchors if a.get("split") == "test"][: args.max_test_anchors]
    metrics = ({row["tag"]: row for row in read_csv(layout.metrics_csv)}
               if layout.metrics_csv.is_file() else {})
    tags = ([value.strip() for value in args.models.split(",") if value.strip()]
            if args.models != "auto" else steered_tags(config))
    log("r2", f"short {scale}: layer {meta['layer']}, {len(test)} test anchors; "
              f"close-token bias fixed earlier on {len(dev)} dev anchors")

    bias_choice: dict[str, float] = {}
    bias_file = layout.r2 / f"R2_bias_choice_{scale}.json"
    if bias_file.is_file():
        bias_choice = json.loads(bias_file.read_text(encoding="utf-8"))

    for tag in tags:
        out_path = layout.r2 / f"R2_short_{scale}_{tag}_shard{args.shard}.jsonl"
        if args.overwrite and out_path.exists():
            out_path.unlink()
        done = ({(str(row["anchor_id"]), str(row["condition"]))
                 for row in iter_jsonl(out_path)} if out_path.exists() else set())
        model_path = model_path_for(layout, tag, metrics_row=metrics.get(tag))
        tokenizer = load_tokenizer(model_path)
        audit_chat_template(tokenizer)
        stops = stop_token_ids(model_path, tokenizer)
        model, backend = load_model(model_path, device=args.device,
                                    attn_implementation=args.attn)
        hook = SteeringHook(model, int(meta["layer"]), args.device)
        sign = steer_sign(tag, config)
        chat_cache: dict[str, list[int]] = {}

        def chat_builder(anchor: Mapping[str, Any]) -> list[int]:
            sha = str(anchor["prompt_sha256"])
            ids = chat_cache.get(sha)
            if ids is None:
                ids = build_chat_ids(tokenizer, anchor["prompt_text"])
                chat_cache[sha] = ids
            return ids

        try:
            if tag not in bias_choice:
                raise SystemExit(
                    f"no calibrated close-token bias for {tag}; run "
                    f"`run_r2.py bias --scale {scale}` first.  The bias is fixed "
                    "once on dev and reused by every shard, so calibrating it "
                    "inside a shard would give the shards different controls.")
            conditions = short_conditions(
                d, sign, include_alpha=ALPHA_GRID, randoms=randoms,
                close_bias=bias_choice.get(tag))
            shard = test[args.shard::args.num_shards]
            log("r2", f"short {tag}: {len(shard)} anchors x {len(conditions)} "
                      f"conditions ({backend}, sign={sign:+.0f}, "
                      f"bias={bias_choice.get(tag)})")
            for index, anchor in enumerate(shard, start=1):
                pending = [c for c in conditions
                           if (str(anchor["anchor_id"]), c["condition"]) not in done]
                if not pending:
                    continue
                rows = run_conditions(
                    model, tokenizer, anchor, chat_ids=chat_builder(anchor),
                    stop_ids=stops, hook=hook, device=args.device,
                    conditions=pending, k_resample=args.k_resample,
                    max_new_tokens=args.short_tokens,
                    sample_batch=args.sample_batch)
                for row in rows:
                    row.update({"model_tag": tag, "scale": scale,
                                "layer": int(meta["layer"]), "sign": sign})
                    append_jsonl(out_path, row)
                if index % 5 == 0 or index == len(shard):
                    log("r2", f"short {tag}: {index}/{len(shard)}")
        finally:
            hook.remove()
            del model
            _empty_cache()
    return 0


def stage_bias(layout: Layout, args) -> int:
    """Fix one close-token bias per model, on dev, before any test anchor runs."""
    scale = args.scale
    config = r2_models(scale)
    d, meta = load_direction(layout, scale)
    anchors = load_pool(layout, scale)
    dev = [a for a in anchors if a.get("split") == "dev"][: args.calibration_anchors]
    metrics = ({row["tag"]: row for row in read_csv(layout.metrics_csv)}
               if layout.metrics_csv.is_file() else {})
    tags = ([value.strip() for value in args.models.split(",") if value.strip()]
            if args.models != "auto" else steered_tags(config))
    bias_file = layout.r2 / f"R2_bias_choice_{scale}.json"
    bias_choice = (json.loads(bias_file.read_text(encoding="utf-8"))
                   if bias_file.is_file() else {})

    for tag in tags:
        if tag in bias_choice and not args.overwrite:
            log("r2", f"bias {tag}: already {bias_choice[tag]:g}, skipping")
            continue
        model_path = model_path_for(layout, tag, metrics_row=metrics.get(tag))
        tokenizer = load_tokenizer(model_path)
        audit_chat_template(tokenizer)
        stops = stop_token_ids(model_path, tokenizer)
        model, _backend = load_model(model_path, device=args.device,
                                     attn_implementation=args.attn)
        hook = SteeringHook(model, int(meta["layer"]), args.device)
        chat_cache: dict[str, list[int]] = {}

        def chat_builder(anchor: Mapping[str, Any]) -> list[int]:
            sha = str(anchor["prompt_sha256"])
            ids = chat_cache.get(sha)
            if ids is None:
                ids = build_chat_ids(tokenizer, anchor["prompt_text"])
                chat_cache[sha] = ids
            return ids

        try:
            log("r2", f"bias {tag}: {len(dev)} dev anchors, grid={BIAS_GRID}")
            value, _rows = calibrate_bias(
                model, tokenizer, dev, chat_builder=chat_builder, stop_ids=stops,
                hook=hook, device=args.device, d=d, sign=steer_sign(tag, config),
                k_resample=args.k_resample, max_new_tokens=args.short_tokens,
                sample_batch=args.sample_batch,
                out_path=layout.r2 / f"R2_bias_dev_{scale}_{tag}.jsonl")
            bias_choice[tag] = value
            write_json(bias_file, bias_choice)
        finally:
            hook.remove()
            del model
            _empty_cache()
    return 0


# ----------------------------------------------------------------------- long

def select_long_anchors(anchors: Sequence[Mapping[str, Any]],
                        quota: Mapping[str, int]) -> list[dict[str, Any]]:
    """Fixed, result-independent selection: sorted by anchor id, quota per role."""
    chosen: list[dict[str, Any]] = []
    for role, count in quota.items():
        picked = [a for a in anchors if a.get("split") == "test" and a.get("role") == role]
        chosen.extend(sorted(picked, key=lambda row: str(row["anchor_id"]))[:count])
    return sorted(chosen, key=lambda row: str(row["anchor_id"]))


def analyse_continuation(*, prefix_text: str, tail: Mapping[str, Any],
                         anchor: Mapping[str, Any], record: goldlib.GoldRecord,
                         tokenizer, build_event_row) -> dict[str, Any]:
    """Score one full continuation as a whole response, then attribute the tail.

    Prefix and tail are parsed TOGETHER: a motif that starts in the prefix and
    is confirmed in the tail is a repetition the continuation produced, and
    scoring the tail alone would miss it.  Only a terminal event confirmed at a
    character position past the prefix counts as the tail's.
    """
    text = prefix_text + str(tail["text"])
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    finish_reason = "stop" if tail["ended_with_stop_token"] else "length"
    row, _snippets = build_event_row(
        model_tag="R2", key=anchor.get("key"), seed=int(tail["sample_index"]),
        text=text, token_ids=token_ids, offsets=offsets,
        finish_reason=finish_reason, gen_tokens_engine=None)

    cut = len(prefix_text)
    spans = list(row.get("block_index", []) or [])
    values = goldlib.blocks_from_spans(text, spans)
    covered_pairs: set[tuple[str, str]] = set()
    covered_triples: set[tuple[str, str, str]] = set()
    seen: set[tuple[str, str, str]] = set()
    new_gold_pairs = new_gold_triples = novel_tail_triples = tail_blocks = 0
    for span, block in zip(spans, values):
        identity = goldlib.block_identity(block) if isinstance(block, Mapping) else None
        in_tail = int(span["char_start"]) >= cut
        tail_blocks += int(in_tail)
        if identity is None:
            continue
        first_time = identity not in seen
        seen.add(identity)
        if in_tail and first_time:
            novel_tail_triples += 1
        if identity in record.triples and identity not in covered_triples:
            covered_triples.add(identity)
            if in_tail:
                new_gold_triples += 1
        if identity[:2] in record.pairs and identity[:2] not in covered_pairs:
            covered_pairs.add(identity[:2])
            if in_tail:
                new_gold_pairs += 1

    capture = row.get("motif_capture_triple", {}) or {}
    legacy = row.get("legacy_orbit", {}) or {}
    tail_capture = bool(capture.get("exists")) and \
        int(capture.get("confirmed_at_char_exclusive", 0)) > cut
    orbit_char = legacy.get("orbit_confirmed_at_char_exclusive")
    tail_orbit = bool(legacy.get("exists")) and orbit_char is not None and \
        int(orbit_char) > cut
    remaining = int(anchor.get("remaining_gold_pairs") or 0)
    return {
        "anchor_id": anchor["anchor_id"], "prompt_id": anchor["prompt_id"],
        "role": anchor["role"], "source_tag": anchor["source_tag"],
        "draw": int(tail["sample_index"]),
        "tail_tokens": int(tail["n_tokens"]),
        "actual_eos": bool(tail["ended_with_stop_token"]),
        "tail_hit_max": bool(tail["hit_max"]),
        "tail_capture": tail_capture,
        "tail_orbit": tail_orbit,
        "json_complete": bool(row.get("full_json_list_valid")),
        "top_level_list_complete": bool(row.get("top_level_list_complete")),
        "n_blocks_total": int(row.get("n_blocks", 0)),
        "n_blocks_tail": tail_blocks,
        "novel_tail_triples": novel_tail_triples,
        "new_gold_triples": new_gold_triples,
        "new_gold_pairs": new_gold_pairs,
        "remaining_gold_pairs_at_boundary": remaining,
        "remaining_gold_pair_recall": (new_gold_pairs / remaining) if remaining else None,
    }


def complete_long_conditions(path: Path, n_draws: int) -> set[tuple[str, str]]:
    """The (anchor, condition) cells that already hold ALL their draws.

    Draws are appended one at a time, so an interrupted run leaves a cell with
    one or two of its four continuations.  Treating that cell as finished would
    quietly leave the quality matrix ragged, so the partial rows are dropped
    from the file and the cell is redone.  Rewriting through a temporary file
    keeps an interrupted cleanup from destroying the finished cells.
    """
    if not path.is_file():
        return set()
    rows = list(iter_jsonl(path))
    draws: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        draws[(str(row["anchor_id"]), str(row["condition"]))].add(int(row.get("draw", -1)))
    complete = {cell for cell, seen in draws.items() if len(seen) >= n_draws}
    partial = {cell for cell in draws if cell not in complete}
    if partial:
        kept = [row for row in rows
                if (str(row["anchor_id"]), str(row["condition"])) in complete]
        write_jsonl(path, kept)
        log("r2", f"  {path.name}: dropped {len(rows) - len(kept)} row(s) from "
                  f"{len(partial)} interrupted condition(s); they will be redone")
    return complete


def collapse_draws(rows: Sequence[Mapping[str, Any]], metric: str
                   ) -> list[dict[str, Any]]:
    """One value per (anchor, condition) -- the mean over that cell's draws.

    The paired test compares CONDITIONS on an anchor, so the four continuations
    of a cell are its within-cell replicates and belong inside its single value.
    Passing them as separate rows would make the anchor appear four times and
    the pairing would keep only one of them.
    """
    out: list[dict[str, Any]] = []
    for condition in sorted({str(row["condition"]) for row in rows}):
        subset = [row for row in rows if str(row["condition"]) == condition]
        for entry in stats.mean_per_anchor(subset, value_field=metric):
            entry["condition"] = condition
            out.append(entry)
    return out


def stage_long(layout: Layout, args) -> int:
    scale = args.scale
    config = r2_models(scale)
    if not config.get("long_form", False) and not args.force:
        log("r2", f"long: scale {scale} is not configured for the full-continuation "
                  "matrix (§7 R2.7); pass --force to run it anyway")
        return 0
    from tcr.evaluation.structured import build_event_row       # noqa: E402

    d, meta = load_direction(layout, scale)
    randoms = random_directions(d)
    anchors = load_pool(layout, scale)
    selected = select_long_anchors(anchors, LONG_QUOTA)
    gold = goldlib.load_gold(layout.gold)
    bias_file = layout.r2 / f"R2_bias_choice_{scale}.json"
    bias_choice = (json.loads(bias_file.read_text(encoding="utf-8"))
                   if bias_file.is_file() else {})
    metrics = ({row["tag"]: row for row in read_csv(layout.metrics_csv)}
               if layout.metrics_csv.is_file() else {})
    tags = ([value.strip() for value in args.models.split(",") if value.strip()]
            if args.models != "auto" else steered_tags(config))
    without_bias = [tag for tag in tags if bias_choice.get(tag) is None]
    if without_bias:
        raise SystemExit(
            "no calibrated close-token bias for " + ", ".join(without_bias)
            + f"; run `run_r2.py bias --scale {scale}` first.  The "
            "full-continuation matrix has four conditions and the bias control "
            "is one of them; running three would leave the content-cost "
            "comparison without the cheap-trick it has to beat.")
    log("r2", f"long {scale}: {len(selected)} anchors "
              f"({dict(Counter(a['role'] for a in selected))}), "
              f"{len(LONG_CONDITIONS)} conditions, {LONG_DRAWS} draws")

    for tag in tags:
        out_path = layout.r2 / f"R2_long_{scale}_{tag}_shard{args.shard}.jsonl"
        if args.overwrite and out_path.exists():
            out_path.unlink()
        done = complete_long_conditions(out_path, LONG_DRAWS)
        model_path = model_path_for(layout, tag, metrics_row=metrics.get(tag))
        tokenizer = load_tokenizer(model_path)
        audit_chat_template(tokenizer)
        stops = stop_token_ids(model_path, tokenizer)
        model, backend = load_model(model_path, device=args.device,
                                    attn_implementation=args.attn)
        hook = SteeringHook(model, int(meta["layer"]), args.device)
        sign = steer_sign(tag, config)
        shard = selected[args.shard::args.num_shards]
        try:
            for index, anchor in enumerate(shard, start=1):
                chat_ids = build_chat_ids(tokenizer, anchor["prompt_text"])
                prefix_ids = [int(v) for v in anchor["prefix_response_ids"]]
                budget = max(CONTEXT_LIMIT - len(chat_ids) - len(prefix_ids), 0)
                if args.long_tokens:
                    budget = min(budget, args.long_tokens)
                if budget < 32:
                    log("r2", f"long {tag}: {anchor['anchor_id']} has {budget} tokens "
                              "of context left; skipping")
                    continue
                # One random direction per anchor, balanced across the eight, so
                # the control costs one condition rather than eight.
                which = anchor_seed(str(anchor["anchor_id"]), "random") % len(randoms)
                conditions = [
                    {"condition": "baseline", "family": "baseline", "alpha": 0.0,
                     "vector": None, "close_bias": 0.0},
                    {"condition": f"direction_a{MAIN_ALPHA:g}", "family": "direction",
                     "alpha": MAIN_ALPHA, "vector": sign * MAIN_ALPHA * d,
                     "close_bias": 0.0},
                    {"condition": "random_matched", "family": "random",
                     "alpha": MAIN_ALPHA, "vector": sign * MAIN_ALPHA * randoms[which],
                     "close_bias": 0.0},
                ]
                conditions.append({"condition": "close_bias", "family": "bias",
                                   "alpha": 0.0, "vector": None,
                                   "close_bias": sign * float(bias_choice[tag])})
                record = gold.by_prompt_id(str(anchor["prompt_id"]))
                for condition in conditions:
                    if (str(anchor["anchor_id"]), condition["condition"]) in done:
                        continue
                    tails = generate_long(
                        model, tokenizer, chat_ids=chat_ids,
                        prefix_response_ids=prefix_ids, stop_ids=stops,
                        device=args.device, n_samples=LONG_DRAWS,
                        max_new_tokens=budget,
                        base_seed=anchor_seed(str(anchor["anchor_id"]),
                                              str(condition["condition"])),
                        hook=hook, vector=condition["vector"],
                        close_bias=float(condition["close_bias"]),
                        close_token_id=int(anchor["close_tail_ids"][0]))
                    for tail in tails:
                        row = analyse_continuation(
                            prefix_text=str(anchor["prefix_response_text"]),
                            tail=tail, anchor=anchor, record=record,
                            tokenizer=tokenizer, build_event_row=build_event_row)
                        row.update({
                            "model_tag": tag, "scale": scale,
                            "condition": condition["condition"],
                            "family": condition["family"], "alpha": condition["alpha"],
                            "close_bias_value": condition["close_bias"],
                            "random_index": which if condition["family"] == "random" else None,
                            "context_budget": budget, "layer": int(meta["layer"]),
                        })
                        append_jsonl(out_path, row)
                log("r2", f"long {tag}: {index}/{len(shard)} anchors")
        finally:
            hook.remove()
            del model
            _empty_cache()
    return 0


# ------------------------------------------------------------------------ sae

def stage_sae(layout: Layout, args) -> int:
    scale = args.scale
    config = r2_models(scale)
    if not config.get("sae", False) and not args.force:
        log("r2", f"sae: no matching SAE resource declared for {scale}; skipping. "
                  "A dictionary trained for another size is not a substitute.")
        return 0
    from tcr.steering.sae import (differential_view, load_sae_layer,     # noqa: E402
                              merge_feature_views, projection_view, sae_encode_np)

    d, meta = load_direction(layout, scale)
    layer = int(meta["layer"])
    sae = load_sae_layer(layout.sae_root, layer)
    anchors = {str(row["anchor_id"]): row for row in load_pool(layout, scale)}
    clean_acts, layers = load_acts(layout, scale, config["clean"])
    raw_acts, _ = load_acts(layout, scale, config["raw"])
    position = layers.index(layer)

    dev_ids = [a for a in sorted(clean_acts) if a in raw_acts
               and anchors.get(a, {}).get("split") == "dev"]
    test_ids = [a for a in sorted(clean_acts) if a in raw_acts
                and anchors.get(a, {}).get("split") == "test"]
    dev_clean = np.stack([clean_acts[a][position] for a in dev_ids]).astype(np.float32)
    dev_raw = np.stack([raw_acts[a][position] for a in dev_ids]).astype(np.float32)
    acts_clean = sae_encode_np(dev_clean, sae)
    acts_raw = sae_encode_np(dev_raw, sae)

    projection = projection_view(d, sae)
    differential = differential_view(acts_raw, acts_clean)
    feature_ids = merge_feature_views(projection, differential, cap=32)

    test_clean = sae_encode_np(
        np.stack([clean_acts[a][position] for a in test_ids]).astype(np.float32), sae)
    test_raw = sae_encode_np(
        np.stack([raw_acts[a][position] for a in test_ids]).astype(np.float32), sae)
    by_id = {int(row["feature_id"]): row for row in projection}
    for row in differential:
        by_id.setdefault(int(row["feature_id"]), {}).update(row)

    rows: list[dict[str, Any]] = []
    for feature in feature_ids:
        info = by_id.get(feature, {})
        rows.append({
            "scale": scale, "layer": layer, "feature_id": feature,
            "view": info.get("view", "merged"),
            "cos_with_direction": info.get("cos_with_d_stop"),
            "dev_mean_act_clean": float(acts_clean[:, feature].mean()),
            "dev_mean_act_raw": float(acts_raw[:, feature].mean()),
            "test_mean_act_clean": float(test_clean[:, feature].mean()),
            "test_mean_act_raw": float(test_raw[:, feature].mean()),
            "test_delta_clean_minus_raw": float(
                test_clean[:, feature].mean() - test_raw[:, feature].mean()),
            "test_active_share_clean": float((test_clean[:, feature] > 0).mean()),
            "test_active_share_raw": float((test_raw[:, feature] > 0).mean()),
            "dev_test_sign_agrees": bool(
                np.sign(acts_clean[:, feature].mean() - acts_raw[:, feature].mean())
                == np.sign(test_clean[:, feature].mean() - test_raw[:, feature].mean())),
            # Filled in by hand after reading the activating contexts; "unknown"
            # is a legitimate final answer and is not a failure of the analysis.
            "manual_label": "",
        })
    write_csv(layout.r2 / f"R2_sae_features_{scale}.csv", rows)
    write_json(layout.r2 / f"R2_sae_manifest_{scale}.json", {
        "scale": scale, "layer": layer, "sae_root": str(layout.sae_root),
        "n_features": len(rows), "n_dev": len(dev_ids), "n_test": len(test_ids),
        "note": "Feature views describe what the direction runs along.  They do "
                "not establish a causal role; the intervention does.",
    })
    log("r2", f"sae {scale}: {len(rows)} features at layer {layer}")
    return 0


# --------------------------------------------------------------------- report

def long_matrix_gaps(rows: Sequence[Mapping[str, Any]],
                     selected: Sequence[Mapping[str, Any]],
                     tags: Sequence[str]) -> list[dict[str, Any]]:
    """Per model: what the anchor x condition x draw matrix is still missing.

    Counting anchors alone hides two failure shapes that matter -- an anchor
    that ran under three of the four conditions, and a cell that holds two of
    its four draws -- and both of them change what the quality contrast means.

    `tags` is the list of models that were SUPPOSED to run.  Deriving it from
    the rows instead would make a model with no output at all disappear from
    its own completeness report, which is the one case most worth seeing.
    """
    wanted_anchors = {str(anchor["anchor_id"]) for anchor in selected}
    wanted_draws = set(range(LONG_DRAWS))
    scale = str(rows[0].get("scale")) if rows else ""
    out: list[dict[str, Any]] = []
    for tag in sorted(set(tags) | {str(row["model_tag"]) for row in rows}):
        mine = [row for row in rows if str(row["model_tag"]) == tag]
        if not mine:
            out.append({
                "scale": scale, "model_tag": tag,
                "expected_anchors": len(wanted_anchors), "measured_anchors": 0,
                "missing_anchors": len(wanted_anchors),
                "expected_cells": len(wanted_anchors) * len(LONG_CONDITIONS),
                "measured_cells": 0,
                "missing_cells": len(wanted_anchors) * len(LONG_CONDITIONS),
                "ragged_cells": 0, "example_missing_cell": "(no output at all)",
                "example_ragged_cell": "",
            })
            continue
        draws: dict[tuple[str, str], set[int]] = defaultdict(set)
        for row in mine:
            draws[(str(row["anchor_id"]), str(row["condition"]))].add(
                int(row.get("draw", -1)))
        seen_anchors = {anchor for anchor, _ in draws}
        missing_cells = [(anchor, condition)
                         for anchor in sorted(seen_anchors)
                         for condition in LONG_CONDITIONS
                         if (anchor, condition) not in draws]
        ragged = [cell for cell, seen in draws.items() if seen != wanted_draws]
        out.append({
            "scale": mine[0].get("scale", scale), "model_tag": tag,
            "expected_anchors": len(wanted_anchors),
            "measured_anchors": len(seen_anchors & wanted_anchors),
            "missing_anchors": len(wanted_anchors - seen_anchors),
            "expected_cells": len(wanted_anchors) * len(LONG_CONDITIONS),
            "measured_cells": len(draws),
            "missing_cells": len(missing_cells),
            "ragged_cells": len(ragged),
            "example_missing_cell": (f"{missing_cells[0][0]}/{missing_cells[0][1]}"
                                     if missing_cells else ""),
            "example_ragged_cell": (f"{ragged[0][0]}/{ragged[0][1]}"
                                    if ragged else ""),
        })
    return out


SHORT_METRICS = ("next_event_close_rate", "margin_first_policy",
                 "close_sampler_prob", "eos_without_close_rate", "unresolved_rate")
LONG_METRICS = ("tail_capture", "tail_orbit", "actual_eos", "tail_hit_max",
                "json_complete", "new_gold_triples", "new_gold_pairs",
                "remaining_gold_pair_recall", "tail_tokens", "n_blocks_tail")

#: Pre-registered tolerance for "content was preserved": the early-position
#: recall of still-uncovered reference pairs may not fall more than this below
#: the untouched condition.  It is this study's chosen bound, not a general one.
CONTENT_TOLERANCE = 0.02


def stage_report(layout: Layout, args) -> int:
    scale = args.scale
    config = r2_models(scale)
    short_rows: list[dict[str, Any]] = []
    for path in sorted(layout.r2.glob(f"R2_short_{scale}_*.jsonl")):
        short_rows.extend(iter_jsonl(path))
    long_rows: list[dict[str, Any]] = []
    for path in sorted(layout.r2.glob(f"R2_long_{scale}_*.jsonl")):
        long_rows.extend(iter_jsonl(path))
    if not short_rows:
        raise SystemExit(f"no R2 short readouts for {scale}")

    # A shard that died leaves a smaller matrix that still aggregates cleanly,
    # so say so loudly rather than letting a partial result read as a result.
    anchors = load_pool(layout, scale)
    n_test = len([a for a in anchors if a.get("split") == "test"])
    # Iterate the models that were supposed to run.  A model whose every anchor
    # was skipped writes no rows at all, and a loop over the rows would then
    # leave it out of its own coverage report.
    expected_tags = steered_tags(config)
    for tag in sorted(set(expected_tags) | {str(row["model_tag"]) for row in short_rows}):
        measured = len({str(row["anchor_id"]) for row in short_rows
                        if str(row["model_tag"]) == tag})
        if measured < min(n_test, args.max_test_anchors):
            log("r2", f"[!] {tag}: short readouts cover {measured} of "
                      f"{min(n_test, args.max_test_anchors)} test anchors"
                      + ("; this model produced nothing" if measured == 0
                         else "; a shard is missing or failed"))
    if long_rows:
        gaps = long_matrix_gaps(long_rows, select_long_anchors(anchors, LONG_QUOTA),
                                expected_tags)
        write_csv(layout.r2 / f"R2_long_completeness_{scale}.csv", gaps)
        for row in gaps:
            if row["missing_anchors"] or row["missing_cells"] or row["ragged_cells"]:
                log("r2", f"[!] {row['model_tag']}: full continuations are "
                          f"incomplete -- {row['missing_anchors']} anchor(s) "
                          f"absent, {row['missing_cells']} (anchor, condition) "
                          f"cell(s) absent, {row['ragged_cells']} cell(s) without "
                          f"all {LONG_DRAWS} draws.  The quality table below is "
                          "computed on what exists.")

    # R2 re-measures its own untouched baseline rather than reusing R1's row.
    # The intervened conditions read their boundary distribution off the
    # sampler's prefill; R1 reads its margins off a teacher-forced forward.
    # Mixing the two would put a systematic difference between the baseline and
    # the conditions it is subtracted from, which is a bad trade for the ~8% of
    # GPU time the reuse would save.  The two are compared here instead, so
    # "they agree" is a checked fact rather than an assumption.
    r1_path = layout.r1 / f"R1_anchor_readouts_{scale}.jsonl"
    baseline_check: list[dict[str, Any]] = []
    if r1_path.is_file():
        r1_rows = {(str(row["model_tag"]), str(row["anchor_id"])): row
                   for row in iter_jsonl(r1_path)}
        for tag in sorted({str(row["model_tag"]) for row in short_rows}):
            pairs = [(row, r1_rows[(tag, str(row["anchor_id"]))])
                     for row in short_rows
                     if str(row["model_tag"]) == tag
                     and str(row["condition"]) == "baseline"
                     and (tag, str(row["anchor_id"])) in r1_rows]
            if not pairs:
                continue
            for field in ("next_event_close_rate", "margin_first_policy"):
                gaps = [float(a[field]) - float(b[field]) for a, b in pairs
                        if a.get(field) is not None and b.get(field) is not None]
                if not gaps:
                    continue
                baseline_check.append({
                    "scale": scale, "model_tag": tag, "metric": field,
                    "n_anchors": len(gaps),
                    "r2_baseline_mean": sum(float(a[field]) for a, _ in pairs) / len(pairs),
                    "r1_mean": sum(float(b[field]) for _, b in pairs) / len(pairs),
                    "mean_gap": sum(gaps) / len(gaps),
                    "max_abs_gap": max(abs(value) for value in gaps),
                })
        write_csv(layout.r2 / f"R2_baseline_vs_r1_{scale}.csv", baseline_check)

    effects: list[dict[str, Any]] = []
    for tag in sorted({str(row["model_tag"]) for row in short_rows}):
        rows = [row for row in short_rows if str(row["model_tag"]) == tag]
        conditions = sorted({str(row["condition"]) for row in rows})
        for metric in SHORT_METRICS:
            for condition in conditions:
                if condition == "baseline":
                    continue
                try:
                    effect = stats.paired_condition_effect(
                        rows, metric=metric, value_field=metric,
                        condition_field="condition", baseline="baseline",
                        treatment=condition)
                except ValueError:
                    continue
                effects.append(effect.as_dict(scale=scale, model_tag=tag,
                                              condition=condition,
                                              comparison="vs_baseline"))
            main = [row for row in rows
                    if row["condition"] == f"direction_a{MAIN_ALPHA:g}"]
            controls = [row for row in rows if row["family"] == "random"]
            if main and controls:
                try:
                    effect = stats.specificity_effect(
                        main, controls, metric=metric, value_field=metric)
                    effects.append(effect.as_dict(
                        scale=scale, model_tag=tag,
                        condition=f"direction_a{MAIN_ALPHA:g}",
                        comparison="vs_random_mean"))
                except ValueError:
                    pass
    write_csv(layout.r2 / f"R2_short_effects_{scale}.csv", effects)

    quality: list[dict[str, Any]] = []
    tradeoff: list[dict[str, Any]] = []
    if long_rows:
        for tag in sorted({str(row["model_tag"]) for row in long_rows}):
            rows = [row for row in long_rows if str(row["model_tag"]) == tag]
            for role in ("all", *sorted({str(row["role"]) for row in rows})):
                group = rows if role == "all" else [r for r in rows if r["role"] == role]
                for condition in sorted({str(row["condition"]) for row in group}):
                    bucket = [r for r in group if r["condition"] == condition]
                    entry: dict[str, Any] = {
                        "scale": scale, "model_tag": tag, "role": role,
                        "condition": condition, "n_continuations": len(bucket),
                        "n_anchors": len({r["anchor_id"] for r in bucket}),
                    }
                    for metric in LONG_METRICS:
                        values = [float(r[metric]) for r in bucket
                                  if r.get(metric) is not None]
                        entry[metric] = (sum(values) / len(values)) if values else None
                        entry[f"{metric}_n"] = len(values)
                    quality.append(entry)
                for metric in LONG_METRICS:
                    # The four draws of a cell are collapsed into that cell's
                    # mean FIRST; the pairing is over anchors, not draws.
                    collapsed = collapse_draws(group, metric)
                    for condition in sorted({str(row["condition"]) for row in group}):
                        if condition == "baseline":
                            continue
                        try:
                            effect = stats.paired_condition_effect(
                                collapsed, metric=metric, value_field=metric,
                                condition_field="condition", baseline="baseline",
                                treatment=condition, anchor_key="anchor_id")
                        except ValueError as exc:
                            log("r2", f"  {tag}/{role}/{condition}/{metric}: "
                                      f"skipped ({exc})")
                            continue
                        n_draws = sum(int(row.get("n_draws", 0)) for row in collapsed
                                      if row["condition"] == condition)
                        tradeoff.append(effect.as_dict(
                            scale=scale, model_tag=tag, role=role,
                            condition=condition, n_draws=n_draws))
        write_csv(layout.r2 / f"R2_long_quality_{scale}.csv", quality)
        write_csv(layout.r2 / f"R2_quality_tradeoff_{scale}.csv", tradeoff)

    meta_path = layout.r2 / f"R2_direction_{scale}.json"
    meta = (json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.is_file() else {})
    lines = [direction_summary_line(meta)]
    for tag in steered_tags(config):
        sign = "restore" if tag != config["clean"] else "reverse"
        picked = [row for row in effects if row["model_tag"] == tag
                  and row["condition"] == f"direction_a{MAIN_ALPHA:g}"]
        close_base = next((row for row in picked
                           if row["metric"] == "next_event_close_rate"
                           and row["comparison"] == "vs_baseline"), None)
        close_rand = next((row for row in picked
                           if row["metric"] == "next_event_close_rate"
                           and row["comparison"] == "vs_random_mean"), None)
        if close_base is None:
            lines.append(f"{registry.label(tag)} ({sign}): not measured")
            continue
        lines.append(
            f"{registry.label(tag)} ({sign}): close {fmt_pct(close_base['m0'])}"
            f"->{fmt_pct(close_base['m1'])} d={fmt_pct(close_base['diff'])}"
            f"[{fmt_pct(close_base['ci_low'])},{fmt_pct(close_base['ci_high'])}]"
            + (f"; vs random {fmt_pct(close_rand['diff'])}"
               f"[{fmt_pct(close_rand['ci_low'])},{fmt_pct(close_rand['ci_high'])}]"
               if close_rand else "; no random control"))
    write_compact(layout.compact / f"R2_short_{scale}.txt",
                  f"[R2 single-pulse local effect, {scale}]", lines)

    if tradeoff:
        qlines: list[str] = [f"content tolerance={CONTENT_TOLERANCE:.0%} of the "
                             "still-uncovered reference pairs (this study's bound)"]
        for tag in steered_tags(config):
            for role in ("early_remaining", "all"):
                picked = [row for row in tradeoff if row["model_tag"] == tag
                          and row["role"] == role
                          and row["condition"] == f"direction_a{MAIN_ALPHA:g}"]
                if not picked:
                    continue
                def grab(metric: str):
                    return next((row for row in picked if row["metric"] == metric), None)

                capture, recall, eos = grab("tail_capture"), \
                    grab("remaining_gold_pair_recall"), grab("actual_eos")
                parts = [f"{registry.label(tag)} [{role}]"]
                if capture:
                    parts.append(f"tail capture d={fmt_pct(capture['diff'])}"
                                 f"[{fmt_pct(capture['ci_low'])},{fmt_pct(capture['ci_high'])}]")
                if eos:
                    parts.append(f"actual EOS d={fmt_pct(eos['diff'])}")
                if recall:
                    verdict = ("within tolerance"
                               if recall["ci_low"] >= -CONTENT_TOLERANCE
                               else "content loss beyond the stated bound")
                    parts.append(f"remaining-pair recall d={fmt_pct(recall['diff'])}"
                                 f"[{fmt_pct(recall['ci_low'])},{fmt_pct(recall['ci_high'])}]"
                                 f" -> {verdict}")
                qlines.append("; ".join(parts))
        write_compact(layout.compact / f"R2_quality_{scale}.txt",
                      f"[R2 full-continuation cost, {scale}]", qlines)
    log("r2", f"report {scale}: {len(effects)} short effects, "
              f"{len(tradeoff)} quality contrasts")
    return 0


# ---------------------------------------------------------------------- entry

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("collect", "direction", "bias", "short",
                                          "long", "sae", "report", "all"))
    parser.add_argument("--scale", default="4B")
    parser.add_argument("--models", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--k-resample", type=int, default=16)
    parser.add_argument("--short-tokens", type=int, default=32,
                        help="tokens per short probe")
    parser.add_argument("--long-tokens", type=int, default=0,
                        help="cap on a full continuation; 0 uses the whole "
                             "remaining context, which is what the free-generation "
                             "evaluation gives a response")
    parser.add_argument("--sample-batch", type=int, default=8)
    parser.add_argument("--max-test-anchors", type=int, default=128)
    parser.add_argument("--calibration-anchors", type=int, default=32)
    parser.add_argument("--attn", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="run a stage the registry marks as not applicable")
    args = parser.parse_args()

    assert_frozen_sampling(NOTHINK_SAMPLING)
    layout = Layout.from_env()
    layout.ensure()
    stages = {"collect": stage_collect, "direction": stage_direction,
              "bias": stage_bias, "short": stage_short, "long": stage_long,
              "sae": stage_sae, "report": stage_report}
    if args.stage == "all":
        for name in ("collect", "direction", "bias", "short", "long", "sae", "report"):
            log("r2", f"=== stage {name} ===")
            stages[name](layout, args)
        return 0
    return stages[args.stage](layout, args)


if __name__ == "__main__":
    raise SystemExit(main())
