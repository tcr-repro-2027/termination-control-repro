# coding: utf-8
"""Turn every closeout result into the tables, figures and summaries the paper uses.

One entry, run once, producing everything: the two main tables, the four main
figures, the four appendices and the low-bandwidth summaries.  Running it again
after a later stage finishes simply refreshes whatever now exists.

Two rules it follows everywhere
-------------------------------
*Nothing is invented.*  A panel whose experiment has not run says so on the
figure; it never gets a zero, an interpolation or the shape the hypothesis
predicts.  A table cell with no measurement is empty, not 0.

*Every headline difference is paired.*  Rates and F1 are recomputed inside a
prompt-clustered bootstrap that resamples both arms together, because the two
models saw the same 1,106 records and subtracting two independent intervals
would not be an interval for the difference.  F1 is pooled from resampled
TP / predicted / gold counts rather than averaged over per-record F1 values.

Output tree (all of it under PAPER_ROOT)
----------------------------------------
    tables/T1_natural.{csv,tex,md}     tables/T2_obr.{csv,tex,md}
    figures/F1_overview .. F4_intervention  (.pdf and .png)
    appendix/A_controls, B_motif_gain, C_e2_limits, D_sae_features
    compact/N1_01.txt, O1_01.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (REPO_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tcr.paper import figures as figlib                                    # noqa: E402
from tcr.paper import registry, stats                                      # noqa: E402
from tcr.paper.io import (fmt, fmt_pct, iter_jsonl, log, read_csv,          # noqa: E402
                   write_compact, write_csv, write_json, write_table)
from tcr.paper.layout import Layout                                        # noqa: E402


# ------------------------------------------------------------------- loading

def load_metrics(layout: Layout) -> dict[str, dict[str, str]]:
    if not layout.metrics_csv.is_file():
        raise SystemExit(f"{layout.metrics_csv} is missing; E1 has to finish first")
    rows = {row["tag"]: row for row in read_csv(layout.metrics_csv)
            if str(row.get("is_final", "True")).lower() in ("true", "1", "")}
    log("assets", f"e1_metrics: {len(rows)} final rows")
    return rows


def number(row: Mapping[str, Any], field: str) -> float:
    try:
        value = row[field]
    except KeyError:
        return math.nan
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def response_rows(layout: Layout, tag: str) -> list[dict[str, Any]]:
    """Per-response numbers for the paired bootstrap, from the E1 event rows.

    The event rows already carry both halves the paper needs: the structured
    repetition events and the per-response quality counts the pooled F1 is built
    from.  Reading them here keeps the closeout on exactly the numbers the CSV
    summarises rather than on a second, slightly different computation.
    """
    path = layout.events(tag)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        flags = row.get("stage_flags", {}) or {}
        quality = row.get("quality", {}) or {}
        strict = quality.get("strict", {}) or {}
        relaxed = quality.get("relaxed", {}) or {}
        out.append({
            "stable_prompt_id": row.get("stable_prompt_id"),
            "seed": row.get("seed"),
            "semantic_capture": float(bool(flags.get("capture"))),
            "stable_orbit": float(bool(flags.get("stable_orbit"))),
            "hit_max": float(bool(flags.get("hit_max"))),
            "runaway_any": float(bool(flags.get("runaway_any"))),
            "gen_tokens": float(row.get("gen_tokens", 0) or 0),
            "n_blocks": float(row.get("n_blocks", 0) or 0),
            "json_list_complete": float(bool(row.get("full_json_list_valid"))),
            "strict_tp": float(strict.get("tp", 0) or 0),
            "strict_pred": float(strict.get("pred_n", 0) or 0),
            "strict_gold": float(strict.get("gold_n", 0) or 0),
            "relaxed_tp": float(relaxed.get("tp", 0) or 0),
            "relaxed_pred": float(relaxed.get("pred_n", 0) or 0),
            "relaxed_gold": float(relaxed.get("gold_n", 0) or 0),
            "n_pred_blocks": float(quality.get("n_pred_blocks", 0) or 0),
            "n_out_of_candidate": float((quality.get("n_pred_blocks", 0) or 0)
                                        - (quality.get("n_in_candidate", 0) or 0)),
            "n_out_of_text": float((quality.get("n_pred_blocks", 0) or 0)
                                   - (quality.get("n_in_text", 0) or 0)),
        })
    return out


RATE_METRICS = (
    ("semantic_capture_rate", "semantic_capture"),
    ("stable_orbit_rate", "stable_orbit"),
    ("hit_max_rate", "hit_max"),
    ("runaway_any_rate", "runaway_any"),
    ("json_list_complete_rate", "json_list_complete"),
)
MEAN_METRICS = (("gen_tokens_mean", "gen_tokens"), ("n_blocks_mean", "n_blocks"))
BLOCK_RATE_METRICS = (
    ("out_of_candidate_block_rate", "n_out_of_candidate", "n_pred_blocks"),
    ("out_of_text_endpoint_rate", "n_out_of_text", "n_pred_blocks"),
)


def paired_block(layout: Layout, m0: str, m1: str, *, label: str,
                 cache: dict[str, list[dict[str, Any]]], n_boot: int
                 ) -> list[dict[str, Any]]:
    """Every paired effect of one contrast, or an empty list if a side is absent."""
    for tag in (m0, m1):
        if tag not in cache:
            cache[tag] = response_rows(layout, tag)
    left, right = cache[m0], cache[m1]
    if not left or not right:
        log("assets", f"[!] {label}: no event rows for "
                      f"{m0 if not left else m1}; contrast skipped")
        return []
    rows: list[dict[str, Any]] = []
    shared = {"contrast": label, "m0_tag": m0, "m1_tag": m1}
    for name, field in RATE_METRICS:
        rows.append(stats.paired_rate(left, right, metric=name, value_field=field,
                                      n_boot=n_boot).as_dict(**shared))
    for name, field in MEAN_METRICS:
        rows.append(stats.paired_mean(left, right, metric=name, value_field=field,
                                      n_boot=n_boot).as_dict(**shared))
    for name, numerator, denominator in BLOCK_RATE_METRICS:
        rows.append(stats.paired_rate(left, right, metric=name,
                                      value_field=numerator,
                                      count_field=denominator,
                                      n_boot=n_boot).as_dict(**shared))
    rows.append(stats.paired_f1(left, right, metric="strict_f1", tp_field="strict_tp",
                                pred_field="strict_pred", gold_field="strict_gold",
                                n_boot=n_boot).as_dict(**shared))
    rows.append(stats.paired_f1(left, right, metric="relaxed_f1", tp_field="relaxed_tp",
                                pred_field="relaxed_pred", gold_field="relaxed_gold",
                                n_boot=n_boot).as_dict(**shared))
    return rows


# -------------------------------------------------------------------- tables

T1_FIELDS = ("scale", "condition", "train_seed", "n_responses",
             "continuous_pattern_repetition", "stable_token_orbit",
             "hit_context_limit", "gen_tokens_mean", "triple_f1", "entity_pair_f1")


def table1(layout: Layout, metrics: Mapping[str, Mapping[str, str]],
           effects: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tag in registry.T1_ROWS:
        row = metrics.get(tag)
        if row is None:
            log("assets", f"[!] Table 1: no E1 row for {tag}")
            continue
        spec = registry.spec(tag)
        rows.append({
            "scale": spec.size,
            "condition": {"notrain": "no SFT", "cleanv2": "entity-constraint cleaned",
                          "keep4": "raw"}.get(spec.condition, spec.condition),
            "train_seed": spec.train_seed if spec.train_seed is not None else "--",
            "n_responses": int(number(row, "n_responses")),
            "continuous_pattern_repetition": number(row, "semantic_capture_rate"),
            "stable_token_orbit": number(row, "stable_orbit_rate"),
            "hit_context_limit": number(row, "hit_max_rate"),
            "gen_tokens_mean": number(row, "gen_tokens_mean"),
            "triple_f1": number(row, "strict_f1"),
            "entity_pair_f1": number(row, "relaxed_f1"),
            "tag": tag,
        })
    write_table(layout.tables / "T1_natural", rows, fields=T1_FIELDS,
                caption="Raw versus entity-constraint-cleaned supervision. Rates "
                        "are over all responses; F1 is pooled micro-F1.",
                align_right=T1_FIELDS[3:])
    write_csv(layout.tables / "T1_natural_full.csv", rows)
    write_csv(layout.tables / "N1_paired_effects.csv", effects)
    return rows


T2_FIELDS = ("condition", "replacement_rate", "train_seed", "records", "target_blocks",
             "continuous_pattern_repetition", "stable_token_orbit",
             "hit_context_limit", "entity_pair_f1", "out_of_candidate_blocks")


def table2(layout: Layout, metrics: Mapping[str, Mapping[str, str]],
           effects: Sequence[Mapping[str, Any]],
           dose_table: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, tag, dose in registry.T2_ROWS:
        row = metrics.get(tag)
        if row is None:
            log("assets", f"[!] Table 2: no E1 row for {tag}")
            continue
        spec = registry.spec(tag)
        data = dose_table.get(spec.condition, {})
        rows.append({
            "condition": label,
            "replacement_rate": data.get("realized_rho", dose),
            "train_seed": spec.train_seed,
            "records": data.get("records", ""),
            "target_blocks": data.get("blocks", ""),
            "continuous_pattern_repetition": number(row, "semantic_capture_rate"),
            "stable_token_orbit": number(row, "stable_orbit_rate"),
            "hit_context_limit": number(row, "hit_max_rate"),
            "entity_pair_f1": number(row, "relaxed_f1"),
            "out_of_candidate_blocks": number(row, "out_of_candidate_block_rate"),
            "triple_f1": number(row, "strict_f1"),
            "gen_tokens_mean": number(row, "gen_tokens_mean"),
            "tag": tag,
        })
    write_table(layout.tables / "T2_obr", rows, fields=T2_FIELDS,
                caption="One-for-one target replacement. Input, record count and "
                        "per-record block count are held fixed; only which target "
                        "blocks are out-of-candidate moves. 24.3% is in Appendix A.",
                align_right=T2_FIELDS[1:])
    write_csv(layout.tables / "T2_obr_full.csv", rows)
    write_csv(layout.tables / "O1_dose_effects.csv", effects)
    return rows


def appendix_a(layout: Layout, metrics: Mapping[str, Mapping[str, str]],
               cache: dict[str, list[dict[str, Any]]], n_boot: int) -> None:
    rows: list[dict[str, Any]] = []
    reference = "qwen3-4b-cleanv2-s42"
    for tag in registry.APPENDIX_A_ROWS:
        row = metrics.get(tag)
        if row is None:
            continue
        spec = registry.spec(tag)
        rows.append({
            "condition": spec.label,
            "what_is_manipulated": {
                "cleanv2": "reference (entity-constraint cleaned)",
                "obr": "24.3% one-for-one target replacement",
                "isc_a": "input edited so a target loses candidate support",
                "isc_e": "input edited so a target loses literal support",
                "isc_ae": "input edited on both axes",
                "benign_input": "input edited without changing support status",
                "generic_noise": "relation labels swapped within the record; "
                                 "clauses of the original description reordered",
                "keep4_a": "raw entries whose endpoints are out of candidate only",
                "keep4_ae": "raw entries out of candidate and out of text",
                "keep4": "raw data, nothing removed",
            }.get(spec.condition, spec.condition),
            "continuous_pattern_repetition": number(row, "semantic_capture_rate"),
            "stable_token_orbit": number(row, "stable_orbit_rate"),
            "hit_context_limit": number(row, "hit_max_rate"),
            "entity_pair_f1": number(row, "relaxed_f1"),
            "out_of_candidate_blocks": number(row, "out_of_candidate_block_rate"),
            "episodes_per_response": number(row, "episodes_per_response"),
            "per_episode_capture_hazard": number(row, "per_episode_capture_hazard"),
            "gap_stop_hazard": number(row, "gap_stop_hazard"),
            "tag": tag,
        })
    effects: list[dict[str, Any]] = []
    for tag in registry.APPENDIX_A_ROWS:
        if tag == reference or tag not in metrics:
            continue
        effects.extend(paired_block(layout, reference, tag,
                                    label=f"{registry.label(tag)} - cleaned",
                                    cache=cache, n_boot=n_boot))
    write_table(layout.appendix / "A_controls", rows,
                caption="Conditions that bound the claim rather than carry it. "
                        "The doses and the amount of semantic damage are not "
                        "matched across these arms, so they are not a ranking.",
                align_right=[f for f in (rows[0].keys() if rows else [])
                             if f not in ("condition", "what_is_manipulated", "tag")])
    write_csv(layout.appendix / "A_controls_effects.csv", effects)
    log("assets", f"appendix A: {len(rows)} conditions, {len(effects)} effects")


def appendix_c(layout: Layout, e2_csv: Path | None) -> None:
    """E2 as a limit on interpretation, not as mechanism evidence."""
    if e2_csv is None or not Path(e2_csv).is_file():
        log("assets", "appendix C: no E2 CSV supplied; skipped")
        return
    rows = read_csv(e2_csv)
    wanted = {spec.tag for spec in registry.MODELS}
    kept = [row for row in rows if row.get("tag") in wanted]
    write_csv(layout.appendix / "C_e2_limits.csv", kept)
    hazard = sum(1 for row in rows if str(row.get("has_hazard")).lower() == "true")
    write_json(layout.appendix / "C_e2_manifest.json", {
        "source": str(e2_csv), "rows_in_file": len(rows),
        "rows_used": len(kept), "rows_with_hazard": hazard,
        "reading": "Per-point net effect is manip minus neutral; the two removal "
                   "levels give a through-the-origin slope. The readout is the "
                   "first-divergence processed logit difference, NOT S1's "
                   "multi-token path score, and there is no short-continuation "
                   "measurement here. What this bounds is the original "
                   "support-conditioning explanation; it does not show that input "
                   "information is irrelevant to termination.",
    })
    log("assets", f"appendix C: {len(kept)} E2 rows for this paper's models")


def appendix_b(layout: Layout) -> list[dict[str, Any]]:
    path = layout.x1 / "X1_paired_effects.csv"
    if not path.is_file():
        log("assets", "appendix B: X1 has not run; skipped")
        return []
    effects = read_csv(path)
    by_m = read_csv(layout.x1 / "X1_gain_by_m.csv") if \
        (layout.x1 / "X1_gain_by_m.csv").is_file() else []
    write_table(layout.appendix / "B_motif_gain",
                [row for row in effects if row.get("motif_length") == "all"],
                fields=("purpose", "metric", "m0", "m1", "diff", "ci_low",
                        "ci_high", "n_prompts", "direction", "m0_tag", "m1_tag"),
                caption="Artificial complete-motif gain G and its two components. "
                        "G = E + P exactly; a negative contrast means the extra "
                        "repetition made repeating relatively LESS attractive. "
                        "m0/m1 are the two arms' mean values in nats per token.",
                align_right=("m0", "m1", "diff", "ci_low", "ci_high", "n_prompts"))
    write_csv(layout.appendix / "B_motif_gain_by_m.csv", by_m)
    return effects


def appendix_d(layout: Layout, scale: str = "8B") -> list[dict[str, Any]]:
    path = layout.r2 / f"R2_sae_features_{scale}.csv"
    if not path.is_file():
        log("assets", f"appendix D: no SAE features for {scale}; skipped")
        return []
    rows = read_csv(path)
    write_table(layout.appendix / "D_sae_features", rows,
                caption="Dictionary features the direction runs along. Descriptive: "
                        "an activation change is not a causal role, and 'unknown' is "
                        "a legitimate label.",
                align_right=("cos_with_direction", "test_delta_clean_minus_raw"))
    return rows


# ------------------------------------------------------------------- figures

def figure1(layout: Layout, plt) -> None:
    """What the object of study is: a trajectory, not a single wrong answer."""
    example_path = layout.r0 / "F1_example.json"
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))
    ax = axes[0]
    ax.set_title("One-for-one target replacement (OBR)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for index, (label, colour) in enumerate([("cleaned target list", figlib.ROLE_COLORS["C"]),
                                             ("replaced target list", figlib.ROLE_COLORS["O"])]):
        y = 2.6 - 1.6 * index
        for block in range(8):
            swapped = index == 1 and block in (2, 5)
            ax.add_patch(plt.Rectangle((0.4 + block * 1.1, y), 0.95, 0.6,
                                       facecolor=("#D2694A" if swapped else colour),
                                       edgecolor="white", linewidth=0.8))
        ax.text(0.4, y + 0.75, label, fontsize=7.5)
    ax.text(0.4, 0.45, "held fixed: the input, the record count, the number of\n"
                       "blocks per record, one list close and one EOS",
            fontsize=6.6, color="#444444", va="top")

    ax = axes[1]
    if example_path.is_file():
        example = json.loads(example_path.read_text(encoding="utf-8"))
        ordinals = list(range(1, len(example["novel"]) + 1))
        ax.step(ordinals, example["gold_pairs_hit"], where="post",
                color=figlib.ROLE_COLORS["C"], label="reference pairs covered")
        reuse = [i for i, novel in enumerate(example["novel"], start=1) if not novel]
        if reuse:
            ax.plot(reuse, [0.0] * len(reuse), "|", color=figlib.ROLE_COLORS["K"],
                    markersize=6, label="reused relation")
        if example.get("capture_block"):
            ax.axvline(example["capture_block"], color="#333333", linestyle="--",
                       linewidth=1.0, label="continuous repetition confirmed")
        ax.set_xlabel("complete relation block")
        ax.set_ylabel("reference pairs covered")
        ax.set_title(f"A real trajectory ({registry.label(example['tag'])})")
        ax.legend(loc="upper left")
    else:
        figlib.annotate_missing(ax, "no example selected yet (run run_r0.py)")
    figlib.note(fig, "Left: the controlled construction. Right: one real response, "
                     "not a schematic.")
    figlib.save(fig, layout.figures / "F1_overview")


def figure2(layout: Layout, metrics: Mapping[str, Mapping[str, str]], plt) -> None:
    """Exposure and persistence are not the same quantity."""
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.0))

    ax = axes[0]
    plotted = 0
    for tag in ("qwen3-4b-notrain", "qwen3-4b-cleanv2-s42", "qwen3-4b-keep4-s42",
                "qwen3-4b-obr-p5-s42", "qwen3-4b-obr-p10-s42", "qwen3-4b-obr-p15-s42",
                "qwen3-4b-generic-noise-s42", "qwen3-4b-isc-a-s42"):
        row = metrics.get(tag)
        if row is None:
            continue
        x = number(row, "episodes_per_response")
        y = number(row, "per_episode_capture_hazard")
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        spec = registry.spec(tag)
        ax.scatter(x, 100.0 * y, s=34, color=figlib.color_for(spec.role),
                   edgecolor="white", linewidth=0.6, zorder=3)
        # Alternate the label side so near-coincident points stay readable.
        offset = (5, 5) if plotted % 2 == 0 else (5, -9)
        ax.annotate(spec.label, (x, 100.0 * y), textcoords="offset points",
                    xytext=offset, fontsize=6.4)
        plotted += 1
    if plotted:
        ax.set_xlabel("reuse episodes per response")
        ax.set_ylabel("per-episode capture hazard (%)")
        ax.set_title("More chances, or more dangerous chances?")
    else:
        figlib.annotate_missing(ax, "no episode columns in e1_metrics.csv")

    curves = {"4b_CK_s42": ("raw vs cleaned", figlib.ROLE_COLORS["K"]),
              "4b_CO15_s42": ("OBR 15% vs cleaned", figlib.ROLE_COLORS["O"])}
    ax = axes[1]
    drawn = False
    for pair, (label, colour) in curves.items():
        path = layout.r0 / pair / "p0d_episode_hazard" / "P0D_EPISODE_SUMMARY.csv"
        if not path.is_file():
            continue
        rows = [row for row in read_csv(path)
                if row.get("episode_variant") == "lineage"
                and row.get("capture_kind") == "primary"]
        for model, style in (("M0", ":"), ("M1", "-")):
            picked = sorted((row for row in rows if row.get("model") == model),
                            key=lambda row: int(row["episode_ordinal"]))
            picked = [row for row in picked if int(row.get("episodes_started", 0)) >= 20]
            if not picked:
                continue
            # A marker as well as a line: an ordinal range that survives the
            # at-risk floor can be a single point, and a lone point on a line
            # plot draws nothing at all.
            ax.plot([int(row["episode_ordinal"]) for row in picked],
                    [100.0 * float(row["capture_hazard_given_episode"]) for row in picked],
                    style, marker="o", markersize=2.6,
                    color=colour if model == "M1" else "#888888",
                    label=f"{label} [{model}]" if model == "M1" else None)
            drawn = True
    if drawn:
        ax.set_xlabel("reuse episode ordinal k")
        ax.set_ylabel("capture hazard given the episode (%)")
        ax.set_title("Per-episode risk along the process")
        ax.legend()
    else:
        figlib.annotate_missing(ax, "P0d episode curves not available yet")

    ax = axes[2]
    labels, values, lows, highs, colours = [], [], [], [], []
    for pair, (label, colour) in curves.items():
        path = layout.r0 / pair / "p0d_episode_hazard" / "P0D_SHAPLEY.csv"
        if not path.is_file():
            continue
        rows = [row for row in read_csv(path)
                if row.get("episode_variant") == "lineage"
                and row.get("capture_kind") == "primary"]
        for component, name in (("exposure_component", "exposure"),
                                ("propensity_component", "persistence")):
            picked = next((row for row in rows if row["component"] == component), None)
            if picked is None:
                continue
            labels.append(f"{label}\n{name}")
            values.append(float(picked["value"]))
            lows.append(float(picked["ci_low"]))
            highs.append(float(picked["ci_high"]))
            colours.append(colour if name == "persistence" else "#9FB6D4")
    if labels:
        figlib.bar_with_ci(axes[2], labels, values, lows, highs, colours,
                           ylabel="descriptive component of the CIF difference",
                           rotation=18)
        axes[2].set_title("Standardised decomposition")
    else:
        figlib.annotate_missing(ax, "Shapley components not available yet")
    figlib.note(fig, "Right panel is a descriptive standardisation on the common "
                     "risk range, not a mediated causal share.")
    figlib.save(fig, layout.figures / "F2_dynamics")


def figure3(layout: Layout, plt, scale: str = "4B") -> None:
    """Same prefix, different exit choice?"""
    path = layout.r1 / f"R1_paired_effects_{scale}.csv"
    execution = layout.r1 / f"R1_close_execution_{scale}.csv"
    roles = ("natural_stop", "early_remaining", "pre_first_reuse")

    def rows_for(effects, metric):
        """Test-split treatment contrasts for one metric, in a fixed order.

        The seed-only contrast is deliberately excluded: it is the scale the
        others are read against, reported in the text, not another bar that
        would invite reading training randomness as a treatment.
        """
        picked = [row for row in effects
                  if row["metric"] == metric and row["split"] == "test"
                  and row["source"] == "all" and row["role"] in roles
                  and row.get("kind") == "treatment"]
        return sorted(picked, key=lambda row: (row["contrast"],
                                               roles.index(row["role"])))

    if not path.is_file():
        fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.0))
        for ax in axes:
            figlib.annotate_missing(ax, "R1 has not run yet")
        figlib.save(fig, layout.figures / "F3_termination")
        return

    effects = read_csv(path)
    ordered = rows_for(effects, "next_event_close_rate")
    # One row per (contrast, role); the panel has to grow with them or twelve
    # labels end up on top of each other.
    height = max(3.0, 0.30 * len(ordered) + 1.1)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, height))

    labels = [f"{row['contrast']}  |  {row['role'].replace('_', ' ')}"
              for row in ordered]
    for index, (metric, title, percent) in enumerate((
            ("next_event_close_rate", "Local close frequency (16 draws)", True),
            ("margin_first_policy", "First-token stop margin", False))):
        ax = axes[index]
        picked = rows_for(effects, metric)
        if not picked:
            figlib.annotate_missing(ax, "no paired test effects")
            continue
        figlib.diff_panel(ax, labels if index == 0 else [""] * len(picked),
                          [float(row["diff"]) for row in picked],
                          [float(row["ci_low"]) for row in picked],
                          [float(row["ci_high"]) for row in picked],
                          xlabel="vs cleaned, same seed", percent=percent)
        ax.set_title(title)
        if index:
            ax.tick_params(axis="y", length=0)

    ax = axes[2]
    if execution.is_file():
        rows = [row for row in read_csv(execution) if row["role"] == "all"]
        names = [registry.label(row["model_tag"]) for row in rows]
        values = [float(row["close_reachable_share"]) for row in rows]
        colours = [figlib.color_for(registry.BY_TAG[row["model_tag"]].role)
                   if row["model_tag"] in registry.BY_TAG else "#888888"
                   for row in rows]
        figlib.bar_with_ci(ax, names, values, values, values, colours,
                           ylabel="close token inside top-k/top-p", percent=True,
                           rotation=30)
        ax.set_title("Is closing even reachable?")
    else:
        figlib.annotate_missing(ax, "close-execution table missing")
    figlib.note(fig, "Each contrast is against the cleaned arm of the SAME "
                     "training seed. Prefixes come from both source models; a "
                     "text prefix is on-policy for the model that produced it only.")
    figlib.save(fig, layout.figures / "F3_termination")


def figure4(layout: Layout, plt, scale: str = "4B") -> None:
    """Does the direction do anything, and what does it cost?"""
    short_path = layout.r2 / f"R2_short_effects_{scale}.csv"
    long_path = layout.r2 / f"R2_quality_tradeoff_{scale}.csv"
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.0))

    ax = axes[0]
    if short_path.is_file():
        effects = read_csv(short_path)
        labels, diffs, lows, highs, colours = [], [], [], [], []
        for row in effects:
            if row["metric"] != "next_event_close_rate":
                continue
            if row["condition"] not in ("direction_a1", "close_bias"):
                continue
            if row["condition"] == "direction_a1" and row["comparison"] != "vs_random_mean":
                continue
            if row["condition"] == "close_bias" and row["comparison"] != "vs_baseline":
                continue
            labels.append(f"{registry.label(row['model_tag'])}\n"
                          + ("direction - random" if row["condition"] == "direction_a1"
                             else "close bias - baseline"))
            diffs.append(float(row["diff"]))
            lows.append(float(row["ci_low"]))
            highs.append(float(row["ci_high"]))
            colours.append(figlib.ROLE_COLORS["bias"] if row["condition"] == "close_bias"
                           else figlib.ROLE_COLORS["C"])
        if labels:
            figlib.diff_panel(ax, labels, diffs, lows, highs,
                              xlabel="change in local close frequency", percent=True)
            ax.set_title("Single pulse vs equal-norm random")
        else:
            figlib.annotate_missing(ax, "no short effects")
    else:
        figlib.annotate_missing(ax, "R2 short stage has not run")

    if long_path.is_file():
        tradeoff = read_csv(long_path)
        for index, (metric, title, percent) in enumerate((
                ("tail_capture", "Repetition after the boundary", True),
                ("remaining_gold_pair_recall", "Still-owed reference pairs recalled", True))):
            ax = axes[index + 1]
            labels, diffs, lows, highs, colours = [], [], [], [], []
            for row in tradeoff:
                if row["metric"] != metric or row["condition"] != "direction_a1":
                    continue
                if row["role"] not in ("early_remaining", "all"):
                    continue
                labels.append(f"{registry.label(row['model_tag'])}\n{row['role']}")
                diffs.append(float(row["diff"]))
                lows.append(float(row["ci_low"]))
                highs.append(float(row["ci_high"]))
                spec = registry.BY_TAG.get(row["model_tag"])
                colours.append(figlib.color_for(spec.role if spec else "C"))
            if labels:
                figlib.diff_panel(ax, labels, diffs, lows, highs,
                                  xlabel="change from no intervention", percent=percent)
                ax.set_title(title)
            else:
                figlib.annotate_missing(ax, "no full-continuation contrast")
    else:
        for ax in axes[1:]:
            figlib.annotate_missing(ax, "R2 long stage has not run")
    figlib.note(fig, "The stop rate and the content cost are shown together on "
                     "purpose; either one alone would be misleading.")
    figlib.save(fig, layout.figures / "F4_intervention")


def pick_example(layout: Layout) -> None:
    """One real trajectory for Figure 1, chosen by a fixed rule, never written by hand."""
    tag = "qwen3-4b-keep4-s42"
    path = layout.r0 / f"R0_prefix_candidates_{tag}.jsonl"
    if not path.is_file():
        return
    best: dict[str, Any] | None = None
    for row in iter_jsonl(path):
        if not row.get("semantic_capture") or not row.get("capture_confirmed_block_exclusive_0based"):
            continue
        confirmed = int(row["capture_confirmed_block_exclusive_0based"])
        hits = row.get("gold_pairs_hit") or []
        # A readable example: the response covered some reference material first
        # and the repetition was confirmed early enough to fit on an axis.
        if confirmed >= len(hits) or confirmed > 80 or not hits:
            continue
        covered = hits[min(confirmed, len(hits) - 1)]
        if covered < 5:
            continue
        score = (covered, -confirmed)
        if best is None or score > best["_score"]:
            best = {"_score": score, "tag": tag,
                    "sample_id": row["sample_id"],
                    "capture_block": confirmed,
                    "gold_pairs_hit": hits[: confirmed + 12],
                    "novel": (row.get("first_occurrence") or [])[: confirmed + 12],
                    "gold_pairs_total": row.get("gold_pairs")}
    if best is None:
        return
    best.pop("_score")
    write_json(layout.r0 / "F1_example.json", best)
    log("assets", f"figure 1 example: {best['sample_id']} "
                  f"(repetition confirmed at block {best['capture_block']})")


# ------------------------------------------------------------------- compacts

def compacts(layout: Layout, t1: Sequence[Mapping[str, Any]],
             n1: Sequence[Mapping[str, Any]], t2: Sequence[Mapping[str, Any]],
             o1: Sequence[Mapping[str, Any]]) -> None:
    lines: list[str] = []
    for label, m0, m1 in registry.T1_PAIRS:
        picked = [row for row in n1 if row["contrast"] == label]
        if not picked:
            lines.append(f"{label}: not available")
            continue
        def grab(metric: str):
            return next((row for row in picked if row["metric"] == metric), None)

        capture, pair_f1, hit = grab("semantic_capture_rate"), \
            grab("relaxed_f1"), grab("hit_max_rate")
        lines.append(
            f"{label} raw-cleaned n={capture['n_prompts']}: repetition "
            f"{fmt_pct(capture['m0'])}->{fmt_pct(capture['m1'])} "
            f"d={fmt_pct(capture['diff'])}[{fmt_pct(capture['ci_low'])},"
            f"{fmt_pct(capture['ci_high'])}]"
            + (f"; hit-max d={fmt_pct(hit['diff'])}" if hit else "")
            + (f"; pair F1 d={fmt_pct(pair_f1['diff'])}"
               f"[{fmt_pct(pair_f1['ci_low'])},{fmt_pct(pair_f1['ci_high'])}]"
               if pair_f1 else ""))
    write_compact(layout.compact / "N1_01.txt",
                  "[N1 raw vs entity-constraint-cleaned]", lines)

    lines = []
    for row in t2:
        lines.append(f"{row['condition']}: repetition "
                     f"{fmt_pct(row['continuous_pattern_repetition'])}, "
                     f"orbit {fmt_pct(row['stable_token_orbit'])}, "
                     f"hit-max {fmt_pct(row['hit_context_limit'])}, "
                     f"pair F1 {fmt_pct(row['entity_pair_f1'])}")
    for label, m0, m1 in registry.T2_PAIRS:
        picked = [row for row in o1 if row["contrast"] == label]
        capture = next((row for row in picked
                        if row["metric"] == "semantic_capture_rate"), None)
        if capture is None:
            continue
        lines.append(f"{label}: d={fmt_pct(capture['diff'])}"
                     f"[{fmt_pct(capture['ci_low'])},{fmt_pct(capture['ci_high'])}] "
                     f"{capture['direction']}")
    lines.append("24.3% is reported in Appendix A; the response is not claimed "
                 "monotone across the whole range.")
    write_compact(layout.compact / "O1_01.txt", "[O1 target-replacement dose]", lines)


# ----------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--e2-csv", default=None,
                        help="e2_metrics.csv, for Appendix C")
    parser.add_argument("--dose-table", default=None,
                        help="obr_dose_data_table.csv from obr_dose_table.py")
    parser.add_argument("--scale", default="4B", help="scale for Figures 3 and 4")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    layout = Layout.from_env()
    layout.ensure()
    metrics = load_metrics(layout)
    cache: dict[str, list[dict[str, Any]]] = {}

    n1: list[dict[str, Any]] = []
    for label, m0, m1 in registry.T1_PAIRS:
        if m0 in metrics and m1 in metrics:
            n1.extend(paired_block(layout, m0, m1, label=label, cache=cache,
                                   n_boot=args.bootstrap))
    o1: list[dict[str, Any]] = []
    for label, m0, m1 in registry.T2_PAIRS:
        if m0 in metrics and m1 in metrics:
            o1.extend(paired_block(layout, m0, m1, label=label, cache=cache,
                                   n_boot=args.bootstrap))

    dose_table: dict[str, dict[str, Any]] = {}
    if args.dose_table and Path(args.dose_table).is_file():
        for row in read_csv(args.dose_table):
            dose_table[str(row.get("dataset", "")).replace("train_obr", "obr")
                       .replace("cleanv2_train", "cleanv2").strip("_")] = row

    t1 = table1(layout, metrics, n1)
    t2 = table2(layout, metrics, o1, dose_table)
    appendix_a(layout, metrics, cache, args.bootstrap)
    appendix_b(layout)
    appendix_c(layout, Path(args.e2_csv) if args.e2_csv else None)
    appendix_d(layout)
    compacts(layout, t1, n1, t2, o1)

    if not args.skip_figures:
        plt = figlib.pyplot()
        pick_example(layout)
        figure1(layout, plt)
        figure2(layout, metrics, plt)
        figure3(layout, plt, args.scale)
        figure4(layout, plt, args.scale)

    write_json(layout.paper_root / "assets_manifest.json", {
        "layout": layout.describe(),
        "bootstrap": args.bootstrap,
        "n_table1_rows": len(t1), "n_table2_rows": len(t2),
        "n_natural_effects": len(n1), "n_dose_effects": len(o1),
        "models_missing_from_e1": registry.missing(registry.all_tags(), metrics),
    })
    log("assets", f"done -> {layout.paper_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
