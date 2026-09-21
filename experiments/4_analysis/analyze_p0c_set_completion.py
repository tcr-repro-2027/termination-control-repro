#!/usr/bin/env python3
"""P0c set-completion utility, overrun, and multi-seed capture audit."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tcr.events.io_utils import sha256_file, write_csv, write_json, write_jsonl
from tcr.events.p0c_set_completion import (
    MODEL_TAGS,
    PROGRESS_BINS,
    analyze_response_files,
    bootstrap_opportunity_rates,
    bootstrap_response_metrics,
    build_opportunity_array,
    build_utility_array,
    decision_from_results,
    descriptive_summary,
    load_event_meta,
    load_gold_dataset,
    opportunity_contrasts,
    prompt_ids_and_validate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--gold-data", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gold-key-field", default="auto")
    parser.add_argument("--responses-m0", default=None)
    parser.add_argument("--responses-m1", default=None)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--bootstrap-batch-size", type=int, default=200)
    parser.add_argument("--max-return-chars", type=int, default=1950)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NA"
    return f"{100.0 * number:.{digits}f}%"


def gfmt(value: Any, digits: int = 2) -> str:
    """Format a gold-normalized block position without producing ``NAG``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}G"


def ci_text(result: Mapping[str, float]) -> str:
    return f"{pct(result['diff'])}[{pct(result['ci_low'])},{pct(result['ci_high'])}]"


def direction(low: float, high: float) -> str:
    if low > 0:
        return "UP"
    if high < 0:
        return "DOWN"
    return "UNRESOLVED"


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def key_metric_rows(metrics: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, result in metrics.items():
        rows.append(
            {
                "metric": name,
                "m0": result["m0"],
                "m1": result["m1"],
                "diff_m1_minus_m0": result["diff"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "m0_ci_low": result["m0_ci_low"],
                "m0_ci_high": result["m0_ci_high"],
                "m1_ci_low": result["m1_ci_low"],
                "m1_ci_high": result["m1_ci_high"],
                "n_m0": result["n_m0"],
                "n_m1": result["n_m1"],
                "direction": direction(result["ci_low"], result["ci_high"]),
            }
        )
    return rows


def invalid_object_rows(analyses) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_TAGS:
        model_rows = [row for row in analyses if row.model_tag == model]
        total = sum(row.parser_invalid_objects for row in model_rows)
        for index, (label, left, right) in enumerate(PROGRESS_BINS):
            count = sum(row.invalid_object_bin_counts[index] for row in model_rows)
            rows.append(
                {
                    "model": model,
                    "progress_range": label,
                    "progress_left": left,
                    "progress_right": right,
                    "invalid_complete_object_candidates": count,
                    "all_invalid_complete_object_candidates": total,
                    "share_within_model": count / total if total else math.nan,
                    "n_responses": len(model_rows),
                }
            )
    return rows


def seed_progress_rows(analyses) -> list[dict[str, Any]]:
    bins = (
        ("<0.75G", 0.0, 0.75),
        ("0.75-1.00G", 0.75, 1.0),
        ("1.00-1.25G", 1.0, 1.25),
        (">=1.25G", 1.25, math.inf),
    )
    rows: list[dict[str, Any]] = []
    for model in MODEL_TAGS:
        seeded = [row for row in analyses if row.model_tag == model and row.first_seed_progress is not None]
        for label, left, right in bins:
            selected = [row for row in seeded if left <= float(row.first_seed_progress) < right]
            rows.append(
                {
                    "model": model,
                    "progress_range": label,
                    "count": len(selected),
                    "denominator_seeded": len(seeded),
                    "share": len(selected) / len(seeded) if seeded else math.nan,
                    "mean_strict_coverage": (
                        sum(float(row.first_seed_strict_coverage) for row in selected) / len(selected)
                        if selected
                        else math.nan
                    ),
                    "mean_relaxed_coverage": (
                        sum(float(row.first_seed_relaxed_coverage) for row in selected) / len(selected)
                        if selected
                        else math.nan
                    ),
                    "mean_relaxed_fraction_of_final": (
                        sum(float(row.first_seed_relaxed_fraction_of_final) for row in selected if row.first_seed_relaxed_fraction_of_final is not None)
                        / sum(row.first_seed_relaxed_fraction_of_final is not None for row in selected)
                        if any(row.first_seed_relaxed_fraction_of_final is not None for row in selected)
                        else math.nan
                    ),
                }
            )
    return rows


def capture_distribution_rows(descriptive: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_TAGS:
        for kind, field in (
            ("primary_semantic_capture", "capture_seed_ordinal_bins"),
            ("legacy_aligned_capture", "legacy_capture_seed_ordinal_bins"),
        ):
            counts = descriptive[model][field]
            denominator = sum(counts.values())
            for ordinal_bin, count in counts.items():
                rows.append(
                    {
                        "model": model,
                        "capture_kind": kind,
                        "seed_ordinal_bin": ordinal_bin,
                        "count": count,
                        "denominator": denominator,
                        "share": count / denominator if denominator else math.nan,
                    }
                )
    return rows


def compact_text(
    *,
    n_prompts: int,
    n_m0: int,
    n_m1: int,
    gold_field: str,
    descriptive: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, float]],
    opportunity: Mapping[str, Any],
    contrasts: Sequence[Mapping[str, Any]],
    decision: Mapping[str, str],
    max_chars: int,
) -> str:
    post = opportunity["aggregate"][">=1.00G"]
    deep = opportunity["aggregate"][">=1.25G"]
    contrast = {str(row["contrast"]): row for row in contrasts}
    utility = contrast["M1_post_ge1_minus_pre_0.5_1.0::new_valid_any"]
    reuse = contrast["M1_post_ge1_minus_pre_0.5_1.0::seen_triple_reuse"]
    lines = [
        "[01b-P0c COMPACT]",
        f"完整性=PASS; prompts={n_prompts}; rows M0/M1={n_m0}/{n_m1}; gold_key={gold_field}; response/block/hash全校验",
        f"L/G中位数 M0/M1={fmt(descriptive['M0']['output_gold_ratio_median'],2)}/{fmt(descriptive['M1']['output_gold_ratio_median'],2)}; parser-invalid={descriptive['M0']['parser_invalid_objects']}/{descriptive['M1']['parser_invalid_objects']}",
        f"relaxed覆盖@1G M0/M1={pct(metrics['relaxed_coverage_at_1g']['m0'])}/{pct(metrics['relaxed_coverage_at_1g']['m1'])}; G后新增覆盖={pct(metrics['post_1g_relaxed_coverage_gain']['m0'])}/{pct(metrics['post_1g_relaxed_coverage_gain']['m1'])}, Δ{ci_text(metrics['post_1g_relaxed_coverage_gain'])}",
        f">=1.00G 新gold效用块 M0/M1={pct(post['new_valid_any']['m0'])}/{pct(post['new_valid_any']['m1'])}, Δ{ci_text(post['new_valid_any'])}",
        f">=1.00G seen-reuse M0/M1={pct(post['seen_triple_reuse']['m0'])}/{pct(post['seen_triple_reuse']['m1'])}, Δ{ci_text(post['seen_triple_reuse'])}",
        f">=1.00G new-unmatched M0/M1={pct(post['new_unmatched']['m0'])}/{pct(post['new_unmatched']['m1'])}, Δ{ci_text(post['new_unmatched'])}",
        f">=1.00G normal-stop M0/M1={pct(post['normal_stop']['m0'])}/{pct(post['normal_stop']['m1'])}, Δ{ci_text(post['normal_stop'])}",
        f"M1 post-pre: 新gold-match Δ{pct(utility['value'])}[{pct(utility['ci_low'])},{pct(utility['ci_high'])}]; reuse Δ{pct(reuse['value'])}[{pct(reuse['ci_low'])},{pct(reuse['ci_high'])}]",
        f">=1.25G M1: gold-match={pct(deep['new_valid_any']['m1'])}; reuse={pct(deep['seen_triple_reuse']['m1'])}; unmatched={pct(deep['new_unmatched']['m1'])}; stop={pct(deep['normal_stop']['m1'])}",
        f"First-seed中位位置 M0/M1={gfmt(descriptive['M0']['seed_progress_median'])}/{gfmt(descriptive['M1']['seed_progress_median'])}; >=1G={pct(metrics['seed_ge_1g_rate']['m0'])}/{pct(metrics['seed_ge_1g_rate']['m1'])}; raw=normalized-first={pct(metrics['frozen_seed_equals_normalized_first_rate']['m0'])}/{pct(metrics['frozen_seed_equals_normalized_first_rate']['m1'])}",
        f"Seed时relaxed覆盖 M0/M1={pct(descriptive['M0']['seed_relaxed_coverage_median'])}/{pct(descriptive['M1']['seed_relaxed_coverage_median'])}; 已达最终覆盖={pct(metrics['seed_relaxed_fraction_of_final']['m0'])}/{pct(metrics['seed_relaxed_fraction_of_final']['m1'])}; seed后无新relaxed={pct(metrics['no_new_relaxed_after_seed_rate']['m0'])}/{pct(metrics['no_new_relaxed_after_seed_rate']['m1'])}",
        f"Primary capture: seed#2+ M0/M1={pct(metrics['capture_later_than_first_rate']['m0'])}/{pct(metrics['capture_later_than_first_rate']['m1'])}; after-last-new={pct(metrics['capture_after_last_new_relaxed_rate']['m0'])}/{pct(metrics['capture_after_last_new_relaxed_rate']['m1'])}; ordinal中位={fmt(descriptive['M0']['capture_seed_ordinal_median'],1)}/{fmt(descriptive['M1']['capture_seed_ordinal_median'],1)}",
        f"Legacy capture: seed#2+ M0/M1={pct(metrics['legacy_capture_later_than_first_rate']['m0'])}/{pct(metrics['legacy_capture_later_than_first_rate']['m1'])}; after-last-new={pct(metrics['legacy_capture_after_last_new_relaxed_rate']['m0'])}/{pct(metrics['legacy_capture_after_last_new_relaxed_rate']['m1'])}; ordinal中位={fmt(descriptive['M0']['legacy_capture_seed_ordinal_median'],1)}/{fmt(descriptive['M1']['legacy_capture_seed_ordinal_median'],1)}",
        f"判定={decision['label']}; completion={decision['completion_component']}; capture={decision['capture_origin']}",
        f"下一步={decision['next_step']}",
    ]
    text = "\n".join(lines) + "\n"
    if len(text) > max_chars:
        # Remove lower-priority deep-overrun details while retaining the main Gate.
        lines = [line for line in lines if not line.startswith(">=1.25G M1:")]
        text = "\n".join(lines) + "\n"
    if len(text) > max_chars:
        # This diagnostic remains available in P0C_KEY_METRICS.csv.
        lines = [line for line in lines if "raw=normalized-first" not in line]
        text = "\n".join(lines) + "\n"
    if len(text) > max_chars:
        # Final compact fallback: parser-invalid remains available in the report/manifest.
        lines = [line.replace(f"; parser-invalid={descriptive['M0']['parser_invalid_objects']}/{descriptive['M1']['parser_invalid_objects']}", "") for line in lines]
        text = "\n".join(lines) + "\n"
    if len(text) > max_chars:
        raise ValueError(f"compact return exceeds {max_chars} characters: {len(text)}")
    return text


def main() -> int:
    args = parse_args()
    if args.bootstrap < 100:
        raise ValueError("bootstrap must be at least 100")
    if args.bootstrap_batch_size < 1:
        raise ValueError("bootstrap-batch-size must be positive")
    if args.max_return_chars < 1000:
        raise ValueError("max-return-chars must be at least 1000")

    result_dir = Path(args.result_dir).expanduser().resolve()
    gold_path = Path(args.gold_data).expanduser().resolve()
    if not result_dir.exists():
        raise FileNotFoundError(result_dir)
    if not gold_path.exists():
        raise FileNotFoundError(gold_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else result_dir / "p0c_set_completion"
    )
    prepare_output(output_dir, args.overwrite)

    event_index, run_manifest = load_event_meta(result_dir)
    target = str(run_manifest.get("protocol", {}).get("target", "answer"))
    inputs = run_manifest.get("inputs", {})
    responses_m0 = Path(args.responses_m0 or inputs.get("responses_m0", "")).expanduser().resolve()
    responses_m1 = Path(args.responses_m1 or inputs.get("responses_m1", "")).expanduser().resolve()
    for path in (responses_m0, responses_m1):
        if not path.exists():
            raise FileNotFoundError(
                f"response path from run_manifest does not exist: {path}; override RESPONSES_M0/RESPONSES_M1"
            )

    # 01b freezes whole-file hashes.  Per-response SHA/block/hash checks are
    # still performed below, but this gives an early, explicit failure when a
    # similarly named response file is supplied by mistake.
    response_hashes = {
        "responses_m0": sha256_file(responses_m0),
        "responses_m1": sha256_file(responses_m1),
    }
    frozen_hashes = run_manifest.get("input_file_sha256", {})
    for key, actual in response_hashes.items():
        expected = frozen_hashes.get(key)
        if expected and actual != expected:
            raise ValueError(
                f"{key} SHA-256 mismatch: actual={actual}, frozen_by_01b={expected}"
            )

    all_event_keys = [meta.key for model in MODEL_TAGS for meta in event_index[model].values()]
    gold = load_gold_dataset(gold_path, all_event_keys, key_field=args.gold_key_field)

    p0b_manifest_path = result_dir / "p0b_competing_risk" / "P0B_MANIFEST.json"
    if p0b_manifest_path.exists():
        p0b_manifest = json.loads(p0b_manifest_path.read_text(encoding="utf-8"))
        previous_field = p0b_manifest.get("gold_key_field")
        if previous_field and previous_field != gold.key_field:
            raise ValueError(
                f"P0c gold mapping {gold.key_field!r} disagrees with P0b {previous_field!r}"
            )

    analyses, capture_cases = analyze_response_files(
        event_index=event_index,
        gold=gold,
        responses_m0=responses_m0,
        responses_m1=responses_m1,
        target=target,
    )
    prompt_ids = prompt_ids_and_validate(analyses)
    n_m0 = sum(row.model_tag == "M0" for row in analyses)
    n_m1 = sum(row.model_tag == "M1" for row in analyses)

    opportunity_counts = build_opportunity_array(analyses, prompt_ids)
    utility_counts = build_utility_array(analyses, prompt_ids)
    opportunity = bootstrap_opportunity_rates(
        opportunity_counts,
        utility_counts=utility_counts,
        n_bootstrap=args.bootstrap,
        random_seed=args.seed,
        batch_size=args.bootstrap_batch_size,
    )
    contrasts = opportunity_contrasts(opportunity)
    response_metrics = bootstrap_response_metrics(
        analyses,
        prompt_ids,
        n_bootstrap=args.bootstrap,
        random_seed=args.seed + 101,
        batch_size=args.bootstrap_batch_size,
    )
    descriptive = descriptive_summary(analyses)
    decision = decision_from_results(opportunity, contrasts, response_metrics)

    write_jsonl(output_dir / "P0C_RESPONSE_SUMMARY.jsonl", (row.to_dict() for row in analyses))
    write_csv(output_dir / "P0C_CAPTURE_CASES.csv", [row.to_dict() for row in capture_cases])
    write_csv(output_dir / "P0C_OPPORTUNITY_RATES.csv", opportunity["rows"])
    write_csv(output_dir / "P0C_CONTRASTS.csv", contrasts)
    write_csv(output_dir / "P0C_KEY_METRICS.csv", key_metric_rows(response_metrics))
    write_csv(output_dir / "P0C_INVALID_OBJECTS.csv", invalid_object_rows(analyses))
    write_csv(output_dir / "P0C_SEED_PROGRESS.csv", seed_progress_rows(analyses))
    write_csv(output_dir / "P0C_CAPTURE_ORDINAL_DISTRIBUTION.csv", capture_distribution_rows(descriptive))

    compact = compact_text(
        n_prompts=len(prompt_ids),
        n_m0=n_m0,
        n_m1=n_m1,
        gold_field=gold.key_field,
        descriptive=descriptive,
        metrics=response_metrics,
        opportunity=opportunity,
        contrasts=contrasts,
        decision=decision,
        max_chars=args.max_return_chars,
    )

    post = opportunity["aggregate"][">=1.00G"]
    deep = opportunity["aggregate"][">=1.25G"]
    contrast_map = {str(row["contrast"]): row for row in contrasts}
    report = [
        "# P0c：参考集合完成、边际 gold-match 产出与过度生成审计",
        "",
        "## 1. 冻结定义",
        "",
        "- 输入：all-mode `event_rows.jsonl`、其 `run_manifest.json` 指向的原始 M0/M1 responses、以及 `prompt_eval_dedup.jsonl`。",
        "- 不加载模型或 SAE，不重新生成，不重新运行 01b，不移动 legacy raw onset。",
        "- 每条 response 重新使用已验证的四字段状态机 parser；response SHA、block 数、每个 triple/quad hash、continuity segment 全部必须与 01b 一致。",
        "- Strict gold match：沿用 protocol v1.0 归一化后的 `(source,target,relation)`；空 identity 不作为有效关系。",
        "- Relaxed gold match：沿用 protocol v1.0 归一化后的 `(source,target)`；空 identity 不作为有效关系。",
        "- 归一化规则：strip、lowercase、collapse whitespace。",
        "- 关系机会 j 的进度为 `j/G`；正常结束位于 `(n_blocks+1)/G`；hit-max 只作行政删失，不作为正常停止。",
        "- 主 block 分类互斥：new strict gold、new relaxed-pair-only、seen normalized-triple reuse、gold-pair-covered variant、new unmatched、invalid identity；parser-invalid object 另行报告。",
        "- `new_relaxed_gold_pair` 是重叠边际效用计数：首次覆盖 gold pair 的 strict block 也计入；同一 block 在自然事件分母中仍只出现一次。",
        "- Capture seed ordinal：按所有 raw-exact nonempty triple reuse block 的时间顺序编号；primary capture 使用其 second-copy start；legacy stable orbit 使用 alignment evidence 中的 matching quad run。",
        "",
        "## 2. 完整性",
        "",
        f"- selection mode：all；prompts={len(prompt_ids)}；responses M0/M1={n_m0}/{n_m1}。",
        f"- gold key mapping：`{gold.key_field}`；gold rows={gold.n_rows}。",
        "- M0/M1 prompt 和 seed 完全配对：PASS。",
        "- 原 response 与 01b event row 全量逐 response/逐 block 校验：PASS。",
        f"- parser-invalid complete object candidates：M0={descriptive['M0']['parser_invalid_objects']}，M1={descriptive['M1']['parser_invalid_objects']}。",
        "",
        "## 3. 输出与最终 gold 覆盖",
        "",
        f"- L/G 中位数：M0={fmt(descriptive['M0']['output_gold_ratio_median'])}，M1={fmt(descriptive['M1']['output_gold_ratio_median'])}。",
        f"- strict coverage@G：M0={pct(response_metrics['strict_coverage_at_1g']['m0'])}，M1={pct(response_metrics['strict_coverage_at_1g']['m1'])}，Δ={ci_text(response_metrics['strict_coverage_at_1g'])}；G 后增益：M0={pct(response_metrics['post_1g_strict_coverage_gain']['m0'])}，M1={pct(response_metrics['post_1g_strict_coverage_gain']['m1'])}，Δ={ci_text(response_metrics['post_1g_strict_coverage_gain'])}；最终：M0={pct(response_metrics['final_strict_coverage']['m0'])}，M1={pct(response_metrics['final_strict_coverage']['m1'])}。",
        f"- relaxed coverage@G：M0={pct(response_metrics['relaxed_coverage_at_1g']['m0'])}，M1={pct(response_metrics['relaxed_coverage_at_1g']['m1'])}，Δ={ci_text(response_metrics['relaxed_coverage_at_1g'])}；G 后增益：M0={pct(response_metrics['post_1g_relaxed_coverage_gain']['m0'])}，M1={pct(response_metrics['post_1g_relaxed_coverage_gain']['m1'])}，Δ={ci_text(response_metrics['post_1g_relaxed_coverage_gain'])}；最终：M0={pct(response_metrics['final_relaxed_coverage']['m0'])}，M1={pct(response_metrics['final_relaxed_coverage']['m1'])}。",
        "",
        "## 4. 关系机会的边际事件率（相对于教师参考集合）",
        "",
        f"- >=1.00G new strict gold triple：M0={pct(post['new_strict_gold']['m0'])}，M1={pct(post['new_strict_gold']['m1'])}，Δ={ci_text(post['new_strict_gold'])}。",
        f"- >=1.00G newly covered relaxed gold pair：M0={pct(post['new_relaxed_gold_pair']['m0'])}，M1={pct(post['new_relaxed_gold_pair']['m1'])}，Δ={ci_text(post['new_relaxed_gold_pair'])}。",
        f"- >=1.00G new gold utility block union：M0={pct(post['new_valid_any']['m0'])}，M1={pct(post['new_valid_any']['m1'])}，Δ={ci_text(post['new_valid_any'])}。",
        f"- >=1.00G seen-reuse：M0={pct(post['seen_triple_reuse']['m0'])}，M1={pct(post['seen_triple_reuse']['m1'])}，Δ={ci_text(post['seen_triple_reuse'])}。",
        f"- >=1.00G gold-pair-covered variant：M0={pct(post['gold_pair_already_covered_variant']['m0'])}，M1={pct(post['gold_pair_already_covered_variant']['m1'])}，Δ={ci_text(post['gold_pair_already_covered_variant'])}。",
        f"- >=1.00G new-unmatched：M0={pct(post['new_unmatched']['m0'])}，M1={pct(post['new_unmatched']['m1'])}，Δ={ci_text(post['new_unmatched'])}。",
        f"- >=1.00G normal-stop：M0={pct(post['normal_stop']['m0'])}，M1={pct(post['normal_stop']['m1'])}，Δ={ci_text(post['normal_stop'])}。",
        f"- >=1.25G M1：new gold-match={pct(deep['new_valid_any']['m1'])}，seen-reuse={pct(deep['seen_triple_reuse']['m1'])}，new-unmatched={pct(deep['new_unmatched']['m1'])}，normal-stop={pct(deep['normal_stop']['m1'])}。",
        f"- M1 post(>=1G)−pre(0.5–1G) new gold-match：{pct(contrast_map['M1_post_ge1_minus_pre_0.5_1.0::new_valid_any']['value'])} [{pct(contrast_map['M1_post_ge1_minus_pre_0.5_1.0::new_valid_any']['ci_low'])}, {pct(contrast_map['M1_post_ge1_minus_pre_0.5_1.0::new_valid_any']['ci_high'])}]。",
        f"- M1 post(>=1G)−pre(0.5–1G) seen-reuse：{pct(contrast_map['M1_post_ge1_minus_pre_0.5_1.0::seen_triple_reuse']['value'])} [{pct(contrast_map['M1_post_ge1_minus_pre_0.5_1.0::seen_triple_reuse']['ci_low'])}, {pct(contrast_map['M1_post_ge1_minus_pre_0.5_1.0::seen_triple_reuse']['ci_high'])}]。",
        "",
        "## 5. First seed 时的集合状态",
        "",
        f"- first-seed progress 中位数：M0={fmt(descriptive['M0']['seed_progress_median'])}G，M1={fmt(descriptive['M1']['seed_progress_median'])}G。",
        f"- first-seed strict coverage：M0={pct(response_metrics['seed_strict_coverage']['m0'])}，M1={pct(response_metrics['seed_strict_coverage']['m1'])}。",
        f"- first-seed relaxed coverage：M0={pct(response_metrics['seed_relaxed_coverage']['m0'])}，M1={pct(response_metrics['seed_relaxed_coverage']['m1'])}。",
        f"- first seed 时已完成最终 relaxed coverage 的比例：M0={pct(response_metrics['seed_relaxed_fraction_of_final']['m0'])}，M1={pct(response_metrics['seed_relaxed_fraction_of_final']['m1'])}。",
        f"- seed 后不再产生新 relaxed gold pair：M0={pct(response_metrics['no_new_relaxed_after_seed_rate']['m0'])}，M1={pct(response_metrics['no_new_relaxed_after_seed_rate']['m1'])}。",
        f"- first seed 位于最后一个新 relaxed pair 之后：M0={pct(response_metrics['seed_after_last_new_relaxed_rate']['m0'])}，M1={pct(response_metrics['seed_after_last_new_relaxed_rate']['m1'])}。",
        f"- first seed 位于 >=1.00G：M0={pct(response_metrics['seed_ge_1g_rate']['m0'])}，M1={pct(response_metrics['seed_ge_1g_rate']['m1'])}。",
        f"- raw-exact seed 与 normalized earliest reuse 一致率：M0={pct(response_metrics['frozen_seed_equals_normalized_first_rate']['m0'])}，M1={pct(response_metrics['frozen_seed_equals_normalized_first_rate']['m1'])}。",
        "",
        "## 6. Capture 来自第几个 seed",
        "",
        f"- primary semantic capture 来自 seed #2+：M0={pct(response_metrics['capture_later_than_first_rate']['m0'])}，M1={pct(response_metrics['capture_later_than_first_rate']['m1'])}；位于最后一个新 relaxed pair 之后：M0={pct(response_metrics['capture_after_last_new_relaxed_rate']['m0'])}，M1={pct(response_metrics['capture_after_last_new_relaxed_rate']['m1'])}。",
        f"- primary capture seed ordinal 中位数：M0={fmt(descriptive['M0']['capture_seed_ordinal_median'],1)}，M1={fmt(descriptive['M1']['capture_seed_ordinal_median'],1)}。",
        f"- legacy-aligned stable capture 来自 seed #2+：M0={pct(response_metrics['legacy_capture_later_than_first_rate']['m0'])}，M1={pct(response_metrics['legacy_capture_later_than_first_rate']['m1'])}；位于最后一个新 relaxed pair 之后：M0={pct(response_metrics['legacy_capture_after_last_new_relaxed_rate']['m0'])}，M1={pct(response_metrics['legacy_capture_after_last_new_relaxed_rate']['m1'])}。",
        f"- legacy-aligned seed ordinal 中位数：M0={fmt(descriptive['M0']['legacy_capture_seed_ordinal_median'],1)}，M1={fmt(descriptive['M1']['legacy_capture_seed_ordinal_median'],1)}。",
        "",
        "## 7. 判定",
        "",
        f"- `{decision['label']}`",
        f"- completion component：`{decision['completion_component']}`",
        f"- capture origin：`{decision['capture_origin']}`",
        f"- 下一步：{decision['next_step']}。",
        "",
        "## 8. 解释边界",
        "",
        "- strict/relaxed 都是教师参考集合上的蒸馏一致性匹配；relaxed 不是语义 embedding/fuzzy match，new-unmatched 也不能自动判成语义错误。空 identity 从有效 gold utility 中排除，但仍计入原始 G 并单独审计。",
        "- 各进度区间的 event rate 条件于 response 已到达该区间；P0b 的 competing-risk 结果仍负责解释停止/暴露差异，P0c 不把这些条件率单独解释成因果 hazard。",
        "- G 是参考输出字典数，不被事后调整；P0c 用边际效用检验它是否近似 completion 位置。",
        "- seed ordinal 是 raw-exact nonempty triple reuse event 的顺序；它不把同一循环中的所有 block 合并成主观 episode。",
        "- 本步骤仍是行为分解，不等价于 SAE 中介或训练数据归因。",
    ]
    (output_dir / "P0C_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output_dir / "P0C_COMPACT_RETURN.txt").write_text(compact, encoding="utf-8")

    gold_empty = sum(record.n_empty_identity for record in gold.records.values())
    manifest = {
        "protocol": "01b-p0c-v0.1",
        "result_dir": str(result_dir),
        "selection_mode": "all",
        "gold_data": str(gold_path),
        "gold_key_field": gold.key_field,
        "responses_m0": str(responses_m0),
        "responses_m1": str(responses_m1),
        "target": target,
        "bootstrap": args.bootstrap,
        "bootstrap_unit": "prompt (all paired seeds retained)",
        "random_seed": args.seed,
        "definitions": {
            "strict_match": "protocol-v1.0 normalized (source,target,relation), excluding empty identity from valid utility",
            "relaxed_match": "protocol-v1.0 normalized (source,target), excluding empty identity from valid utility",
            "normalization": "strip + lowercase + collapse whitespace",
            "new_relaxed_gold_pair": "overlapping utility count; every first-covered gold pair, including strict hits",
            "seen_triple_reuse": "evaluation-normalized nonempty triple seen earlier in the same response",
            "progress": "next relation opportunity j / gold n_output_dicts",
            "normal_stop": "non-hit-max terminal event at (n_blocks+1)/G",
            "admin_censor": "hit-max terminal event; excluded from natural-event denominator",
            "seed": "frozen 01b first_nonempty_triple_reuse (raw exact)",
            "capture_seed_ordinal": "ordinal among all raw-exact nonempty triple reuse blocks",
            "legacy_raw_onset_moved": False,
        },
        "counts": {
            "prompts": len(prompt_ids),
            "responses_m0": n_m0,
            "responses_m1": n_m1,
            "capture_case_rows": len(capture_cases),
            "gold_rows": gold.n_rows,
            "gold_empty_normalized_identities": gold_empty,
        },
        "decision": decision,
        "input_sha256": {
            "event_rows": sha256_file(result_dir / "event_rows.jsonl"),
            "run_manifest": sha256_file(result_dir / "run_manifest.json"),
            "selection_manifest": sha256_file(result_dir / "selection_manifest.json"),
            "gold_data": sha256_file(gold_path),
            "responses_m0": response_hashes["responses_m0"],
            "responses_m1": response_hashes["responses_m1"],
        },
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    write_json(output_dir / "P0C_MANIFEST.json", manifest)

    print("=" * 72)
    print("P0c set-completion audit completed")
    print(f"Output: {output_dir}")
    print(f"Decision: {decision['label']}")
    print("Only return: P0C_COMPACT_RETURN.txt")
    print("=" * 72)
    print(compact, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
