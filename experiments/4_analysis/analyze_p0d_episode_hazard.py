#!/usr/bin/env python3
"""P0d reuse-episode capture-hazard decomposition (exposure vs propensity)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tcr.events.io_utils import iter_jsonl, sha256_file, write_csv, write_json, write_jsonl
from tcr.events.p0d_episode_hazard import (
    CAPTURE_KINDS,
    EPISODE_VARIANTS,
    HAZARD_ORDINALS,
    analyze_sequences,
    bootstrap_episode_analysis,
    build_slot_arrays,
    decision_from_results,
    descriptive_episode_summary,
    distance_rows,
    episode_count_rows,
    episode_curve_rows,
    extract_block_sequence,
    load_event_rows,
    reparse_pair_novelty,
    response_metric_arrays,
    validate_capture_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-batch-size", type=int, default=200)
    parser.add_argument("--tail-pool-min", type=int, default=50)
    parser.add_argument("--max-return-chars", type=int, default=1950)
    parser.add_argument("--reparse", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--responses-m0", default=None)
    parser.add_argument("--responses-m1", default=None)
    parser.add_argument("--p0c-summary", default=None)
    parser.add_argument("--allow-sample", action="store_true")
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


def ci_text(result: Mapping[str, float]) -> str:
    return f"{pct(result['diff'])}[{pct(result['ci_low'])},{pct(result['ci_high'])}]"


def sci_text(result: Mapping[str, float]) -> str:
    return f"{pct(result['value'])}[{pct(result['ci_low'])},{pct(result['ci_high'])}]"


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


def cross_check_p0c(sequences, outcomes_primary, p0c_summary_path: Path) -> dict[str, Any]:
    """Validate P0d recomputation against the frozen P0c response summary."""
    reuse_by_sample = {o.sample_id: o.n_reuse_blocks for o in outcomes_primary}
    capture_by_sample = {o.sample_id: o.terminal_event == "capture" for o in outcomes_primary}
    checked = 0
    for row in iter_jsonl(p0c_summary_path):
        sample_id = str(row.get("sample_id"))
        if sample_id not in reuse_by_sample:
            raise ValueError(f"P0c summary contains unknown sample: {sample_id}")
        if int(row.get("n_nonempty_reuse_events", -1)) != reuse_by_sample[sample_id]:
            raise ValueError(
                f"P0c reuse-event count disagrees with P0d recomputation: {sample_id}"
            )
        if bool(row.get("semantic_capture")) != capture_by_sample[sample_id]:
            raise ValueError(f"P0c semantic-capture flag disagrees with P0d: {sample_id}")
        checked += 1
    if checked != len(reuse_by_sample):
        raise ValueError(
            f"P0c summary covers {checked} samples but P0d has {len(reuse_by_sample)}"
        )
    return {"status": "PASS", "n_checked": checked}


def key_metric_rows(results_by_combo: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (variant, kind), results in results_by_combo.items():
        for scope, metrics in (("slot", results["slot"]), ("response", results["response"])):
            for name, value in metrics.items():
                if "value" in value:  # decomposition scalar
                    rows.append(
                        {
                            "episode_variant": variant,
                            "capture_kind": kind,
                            "scope": scope,
                            "metric": name,
                            "m0": math.nan,
                            "m1": math.nan,
                            "diff_or_value": value["value"],
                            "ci_low": value["ci_low"],
                            "ci_high": value["ci_high"],
                            "n_m0": math.nan,
                            "n_m1": math.nan,
                            "direction": direction(value["ci_low"], value["ci_high"]),
                        }
                    )
                else:
                    rows.append(
                        {
                            "episode_variant": variant,
                            "capture_kind": kind,
                            "scope": scope,
                            "metric": name,
                            "m0": value["m0"],
                            "m1": value["m1"],
                            "diff_or_value": value["diff"],
                            "ci_low": value["ci_low"],
                            "ci_high": value["ci_high"],
                            "n_m0": value.get("n_m0", math.nan),
                            "n_m1": value.get("n_m1", math.nan),
                            "direction": direction(value["ci_low"], value["ci_high"]),
                        }
                    )
    return rows


def shapley_rows(results_by_combo: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (variant, kind), results in results_by_combo.items():
        slot = results["slot"]
        total = slot["decomposition_total"]["value"]
        for component in (
            "propensity_component",
            "exposure_component",
            "decomposition_total",
            "raw_cif_diff",
            "identity_error",
        ):
            value = slot[component]
            rows.append(
                {
                    "episode_variant": variant,
                    "capture_kind": kind,
                    "component": component,
                    "value": value["value"],
                    "ci_low": value["ci_low"],
                    "ci_high": value["ci_high"],
                    "share_of_total": value["value"] / total if total else math.nan,
                    "direction": direction(value["ci_low"], value["ci_high"]),
                }
            )
    return rows


def hazard_line(results: Mapping[str, Any], model_key: str) -> str:
    slot = results["slot"]
    parts = []
    for ordinal in (1, 2, 3, 4, 5):
        name = f"hazard_k{ordinal}"
        if name in slot:
            parts.append(pct(slot[name][model_key]))
    tail = slot.get("hazard_k9plus_pooled")
    pooled = slot["pooled_capture_hazard"]
    text = "/".join(parts)
    if tail is not None:
        text += f"; k9+={pct(tail[model_key])}"
    text += f"; pooled={pct(pooled[model_key])}"
    return text


def compact_text(
    *,
    n_prompts: int,
    n_m0: int,
    n_m1: int,
    primary: Mapping[str, Any],
    legacy: Mapping[str, Any],
    descriptive: Mapping[str, Any],
    dist_rows: list[dict[str, Any]],
    decision: Mapping[str, Any],
    pair_source: str,
    p0c_check: str,
    max_chars: int,
) -> str:
    slot = primary["slot"]
    resp = primary["response"]
    total = slot["decomposition_total"]["value"]
    prop = slot["propensity_component"]
    expo = slot["exposure_component"]
    kmax = primary["kmax"]
    dist = {row["model"]: row for row in dist_rows}
    d0, d1 = dist.get("M0", {}), dist.get("M1", {})
    orbit = decision["orbit_given_captured"]
    lines = [
        "[01b-P0d COMPACT]",
        (
            f"完整性=PASS; prompts={n_prompts}; rows M0/M1={n_m0}/{n_m1}; "
            f"capture一致性=PASS; 首seed重算=PASS; P0c交叉核对={p0c_check}; pair距离源={pair_source}"
        ),
        (
            f"Episodes/resp 中位 M0/M1={fmt(descriptive['M0']['episodes_experienced']['median'],1)}/"
            f"{fmt(descriptive['M1']['episodes_experienced']['median'],1)}; "
            f"mean={fmt(resp['mean_episodes_experienced']['m0'],2)}/{fmt(resp['mean_episodes_experienced']['m1'],2)}, "
            f"Δ{fmt(resp['mean_episodes_experienced']['diff'],2)}[{fmt(resp['mean_episodes_experienced']['ci_low'],2)},{fmt(resp['mean_episodes_experienced']['ci_high'],2)}]; "
            f"0-episode={pct(resp['zero_episode_rate']['m0'])}/{pct(resp['zero_episode_rate']['m1'])}"
        ),
        (
            f"Primary capture率 M0/M1={pct(resp['capture_rate']['m0'])}/{pct(resp['capture_rate']['m1'])}; "
            f"capture episode ordinal中位={fmt(descriptive['M0']['captured_episode_ordinal']['median'],1)}/"
            f"{fmt(descriptive['M1']['captured_episode_ordinal']['median'],1)}; "
            f"capture在第1个episode={pct(resp['capture_in_first_episode_rate']['m0'])}/{pct(resp['capture_in_first_episode_rate']['m1'])}"
        ),
        f"h(k)捕获 k1-5 M0: {hazard_line(primary, 'm0')}",
        f"h(k)捕获 k1-5 M1: {hazard_line(primary, 'm1')}",
        (
            f"pooled h(capture|episode) M0/M1={pct(slot['pooled_capture_hazard']['m0'])}/{pct(slot['pooled_capture_hazard']['m1'])}, "
            f"Δ{ci_text(slot['pooled_capture_hazard'])}; gap-stop hazard M0/M1={pct(slot['pooled_gap_stop_hazard']['m0'])}/"
            f"{pct(slot['pooled_gap_stop_hazard']['m1'])}, Δ{ci_text(slot['pooled_gap_stop_hazard'])}"
        ),
        (
            f"CIF@K{kmax}(capture) M0/M1={pct(slot['cif_capture_final']['m0'])}/{pct(slot['cif_capture_final']['m1'])}, "
            f"Δ{ci_text(slot['cif_capture_final'])}"
        ),
        (
            f"Shapley(ΔCIF,尾池化K>{primary['k_pool']}): propensity={sci_text(prop)} share={pct(prop['value']/total if total else math.nan)}; "
            f"exposure={sci_text(expo)} share={pct(expo['value']/total if total else math.nan)}"
        ),
        (
            f"Legacy口径: propensity={sci_text(legacy['slot']['propensity_component'])}; "
            f"pooled hazard Δ{ci_text(legacy['slot']['pooled_capture_hazard'])}"
        ),
        (
            f"Orbit flag={'ON' if decision['orbit_stage_flag'] else 'OFF'}: P(orbit|capture) M0/M1={pct(orbit['m0'])}/{pct(orbit['m1'])}, "
            f"Δ{pct(orbit['diff'])}[{pct(orbit['ci_low'])},{pct(orbit['ci_high'])}]"
        ),
        (
            f"距离: last新triple→capture中位blocks M0/M1={fmt(d0.get('dist_last_new_triple_blocks_median'),1)}/"
            f"{fmt(d1.get('dist_last_new_triple_blocks_median'),1)}; "
            f"capture=last新后首个episode占比={pct(d0.get('capture_first_episode_after_last_new_share'))}/"
            f"{pct(d1.get('capture_first_episode_after_last_new_share'))}; "
            f"last新pair→capture中位={fmt(d0.get('dist_last_new_pair_blocks_median'),1)}/{fmt(d1.get('dist_last_new_pair_blocks_median'),1)}"
        ),
        f"敏感性: lineage分支={decision['primary_branch']}; contiguous分支={decision['sensitivity_branch']}",
        f"判定={decision['label']}",
        f"下一步={decision['next_step']}",
    ]
    text = "\n".join(lines) + "\n"
    if len(text) > max_chars:
        lines = [line for line in lines if not line.startswith("Legacy口径")]
        text = "\n".join(lines) + "\n"
    if len(text) > max_chars:
        lines = [line for line in lines if not line.startswith("h(k)捕获")]
        text = "\n".join(lines) + "\n"
    if len(text) > max_chars:
        lines = [line for line in lines if not line.startswith("距离")]
        text = "\n".join(lines) + "\n"
    if len(text) > max_chars:
        raise ValueError(f"compact return exceeds {max_chars} characters: {len(text)}")
    return text


def build_report(
    *,
    n_prompts: int,
    n_m0: int,
    n_m1: int,
    results_by_combo: Mapping[tuple[str, str], Mapping[str, Any]],
    descriptive_by_combo: Mapping[tuple[str, str], Mapping[str, Any]],
    dist_by_combo: Mapping[tuple[str, str], list[dict[str, Any]]],
    decision: Mapping[str, Any],
    pair_source: str,
    p0c_check: str,
) -> str:
    primary = results_by_combo[("lineage", "primary")]
    sens = results_by_combo[("contiguous", "primary")]
    legacy = results_by_combo[("lineage", "legacy")]
    desc = descriptive_by_combo[("lineage", "primary")]
    slot = primary["slot"]
    resp = primary["response"]
    total = slot["decomposition_total"]["value"]
    dist = {row["model"]: row for row in dist_by_combo[("lineage", "primary")]}
    orbit = decision["orbit_given_captured"]

    def hazard_table(results: Mapping[str, Any]) -> list[str]:
        rows = ["| k | h_M0(k) | h_M1(k) | Δ [95% CI] | 方向 |", "| --- | --- | --- | --- | --- |"]
        for ordinal in HAZARD_ORDINALS:
            name = f"hazard_k{ordinal}"
            if name not in results["slot"]:
                continue
            value = results["slot"][name]
            rows.append(
                f"| {ordinal} | {pct(value['m0'])} | {pct(value['m1'])} | {ci_text(value)} | "
                f"{direction(value['ci_low'], value['ci_high'])} |"
            )
        tail = results["slot"].get("hazard_k9plus_pooled")
        if tail is not None:
            rows.append(
                f"| 9+ (pooled) | {pct(tail['m0'])} | {pct(tail['m1'])} | {ci_text(tail)} | "
                f"{direction(tail['ci_low'], tail['ci_high'])} |"
            )
        pooled = results["slot"]["pooled_capture_hazard"]
        rows.append(
            f"| all (pooled) | {pct(pooled['m0'])} | {pct(pooled['m1'])} | {ci_text(pooled)} | "
            f"{direction(pooled['ci_low'], pooled['ci_high'])} |"
        )
        return rows

    lines: list[str] = [
        "# P0d：Reuse-Episode 级 Capture Hazard 分解（暴露 vs 倾向）",
        "",
        "## 1. 冻结定义",
        "",
        "- 输入：all-mode `event_rows.jsonl`（01b 冻结产物）。不加载模型/SAE，不重新生成，不移动 legacy raw onset，不改变任何 01b 事件定义。",
        "- reuse block：identity-complete 且 raw-exact triple hash 在此前 identity-complete block 中出现过（与冻结 seed 定义一致）；novel block：identity-complete 首次出现；empty block：identity 不完整。",
        "- episode（lineage 主变体）：连续 reuse block 的最大运行段，按 lineage 进一步切分——相邻 reuse block 若同 segment 且（lag 相同 / triple hash 已在本 episode / prev 指向本 episode 的块或其来源）则延续，否则记 motif 切换并开新 episode。novel/empty/segment break 一律关闭 episode。",
        "- episode（contiguous 敏感性变体）：同上但不做 lineage 切分。",
        "- capture 口径分别分析：primary=`motif_capture_triple` 的 second-copy start（过 `is_semantic_capture` 校验）；legacy=structured 对齐 `matching_quad_run` 的 onset+period。",
        "- 交错两阶段竞争轴：每个周期 k 拆成 gap 半格 G_k（恢复后的 novel 阶段；normal stop=停止事件，hit-max=删失，存活=第 k 个 episode 开始）与 episode 半格 E_k（episode 已存在；capture=捕获事件，response 在 episode 内正常结束=停止事件，hit-max=删失，存活=恢复）。capture 后的 episode 属 orbit 状态，一律截断不计。",
        "- 因此 per-episode capture hazard h(k)=captures_k/episodes_started_k，**条件于 episode 存在**。若不做该拆分，仅停止更少的模型会以 gap 存活因子机械抬高逐 slot capture hazard，伪造 propensity 效应——这正是 P0d 必须排除的混淆（已用合成真值数据验证：同 propensity、不同 stop 的场景在单 slot 模型下误判为 B，在两阶段模型下正确判 A）。",
        "- legacy 口径下 stable orbit 存在但无 structured 对齐（无法定位 block 坐标）的 response 记为删失并单独计数，不记为 stop。",
        "- Shapley 分解：与 P0b 完全相同的 `competing_decomposition` 机制作用在交错轴上——交换 capture-cause hazard（episode 半格）得 per-episode propensity 分量；交换 stop-cause hazard（gap+episode 内停止）得 stop/exposure 分量。",
        "- 统计单位：prompt 级配对 bootstrap（同 prompt 全部 seed 保留，multinomial 权重），与 P0b/P0c 一致。",
        "- 预注册 Gate（只按分支走，不改公式重跑）：",
        "  - 分支 B（propensity 也升高）：propensity 分量 CI 下界 > 0；",
        "  - 分支 A（暴露主导）：非 B，且 exposure 分量 CI 下界 > 0 且占总 ΔCIF 份额 > 50%；",
        "  - 其余 UNRESOLVED；contiguous 变体分支不一致时降级为 SENSITIVITY_DIVERGENT；",
        "  - orbit-stage flag（独立汇报，不参与 A/B 判定）：ΔP(stable orbit | captured) CI 下界 > 0。",
        "",
        "## 2. 完整性",
        "",
        f"- prompts={n_prompts}；responses M0/M1={n_m0}/{n_m1}；M0/M1 prompt 与 seed 完全配对：PASS。",
        "- 每条 response 重算 reuse 序列并与冻结 `first_nonempty_triple_reuse` 对账：PASS（不一致会硬失败）。",
        "- primary capture 三份 motif copy 的 triple hash 校验、legacy matching quad run 第二份 copy 的 quad hash 校验：PASS。",
        "- episode 级 capture 指示与 row 级 capture flag 恒等：PASS。",
        f"- P0c response summary 交叉核对（n_nonempty_reuse_events、semantic_capture 全样本一致）：{p0c_check}。",
        f"- response 内新 pair 距离来源：{pair_source}（reparse 时逐 response 校验 SHA/块数/triple hash/identity flag）。",
        "",
        "## 3. Episode 暴露（lineage 主变体，primary 口径）",
        "",
        f"- 每 response 经历 episodes（capture/stop 前）：M0 中位 {fmt(desc['M0']['episodes_experienced']['median'],1)}（p25–p75 {fmt(desc['M0']['episodes_experienced']['p25'],1)}–{fmt(desc['M0']['episodes_experienced']['p75'],1)}），M1 中位 {fmt(desc['M1']['episodes_experienced']['median'],1)}（{fmt(desc['M1']['episodes_experienced']['p25'],1)}–{fmt(desc['M1']['episodes_experienced']['p75'],1)}）；mean Δ={fmt(resp['mean_episodes_experienced']['diff'],2)} [{fmt(resp['mean_episodes_experienced']['ci_low'],2)}, {fmt(resp['mean_episodes_experienced']['ci_high'],2)}]。",
        f"- 0-episode（从无非空复用）：M0={pct(resp['zero_episode_rate']['m0'])}，M1={pct(resp['zero_episode_rate']['m1'])}，Δ={ci_text(resp['zero_episode_rate'])}。",
        f"- 终局分布：capture M0/M1={pct(resp['capture_rate']['m0'])}/{pct(resp['capture_rate']['m1'])}；normal-stop={pct(resp['stop_rate']['m0'])}/{pct(resp['stop_rate']['m1'])}；hit-max 删失={pct(resp['censor_rate']['m0'])}/{pct(resp['censor_rate']['m1'])}。",
        f"- captured 子集：capture episode ordinal 中位 M0={fmt(desc['M0']['captured_episode_ordinal']['median'],1)}，M1={fmt(desc['M1']['captured_episode_ordinal']['median'],1)}；capture 前已恢复 episodes mean M0={fmt(resp['recovered_before_capture_mean']['m0'],2)}，M1={fmt(resp['recovered_before_capture_mean']['m1'],2)}，Δ={fmt(resp['recovered_before_capture_mean']['diff'],2)} [{fmt(resp['recovered_before_capture_mean']['ci_low'],2)}, {fmt(resp['recovered_before_capture_mean']['ci_high'],2)}]；第 1 个 episode 即被 capture 的占比 M0={pct(resp['capture_in_first_episode_rate']['m0'])}，M1={pct(resp['capture_in_first_episode_rate']['m1'])}，Δ={ci_text(resp['capture_in_first_episode_rate'])}。",
        f"- episode 长度（blocks）中位：M0={fmt(desc['M0']['episode_block_length']['median'],1)}，M1={fmt(desc['M1']['episode_block_length']['median'],1)}；被 capture 的 episode 长度中位：M0={fmt(desc['M0']['captured_episode_n_blocks']['median'],1)}，M1={fmt(desc['M1']['captured_episode_n_blocks']['median'],1)}。",
        f"- capture 后仍出现 novel block 的 captured responses（capture 逃逸迹象）：M0={desc['M0']['post_capture_novel_responses']}，M1={desc['M1']['post_capture_novel_responses']}。",
        "",
        "## 4. Per-episode capture hazard h(k)=P(capture|第k个episode存在)（lineage，primary）",
        "",
        *hazard_table(primary),
        "",
        f"- gap stop hazard（每个恢复间隙的正常停止概率）pooled：M0={pct(slot['pooled_gap_stop_hazard']['m0'])}，M1={pct(slot['pooled_gap_stop_hazard']['m1'])}，Δ={ci_text(slot['pooled_gap_stop_hazard'])}。",
        f"- in-episode stop hazard（episode 内正常结束）pooled：M0={pct(slot['pooled_inepisode_stop_hazard']['m0'])}，M1={pct(slot['pooled_inepisode_stop_hazard']['m1'])}，Δ={ci_text(slot['pooled_inepisode_stop_hazard'])}。",
        "",
        "## 5. Episode-ordinal CIF 与 Shapley 分解（主判定）",
        "",
        f"- capture CIF@K{primary['kmax']}：M0={pct(slot['cif_capture_final']['m0'])}，M1={pct(slot['cif_capture_final']['m1'])}，Δ={ci_text(slot['cif_capture_final'])}。",
        f"- stop CIF@K{primary['kmax']}：M0={pct(slot['cif_stop_final']['m0'])}，M1={pct(slot['cif_stop_final']['m1'])}，Δ={ci_text(slot['cif_stop_final'])}。",
        f"- Shapley（尾部池化边界 K_pool={primary['k_pool']}，双模型 episode at-risk ≥{primary['tail_pool_min']} 的最大 ordinal；边界外每模型每 cause 用其自身尾部池化 hazard，避免空风险集填 0 伪造 propensity）：per-episode propensity 分量={sci_text(slot['propensity_component'])}（share={pct(slot['propensity_component']['value']/total if total else math.nan)}）；stop/exposure 分量={sci_text(slot['exposure_component'])}（share={pct(slot['exposure_component']['value']/total if total else math.nan)}）。",
        f"- 分解模型总差={sci_text(slot['decomposition_total'])}；原始曲线 ΔCIF={sci_text(slot['raw_cif_diff'])}（两者之差为尾部平滑差异，应当很小）；identity error={fmt(slot['identity_error']['value'],6)}。",
        "",
        "## 6. Legacy 口径一致性（lineage，legacy-aligned capture）",
        "",
        f"- capture 率 M0/M1={pct(legacy['response']['capture_rate']['m0'])}/{pct(legacy['response']['capture_rate']['m1'])}；pooled per-episode hazard Δ={ci_text(legacy['slot']['pooled_capture_hazard'])}；propensity 分量={sci_text(legacy['slot']['propensity_component'])}；exposure 分量={sci_text(legacy['slot']['exposure_component'])}。",
        f"- stable orbit 存在但结构未对齐（记删失）：M0={descriptive_by_combo[('lineage','legacy')]['M0']['legacy_orbit_unmapped']}，M1={descriptive_by_combo[('lineage','legacy')]['M1']['legacy_orbit_unmapped']}。",
        "",
        "## 7. Orbit-stage flag（独立于 A/B）",
        "",
        f"- P(stable orbit | primary captured)：M0={pct(orbit['m0'])}，M1={pct(orbit['m1'])}，Δ={pct(orbit['diff'])} [{pct(orbit['ci_low'])}, {pct(orbit['ci_high'])}]；flag={'ON' if decision['orbit_stage_flag'] else 'OFF'}。",
        f"- P(stable orbit | uncaptured)：M0={pct(resp['orbit_given_uncaptured']['m0'])}，M1={pct(resp['orbit_given_uncaptured']['m1'])}。",
        "",
        "## 8. 距离统计（captured responses，lineage，primary；描述性，无 CI）",
        "",
        f"- last response 内新 triple → capture episode 起点：M0 中位 {fmt(dist.get('M0',{}).get('dist_last_new_triple_blocks_median'),1)} blocks（p25–p75 {fmt(dist.get('M0',{}).get('dist_last_new_triple_blocks_p25'),1)}–{fmt(dist.get('M0',{}).get('dist_last_new_triple_blocks_p75'),1)}），M1 中位 {fmt(dist.get('M1',{}).get('dist_last_new_triple_blocks_median'),1)}（{fmt(dist.get('M1',{}).get('dist_last_new_triple_blocks_p25'),1)}–{fmt(dist.get('M1',{}).get('dist_last_new_triple_blocks_p75'),1)}）。",
        f"- capture episode 是 last 新 triple 后第一个 episode 的占比：M0={pct(dist.get('M0',{}).get('capture_first_episode_after_last_new_share'))}，M1={pct(dist.get('M1',{}).get('capture_first_episode_after_last_new_share'))}；之间隔的 episodes 中位：M0={fmt(dist.get('M0',{}).get('episodes_between_last_new_and_capture_median'),1)}，M1={fmt(dist.get('M1',{}).get('episodes_between_last_new_and_capture_median'),1)}。",
        f"- last response 内新 pair → capture episode 起点中位：M0={fmt(dist.get('M0',{}).get('dist_last_new_pair_blocks_median'),1)}，M1={fmt(dist.get('M1',{}).get('dist_last_new_pair_blocks_median'),1)}（来源={pair_source}）。",
        "",
        "## 9. 敏感性（contiguous 变体，primary 口径）",
        "",
        f"- pooled per-episode hazard Δ={ci_text(sens['slot']['pooled_capture_hazard'])}；propensity 分量={sci_text(sens['slot']['propensity_component'])}；exposure 分量={sci_text(sens['slot']['exposure_component'])}。",
        f"- 分支：lineage={decision['primary_branch']}，contiguous={decision['sensitivity_branch']}。",
        "",
        "## 10. 判定",
        "",
        f"- `{decision['label']}`",
        f"- 下一步：{decision['next_step']}。",
        "",
        "## 11. 解释边界",
        "",
        "- episode 合并是确定性状态机（附单测），但『lineage』边界仍是一种建模选择；因此 contiguous 变体作为预注册敏感性对照一并汇报，分支不一致即降级，不挑好看的报。",
        "- capture 作为吸收态截断是建模约定；capture 后的 novel block 数量已单独汇报，供检查『capture 逃逸』是否被该约定掩盖。",
        "- hit-max 删失只移出风险集，不构成事件；M1 删失远多于 M0，CIF 与 hazard 均为删失一致口径下的估计。",
        "- 本步骤仍是行为级分解：它裁定『暴露 vs per-episode 倾向』，不裁定任何 SAE/回路中介，也不是训练数据归因。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.bootstrap < 100:
        raise ValueError("bootstrap must be at least 100")
    if args.bootstrap_batch_size < 1:
        raise ValueError("bootstrap-batch-size must be positive")
    if args.max_return_chars < 1000:
        raise ValueError("max-return-chars must be at least 1000")

    result_dir = Path(args.result_dir).expanduser().resolve()
    if not result_dir.exists():
        raise FileNotFoundError(result_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else result_dir / "p0d_episode_hazard"
    )
    prepare_output(output_dir, args.overwrite)

    rows, prompt_ids, run_manifest = load_event_rows(result_dir, allow_sample=args.allow_sample)
    sequences = [extract_block_sequence(row) for row in rows]
    n_m0 = sum(seq.model_tag == "M0" for seq in sequences)
    n_m1 = sum(seq.model_tag == "M1" for seq in sequences)

    # Optional reparse: response-internal new-pair novelty (raw-exact pairs).
    pair_flags = None
    pair_source = "absent"
    inputs = run_manifest.get("inputs", {})
    responses_m0 = Path(args.responses_m0 or inputs.get("responses_m0", "")).expanduser()
    responses_m1 = Path(args.responses_m1 or inputs.get("responses_m1", "")).expanduser()
    if args.reparse != "off":
        available = responses_m0.is_file() and responses_m1.is_file()
        if args.reparse == "on" and not available:
            raise FileNotFoundError(
                f"--reparse on but response files missing: {responses_m0} / {responses_m1}"
            )
        if available:
            target = str(run_manifest.get("protocol", {}).get("target", "answer"))
            pair_flags = reparse_pair_novelty(
                sequences,
                responses_m0=responses_m0,
                responses_m1=responses_m1,
                target=target,
            )
            pair_source = "reparse"

    results_by_combo: dict[tuple[str, str], dict[str, Any]] = {}
    descriptive_by_combo: dict[tuple[str, str], dict[str, Any]] = {}
    dist_by_combo: dict[tuple[str, str], list[dict[str, Any]]] = {}
    curve_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    all_dist_rows: list[dict[str, Any]] = []
    outcome_store: dict[tuple[str, str], list] = {}
    seed_offset = 0
    for variant in EPISODE_VARIANTS:
        for kind in CAPTURE_KINDS:
            outcomes = analyze_sequences(
                sequences, variant=variant, capture_kind=kind, pair_flags=pair_flags
            )
            validate_capture_identity(sequences, outcomes)
            outcome_store[(variant, kind)] = outcomes
            risk, capture, stop = build_slot_arrays(outcomes, prompt_ids)
            sums, counts, names = response_metric_arrays(outcomes, prompt_ids)
            results = bootstrap_episode_analysis(
                risk=risk,
                capture=capture,
                stop=stop,
                metric_sums=sums,
                metric_counts=counts,
                metric_names=names,
                n_bootstrap=args.bootstrap,
                random_seed=args.seed + seed_offset,
                batch_size=args.bootstrap_batch_size,
                tail_pool_min=args.tail_pool_min,
            )
            seed_offset += 101
            results_by_combo[(variant, kind)] = results
            descriptive_by_combo[(variant, kind)] = descriptive_episode_summary(outcomes)
            combo_dist = distance_rows(outcomes, variant=variant, capture_kind=kind)
            dist_by_combo[(variant, kind)] = combo_dist
            all_dist_rows.extend(combo_dist)
            curve_rows.extend(
                episode_curve_rows(risk, capture, stop, variant=variant, capture_kind=kind)
            )
            count_rows.extend(episode_count_rows(outcomes, variant=variant, capture_kind=kind))

    # Cross-check against the frozen P0c response summary when present.
    p0c_summary_path = (
        Path(args.p0c_summary).expanduser().resolve()
        if args.p0c_summary
        else result_dir / "p0c_set_completion" / "P0C_RESPONSE_SUMMARY.jsonl"
    )
    if p0c_summary_path.is_file():
        p0c_check = cross_check_p0c(
            sequences, outcome_store[("lineage", "primary")], p0c_summary_path
        )["status"]
    else:
        p0c_check = "SKIP(file_absent)"

    decision = decision_from_results(
        results_by_combo[("lineage", "primary")],
        results_by_combo[("contiguous", "primary")],
    )

    write_csv(output_dir / "P0D_EPISODE_SUMMARY.csv", curve_rows)
    write_csv(output_dir / "P0D_KEY_METRICS.csv", key_metric_rows(results_by_combo))
    write_csv(output_dir / "P0D_SHAPLEY.csv", shapley_rows(results_by_combo))
    write_csv(output_dir / "P0D_EPISODE_COUNT_DISTRIBUTION.csv", count_rows)
    write_csv(output_dir / "P0D_DISTANCE_TO_CAPTURE.csv", all_dist_rows)
    write_jsonl(
        output_dir / "P0D_RESPONSE_EPISODES.jsonl",
        (
            outcome.to_dict()
            for kind in CAPTURE_KINDS
            for outcome in outcome_store[("lineage", kind)]
        ),
    )
    sensitivity_rows = []
    for (variant, kind), results in results_by_combo.items():
        slot = results["slot"]
        sensitivity_rows.append(
            {
                "episode_variant": variant,
                "capture_kind": kind,
                "kmax": results["kmax"],
                "pooled_hazard_diff": slot["pooled_capture_hazard"]["diff"],
                "pooled_hazard_ci_low": slot["pooled_capture_hazard"]["ci_low"],
                "pooled_hazard_ci_high": slot["pooled_capture_hazard"]["ci_high"],
                "propensity_component": slot["propensity_component"]["value"],
                "propensity_ci_low": slot["propensity_component"]["ci_low"],
                "propensity_ci_high": slot["propensity_component"]["ci_high"],
                "exposure_component": slot["exposure_component"]["value"],
                "exposure_ci_low": slot["exposure_component"]["ci_low"],
                "exposure_ci_high": slot["exposure_component"]["ci_high"],
                "decomposition_total": slot["decomposition_total"]["value"],
                "raw_cif_diff": slot["raw_cif_diff"]["value"],
                "k_pool": results["k_pool"],
            }
        )
    write_csv(output_dir / "P0D_SENSITIVITY.csv", sensitivity_rows)

    compact = compact_text(
        n_prompts=len(prompt_ids),
        n_m0=n_m0,
        n_m1=n_m1,
        primary=results_by_combo[("lineage", "primary")],
        legacy=results_by_combo[("lineage", "legacy")],
        descriptive=descriptive_by_combo[("lineage", "primary")],
        dist_rows=dist_by_combo[("lineage", "primary")],
        decision=decision,
        pair_source=pair_source,
        p0c_check=p0c_check,
        max_chars=args.max_return_chars,
    )
    report = build_report(
        n_prompts=len(prompt_ids),
        n_m0=n_m0,
        n_m1=n_m1,
        results_by_combo=results_by_combo,
        descriptive_by_combo=descriptive_by_combo,
        dist_by_combo=dist_by_combo,
        decision=decision,
        pair_source=pair_source,
        p0c_check=p0c_check,
    )
    (output_dir / "P0D_EPISODE_HAZARD_REPORT.md").write_text(report, encoding="utf-8")
    (output_dir / "P0D_COMPACT_RETURN.txt").write_text(compact, encoding="utf-8")

    manifest = {
        "protocol": "01b-p0d-v0.1",
        "result_dir": str(result_dir),
        "selection_mode": run_manifest.get("selection", {}).get("mode"),
        "bootstrap": args.bootstrap,
        "bootstrap_unit": "prompt (all paired seeds retained)",
        "random_seed": args.seed,
        "pair_distance_source": pair_source,
        "p0c_cross_check": p0c_check,
        "definitions": {
            "reuse_block": "identity-complete raw-exact triple hash repeat (frozen seed definition)",
            "episode_lineage": (
                "consecutive reuse blocks, same segment, linked by lag equality or "
                "hash membership or prev-in-episode; split on motif switch"
            ),
            "episode_contiguous": "consecutive same-segment reuse blocks (sensitivity variant)",
            "capture_primary": "motif_capture_triple second_copy_start (is_semantic_capture)",
            "capture_legacy": "matching_quad_run block_onset+block_period (structured alignment)",
            "slot_model": (
                "interleaved half-slots per cycle k: gap G_k (stop vs censor vs episode "
                "starts) then episode E_k (capture vs in-episode stop vs censor vs "
                "recovery); capture hazard is conditional on the episode existing; "
                "capture is absorbing"
            ),
            "shapley": "P0b competing_decomposition on the interleaved gap/episode axis",
            "gate": (
                "B if propensity ci_low>0; A if exposure ci_low>0 and share>0.5 and not B; "
                "else UNRESOLVED; contiguous variant must agree or SENSITIVITY_DIVERGENT; "
                "orbit flag independent"
            ),
        },
        "counts": {
            "prompts": len(prompt_ids),
            "responses_m0": n_m0,
            "responses_m1": n_m1,
            "kmax_by_combo": {
                f"{variant}:{kind}": results["kmax"]
                for (variant, kind), results in results_by_combo.items()
            },
            "k_pool_by_combo": {
                f"{variant}:{kind}": results["k_pool"]
                for (variant, kind), results in results_by_combo.items()
            },
            "tail_pool_min": args.tail_pool_min,
        },
        "decision": decision,
        "input_sha256": {
            "event_rows": sha256_file(result_dir / "event_rows.jsonl"),
            "run_manifest": sha256_file(result_dir / "run_manifest.json"),
        },
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    write_json(output_dir / "P0D_MANIFEST.json", manifest)

    print("=" * 72)
    print("P0d reuse-episode capture-hazard decomposition completed")
    print(f"Output: {output_dir}")
    print(f"Decision: {decision['label']}")
    print("Only return: P0D_COMPACT_RETURN.txt")
    print("=" * 72)
    print(compact, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
