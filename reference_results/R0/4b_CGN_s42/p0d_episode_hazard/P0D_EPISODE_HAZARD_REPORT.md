# P0d：Reuse-Episode 级 Capture Hazard 分解（暴露 vs 倾向）

## 1. 冻结定义

- 输入：all-mode `event_rows.jsonl`（01b 冻结产物）。不加载模型/SAE，不重新生成，不移动 legacy raw onset，不改变任何 01b 事件定义。
- reuse block：identity-complete 且 raw-exact triple hash 在此前 identity-complete block 中出现过（与冻结 seed 定义一致）；novel block：identity-complete 首次出现；empty block：identity 不完整。
- episode（lineage 主变体）：连续 reuse block 的最大运行段，按 lineage 进一步切分——相邻 reuse block 若同 segment 且（lag 相同 / triple hash 已在本 episode / prev 指向本 episode 的块或其来源）则延续，否则记 motif 切换并开新 episode。novel/empty/segment break 一律关闭 episode。
- episode（contiguous 敏感性变体）：同上但不做 lineage 切分。
- capture 口径分别分析：primary=`motif_capture_triple` 的 second-copy start（过 `is_semantic_capture` 校验）；legacy=structured 对齐 `matching_quad_run` 的 onset+period。
- 交错两阶段竞争轴：每个周期 k 拆成 gap 半格 G_k（恢复后的 novel 阶段；normal stop=停止事件，hit-max=删失，存活=第 k 个 episode 开始）与 episode 半格 E_k（episode 已存在；capture=捕获事件，response 在 episode 内正常结束=停止事件，hit-max=删失，存活=恢复）。capture 后的 episode 属 orbit 状态，一律截断不计。
- 因此 per-episode capture hazard h(k)=captures_k/episodes_started_k，**条件于 episode 存在**。若不做该拆分，仅停止更少的模型会以 gap 存活因子机械抬高逐 slot capture hazard，伪造 propensity 效应——这正是 P0d 必须排除的混淆（已用合成真值数据验证：同 propensity、不同 stop 的场景在单 slot 模型下误判为 B，在两阶段模型下正确判 A）。
- legacy 口径下 stable orbit 存在但无 structured 对齐（无法定位 block 坐标）的 response 记为删失并单独计数，不记为 stop。
- Shapley 分解：与 P0b 完全相同的 `competing_decomposition` 机制作用在交错轴上——交换 capture-cause hazard（episode 半格）得 per-episode propensity 分量；交换 stop-cause hazard（gap+episode 内停止）得 stop/exposure 分量。
- 统计单位：prompt 级配对 bootstrap（同 prompt 全部 seed 保留，multinomial 权重），与 P0b/P0c 一致。
- 预注册 Gate（只按分支走，不改公式重跑）：
  - 分支 B（propensity 也升高）：propensity 分量 CI 下界 > 0；
  - 分支 A（暴露主导）：非 B，且 exposure 分量 CI 下界 > 0 且占总 ΔCIF 份额 > 50%；
  - 其余 UNRESOLVED；contiguous 变体分支不一致时降级为 SENSITIVITY_DIVERGENT；
  - orbit-stage flag（独立汇报，不参与 A/B 判定）：ΔP(stable orbit | captured) CI 下界 > 0。

## 2. 完整性

- prompts=1106；responses M0/M1=8848/8848；M0/M1 prompt 与 seed 完全配对：PASS。
- 每条 response 重算 reuse 序列并与冻结 `first_nonempty_triple_reuse` 对账：PASS（不一致会硬失败）。
- primary capture 三份 motif copy 的 triple hash 校验、legacy matching quad run 第二份 copy 的 quad hash 校验：PASS。
- episode 级 capture 指示与 row 级 capture flag 恒等：PASS。
- P0c response summary 交叉核对（n_nonempty_reuse_events、semantic_capture 全样本一致）：SKIP(file_absent)。
- response 内新 pair 距离来源：reparse（reparse 时逐 response 校验 SHA/块数/triple hash/identity flag）。

## 3. Episode 暴露（lineage 主变体，primary 口径）

- 每 response 经历 episodes（capture/stop 前）：M0 中位 2.0（p25–p75 0.0–5.0），M1 中位 3.0（1.0–7.0）；mean Δ=1.40 [1.26, 1.54]。
- 0-episode（从无非空复用）：M0=27.5%，M1=15.4%，Δ=-12.1%[-13.2%,-11.1%]。
- 终局分布：capture M0/M1=5.5%/5.7%；normal-stop=94.4%/94.2%；hit-max 删失=0.1%/0.1%。
- captured 子集：capture episode ordinal 中位 M0=4.0，M1=4.0；capture 前已恢复 episodes mean M0=4.87，M1=4.25，Δ=-0.62 [-1.21, -0.05]；第 1 个 episode 即被 capture 的占比 M0=16.2%，M1=13.8%，Δ=-2.4%[-6.7%,2.0%]。
- episode 长度（blocks）中位：M0=1.0，M1=1.0；被 capture 的 episode 长度中位：M0=21.0，M1=20.0。
- capture 后仍出现 novel block 的 captured responses（capture 逃逸迹象）：M0=15，M1=11。

## 4. Per-episode capture hazard h(k)=P(capture|第k个episode存在)（lineage，primary）

| k | h_M0(k) | h_M1(k) | Δ [95% CI] | 方向 |
| --- | --- | --- | --- | --- |
| 1 | 1.2% | 0.9% | -0.3%[-0.6%,0.0%] | UNRESOLVED |
| 2 | 1.1% | 1.3% | 0.2%[-0.2%,0.7%] | UNRESOLVED |
| 3 | 1.7% | 1.5% | -0.2%[-0.7%,0.3%] | UNRESOLVED |
| 4 | 1.5% | 1.4% | -0.1%[-0.6%,0.5%] | UNRESOLVED |
| 5 | 1.6% | 1.2% | -0.4%[-1.1%,0.2%] | UNRESOLVED |
| 6 | 1.6% | 0.9% | -0.7%[-1.3%,-0.2%] | DOWN |
| 7 | 2.0% | 0.9% | -1.0%[-1.8%,-0.3%] | DOWN |
| 8 | 1.5% | 1.5% | -0.0%[-0.9%,0.8%] | UNRESOLVED |
| 9+ (pooled) | 1.7% | 0.8% | -0.8%[-1.2%,-0.5%] | DOWN |
| all (pooled) | 1.5% | 1.1% | -0.4%[-0.5%,-0.2%] | DOWN |

- gap stop hazard（每个恢复间隙的正常停止概率）pooled：M0=15.2%，M1=10.1%，Δ=-5.1%[-5.7%,-4.5%]。
- in-episode stop hazard（episode 内正常结束）pooled：M0=7.7%，M1=7.3%，Δ=-0.4%[-0.7%,-0.1%]。

## 5. Episode-ordinal CIF 与 Shapley 分解（主判定）

- capture CIF@K48：M0=5.5%，M1=5.7%，Δ=0.2%[-0.4%,0.9%]。
- stop CIF@K48：M0=94.5%，M1=94.3%，Δ=-0.2%[-0.9%,0.4%]。
- Shapley（尾部池化边界 K_pool=24，双模型 episode at-risk ≥50 的最大 ordinal；边界外每模型每 cause 用其自身尾部池化 hazard，避免空风险集填 0 伪造 propensity）：per-episode propensity 分量=-1.5%[-2.2%,-0.9%]（share=-695.4%）；stop/exposure 分量=1.8%[1.5%,2.0%]（share=795.4%）。
- 分解模型总差=0.2%[-0.4%,0.9%]；原始曲线 ΔCIF=0.2%[-0.4%,0.9%]（两者之差为尾部平滑差异，应当很小）；identity error=0.000000。

## 6. Legacy 口径一致性（lineage，legacy-aligned capture）

- capture 率 M0/M1=0.6%/0.8%；pooled per-episode hazard Δ=0.0%[-0.0%,0.1%]；propensity 分量=0.0%[-0.2%,0.3%]；exposure 分量=0.2%[0.2%,0.3%]。
- stable orbit 存在但结构未对齐（记删失）：M0=1，M1=0。

## 7. Orbit-stage flag（独立于 A/B）

- P(stable orbit | primary captured)：M0=10.7%，M1=14.8%，Δ=4.1% [0.2%, 8.0%]；flag=ON。
- P(stable orbit | uncaptured)：M0=0.0%，M1=0.0%。

## 8. 距离统计（captured responses，lineage，primary；描述性，无 CI）

- last response 内新 triple → capture episode 起点：M0 中位 1.0 blocks（p25–p75 1.0–5.0），M1 中位 1.0（1.0–4.0）。
- capture episode 是 last 新 triple 后第一个 episode 的占比：M0=51.4%，M1=57.7%；之间隔的 episodes 中位：M0=0.0，M1=0.0。
- last response 内新 pair → capture episode 起点中位：M0=3.0，M1=3.0（来源=reparse）。

## 9. 敏感性（contiguous 变体，primary 口径）

- pooled per-episode hazard Δ=-0.6%[-0.9%,-0.4%]；propensity 分量=-1.6%[-2.3%,-0.9%]；exposure 分量=1.8%[1.6%,2.0%]。
- 分支：lineage=A_EXPOSURE_DOMINATED，contiguous=A_EXPOSURE_DOMINATED。

## 10. 判定

- `P0D_A_EXPOSURE_DOMINATED`
- 下一步：冻结单机制主线：终止失准→暴露→偶发capture；进入 R1/S1 终止决策行为量验证；T-DRS 只做终止成分，F_core/07b 写入 capture→orbit 支撑证据；orbit-stage flag 触发：P2b 瓶颈修复机制保留为 capture→orbit 阶段证据。

## 11. 解释边界

- episode 合并是确定性状态机（附单测），但『lineage』边界仍是一种建模选择；因此 contiguous 变体作为预注册敏感性对照一并汇报，分支不一致即降级，不挑好看的报。
- capture 作为吸收态截断是建模约定；capture 后的 novel block 数量已单独汇报，供检查『capture 逃逸』是否被该约定掩盖。
- hit-max 删失只移出风险集，不构成事件；M1 删失远多于 M0，CIF 与 hazard 均为删失一致口径下的估计。
- 本步骤仍是行为级分解：它裁定『暴露 vs per-episode 倾向』，不裁定任何 SAE/回路中介，也不是训练数据归因。
