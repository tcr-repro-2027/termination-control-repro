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

- 每 response 经历 episodes（capture/stop 前）：M0 中位 6.0（p25–p75 2.0–12.0），M1 中位 7.0（3.0–14.0）；mean Δ=1.07 [0.70, 1.42]。
- 0-episode（从无非空复用）：M0=8.6%，M1=1.8%，Δ=-6.8%[-7.7%,-6.0%]。
- 终局分布：capture M0/M1=25.9%/60.6%；normal-stop=73.9%/38.9%；hit-max 删失=0.1%/0.5%。
- captured 子集：capture episode ordinal 中位 M0=7.0，M1=7.0；capture 前已恢复 episodes mean M0=11.42，M1=9.34，Δ=-2.08 [-2.87, -1.28]；第 1 个 episode 即被 capture 的占比 M0=10.0%，M1=7.8%，Δ=-2.1%[-3.6%,-0.7%]。
- episode 长度（blocks）中位：M0=1.0，M1=1.0；被 capture 的 episode 长度中位：M0=338.5，M1=408.0。
- capture 后仍出现 novel block 的 captured responses（capture 逃逸迹象）：M0=29，M1=58。

## 4. Per-episode capture hazard h(k)=P(capture|第k个episode存在)（lineage，primary）

| k | h_M0(k) | h_M1(k) | Δ [95% CI] | 方向 |
| --- | --- | --- | --- | --- |
| 1 | 2.8% | 4.8% | 2.0%[1.4%,2.6%] | UP |
| 2 | 2.9% | 5.8% | 2.9%[2.2%,3.5%] | UP |
| 3 | 3.2% | 6.5% | 3.3%[2.6%,4.0%] | UP |
| 4 | 3.0% | 6.0% | 3.0%[2.3%,3.8%] | UP |
| 5 | 2.8% | 6.3% | 3.5%[2.7%,4.2%] | UP |
| 6 | 2.7% | 6.0% | 3.3%[2.5%,4.1%] | UP |
| 7 | 2.4% | 6.4% | 4.0%[3.2%,4.9%] | UP |
| 8 | 2.8% | 6.3% | 3.5%[2.6%,4.5%] | UP |
| 9+ (pooled) | 2.9% | 6.1% | 3.2%[2.9%,3.5%] | UP |
| all (pooled) | 2.9% | 6.0% | 3.1%[2.9%,3.4%] | UP |

- gap stop hazard（每个恢复间隙的正常停止概率）pooled：M0=4.0%，M1=1.5%，Δ=-2.5%[-2.7%,-2.2%]。
- in-episode stop hazard（episode 内正常结束）pooled：M0=4.0%，M1=2.3%，Δ=-1.7%[-1.9%,-1.5%]。

## 5. Episode-ordinal CIF 与 Shapley 分解（主判定）

- capture CIF@K145：M0=26.0%，M1=60.9%，Δ=34.9%[33.3%,36.4%]。
- stop CIF@K145：M0=74.0%，M1=39.1%，Δ=-34.9%[-36.4%,-33.3%]。
- Shapley（尾部池化边界 K_pool=56，双模型 episode at-risk ≥50 的最大 ordinal；边界外每模型每 cause 用其自身尾部池化 hazard，避免空风险集填 0 伪造 propensity）：per-episode propensity 分量=15.8%[14.7%,17.0%]（share=45.4%）；stop/exposure 分量=19.0%[17.9%,20.3%]（share=54.6%）。
- 分解模型总差=34.9%[33.3%,36.5%]；原始曲线 ΔCIF=34.9%[33.3%,36.4%]（两者之差为尾部平滑差异，应当很小）；identity error=0.000000。

## 6. Legacy 口径一致性（lineage，legacy-aligned capture）

- capture 率 M0/M1=6.5%/22.8%；pooled per-episode hazard Δ=0.8%[0.7%,0.9%]；propensity 分量=13.1%[11.7%,14.5%]；exposure 分量=8.8%[8.0%,9.7%]。
- stable orbit 存在但结构未对齐（记删失）：M0=3，M1=7。

## 7. Orbit-stage flag（独立于 A/B）

- P(stable orbit | primary captured)：M0=24.9%，M1=37.6%，Δ=12.7% [10.1%, 15.3%]；flag=ON。
- P(stable orbit | uncaptured)：M0=0.0%，M1=0.2%。

## 8. 距离统计（captured responses，lineage，primary；描述性，无 CI）

- last response 内新 triple → capture episode 起点：M0 中位 4.0 blocks（p25–p75 1.0–12.0），M1 中位 4.0（1.0–11.0）。
- capture episode 是 last 新 triple 后第一个 episode 的占比：M0=38.5%，M1=37.9%；之间隔的 episodes 中位：M0=2.0，M1=2.0。
- last response 内新 pair → capture episode 起点中位：M0=6.0，M1=5.0（来源=reparse）。

## 9. 敏感性（contiguous 变体，primary 口径）

- pooled per-episode hazard Δ=7.3%[6.9%,7.8%]；propensity 分量=18.7%[17.4%,20.0%]；exposure 分量=16.1%[15.1%,17.1%]。
- 分支：lineage=B_PROPENSITY_ALSO_ELEVATED，contiguous=B_PROPENSITY_ALSO_ELEVATED。

## 10. 判定

- `P0D_B_PROPENSITY_ALSO_ELEVATED`
- 下一步：两阶段机制：终止失准+捕获增强；R1/S1 照常，R2 后追加 episode 起点固定前缀 F_core clamp 干预式验证（须过随机匹配对照）；orbit-stage flag 触发：P2b 瓶颈修复机制保留为 capture→orbit 阶段证据。

## 11. 解释边界

- episode 合并是确定性状态机（附单测），但『lineage』边界仍是一种建模选择；因此 contiguous 变体作为预注册敏感性对照一并汇报，分支不一致即降级，不挑好看的报。
- capture 作为吸收态截断是建模约定；capture 后的 novel block 数量已单独汇报，供检查『capture 逃逸』是否被该约定掩盖。
- hit-max 删失只移出风险集，不构成事件；M1 删失远多于 M0，CIF 与 hazard 均为删失一致口径下的估计。
- 本步骤仍是行为级分解：它裁定『暴露 vs per-episode 倾向』，不裁定任何 SAE/回路中介，也不是训练数据归因。
