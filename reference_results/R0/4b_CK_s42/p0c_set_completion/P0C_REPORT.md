# P0c：参考集合完成、边际 gold-match 产出与过度生成审计

## 1. 冻结定义

- 输入：all-mode `event_rows.jsonl`、其 `run_manifest.json` 指向的原始 M0/M1 responses、以及 `prompt_eval_dedup.jsonl`。
- 不加载模型或 SAE，不重新生成，不重新运行 01b，不移动 legacy raw onset。
- 每条 response 重新使用已验证的四字段状态机 parser；response SHA、block 数、每个 triple/quad hash、continuity segment 全部必须与 01b 一致。
- Strict gold match：沿用 protocol v1.0 归一化后的 `(source,target,relation)`；空 identity 不作为有效关系。
- Relaxed gold match：沿用 protocol v1.0 归一化后的 `(source,target)`；空 identity 不作为有效关系。
- 归一化规则：strip、lowercase、collapse whitespace。
- 关系机会 j 的进度为 `j/G`；正常结束位于 `(n_blocks+1)/G`；hit-max 只作行政删失，不作为正常停止。
- 主 block 分类互斥：new strict gold、new relaxed-pair-only、seen normalized-triple reuse、gold-pair-covered variant、new unmatched、invalid identity；parser-invalid object 另行报告。
- `new_relaxed_gold_pair` 是重叠边际效用计数：首次覆盖 gold pair 的 strict block 也计入；同一 block 在自然事件分母中仍只出现一次。
- Capture seed ordinal：按所有 raw-exact nonempty triple reuse block 的时间顺序编号；primary capture 使用其 second-copy start；legacy stable orbit 使用 alignment evidence 中的 matching quad run。

## 2. 完整性

- selection mode：all；prompts=1106；responses M0/M1=8848/8848。
- gold key mapping：`key`；gold rows=1106。
- M0/M1 prompt 和 seed 完全配对：PASS。
- 原 response 与 01b event row 全量逐 response/逐 block 校验：PASS。
- parser-invalid complete object candidates：M0=200，M1=743。

## 3. 输出与最终 gold 覆盖

- L/G 中位数：M0=1.355，M1=2.185。
- strict coverage@G：M0=9.0%，M1=8.7%，Δ=-0.3%[-0.5%,-0.1%]；G 后增益：M0=1.4%，M1=2.0%，Δ=0.6%[0.5%,0.8%]；最终：M0=10.3%，M1=10.7%。
- relaxed coverage@G：M0=33.3%，M1=31.4%，Δ=-1.9%[-2.2%,-1.5%]；G 后增益：M0=4.8%，M1=6.8%，Δ=2.0%[1.7%,2.2%]；最终：M0=38.2%，M1=38.3%。

## 4. 关系机会的边际事件率（相对于教师参考集合）

- >=1.00G new strict gold triple：M0=2.2%，M1=0.7%，Δ=-1.5%[-1.7%,-1.3%]。
- >=1.00G newly covered relaxed gold pair：M0=7.5%，M1=2.3%，Δ=-5.2%[-5.8%,-4.8%]。
- >=1.00G new gold utility block union：M0=7.7%，M1=2.4%，Δ=-5.3%[-5.9%,-4.8%]。
- >=1.00G seen-reuse：M0=34.4%，M1=73.1%，Δ=38.7%[35.6%,41.7%]。
- >=1.00G gold-pair-covered variant：M0=1.9%，M1=0.8%，Δ=-1.2%[-1.4%,-1.0%]。
- >=1.00G new-unmatched：M0=53.2%，M1=23.2%，Δ=-30.0%[-32.6%,-27.4%]。
- >=1.00G normal-stop：M0=2.7%，M1=0.5%，Δ=-2.2%[-2.3%,-2.0%]。
- >=1.25G M1：new gold-match=1.5%，seen-reuse=78.1%，new-unmatched=19.2%，normal-stop=0.5%。
- M1 post(>=1G)−pre(0.5–1G) new gold-match：-19.1% [-19.9%, -18.4%]。
- M1 post(>=1G)−pre(0.5–1G) seen-reuse：64.1% [62.8%, 65.4%]。

## 5. First seed 时的集合状态

- first-seed progress 中位数：M0=0.909G，M1=1.000G。
- first-seed strict coverage：M0=8.2%，M1=8.4%。
- first-seed relaxed coverage：M0=31.2%，M1=31.0%。
- first seed 时已完成最终 relaxed coverage 的比例：M0=85.7%，M1=84.0%。
- seed 后不再产生新 relaxed gold pair：M0=42.0%，M1=40.1%。
- first seed 位于最后一个新 relaxed pair 之后：M0=41.6%，M1=39.6%。
- first seed 位于 >=1.00G：M0=41.9%，M1=51.5%。
- raw-exact seed 与 normalized earliest reuse 一致率：M0=100.0%，M1=100.0%。

## 6. Capture 来自第几个 seed

- primary semantic capture 来自 seed #2+：M0=86.3%，M1=92.4%；位于最后一个新 relaxed pair 之后：M0=97.9%，M1=96.9%。
- primary capture seed ordinal 中位数：M0=7.0，M1=8.0。
- legacy-aligned stable capture 来自 seed #2+：M0=86.5%，M1=95.2%；位于最后一个新 relaxed pair 之后：M0=100.0%，M1=100.0%。
- legacy-aligned seed ordinal 中位数：M0=4.0，M1=7.0。

## 7. 判定

- `P0C_COMPLETION_FAILURE_PLUS_MULTISEED_CAPTURE`
- completion component：`SET_COMPLETION_FAILURE_SUPPORTED`
- capture origin：`LATER_SEED_DOMINANT`
- 下一步：进入模型级 set-completion/close-list vs continue-dict 边界分析；capture 作为下游多次 seed 暴露结果。

## 8. 解释边界

- strict/relaxed 都是教师参考集合上的蒸馏一致性匹配；relaxed 不是语义 embedding/fuzzy match，new-unmatched 也不能自动判成语义错误。空 identity 从有效 gold utility 中排除，但仍计入原始 G 并单独审计。
- 各进度区间的 event rate 条件于 response 已到达该区间；P0b 的 competing-risk 结果仍负责解释停止/暴露差异，P0c 不把这些条件率单独解释成因果 hazard。
- G 是参考输出字典数，不被事后调整；P0c 用边际效用检验它是否近似 completion 位置。
- seed ordinal 是 raw-exact nonempty triple reuse event 的顺序；它不把同一循环中的所有 block 合并成主观 episode。
- 本步骤仍是行为分解，不等价于 SAE 中介或训练数据归因。
