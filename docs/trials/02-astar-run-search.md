# 固定种子的加权 A* 整局搜索做示范（含加大失败入口的求解预算）

- **时间**：2026-09-23（`181ad91` “A*+combatsolver for bootstrap”）至 2026-09-27；2026-09-26 ADR 0003 定下唯一主线为 MCTS 后删除。
- **来源**：`bb0b1ee^:docs/archive/search/ASTAR_COMPLETION_ANALYSIS.md`、`COMBAT_SEARCH_ANALYSIS.md`、`EARLY_ROUTE_DIVERSITY.md`；`bb0b1ee^:.scratch/search-speed/evidence/diagnosis.md`、`manual-budget-comparison.md`。
- **状态**：A* 整局搜索被删除；加大战斗求解预算没带来提升。

## 动机

录制示范只有十几局（[01](01-1b-first-design.md)）。想法是在同一个种子上用搜索找出第一条能独立重放的通关轨迹，把它当示范：局外（走图、奖励、事件、商店、营火）由整局搜索决定，战斗交给 CombatSolver。

## 做法

### 加权 A* 整局搜索

- 打分 `f = g + 0.1 + weight × max(0, h(parent) − 0.25 × preference)`，`weight=12`、`rollout_decisions=256`、`budget_ms=1000`、`reuse_turn_plan=true`。
- 之后陆续加入：失败回跳（3、5、7……层）、exhaust 选择的方向修正、同一种子 4 条静态分片的搜索线、早期路线多样性、Boss 战 5000 ms。

### 给失败入口加大战斗求解预算

猜测 Boss 战失败是因为求解预算不够，于是在失败的入口上把单次搜索预算调大重打；反过来也试了把整局的预算调小。

## 结果

### 加权 A* 整局搜索

- 单线 180 s 基线（种子 `7E4A91CDAE1225F0`，Ironclad A0）：“180.034 秒内展开 323 个节点、原生死亡 10 次、Boss 失败计数 9、未解决分支 2、队列剩余 3,971 个节点”，没有通关。时间占比 solver_step 81.16%、restore 17.32%、heuristic+preference 0.11%。
- 四线从头运行（`perf-completion-4lanes`）：“424.680 秒（约 7 分 5 秒）完成搜索、全新原生进程独立重放、数据导出”，总展开 2,480 个节点，真实战败 97 次，胜利前缀 545 个原生动作、21 场遭遇。文档注明这是一次运行的时间，“不是已证明的稳定平均时间”。
- 其余几次（时间，战败／未解决）：`perf-before-selection-fix` 180.026 s 未通关（24／1）；`perf-selection-fixed` 180.935 s（11／0）；`perf-location-backjump` 467.681 s（27／0）；`perf-exhaust-priority` 207.146 s（11／0）；诊断合计约 25.55 分钟。
- 早期多样性加 Boss 5000 ms 的四线 120 s 检查：展开 963 个节点，最远到第三幕 12 层，没有通关。

### 加大失败入口的求解预算

| 入口 | 预算 | 结果 |
| --- | --- | --- |
| 第二幕 Boss 前 310 步前缀 | 1000 ms／2500 ms | 都在 18 个 solver_step 后死亡 |
| 第三幕 12 层 | 1000 ms（4.670 s）／4000 ms（12.776 s） | 都在 23 步后死亡 |
| 第三幕 15 层 AEONGLASS，88／92 HP | 10000 ms（42.068 s） | 29 步后死亡 |

减预算的方向：

- 固定战斗 1000 ms 对 250 ms：第一幕第 2 层 1.953 s 对 0.797 s（2.45 倍，战后 HP 都是 80）；第三幕第 2 层 4.277 s 对 1.437 s（2.98 倍，HP 88 → 95）。
- 整局对照（种子 `UNIFICATION-HOLDOUT-01`，单线）：250 ms 742.907 s（12.38 分钟）`verified_victory`；1000 ms 1327.446 s（22.12 分钟）`stopped`。

## 原因分析

- **启发式不估通关概率**。h 由剩余楼层主导：“每层约 3，全血到零血的风险差只有 6”，`weight=12` 时一层进度约 36 分，而每次局外决策的 g 只加 0.1，搜索天然偏向快速深入。
- **失败经验没有隔离**。Boss 回跳不按遭遇区分失败原因。
- **选择边界中断分支**。求解器没有计划的合法选择（例如“从手牌 exhaust 一张”）被当作搜索错误，分支随之中断。
- **时间不在打分上**。solver_step 与 restore 占了约 98% 的时间，“打分全部消除也只省约 0.19 秒/180 秒”。
- **失败入口的问题在局外**。牌组和路线决定了 Boss 打不过，更深的战斗搜索不改变结果；预算加大“不保证所选路线质量单调改善”。
- 一个种子约 7 分钟，且只在 Ironclad A0 的少数种子上成功，规模上不去。

## 结论与后续

- ADR 0003（2026-09-26）定下唯一整局搜索为 MCTS + CombatSolver，A* 与 `lanes-v1` 入口删除，旧检查点明确拒绝；MCTS 见 [04](04-mcts-backtracking-search.md)。整条搜索示范路线在 2026-10-04 被摘要锚点取代（[05](05-replay-restoration.md)）。
- 预算：最终获胜线保持 1000 ms；250 ms 只“作为下一轮开发实验的候选配置”，没有改默认。
- 现在示范战斗仍由 CombatSolver 在重建入场状态上求出，预算是普通与精英 5000 ms、Boss 20000 ms，见 [demos](../implementation/demos.md#生成样本)、[solver](../implementation/solver.md)。加大预算的另一次尝试见 [11](11-solver-potion-demos.md)。

## 教训

- 先剖析时间花在哪里：这里几乎全在原生求解与恢复，优化打分没有意义。
- 求解器没有计划的选择边界要显式交回搜索器处理，不能当错误。
- 获胜路线里“没有 `operation=exhaust` 的选择”，修好的问题不一定是通关的原因。
- 战斗预算不是失败入口的主杠杆；后来 A10 大预算战斗的耗时只涨 1.5–2.3 倍、多数搜索在预算用完前就结束（[11](11-solver-potion-demos.md)），与这里一致。
