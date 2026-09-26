# 固定 seed 完整通关搜索分析

2026-09-24。本次范围为算法分析与原生测量，未修改搜索策略。目标是缩短同一 seed 首条独立重放验收通过的完整通关轨迹的时间，不允许换 seed。

## 原生基准

已配置仓库 sts2-cli 游戏程序集、CombatSolver 0.44.0、RitsuLib compat/0.111.0；.NET 9.0.121。使用 Ironclad、A0、seed `7E4A91CDAE1225F0`、单线、weight=12、rollout_decisions=256、reuse_turn_plan=true、budget_ms=1000、180 秒搜索预算。

计时脚本：`combat_solver_cli/artifacts/perf-analysis/profile_search.py`。从仓库根目录运行，输出目录必须不存在：

```bash
PYTHONPATH=. /home/nastukoi/miniconda3/envs/sts2/bin/python \
  combat_solver_cli/artifacts/perf-analysis/profile_search.py \
  --output combat_solver_cli/artifacts/perf-repeat-1000 --seconds 180
```

脚本固定上述角色、seed、weight 和单线设置；支持 `--budget`、`--rollout`、`--no-reuse`、`--prefix`、`--expansions`。计时包装仅在脚本进程中生效。原生求解受时间预算和运行负载影响，固定 seed 不代表所有求解分支逐次完全一致。

| 项目 | 秒 | 占端到端时间 |
|---|---:|---:|
| solver_step：原生求解/执行/通信 | 146.110 | 81.16% |
| restore：进程启动、重放和状态检查 | 31.188 | 17.32% |
| heuristic + preference | 0.192 | 0.11% |
| save_frontier | 0.077 | 0.04% |
| 其余 | 2.467 | 1.37% |

`combat_and_forced` 的 147.529 秒包含 solver_step，不能重复相加。solver_step 也包含复用计划的执行调用，665 次调用不等于 665 次完整原生搜索。

180.034 秒内展开 323 个节点、原生死亡 10 次、Boss 失败计数 9、未解决分支 2、队列剩余 3,971 个节点、重放 2,515 步。最佳已记录决策点为第二幕 17 层；进入第二幕 18 层 Boss 后失败。没有获得完整通关或 verified_victory。

结果：`combat_solver_cli/artifacts/perf-baseline-1000/{profile.json,summary.json,search.jsonl,frontier.json}`。

## 当前算法的实际行为

`astar.py:428` 使用父观察给尚未执行的候选打分：

```text
f(candidate) = g + 0.1 + weight × max(0, h(parent) − 0.25 × preference(parent, candidate))
```

节点保存动作历史前缀与待执行动作，展开后才真正执行动作、自动推进战斗和唯一合法动作，再生成下一批候选。它是延迟展开、带启发式先验的历史树搜索。同父候选通常完全按 preference 排序；增大 weight 不会改善这些候选的相对偏好。

h 主要由剩余楼层决定：每层约 3，全血到零血的风险差只有 6。weight=12 时，一层进度约影响 36 分，而每次局外决策的 g 只加 0.1。该尺度天然偏向快速深入；h 不估计实际剩余秒数或最终通关概率，也不具备可采纳性。

rollout 强制连续选择本次新生成的最佳子节点，其他分支保留在队列。沿同一条路线前进时，restore 会复用当前 worker，通常无需重放；跳到另一条历史时则关闭进程、从 seed 重建并执行完整前缀。重放不会重新调用战斗求解器，但仍付出引擎执行、通信、哈希检查和启动成本。

Boss 回跳按全局失败次数计算距离 3、5、7……，只用楼层决定候选是否有资格，再按原有 f 选择。它没有记录哪次抓牌、遗物、升级或资源决策可能改变当前失败，也没有按不同 Boss 隔离失败经验。floor>=15 只是判定条件，不能严格等同于 Boss 遭遇。

## 三项受控检查

### 提高 Boss 战斗预算没有解决这次失败

从首次死亡轨迹提取第二幕 Boss 战斗前的 310 步原生前缀，独立重放至同一状态，只展开该场战斗。

| 单次求解预算 | restore | 战斗宏 | 结果 |
|---|---:|---:|---|
| 1000 ms | 2.856 秒 | 5.630 秒 | 18 个 solver_step 后死亡 |
| 2500 ms | 2.862 秒 | 9.401 秒 | 18 个 solver_step 后死亡 |

复现命令（改变预算与输出目录即可对照）：

```bash
PYTHONPATH=. /home/nastukoi/miniconda3/envs/sts2/bin/python \
  combat_solver_cli/artifacts/perf-analysis/profile_search.py \
  --output combat_solver_cli/artifacts/perf-boss-repeat \
  --seconds 60 --budget 1000 --expansions 1 \
  --prefix combat_solver_cli/artifacts/perf-baseline-1000/boss-entry.json
```

该结果只能排除“这个入口统一增加到 2500 ms 就能过关”；不能证明这个入口数学上无解，也不能推出所有战斗都应降低预算。

### 完整战斗计划已存在，但当前只复用本回合

同 seed 首战 11 次 solver_step 中实际搜索 3 次。首次搜索已经返回跨第 1、2、3 回合的 11 步完整胜利方案；适配器只将当前回合动作放入队列，换回合再次搜索。

三个搜索调用耗时约 0.810、0.026、0.016 秒。这个样本中，跨回合复用最多省掉后两次调用的大约 42 毫秒，不能根据“搜索次数从 3 降到 1”宣称 3 倍加速。跨回合复用仍值得在困难战斗上测量，但必须处理预测缺口、抽牌/敌方行为偏差、选择计划与状态验证。

原始证据：`combat_solver_cli/artifacts/perf-first-combat-plans.json`。

### 有可稳定复现的合法分支中断

原生基准两次出现 `Combat selection has no solver plan; complete it externally before solving`。从 unresolved 保存的原始前缀和 pending_action 独立重放，复现第二幕第 2 层战斗开始时的 `card_select`：从手牌中 exhaust 一张，5 个合法 SELECT_ONE 候选，尚未建立求解器 selector。

`ReplayWorker.combat_and_forced` 将战斗中的选牌一律交给 solver_step；`SolverAdapter.SelectCandidate` 在 selector 为空时抛错。当前搜索记录 unresolved 后移除活动分支，普通 frontier 恢复不会自动重试这条路径。

这是完整通关搜索的覆盖缺口：原生游戏尚未失败，路线却不能继续。建议让“没有原生选择计划的选择边界”显式返回给搜索器，保留全部合法候选，选完后继续战斗宏；已有计划的选择仍按计划执行。需要专门的能力/状态标志，不应把任意原生异常都吞掉并随机选牌。可重试的未支持分支应进入持久化待处理队列。

证据：`combat_solver_cli/artifacts/perf-baseline-1000/unresolved/` 与 `unresolved-selection-frame.json`。

## 优化顺序

1. **先补分支覆盖，再量化每种战斗成本。** 修复上述无计划选择边界；统计实际 search 次数、计划复用次数、战斗根状态、每战耗时、战斗后生命、失败位置、restore 次数与耗时。当前 summary 的 solver_steps 不能直接代表搜索次数。
2. **改进失败后的分支调度。** 以遭遇为单位维护失败记录；比较死亡路线的牌组、遗物、生命和资源，优先回到能改变这些因素的决策。保留少量不同牌组/路线的候选，避免反复把几乎相同的配置送进同一 Boss。所有替代分支仍可保存在 frontier，避免不可逆剪掉唯一可赢路线。
3. **把“通关机会/额外时间”纳入优先级。** 同时考虑预计存活能力、当前 worker 可继续的程度和恢复成本。具体权重必须用固定 seed 对照验证，不能把较高楼层直接当成更接近可行解。既要减少恢复，也要防止过度黏着已反复失败的路线。
4. **减少恢复成本。** 优先调查在地图/房间边界建立能精确恢复的原生检查点，再重放短后缀；保留完整动作历史用于最终独立验证。现有 SaveCheckpoint 在非地图房间会回退到房间前，不能直接当作任意决策快照。必须验证隐藏 RNG、待处理操作、遗物计数与后续轨迹一致性。
5. **战斗预算与计划复用做局部对照。** 在同一失败入口先尝试有限预算升级，再决定回跳；预算上限与收益记录按遭遇管理。完整胜利计划可试验跨回合复用并在状态/候选不匹配时重规划。实际改善取决于避免了多少昂贵搜索，以及是否增加后续死亡。
6. **最后清理 Python 调度冗余。** 每帧 heuristic 只算一次，地图 DAG 共享一次后向计算；rollout 最佳子节点可单独持有，避免先入堆再线性查找、删除、heapify。这样能降低大 frontier 下的开销，但当前测量中打分全部消除也只省约 0.19 秒/180 秒。

两/四线当前采用一次性静态分片，各自持有独立 worker；没有动态窃取、共享失败信息或重新分配展开预算。下一步应与单线做固定 CPU 配额对照，并以保持正在推进的 worker 为前提分配备用分支。增加线数的收益不能从代码推定；多个原生求解器会争用 CPU。

## 验收指标与边界

主指标为同一 seed 从启动到 `verified_victory` 的端到端秒数，包含最终独立重放与导出。当前单线 summary.seconds 在验证前计算，不能直接用它衡量该指标；基准脚本的 total_seconds 覆盖整个 search 调用。

同时报告固定时限内是否通关、死亡次数、重复失败遭遇、恢复时间、实际原生搜索时间和峰值 frontier。未通关的运行是受预算截断的结果，不能只比较成功样本的平均时间。先保持同一 seed、相同 CPU 配额做对照，再跨角色与 seed 检查普遍性。

本次没有得出已经提速的实现，也没有生成验收通过的完整通关轨迹。已完成原生基线、相同 Boss 入口预算对照、完整战斗计划检查和合法分支中断复现。固定 seed 下，当前有限预算战斗教师和未支持分支都使“队列耗尽”不能证明游戏本身无解。
