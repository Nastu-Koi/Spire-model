# 同 seed A* 优化与完整通关实测

2026-09-24。目标是尽快找到第一条可独立原生重放的完整通关轨迹，允许次优路线；不是证明最短路径。保留既有 `COMBAT_SEARCH_ANALYSIS.md` 作为此前未通关阶段的记录。

## 已达到的结果

- seed：`7E4A91CDAE1225F0`；Ironclad；A0。
- 最终从头运行：4 条同 seed 独立搜索线，`weight=12`、`rollout_decisions=256`、`budget_ms=1000`、`reuse_turn_plan=true`。
- **424.680 秒（约 7 分 5 秒）完成搜索、全新原生进程独立重放、数据导出以及其余搜索线停止。** 这是最终配置的一次运行时间，不是整个诊断工作的总耗时，也不是已证明的稳定平均时间。
- 总展开 2,480 个节点，真实战败 97 次；lane-1 获胜，该线展开 618 个节点、战败 24 次、未解决分支 0。
- 全部搜索线合计还有 2 次原生计划一致性错误，均发生于未获胜的 lane-2，保存于其 `unresolved/`；没有吞掉这些异常。
- 胜利前缀 **545 个原生动作**；独立回放记录终局 `victory=true`、`bosses=[1,2,3]`、`victory_paid=true`，覆盖 21 场遭遇。
- `accepted.jsonl` 非空，包含 1 条轨迹；`validate_run` 和原始回放 SHA-256 复核通过。

产物位于 `combat_solver_cli/artifacts/perf-completion-4lanes/`：

| 文件 | 内容 |
|---|---|
| `winning_prefix.json` | 同 seed 的完整可回放动作历史 |
| `verified_trace.jsonl` | 全新进程的独立重放与终局证据 |
| `accepted.jsonl` | 已验收的 Bootstrap 数据 |
| `summary.json` | `status=verified_victory`，包含四条搜索线结果 |
| `profile.json` | 包含验收及停止其他线的端到端计时 |

现有训练 schema 将 recorder_bc 写成 `status=partial, victory=null`；真实胜利证据在 `provenance.verified_outcome=A0_final_boss_victory`、`verified_by=fresh_native_seed_replay` 和原始终局记录中。不能只看训练样本顶层 victory 字段。

## 获胜路线透露了什么

胜利轨迹从 Neow 选择 `NEOW.pages.INITIAL.options.LEAD_PAPERWEIGHT` 开始。此前检查的失败路线拥有 `NEW_LEAF`，两者属于同一 seed 下不同的早期决策，而非更换 seed。

获胜路线第三幕第 14 层地图上为 90/90 HP，首次观察到 AEONGLASS 战斗帧时为 86/90 HP。牌组 26 张、遗物 9 件，仍含 8 张基础 Strike/Defend；包含升级的 RUPTURE、HEMOKINESIS、SETUP_STRIKE 等牌。不能从此前某条失败路线基础牌较多，直接断言“必须大幅压缩牌组才能通关”。

获胜轨迹没有 `operation=exhaust` 的选择。故本次不能把最终胜利归因于新修复的 exhaust 排序；该修复解决了其他路线的明确排序问题。更有力的观察是：**不同开局、抓牌和资源组合的探索，能在同 seed、同样 1000 ms 战斗预算下找到完整胜利。** 四线中的一条在第三幕第 15 层失败过一次，之后找到通过路线；其他线仍在回溯，协调器在独立验收后停止它们。

## 实验过程与边界

以下全部保持 seed、角色和难度一致。除明确注明外均为单线，1000 ms、weight=12、rollout=256、回合计划复用。`stopped` 是为了切换实验而在动作边界保存队列，不是队列耗尽。

| 实验目录 | 端到端秒数 | 结果 | 战败 / 未解决 |
|---|---:|---|---:|
| `perf-before-selection-fix` | 180.026 | 未通关，最好第三幕 5 层 | 24 / 1 |
| `perf-selection-fixed` | 180.935 | 未通关，最好第三幕 14 层 | 11 / 0 |
| `perf-location-backjump` | 467.681 | 保存停止，最好第三幕 12 层 | 27 / 0 |
| `perf-completion-resume` | 72.331 | 从第二行队列恢复，保存停止，最好第三幕 14 层 | 6 / 1 |
| `perf-exhaust-priority` | 207.146 | 从头运行，保存停止，最好第二幕 17 层 | 11 / 0 |
| `perf-completion-4lanes` | **424.680** | **从头运行，完整通关并独立验收** | **97 / 2** |

这些主要搜索实验合计约 25.55 分钟计算墙钟时间，另有局部预算、回归和检查开销；最终运行的 7 分 5 秒不能代表整次诊断耗时。

原生战斗求解依赖墙钟预算，相同 seed 不保证每次找到完全相同的计划。上表既改变了实现／配置，也包含不同时间截断，**不能据此宣称固定倍数加速**。旧实现没有跑到通关，因此缺少旧实现的完整 time-to-victory 对照。单线 207 秒未通关，也不能证明单线在 425 秒内必然无法通关。

## 代码问题与已实施修改

### 1. 合法的无计划选择被当作搜索错误

最小原生复现：恢复捕获的 169 步前缀，执行待定走图动作，然后调用真实 `ReplayWorker.combat_and_forced`。修复前两次均在约 2.5 秒内报 `Combat selection has no solver plan`。该边界有 5 个合法 exhaust 选择，游戏并没有死亡。

`SolverAdapter.Step` 现在仅在活动战斗、确实存在选择边界、且 `_selector == null` 时返回显式 `solver_selection_required`。`ReplayWorker` 将多候选边界返回 A*，保留全部分支；单候选按强制动作执行。已有原生选择计划继续执行，真实求解错误仍然报错。没有按异常字符串猜测或随机选择。

修复后同一前缀检查通过，保留 5 个选择，完成选择确认后进入正常 combat 边界。

### 2. 普通战败未触发提前回跳

修改前 180 秒基准在第三幕第 6 层真实死亡 24 次，`boss_failures=0`；仅 `floor>=15` 触发回跳，因此不断重放近似路线。恢复耗时 76.25 秒，重放 8,704 步。

新增 `failure_backtrack`：所有原生战败按“幕:楼层”分别计数，回跳距离为 3、5、7……层。其他位置不继承该位置的次数。只改变下一次展开顺序，不删除未选择的候选，不用公开状态合并隐藏 RNG 状态。计数写入 frontier；多线合并仅累加各线相对于基线的新失败，避免重复计数。

这是按位置的启发式调度，并非精确遭遇或牌组无解证明。不同房间可能共享同一幕／楼层，计数粒度仍可改进。

### 3. exhaust 方向与保留强牌目标相反

真实捕获边界中，原评分为 SETUP_STRIKE 4.1、基础 STRIKE −2.0；直接最大化 card_value 会先消耗力量牌。现对 `operation=exhaust` 与 remove 一样使用负 card_value，让基础弱牌优先成为消耗候选。所有其他选项仍保留，未假设这一先验能理解每张牌的特殊消耗收益。

对应排序检查先失败，再通过。胜利路线未触发该类选择，不能将此项当成通关的必要条件。

### 4. 使用已有的同 seed 多线实现

没有新增多进程架构；最终运行使用仓库已有的 `search_lanes=4` 静态 frontier 分片。机器暴露 32 个逻辑 CPU，单线原生进程主要占约一个核，因此四线是有资源基础的配置尝试。四线只搜索一个 seed，不是四个不同 seed，也没有子代理参与搜索。

目前分片仍是静态的，不共享失败经验、不动态窃取任务；本次结果不能证明 4 线比 2 线普遍更好。

## 时间主要花在哪里

最终四线计时中，下面是跨 worker 累计调用墙钟时间，会重叠，不能直接除以 424.68 秒当成单进程占比：

| 操作 | 累计秒数 |
|---|---:|
| solver_step：原生求解、计划动作执行、通信、选择交接 | 1302.723 |
| restore：重建与前缀重放 | 337.418 |
| heuristic + preference | 1.264 |
| save_frontier | 1.489 |

`combat_and_forced` 包含 solver_step，不能再次相加。7,688 次 solver_step 请求不等于 7,688 次原生搜索，其中包含计划复用与选择交接。下一步性能记录应区分实际 search、计划复用和交接次数。

当前证据不支持优先优化 Python 堆或评分函数来获得显著通关提速。更高价值的方向是：早期分支多样性、失败后的资源选择回溯、严格验证的恢复成本削减，以及只在有收益证据的战斗上增加预算。

## 追加战斗预算的反例

从同一失败轨迹提取完成入场选牌后的真实战斗根状态，独立重放后只推进该战斗：

| 入口 | 单次预算 | 战斗宏耗时 | 结果 |
|---|---:|---:|---|
| 第三幕 12 层，同一 567 步前缀 | 1000 ms | 4.670 秒 | 23 步后死亡 |
| 同一入口 | 4000 ms | 12.776 秒 | 23 步后死亡 |
| 另一条第三幕 15 层 AEONGLASS，88/92 HP | 10000 ms | 42.068 秒 | 29 步后死亡 |

最后一项为四线运行期间的独立局部探测，耗时不是无负载配对基准。它仍清楚表明“把这个失败入口提高到 10000 ms”并未解决通关；不能推出所有 Boss 都不受益于更高预算。最终获胜线保持 1000 ms。

## 复现与验证

从仓库根目录，输出目录必须不存在。使用已配置的原生依赖及训练环境：

```bash
/home/nastukoi/miniconda3/envs/sts2/bin/python -m combat_solver_cli.benchmarks.profile_search \
  --output combat_solver_cli/artifacts/completion-repeat \
  --seconds 1800 --lanes 4 --require-victory
```

脚本固定本报告 seed、Ironclad、A0、weight=12，默认预算 1000 ms、rollout=256、复用回合计划。`--require-victory` 在未验收通关时返回非零。计时覆盖 search 的完整调用，包括独立重放与导出。重复运行可能有不同耗时与路径。

```bash
/home/nastukoi/miniconda3/envs/sts2/bin/python -m unittest discover \
  -s combat_solver_cli/benchmarks -p 'test_*.py' -v

/home/nastukoi/miniconda3/envs/sts2/bin/python -m combat_solver_cli.benchmarks.selection_boundary \
  --branch combat_solver_cli/artifacts/perf-baseline-1000/unresolved/f563072746bbfb2b.json
```

10 项检查通过；原生适配器构建 0 warning / 0 error；真实选择回归通过；最终原生全轨迹独立重放通过；accepted schema、三幕里程碑及 SHA-256 复核通过。没有修改游戏伤害、随机数或胜利规则。未解决的原生“计划选择未消费”错误仍有 2 次保存记录，未宣称搜索覆盖已完全无缺口。
