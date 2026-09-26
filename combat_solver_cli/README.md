# combat_solver_cli

独立的 CombatSolver / sts2-cli 接入模块：局内调用本机 CombatSolver 0.44.0，局外使用有预算的加权 A* 或 MCTS，目标是生成真实 A0-A10 最终 Boss 胜利轨迹；当前批量入口默认先搜索 A0。复用现有 Bootstrap 训练结构与 sts2-cli 决策协议；Steam 录制路径保持独立。它是离线 headless 工具，不是安装进 Steam 的 UI Mod；Steam 录制仍由现有 RunRecorder 负责。

## 本地依赖与配置

需要 .NET 9、本仓库的 `sts2-cli/lib`、已安装的 CombatSolver 0.44.0 及其 RitsuLib 依赖。游戏与外部 Mod 的 DLL 不随模块分发。Python 使用项目训练环境（导出时 `clean_frame` 会加载含 PyTorch 的表示模块）。

```bash
python -m combat_solver_cli configure \
  --solver /path/to/CombatSolver.dll \
  --dependency-dir /path/to/ritsulib/compat/0.111.0 \
  --dependency-dir /path/to/ritsulib/shared
python -m combat_solver_cli smoke --ascension 0 --budget-ms 1000
python -m unittest discover -s combat_solver_cli/tests -v
```

配置固定游戏与求解器 DLL 的 SHA-256；更换依赖需要重新配置并验证。当前接入通过反射绑定求解器内部接口，版本升级不能假定兼容。`configure --lib` 指定运行时游戏目录；构建依然使用仓库 `sts2-cli/lib` 中与 CLI 一致的程序集。

## 单 seed 搜索

```bash
python -m combat_solver_cli search \
  --ascension 0 --character Ironclad --seed 7E4A91CDAE1225F0 \
  --output combat_solver_cli/artifacts/run-001 \
  --budget-ms 2500 --boss-budget-ms 5000 --reuse-turn-plan --rollout-decisions 16 --search-lanes 2 --weight 12 --max-seconds 1800 --max-expansions 2000
```

输出目录必须不存在。`budget-ms` 是普通战斗每次原生搜索的预算；每幕 Boss 默认使用 `--boss-budget-ms 5000`，由原生 `RoomType.Boss` 判断，不能用楼层代替。5 秒是单次搜索软预算，不是整场 Boss 战总预算，提前找到结果可提前返回；进程 I/O 另有上限；软预算结束时使用 CombatSolver 原生的当前回合结果接管机制，额外 10 秒后硬取消；总时间预算在动作边界检查，因此可能超出一个正在执行的动作。`weight` 越大越偏向启发式估计，不能保证最优路径。`--rollout-decisions 16` 开启最多 16 次局外决策的连续试探，其余分支仍全部保留在 A* 队列中；默认 0 关闭试探。连续路线在原生战败后（包括普通战斗），调度器会强制选择更早楼层尚未展开的分支；同一幕／楼层重复失败会逐次扩大回跳距离，各位置分别计数，使抓牌、删牌、商店、事件、营火和走图决策都能被重新搜索。默认启用早期路线多样性：将第一幕前 5 层的前三个非强制关键决策（事件、走图、抓牌或跳过奖励）组成分层路线标识。连续试探结束或失败回跳时，先比较各层路线的尝试次数，优先尝试较少的路线，再按 A* 分数排序；连续试探内保持当前 worker 推进。所有候选仍保留，公开状态相似不会导致状态合并。单线算法标记为 `weighted_astar_early_routes_v3`；`--no-early-route-diversity` 可关闭，恢复 `weighted_astar_location_backjump_v2` 调度用于对照。`--search-lanes 2` 或 `4` 将同一 seed 的 frontier 按早期路线分成相应数量的不重叠分片，先分开不同开局，开局不足时再按后续早期决策及剩余候选拆分；关闭路线多样性时使用原有按分数交错分片；每条线各自持有独立原生进程、CombatSolver 实例、管道、回合缓存和超时，协调器只负责停止信号、胜者验收与检查点合并。它不会同时搜索第二局游戏。

在输出目录创建 `STOP` 文件会在下一个检查点保存队列并停止。恢复到新的输出目录：

```bash
python -m combat_solver_cli search \
  --character Ironclad --seed astar-a10-001 --weight 12 \
  --resume combat_solver_cli/artifacts/run-001/frontier.json \
  --output combat_solver_cli/artifacts/run-001-resumed
```

角色、seed、难度和权重必须与检查点一致。检查点同时保存路线尝试次数、失败次数和下一次回跳位置，因此分段续跑同一局不会丢失跨幕回溯进度。`--prefix-path best_prefix.json` 则仅从该前缀继续搜索，不能恢复前缀之前的其他方案；完整恢复应使用 `--resume`。

引擎依赖补齐了三个 Godot 兼容接口：`CanvasItem.SetSelfModulate`、`CanvasItem.SetVisible` 和 `Node.GetIndex`。它们使用已有颜色、可见性和节点树数据，避免 TestSubject 变色、Crusher 入场及分节敌人退场时因缺失渲染接口中断；不修改伤害、状态或复活规则。

局内默认对齐原生的多束宽组合搜索，并按原生遭遇判定启用第三幕 Boss 专用策略。精炼阶段若抛出原生 `SearchTransitionException`，只允许接管此前发布的完整存活胜利方案；恢复会写入 `refinement_failure`，仍执行搜索前后状态审计和合法候选校验。没有完整方案、发生其他异常或状态被污染时，继续报错。

搜索节点保留动作前缀，不把相同公开观察合并成相同游戏状态。恢复时新建相同难度的原生运行，逐步核对公开状态摘要并执行原始动作语义，不依赖进程间的临时候选引用。战斗行动和已有计划的战斗子选择来自 CombatSolver，并通过 CLI 的完整合法候选执行。入场效果或重放后没有原生选择计划时，适配器显式返回 `solver_selection_required`，由 A* 保留全部合法选择分支，完成后继续战斗；真实求解异常仍按错误处理。求解器快照前后有原生延续状态审计，检测到污染即终止 worker。

A* 展开走图、奖励、事件、商店、营火和局外选牌的合法分支；唯一合法动作与战斗作为宏推进。当前 `g` 每次局外分支增加 0.1，`h` 以剩余楼层为主，结合血量风险、牌组／遗物强度和未完成房间的开销。幕末奖励与下一幕起点采用连续的回血预期估计，真实回血仍由原生引擎执行。商店卡牌通过 `offers` 关系读取实际卡牌属性，不把商品实体当作卡牌。卡牌排序使用公开费用、伤害／格挡、攻击次数、力量／敏捷、抽牌、重复数量及牌组大小作为先验；这些数值不代表完整理解 OPAQUE_RULE 效果。走图候选会在公开地图 DAG 上累计后续战斗风险和营火收益，避免只看眼前房间而走进必经精英路线；Boss 前营火采用更高的回血阈值。候选偏好进一步决定延迟分支的优先级。这个估计不是可采纳启发式，存在商店排列、取消循环和牌组协同估值不足等搜索效率限制。默认战斗每步重规划；`--reuse-turn-plan` 可在同一回合内复用已求解的动作，每步重新映射当前合法候选，换回合、求解选项变化或候选不匹配时重新求解。两种模式都依赖有限预算的教师策略，因此队列耗尽也不是游戏数学意义上的无解证明。

原生死亡、未支持／重放失败的分支、预算结束、基础设施故障分别记录。基础设施故障会保留活动节点和整个队列，不能把启动失败计为游戏失败。`search.jsonl`、`unresolved/`、`best_prefix.json` 和 `frontier.json` 用于诊断与恢复。未解决分支另保留最多 64 KiB 原生 stderr 尾部，以区分引擎异常和实际战败。

## 验证与训练数据

只有发现原生终局胜利，才进入第二个全新进程进行独立重放。验证不重新调用求解器，而是执行胜利前缀中的每一个原生候选；必须收齐三幕 Boss 和最终通关里程碑。独立重放中出现原生 `[ERROR]` 日志也会拒绝验收；失败时不输出 `accepted.jsonl`。也可手动重放：

```bash
python -m combat_solver_cli verify \
  --prefix combat_solver_cli/artifacts/run-001/winning_prefix.json \
  --output combat_solver_cli/artifacts/run-001-verified
```

成功产物：

- `winning_prefix.json`：完整动作历史，可从原生 seed 恢复。
- `verified_trace.jsonl`：独立重放的决策帧与终局里程碑证据。
- `accepted.jsonl`：通过 `model.data.validate_run` 的 Bootstrap 数据，按现有选择宏组织，排除纯强制宏。
- `summary.json`：只有独立验证成功才标记 `verified_victory`。

现有 Bootstrap schema 对 `source=recorder_bc` 强制要求 `status=partial`、`victory=null`、`teacher_visibility=unverified` 和 `provenance.bc_only=true`，本模块保持这些约束。真实胜利证据写入 `provenance.verified_outcome=A{ascension}_final_boss_victory`、原始证据文件及其 SHA-256。导出同时记录游戏、求解器、worker、引擎与 Godot stub 的 SHA-256，区分依赖构建差异。离线搜索试探过未来，不能伪装成仅使用公开信息的教师或 PPO on-policy 数据。训练／验证应按 seed 分组，避免同一 seed 的分支泄漏到两个集合。

## 批量生产

推荐入口默认目标为 100 条独立重放验证通过的轨迹，并使用系统随机源生成 seed。默认五个角色平均分配，即每个角色 20 条。目标不能整除角色数量时，余数按 `--characters` 顺序每个角色增加一条。下例显式写出目标数；提升难度时只需改 `--ascension`。

```bash
python -m combat_solver_cli.generate_bootstrap \
  --target-trajectories 100 --ascension 0 --workers 1 --search-lanes 2 \
  --budget-ms 2500 --weight 12 --rollout-decisions 256 \
  --max-seconds 3600 --max-expansions 10000 \
  --output combat_solver_cli/artifacts/bootstrap-a0-100
```

目标数统计的是 `accepted.jsonl` 中验证成功的轨迹，不是尝试的 seed 数。每局用完单段预算后自动恢复该局 frontier；只有该局通过独立重放验收后才开始下一局，因此整个任务始终只有一个 seed 在搜索。进度写入 `manifest.json` 和 `summary.json`，每条成功轨迹先写到 `accepted-parts/`，随后原子更新汇总 `accepted.jsonl`。

默认使用全部角色；例如 `--characters Ironclad Silent --target-trajectories 101` 会分配为 51/50。目标模式为每条轨迹独立生成一个系统随机 seed，并在整个任务内去重；同一条未完成轨迹的分段续跑保持该 seed。`--seed-prefix` 只适用于旧的 `--seeds-per-character` 有界实验。输出目录必须不存在。在目标目录创建 `STOP` 会让当前多线搜索在动作边界保存 frontier 并停止。之后原地恢复：

```bash
python -m combat_solver_cli.resume_target \
  --output combat_solver_cli/artifacts/bootstrap-a0-100
```

恢复入口会移除顶层 `STOP`，保留已经验收的计数，并从当前轨迹最后一个合并 frontier 继续相同随机 seed。

旧的固定尝试次数模式仍可显式使用 `--seeds-per-character 8`。这个模式不会自动补足失败任务，因此适合有界实验，不适合要求精确训练轨迹数的生产任务。

预算用尽后，用新输出目录继续上一批所有仍有 frontier 的任务；难度、权重、局内预算和 rollout 默认继承上一批，也可显式覆盖：

```bash
python -m combat_solver_cli.resume_bootstrap \
  --previous combat_solver_cli/artifacts/bootstrap-a0-001 \
  --output combat_solver_cli/artifacts/bootstrap-a0-002 \
  --workers 1 --search-lanes 2 --max-seconds 1200
```

底层 `batch` 子命令也可读取显式任务清单：

```bash
python -m combat_solver_cli batch \
  --jobs combat_solver_cli/jobs.example.json \
  --workers 1 --search-lanes 2 --max-seconds 1800 --budget-ms 2500 --reuse-turn-plan \
  --output combat_solver_cli/artifacts/batch-001
```

生成器强制 `workers=1`，同一时刻只搜索一局；任务按 manifest 顺序串行运行。默认 `search_lanes=2`，也可设为 4；每条非空分片使用相互隔离的原生引擎进程，各自的 stdin 串行；设为 1 使用单线。批量脚本仅汇总独立验证成功的对局；失败与未完成任务保留在各自目录。`accepted.jsonl` 可能为空，这不算成功生成通关轨迹。搜索命令仅在验证成功时返回 0；批量命令至少有一条验证通过才返回 0。

## 已验证与尚未完成

测试覆盖五角色原生首战、求解只读与过期执行拒绝、替代路径、断点恢复、基础设施故障保留队列、精炼异常时保留原生完整战斗解，以及拒绝导出未通关前缀。完整验收必须以非空 `accepted.jsonl`、三幕 Boss 里程碑和全新原生进程重放为准；首战、单幕或仅到达第三幕都不能算生成成功。

验证成功的汇总文件可直接供现有 Bootstrap 入口使用：

```bash
python -m model --device cuda bootstrap \
  --data combat_solver_cli/artifacts/bootstrap-a0-001/accepted.jsonl \
  --config configs/rtxpro6000.json --epochs 5 --output runs/bootstrap-solver
```

先确认汇总文件非空且批量摘要包含 `verified_victories`；该训练命令仅为使用示例，不会由生成器自动启动。

## 同 seed 完整通关性能实测

固定 seed `7E4A91CDAE1225F0` 的 Ironclad A0 已在四线配置中生成 545 步完整胜利轨迹，并通过全新原生进程重放；该次从头运行到验收及其他线停止共 424.68 秒。单次结果不代表稳定平均耗时或相对旧实现的加速倍数。代码修改、失败对照、剩余限制和复现命令见 [完整通关分析](../docs/ASTAR_COMPLETION_ANALYSIS.md)。

早期路线调度与 Boss 预算的具体行为、验证记录见 [路线多样性与 Boss 预算](../docs/EARLY_ROUTE_DIVERSITY.md)。此前 424.68 秒通关记录属于修改前配置，不能当作新配置的性能结果。

## 蒙特卡洛树搜索

`search --algorithm mcts` 使用 UCT 选择、公开动作先验引导的随机模拟和沿祖先回传的价值统计。默认仍为 A*；`batch` 入口仍使用 A*。MCTS 默认每次模拟最多推进 256 次决策，`--rollout-decisions 0` 则每次只展开一个节点。两种算法共享原生战斗求解、合法动作、四线开局分片和独立通关验收。

```bash
python -m combat_solver_cli search --algorithm mcts \
  --ascension 0 --character Ironclad --seed 7E4A91CDAE1225F1 \
  --output combat_solver_cli/artifacts/mcts-example \
  --budget-ms 1000 --boss-budget-ms 5000 --reuse-turn-plan \
  --rollout-decisions 256 --search-lanes 4 --weight 12 \
  --max-seconds 900 --max-expansions 100000
```

MCTS 检查点是 `tree.json`，包含整棵树、访问次数、累计价值、精确动作历史和随机数状态。使用 `--algorithm mcts --resume 原输出目录/tree.json` 恢复，保持角色、seed、难度、模拟深度和线数等配置一致，输出到新目录。创建 `STOP` 文件可保存后停止。

算法说明和五个新 seed 的 MCTS 实测见 [MCTS 对比](../docs/MCTS_COMPARISON.md)。

### MCTS 局部修正与风险探索（v2）

默认开启 `--local-repair` 和 `--risk-aware-rollout`。战败后优先选择失败路径祖先中尚未执行的兄弟动作，并与全局 UCT 选择轮换。按失败的幕数和楼层分别计数；第 1–2 次失败回退至少 1 层，第 3–4 次至少 2 层，随后扩大到 4、8 层；没有满足距离的候选时使用最近仍有未探索动作的决策。候选不足时自动回到全局 UCT。检查点保留回溯路径、位置失败次数及轮换状态。

模拟中的随机选择只在距离当前最高先验评分不超过 0.75 的动作之间进行。低于 40% 血量、营火、地图当前节点直连 Boss、删牌或消耗选择时，随机概率降为基础值的 20%（默认从 10% 降至 2%）。评分较低的合法动作仍保留在树中供 UCT 探索。价值函数、战斗预算和通关验收流程不变。

使用 `--no-local-repair` 或 `--no-risk-aware-rollout` 可分别关闭一项做对照；两项都关闭恢复 v1 调度。v1 检查点必须在两项都关闭时恢复；v2 恢复必须保持原先的开关组合。不同策略不直接混用树统计。此前五 seed 的 715.06 秒均值属于 v1，不能作为 v2 的实测结果。
