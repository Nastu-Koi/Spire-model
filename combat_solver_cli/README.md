# 唯一搜索主线：MCTS＋CombatSolver

教师可以在真实引擎中试走同一种子、恢复并搜索其他行动；整局分支由 MCTS 调度，战斗由 CombatSolver 0.44.0 处理。默认验证范围为 Ironclad A0。学生只学习最终选定路线中每步的公开观察、完整合法动作和动作标签；不输入后续步骤或被撤销的分支。

A* 与独立 `public_*` 规则搜索已退役。搜索、批量生成、定量生成和恢复共用同一个 MCTS 后端；不同工作流不代表不同算法。历史证据见 [归档](../docs/archive/search/)，决定与范围见 [ADR 0003](../docs/adr/0003-unify-backtracking-search.md) 和 [合并规格](../.scratch/search-unification/spec.md)。平均五分钟产出一条成功轨迹仍是待独立验收的性能目标。

## 配置

需要游戏 0.111.0 对应的 `sts2-cli/lib`、.NET 9 运行时、.NET 9 或更新 SDK，以及官方 CombatSolver 0.44.0 与包含 `compat/0.111.0` 的 RitsuLib。运行时由共享选择器固定在 .NET 9，不跨大版本使用 .NET 10；构建可以使用较新的 SDK。

```bash
python -m combat_solver_cli configure \
  --solver /path/to/CombatSolver/CombatSolver.dll \
  --dependency-dir /path/to/RitsuLib/compat/0.111.0 \
  --dependency-dir /path/to/RitsuLib/shared
```

默认配置保存在 `combat_solver_cli/artifacts/config.json`，记录本机路径及 solver/game 哈希。依赖或游戏变更后重新配置；不要提交第三方游戏/Mod DLL。本次官方依赖下载来源与哈希见 [依赖记录](../.scratch/search-unification/evidence/dependency-provenance.md)。本机验证副本位于 `combat_solver_cli/artifacts/dependencies/`，不安装到 Steam。

## 搜索与恢复

```bash
python -m combat_solver_cli search --seed MY-SEED \
  --output combat_solver_cli/artifacts/search-001 \
  --max-seconds 120 --budget-ms 1000 --boss-budget-ms 5000 \
  --search-lanes 1 --reuse-turn-plan
```

输出目录必须不存在。默认 Ironclad A0，rollout 上限 256 个决策；所有原生合法分支保留，启发式先验和风险模拟仅影响调度。`--max-expansions`、`--max-steps` 限制展开和单路线动作数。`--search-lanes 2` 或 `4` 将 MCTS 自身展开的开局分配到同种子的独立树，不调用 A*。暂未证明更多 lane 能降低本机成功成本。

```bash
python -m combat_solver_cli search --seed MY-SEED \
  --resume combat_solver_cli/artifacts/search-001/tree.json \
  --output combat_solver_cli/artifacts/search-002 \
  --max-seconds 120 --search-lanes 1 --reuse-turn-plan
```

恢复须保留角色、难度、lane 数和检查点要求的树策略参数。单树使用 `mcts-tree-v1`，并行协调使用 `mcts-lanes-v2`。旧 A* frontier、旧 `mcts-lanes-v1` 以及旧批次清单明确拒绝，不静默转换。历史记录可通过对应 Git 版本研究；它们不是当前生产入口。

在输出目录创建 `STOP` 可请求动作边界停止。时间预算是协作式软边界，当前原生操作及独立验收可能使结束晚于预算；所有实际耗时必须计入费用。搜索状态、失败分支与 `tree.json` 保留，不把“预算耗尽”解释为无解。

## 批量与定量生成

预先指定不同种子的有限批次，所有作业仍走 MCTS：

```bash
python -m combat_solver_cli batch --jobs combat_solver_cli/jobs.example.json \
  --output combat_solver_cli/artifacts/batch-001 --ascension 0 \
  --max-seconds 120 --reuse-turn-plan
```

或生成随机种子，尝试达到指定成功数量：

```bash
python -m combat_solver_cli.generate_bootstrap \
  --output combat_solver_cli/artifacts/target-001 --target-trajectories 3 \
  --budget-ms 1000 --max-seconds 120 --max-total-seconds 900 --max-attempts 20
```

`--max-seconds` 是单搜索段预算；定量模式的 `--max-total-seconds` 和 `--max-attempts` 限制本次调用，尝试数指搜索段而非独立种子数。该模式在当前种子的树上续跑，达到成功后才生成下一个种子；错误或树耗尽会保留现场并停止，不无限重试。目标未完成可以用 `python -m combat_solver_cli.resume_target --output <原目标目录>` 原地恢复。

有限独立作业批次通过 `python -m combat_solver_cli.resume_bootstrap --previous <原批次目录> --output <新目录>` 恢复未完成的树，保留原策略参数。有限批次中的单作业基础设施错误不会删掉其他作业；定量模式遇此错误停止。

批量/定量生成默认在 stderr 报告启动、等待、每段结果和结束；`--progress-interval` 调整等待提示频率，`--quiet` 关闭。stdout 只输出最终 JSON。summary 记录实际墙钟、结果分类、未成功段成本和每条成功轨迹费用，零成功的费用为 `null`；定量恢复累计此前调用的费用，不把各 worker 的时间之和当作墙钟。

## 固定开发／保留集评测

```bash
python -m combat_solver_cli.evaluate \
  --manifest .scratch/search-unification/evidence/dev-manifest.json \
  --split dev --output combat_solver_cli/artifacts/evaluation-001
```

`search-evaluation-v1` manifest 明确角色、难度、互斥的 dev/holdout 种子、重复次数、每局预算和 MCTS profile。profile 只改变同一后端的参数；旧 `public-evaluation-v1` 不再接受。运行中不调参，依次完成所有预定案例，交错 profile 顺序并保留错误/预算停止；报告源码与原生依赖哈希、版本漂移、每局及整个批次真实墙钟。

成功计数要求读取并核对每局独立重放产物、角色/种子/难度、三幕 Boss、最终胜利声明及原始重放 SHA-256；单有 `verified_victory` 状态不足以通过评测。每个案例独立保留 `accepted.jsonl`，汇总只列 `accepted_paths.json`，避免把重复评测误拼成重复训练样本。重复同一 seed 不增加独立样本数。保留集不得用于调参。

需要分解瓶颈时，使用 `python -m combat_solver_cli.profile_search --seed <seed> --seconds 60 --output <新目录>`；它区分实际求解次数、计划/选择执行和恢复耗时。并行 worker 的耗时会重叠，不据此直接计算总墙钟占比。

## 训练数据与独立验收

每条候选路线必须由全新原生进程重放，逐步核对公开观察及合法动作，并再次通过三幕 Boss 和最终胜利检查。任何原生错误或重放差异都会拒绝导出。失败重放留下 `failed_replay_trace.jsonl` 与 `verification_error.json`，不混入训练。

产物继续使用 `source=recorder_bc`、`teacher_visibility=unverified`、`bc_only=true`。顶层 `status=partial`、`victory=null` 是既有 recorder 监督数据契约；实际完整胜利证据保存在 provenance，不表示搜索只完成了半局。不要改标为公开教师 demonstration，也不能拿它充当 PPO on-policy 数据。Bootstrap 使用现有 `python -m model ... bootstrap --data <accepted.jsonl>` 入口。

## 开发边界与验证

- `mcts.py` / `mcts_lanes.py`：唯一搜索与开局分片。
- `search_support.py`：动作语义、公开摘要、历史前缀、原生重放与规则先验，不包含 A*。
- `client.py` / `SolverAdapter.cs`：原生求解、当前回合计划复用和真实状态污染检查。
- `trajectory.py`：独立验收；`batch.py` / `target.py`：采集工作流；`evaluate.py`：冻结评测。

纯 Python 回归：`python -m pytest combat_solver_cli/tests -q`。原生首战检查：`python -m combat_solver_cli smoke --characters Ironclad --ascension 0 --reuse-turn-plan`。本次具体验证结果及未验证项统一见 [迁移证据](../.scratch/search-unification/evidence/merge.md)。

冻结测试尚未达到五分钟成功产出目标，放宽预算后已有独立重放验收成功的路径；实验条件、测量和限制分别见[性能报告](../.scratch/search-unification/evidence/performance.md)与[可行性验证](../.scratch/search-unification/evidence/extended-search.md)。
