# 公开信息搜索与成功轨迹生成

第一阶段针对铁甲战士 A0，目标是在本机允许使用全部线程的条件下，平均每五分钟生成一条通过独立重放验证的成功轨迹。失败、提前放弃、初始化、验证和导出都计入产出成本。该性能目标尚未实测达成。

## 当前实现

`combat_solver_cli/public_search.py` 的规划器只接收公开实体和完整合法动作，不接收游戏 seed、原生引擎或随机数状态。实际执行与搜索隔离，每次只执行一个原生候选，再从新观察规划。

战斗使用有限深度束搜索。每个根动作都有评分，束宽只限制假设的后续分支。生产规划器与效果诊断共用 `CombatModel`：检查原有 17 种卡牌的伤害、格挡、自损、力量与易伤，并补充 Second Wind 按实际消耗手牌计格挡、Spite 按本回合公开损血事实判定连击。Anger 的生成副本、Rampage 的永久增长等未完整建模部分保持近似叶节点。具体覆盖与原生证据见[01–05 验证记录](../.scratch/public-search-repair/evidence/01-05.md)。

常见伤害、格挡、虚弱、力量药水可参与当前战斗收益比较，药水库存也随假设使用消耗。补能会产生根候选之外的新合法动作，因此补能仍在叶节点截断，执行后向原生引擎取得新的完整合法候选。预测分为已建模、近似和未知；未支持的遗物、状态或卡牌修改会触发显式边界，不能把面板参数猜测当作精确效果。

抽牌、其他未知效果和回合结束仍作为叶节点估值，不从真实引擎试走未来，也不把抽牌堆排列当作牌序。未知动作仍参与选择，原生执行后重新规划。当前版本不包含隐藏状态条件采样、跨回合信息集 MCTS 或 POMCP；局部转移正确不等于已具备通关能力。

地图和局外选择复用公开地图 DAG、牌组、血量和资源规则。它们目前是启发式选择，不模拟后续商店库存、奖励和遭遇。只在地图边界、血量不超过上限 8%、没有药水、且所有可走节点都是战斗时提前放弃；这是可关闭的产出优化规则，不代表该局不可通关。

## 运行

需要已构建、与本机游戏版本一致的 `sts2-cli` 原生引擎及 .NET 9 运行时；不需要 CombatSolver 或神经网络检查点。Harmony 与本机 .NET 10 不兼容，不能仅依赖引擎的跨主版本 roll-forward。生成与验证本身只用 Python 标准库，张量训练路径仍需要项目声明的 PyTorch 等依赖。

```bash
python3 -m combat_solver_cli.public_generate \
  --output combat_solver_cli/artifacts/public-a0-001 \
  --target-trajectories 5 --max-seconds 1500 --max-attempts 100 \
  --depth 3 --beam-width 16
```

输出目录必须不存在。默认 worker 数为本机逻辑 CPU 数，不同 worker 执行不同的新 seed，每个 worker 持有自己的原生进程。线程数不是加速倍数保证，Python 规划还受解释器并发限制；需要通过本机吞吐实测决定最佳 worker 数。用 `--workers 1` 可做串行对照，`--no-abandon` 可禁用提前放弃。

本次会话将 .NET 9.0.7 运行时安装在临时目录 `/tmp/spire-model-dotnet9`，未修改系统 .NET。若该目录仍存在，运行命令前设置 `export PATH="/tmp/spire-model-dotnet9:$PATH"`；长期使用应准备持久的 .NET 9 环境。

`--max-seconds` 是整个批次的运行预算，不是每局预算或五分钟产出保证；`--max-steps` 限制单局真实动作数。边界检查与原生 I/O 超时可能使实际结束晚于预算。达到目标或发生基础设施错误后停止发起新任务，并让其他 worker 在动作边界停止；已有成功数据和失败诊断保留。

## 固定种子对照

```bash
python3 -m combat_solver_cli.public_evaluate \
  --manifest .scratch/public-search-repair/evidence/01-05-manifest.json \
  --split dev --output combat_solver_cli/artifacts/evaluation-001
```

manifest 显式指定开发／保留种子、策略配置、独立采样随机源、重复次数、预算与停止规则。`legacy`、`second_wind`、`spite`、`cards`、`potions` 为逐项累积修复的对照配置；默认规划器采用 `potions`。调试用开发集，保留集不参与调参。运行器报告重复差异、分类错误、用药、战斗净耗血及效果误差；敌人跨帧身份不明确时不比较其血量，未知预测不算吻合。

## 验收与输出

只有原生终局胜利、三幕 Boss 里程碑齐全，并且全新原生进程从相同 seed 独立重放动作后再次得到一致结果，才导出训练样本。独立重放仅验证已选动作，不反向修改教师决策。原生 `[ERROR]`、协议异常、重放分歧、预算耗尽和正常死亡分开记录。

`accepted.jsonl` 使用现有 `demonstration` / `teacher_visibility=public` schema，可以交给现有 bootstrap 命令；它不是 PPO on-policy 数据。`summary.json` 报告验收成功数量、失败/放弃/错误以及整个批次墙钟时间除以验收成功数量；零成功时平均成本为 `null`。逐局证据与诊断保存在输出目录，不把失败轨迹混入成功训练集。训练和验证仍应按 seed 隔离。

逐局 `attempts/*/decisions.jsonl` 保存每次公开决策帧、动作及所有根候选评分，`result.json` 保留结果、最后决策状态和规划/验证耗时。成功后的 `evidence.json` 保存原始与独立重放的终局、三幕里程碑，导出记录其 SHA-256。`accepted-parts` 按局保留通过验收的训练样本，即使之后遇到错误也不会丢弃先前成功数据。

## 当前验证状态

2026-09-26，本机 M4 Pro（14 核、24 GB）、游戏 0.111.0 的修复前 14 worker 基线实际耗时 301.68 秒：428 次尝试中 414 次战败、14 次预算停止，零协议错误、零验收成功。五分钟成功产出目标尚未达成。条件伤害与消耗格挡修复的原生对照、固定开发集结果及剩余覆盖限制统一记录在[01–05 验证记录](../.scratch/public-search-repair/evidence/01-05.md)。

后续目标与分阶段验收统一维护在[修复规格](../.scratch/public-search-repair/spec.md)。实施顺序与依赖见[工单索引](../.scratch/public-search-repair/README.md)。逐轮测试、原生故障修复与回归结果见[历史实验记录](archive/search/PUBLIC_SEARCH_BASELINE_2026-09-26.md)；旧固定种子搜索的成绩不能作为公开信息教师的成绩。

## 测试

以下测试不需要本地游戏：

```bash
python3 -m unittest discover -s combat_solver_cli/tests -v
```

测试覆盖公开信息隔离、合法候选完整性、局部卡牌组合、未知未来截断，以及使用可控引擎检查成功重放、失败、验收拒绝和预算处理。这些测试不证明规则教师具备完整通关能力。
