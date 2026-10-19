# solver：CombatSolver worker

`combat_solver_cli/` 在无头引擎中加载第三方 CombatSolver 0.44.0，为示范生成提供战斗决策（[demos](demos.md)）；它看得见隐藏状态，只用于示范生成，不进入正式对局，也不给采样对局出标签。另做轨迹的独立原生重放验证。

## 配置

需要游戏 0.111.0 对应的 `sts2-cli/lib`、.NET 9 运行时、.NET 9 或更新的 SDK，以及官方 CombatSolver 0.44.0 与包含 `compat/0.111.0` 的 RitsuLib。运行时由共享选择器固定在 .NET 9。

```bash
python -m combat_solver_cli configure \
  --solver /path/to/CombatSolver/CombatSolver.dll \
  --dependency-dir /path/to/RitsuLib/compat/0.111.0 \
  --dependency-dir /path/to/RitsuLib/shared
```

该命令只接受清单版本为 `0.44.0` 的求解器，把求解器 DLL 及其清单复制到 `combat_solver_cli/lib/`（固定副本，Steam 自动更新不会改动它），编译 worker，并把本机路径与 solver/game 的 SHA-256 写入 `--config`（默认 `combat_solver_cli/lib/config.json`）。游戏程序集取自 `--lib`（默认 `sts2-cli/lib`）。副本已在 `lib/` 中时可以省略 `--solver`。每次启动 worker 都按配置里的哈希复核两个 DLL，不一致即拒绝。依赖或游戏变化后重新配置并重跑引擎测试。`lib/` 不入库，不要提交第三方 DLL。

## 组成

- `lib/`（不入库）：`CombatSolver.dll`、`CombatSolver.json` 与本机 `config.json`。
- `client.py`：`SolverEngine`，一个原生进程，同一时刻只有一局（开始下一局的条件见[决策协议](../../sts2-cli/docs/decision-protocol.md)）；进程启动、传输出错时抛 `InfrastructureError`（基础设施故障，不是这条分支打坏了），单次请求超时默认 150 s。`step()` 让求解器执行一步战斗（普通/Boss 预算，默认每次搜索 1000 ms、束宽 60、开启 `beam_portfolio`；药水、回合计划复用）。药水：`potions=True` 时按求解器自己的 `Smart` 策略（一瓶药水折成若干点 HP，省不回来就不用），客户端默认 `potions=False` 即 `Disabled`；示范生成的 `battle` 锚点用 `Smart`。`potion_policy` 可直接指定 `Smart`、`Disabled` 或 `RequireAtLeastOne`，`potion_directives` 按栏位指定（`[{slot, potion, directive}]`，`directive` 为 `Force`／`Disabled`／`Smart`，`potion` 是不带 `POTION.` 前缀的 ID）：被 `Force` 的药水必须出现在整场战斗的路线里，找不到这样的路线时求解失败。`potion_costs=[基础, 较高, 最高]` 改 `Smart` 的价格（HP）：求解器自己的是 9／14／18，不传即原价；搜索结果回报所用的价格（`potion_costs`）和路线上药水按该价格合计的代价（`potion_strategic_cost`）。
- `launch.py`：按 `config.json` 设好游戏、求解器与依赖路径，从 `sts2-cli/` 启动 worker；`SolverEngine` 经它起进程。
- `Program.cs` / `SolverAdapter.cs`：worker 主循环、求解请求、当前回合计划复用、真实状态一致性检查；每个决策边界同步求解器状态，出错时重置。
- `ChoiceTransaction.cs` / `NativeSelectionBridge.cs`：统一选择事务（见下）。`PortfolioDeadline.cs`：限时求解的截止处理。`PotionCosts.cs`：药水价格。`AssemblyInfo.cs`：让选择测试工程访问内部类型。CombatSolver 0.44 的三档价格是编译期常量，散在 `PotionUsePolicy` 的四个方法、`CombatSearchCoordinator.MaximumSmartPotionUses` 和插入格挡药水的阈值里；安装时逐处核对常量的个数再改成读取请求的价格，个数对不上（求解器换了版本）就拒绝启动。
- `search_support.py`：动作语义、状态哈希、prefix 记录、公开动作先验（`preference`）。
- `trajectory.py`：独立原生重放验证与导出。

引擎进程的工作目录是 `sts2-cli/`，传给引擎的文件路径必须是绝对路径。

## 独立验收

```bash
python -m combat_solver_cli verify --prefix <prefix.json> --output <新目录>
```

在全新的原生进程中重放整条 prefix，逐步核对公开状态哈希与合法动作，并检查三幕 Boss 与最终胜利。任何原生错误或重放差异都会拒绝导出。产物使用监督数据契约 `source=recorder_bc`、`teacher_visibility=unverified`、`bc_only=true`，胜利证据在 provenance 中；不是策略自己的采样。

当前导出器 `combat-solver-cli-v2` 使用[共用控制规则](env.md#控制分工)：宝箱、水晶球格子及指定单选事件保留原始帧、动作和 actor，写入 `environment_actions`，不成为策略标签。`forced` 只表示引擎只有一个合法候选，与控制器接管分别记录；商店和战斗的策略选择仍保留。

## 统一选择事务

执行层通过 `NativeSelectionBridge` 观察固定游戏版本的 11 个原生选择叶入口，以及无头奖励选择器；找到的叶入口不是 11 个时 worker 启动即失败。候选快照来自游戏已经过滤的集合，不重新过滤或随机抽样；结果在原生选择结束时记录，异步任务返回前也会检查，之后的移牌、升级或抽牌不会破坏对象身份的核销。

`ChoiceTransaction` 是唯一的执行游标：解码动作与回合开始的选择计划，先预留请求，再核对实际对象；显式多步选择与自动选择走同一路径。动作完成和终局帧校验未完成项，原生战斗结束后取消尚未请求的计划项并记录原因；已发生的错选或未完成的请求仍然失败。错误与重置会清理事务。

单卡原生入口中，日志回调的 `[null]` 与异步返回的 `null` 统一解码为 `[]`，覆盖零候选与可跳过选择；多卡结果中的 null 或单卡接口返回多个对象仍会被拒绝。实例序号沿用 CombatSolver 0.44 的 `(CardId, UpgradeLevel)` 分组再验证完整状态键；来源牌堆选择由 `SourceOccurrence` 锁定确切对象。公开协议和学生观察不新增隐藏牌堆信息。新游戏版本、新选择入口或额外奖励替代项需要重新适配。

## 测试

```bash
python -m pytest combat_solver_cli/tests -q
COMBAT_SOLVER_CONFIG=combat_solver_cli/lib/config.json python -m pytest combat_solver_cli/tests -q -m engine
```

`tests/native/` 保存已捕获的原生边界案例（前缀、动作与 C# 回放工具），用作引擎和求解器适配器的回归测试。修改适配器、游戏或依赖后先跑这些测试。

前缀里的 `before_hash` 是整个公开状态加合法动作的指纹。引擎公开的字段变化后动作仍可重放，但哈希全部失效，用 `python combat_solver_cli/tests/native/refresh_hashes.py` 在当前引擎上重放并写回。引擎的随机数消耗变化则会让后续的随机内容（如商店库存）不同，录制的动作本身无法重放，该脚本会报告分歧位置；这种前缀只能在当前引擎上重新捕获同类局面。

## 已知的坑

- `potion_costs` 需要按 `PotionCosts.cs` 改过常量的 worker 构建；旧构建不认识这个字段，会不声不响地按原价搜索。`test_native_potion_costs.py` 必须指向支持它的构建的配置（`COMBAT_SOLVER_CONFIG`）。
- 求解器按毫秒预算搜索：生成战斗时单独运行并固定并发数；并发超过物理核数只会让搜索更浅；新进程里头一两次搜索跑的是未优化的代码，同样的预算展开的节点少得多，所以一局的锚点共用进程。
