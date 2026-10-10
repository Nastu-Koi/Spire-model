# CombatSolver worker

在无头引擎中加载第三方 CombatSolver 0.44.0，为战斗示范和战斗结果数据提供战斗决策，并对轨迹做独立原生重放验证。

## 配置

需要游戏 0.111.0 对应的 `sts2-cli/lib`、.NET 9 运行时、.NET 9 或更新的 SDK，以及官方 CombatSolver 0.44.0 与包含 `compat/0.111.0` 的 RitsuLib。运行时由共享选择器固定在 .NET 9。

```bash
python -m combat_solver_cli configure \
  --solver /path/to/CombatSolver/CombatSolver.dll \
  --dependency-dir /path/to/RitsuLib/compat/0.111.0 \
  --dependency-dir /path/to/RitsuLib/shared
```

该命令把求解器 DLL 及其清单复制到 `combat_solver_cli/lib/`（固定副本，Steam 自动更新不会改动它），编译 worker，并把本机路径与 solver/game 的 SHA-256 写入 `combat_solver_cli/lib/config.json`（默认配置）。副本已在 `lib/` 中时可以省略 `--solver`。依赖或游戏变化后重新配置并重跑引擎测试。`lib/` 不入库，不要提交第三方 DLL。

## 组成

- `lib/`（不入库）：`CombatSolver.dll`、`CombatSolver.json` 与本机 `config.json`。
- `client.py`：`SolverEngine`，一个原生进程，同一时刻只有一局（开始下一局的条件见[决策协议](../sts2-cli/docs/decision-protocol.md)）；`step()` 让求解器执行一步战斗（普通/Boss 预算、药水、回合计划复用）。药水默认按求解器自己的 `Smart` 策略（一瓶药水折成若干点 HP，省不回来就不用）；`potion_policy` 可改为 `Disabled` 或 `RequireAtLeastOne`，`potion_directives` 按栏位指定（`[{slot, potion, directive}]`，`directive` 为 `Force`／`Disabled`／`Smart`，`potion` 是不带 `POTION.` 前缀的 ID）：被 `Force` 的药水必须出现在整场战斗的路线里，找不到这样的路线时求解失败。
- `Program.cs` / `SolverAdapter.cs`：worker 主循环、求解请求、当前回合计划复用、真实状态一致性检查；每个决策边界同步求解器状态，出错时重置。
- `ChoiceTransaction.cs` / `NativeSelectionBridge.cs`：统一选择事务（见下）。`PortfolioDeadline.cs`：限时求解的截止处理。
- `search_support.py`：动作语义、状态哈希、prefix 记录、公开动作先验（`preference`）。
- `trajectory.py`：独立原生重放验证与导出。

引擎进程的工作目录是 `sts2-cli/`，传给引擎的文件路径必须是绝对路径。

## 独立验收

```bash
python -m combat_solver_cli verify --prefix <prefix.json> --output <新目录>
```

在全新的原生进程中重放整条 prefix，逐步核对公开状态哈希与合法动作，并检查三幕 Boss 与最终胜利。任何原生错误或重放差异都会拒绝导出。产物使用监督数据契约 `source=recorder_bc`、`teacher_visibility=unverified`、`bc_only=true`，胜利证据在 provenance 中；不能当作 PPO 的在线数据。

当前导出器 `combat-solver-cli-v2` 使用[主策略的共用控制规则](../model/README.md#策略与控制器)：宝箱、水晶球格子及指定单选事件保留原始帧、动作和 actor，写入 `environment_actions`，不成为策略标签。`forced` 只表示引擎只有一个合法候选，与控制器接管分别记录；商店和战斗的策略选择仍保留。

## 统一选择事务

执行层通过 `NativeSelectionBridge` 观察固定游戏版本的 11 个原生选择叶入口，以及无头奖励选择器。候选快照来自游戏已经过滤的集合，不重新过滤或随机抽样；结果在原生选择结束时记录，异步任务返回前也会检查，之后的移牌、升级或抽牌不会破坏对象身份的核销。

`ChoiceTransaction` 是唯一的执行游标：解码动作与回合开始的选择计划，先预留请求，再核对实际对象；显式多步选择与自动选择走同一路径。动作完成和终局帧校验未完成项，原生战斗结束后取消尚未请求的计划项并记录原因；已发生的错选或未完成的请求仍然失败。错误与重置会清理事务。

单卡原生入口中，日志回调的 `[null]` 与异步返回的 `null` 统一解码为 `[]`，覆盖零候选与可跳过选择；多卡结果中的 null 或单卡接口返回多个对象仍会被拒绝。实例序号沿用 CombatSolver 0.44 的 `(CardId, UpgradeLevel)` 分组再验证完整状态键；来源牌堆选择由 `SourceOccurrence` 锁定确切对象。公开协议和学生观察不新增隐藏牌堆信息。新游戏版本、新选择入口或额外奖励替代项需要重新适配。

## 测试

```bash
python -m pytest combat_solver_cli/tests -q
COMBAT_SOLVER_CONFIG=combat_solver_cli/lib/config.json python -m pytest combat_solver_cli/tests -q -m engine
```

`tests/native/` 保存已捕获的原生边界案例（前缀、动作与 C# 回放工具），用作引擎和求解器适配器的回归测试。修改适配器、游戏或依赖后先跑这些测试。

前缀里的 `before_hash` 是整个公开状态加合法动作的指纹。引擎公开的字段变化后动作仍可重放，但哈希全部失效，用 `python combat_solver_cli/tests/native/refresh_hashes.py` 在当前引擎上重放并写回。引擎的随机数消耗变化（例如商店药水价格改为原生浮动）则会让后续商店的库存不同，录制的动作本身无法重放，该脚本会报告分歧位置；这种前缀只能在当前引擎上重新捕获同类局面。
