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
- `client.py`：`SolverEngine`，一个原生进程，同一时刻只有一局（开始下一局的条件见[决策协议](../../sts2-cli/docs/decision-protocol.md)）；进程启动、传输出错时抛 `InfrastructureError`（基础设施故障，不是这条分支打坏了），单次请求超时默认 150 s。`step()` 让求解器执行一步战斗（普通/Boss 预算，默认每次搜索 1000 ms、束宽 60、开启 `beam_portfolio`；药水、回合计划复用）。药水：`potions=True` 时按求解器自己的 `Smart` 策略（一瓶药水折成若干点 HP，省不回来就不用；实际做法与偏差见下文 [Smart 用药](#smart-用药)），客户端默认 `potions=False` 即 `Disabled`；示范生成的 `battle` 锚点用 `Smart`。`potion_policy` 可直接指定 `Smart`、`Disabled` 或 `RequireAtLeastOne`，`potion_directives` 按栏位指定（`[{slot, potion, directive}]`，`directive` 为 `Force`／`Disabled`／`Smart`，`potion` 是不带 `POTION.` 前缀的 ID）：被 `Force` 的药水必须出现在整场战斗的路线里，找不到这样的路线时求解失败。`potion_costs=[基础, 较高, 最高]` 改 `Smart` 的价格（HP）：求解器自己的是 9／14／18，不传即原价，最低 1（价格 0 会让 `MaximumSmartPotionUses` 除零）；搜索结果回报所用的价格（`potion_costs`）和路线上药水按该价格合计的代价（`potion_strategic_cost`）。`final_boss_potion_costs` 只在整局最后一场战斗（求解器判为 `RunEnding` 的那场：A0 第三幕的 Boss、A10 第三幕的第二个 Boss）取代 `potion_costs`，并让那一场按完整 HP 比较（`BossHpStrategy.MinimizeHpLoss`）；不传时那一场沿用求解器自己的做法，只为赢才用药。回复的 `final_boss_priced` 说明这次搜索是否按它定价。`reuse_route=True`（默认关）时，新回合开始前先用求解器自带的续用核对：上一次完整搜索（不含截止时只给出当前回合的结果）为每个后续回合开头预测了状态戳（`ContinuationStamp`），实际状态与某一回合的预测完全相同，就照原路线执行这一回合，不再搜索；回复的 `search` 带 `reused_from_turn`（找到这条路线的搜索从哪一回合开始），`expanded_nodes` 为 0。续用只在同一场战斗、同一组请求参数内发生。再加 `reuse_within_budget=True` 时只续用在预算内跑完的搜索给出的路线：worker 在预算到点时请求接管，搜索返回时还没有接管请求，说明无药搜索和用药层都没被截止；回复的 `within_budget` 标明最近一次搜索是否如此。两者都默认关闭，示范生成不用。实测（48 局、约 1050 场，`beam_portfolio=false`、药价 1／2／3）：全部续用每场耗时减半（15.1 → 7.8 s），胜场不变，但赢下的战斗平均掉血 5.47 → 6.18、每场用药 0.30 → 0.17 瓶；只续用预算内跑完的路线耗时只省 22%（11.7 s），掉血 6.12、用药 0.21 瓶，损失接近全部续用。逐回合重搜每回合都重新判断用药，剩余回合越少搜得越深，它的价值不只在补救被截止的搜索。
- `launch.py`：按 `config.json` 设好游戏、求解器与依赖路径，从 `sts2-cli/` 启动 worker；`SolverEngine` 经它起进程。
- `Program.cs` / `SolverAdapter.cs`：worker 主循环、求解请求、当前回合计划复用、可选的跨回合续用（`reuse_route`）、回合准备选牌（见[下文](#回合准备选牌)）、真实状态一致性检查；每个决策边界同步求解器状态，出错时重置。
- `ChoiceTransaction.cs` / `NativeSelectionBridge.cs`：统一选择事务（见下）。`PortfolioDeadline.cs`：限时求解的截止处理。`PotionCosts.cs`：药水价格。`PotionReserve.cs`：给用药层预留时间（见 [Smart 用药](#smart-用药)）。`BlockPotionInsertion.cs`、`FinalReplay.cs`：求解器选定路线之后的两处检查，原样会让整次搜索失败（见[选定路线的检查](#选定路线的检查)）。`AssemblyInfo.cs`：让选择测试工程访问内部类型。CombatSolver 0.44 的三档价格是编译期常量，散在 `PotionUsePolicy` 的四个方法、`CombatSearchCoordinator.MaximumSmartPotionUses` 和插入格挡药水的阈值里；安装时逐处核对常量的个数再改成读取请求的价格，个数对不上（求解器换了版本）就拒绝启动。
- `search_support.py`：动作语义、状态哈希、prefix 记录、公开动作先验（`preference`）。
- `trajectory.py`：独立原生重放验证与导出。

引擎进程的工作目录是 `sts2-cli/`，传给引擎的文件路径必须是绝对路径。

## Smart 用药

CombatSolver 0.44 的 `Smart`（无强制药水时）不是在所有用药方案里挑"省下的 HP − 药价"最大的，而是：

1. 先不许用药搜一遍，得到无药路线；它完整胜利且零掉血就直接用它。
2. 无药路线获胜、栏里有格挡药水时，先试把格挡药水插到第一个掉血 ≥ 基础价的回合末，原始 HP 省够基础价就采用并结束，不再看别的药水（不乘下面的 Boss 倍率）。
3. 否则在剩余预算里逐层搜"恰好 k 瓶"（k 从 1 起，上限按无药路线的掉血与基础价估），每层一次新的束搜索；层内按掉血选路线，药价只作末位比较；该路线省下的 HP 不少于所用药水的价格合计就采用，**第一个通过的层即停**。无药路线没赢时，能赢的用药路线一律接受。
4. Boss 倍率：第一、二幕的 Boss 战后回血，药价 ×5；整局最后一场药价无穷大（只为赢才用药），见上文 `final_boss_potion_costs`；A10 第三幕的第一个 Boss 不乘倍率。

预算是整次搜索共用的：用药层只拿"预算 − 已用"。无药部分占满预算的有两处：束宽组合（`beam_portfolio`，`false` 时只跑一个宽度；48 局实测，5000 ms 下进了用药层的普通／精英战搜索被截止打断的比例由约六成降到约四成）；牌组里有能力牌时的能力路线组合（`RunOpeningPowerRoutePortfolio`）。后者每个成员最多 10 s，共 4–20 个成员，没有总时限，一直跑到 worker 在预算处请求接管；这时执行的是正在跑的那个成员的当前回合，用药层一层都没开始。

`potion_reserve`（预算的比例，0–0.9，默认 0）给用药层预留时间，只在 `Smart`、没有强制药水、且腰带有药时生效（`PotionReserve.cs`）。设预算 B、预留 r：

- 无药部分的每次束搜索（`CombatBeamSolver.Solve` 前缀）软时限截到 B·(1−r)，到点按软时限停下、保留已找到的路线，而不是被取消。
- 过了 B·(1−r)，束宽组合与能力路线组合不再开新成员（`CanFinishTargetPortfolio` 后缀）。
- 求解器自己的预算仍是 B，用药层照常拿到"B − 已用"；用药层的束搜索软时限截到 B − 300 ms，赶在 worker 接管之前停下。用药层从第一个审计（`AuditRequiredPotionUse` 前缀）算起；`RunSupplementalAudits` 本身带异常筛选器，Harmony 重新生成它会出错，不能打补丁。

worker 在 B 处请求接管的计时不变。回复的 `potion_reserve_ms` 是这次搜索实际预留的毫秒数（没生效为 0），请求了预留的流水线据此确认 worker 认识这个字段。用药层每层仍是一次完整的束搜索，只接受打完并获胜的路线。

## 选定路线的检查

求解器选定路线后（`MaterializeSelectedRoute`）还有两步，CombatSolver 0.44 里出错都不接，整次搜索失败（第 0 轮正式示范 2,000 场里的 5 场确定性失败）：

- 上文第 2 步补格挡药水（`TryInsertBlockPotion`）要重放补药后的路线。补药改了之后回合的抽牌时，路线计划的选牌在重放里找不到，抛 `InvalidPlannedChoiceBranchException`。搜索里这个异常一律当剪枝吞掉，只有这里没人接。`BlockPotionInsertion.cs` 的 finalizer 把它当作补药不成立，保留原路线（stderr `block_potion_insertion_skipped`）。
- 路线从根重放一遍做遗物标注，重放的结束状态与搜索的不同就失败。打赢的路线里，重放会比搜索多算末个敌人的死亡：每有生物死亡就减费的牌（Melancholy）在消耗堆里结束时少 1 费。`FinalReplay.cs` 在路线打赢时不再要求状态键相同（stderr `final_replay_key_differs_in_won_route`），HP、敌人 HP 与边界仍须相同；其余情况照旧失败。

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

### 回合准备选牌

回合开头由遗物、能力触发的选牌（赌博筹码、烤火手套、工具箱、选择悖论，Tools of the Trade 等能力，带 Imbued 附魔的牌；以求解器自己的 `PlayerTurnSetupCoordinator.RequiresSolverChoice` 为准）由求解器搜索决定，对应游戏里 mod 的回合准备接管（worker 不调 mod 的 `Entry.Initialize`，mod 自己的补丁都不装，这一条是移植过来的）：

- 上一回合的结束回合动作带了下一回合开头的选择（`TurnStartChoices`）时，照这些选择执行。
- 否则（第 1 回合，或路线没预测到）在 `CombatManager.SetupPlayerTurn` 的前缀里、原版准备运行之前取求解器根快照，并激活一个计划待定的事务：原生请求照常登记、候选与来源牌堆照常快照，只是暂不对位。原生选牌在进房间的命令里就已开始，那时 worker 还不知道预算和药价。
- 下一次 `solver_step` 遇到选牌时，用这个根快照和 `IncludeTurnSetup` 按本次请求的参数搜索，把结果的 `TurnSetupChoices` 交给事务对位，之后照常回答；回复的 `search` 带 `turn_setup: true`。出牌阶段照常重搜，不沿用这次搜索的出牌路线。
- 求解器按真实游戏模拟回合准备：第一个选牌一暂停就先跑 side 开始效果，引擎也按这个时机暂停（见 [env](env.md#已知的坑)），所以准备里后面的选择照常对得上。仍对不上时（回合准备的选择，即这个事务或上一回合结束动作带的 `TurnStartChoices`，计划的牌不在原生候选中），事务不报错，记为偏离（stderr `turn_setup_diverged`，带原生候选）；`solver_step` 遇到它的下一个选牌界面时放弃事务，回 `solver_selection_required`，这一选由流水线的规则回答（锚点里该行的 actor 为 `combat_selection_rule`），之后照常重搜。其它对不上（打出的牌带的选择、来源牌堆、效果等）照旧失败。
- 准备阶段没有要回答的选牌时，事务直接结束；还没交给求解器就被重置的事务（重放进程、锚点流水线的兜底）被放弃，之后的原生回调不再报错。
- 求解器只许在游戏主线程取根快照，而游戏把第一个问它的线程认作主线程，所以 worker 在 `PrepareEngine` 里先问一次。
- `resume_save`、`resume_fight` 在命令里就续局或开始战斗，原生选牌在命令返回前已经发生，所以 worker 在这两个命令之前也调 `PrepareEngine`（之后照常再调一次）；否则它们作为 worker 的第一个命令时补丁还没装，药价不生效、回合准备不接管。
- 无头引擎开局时给中和（Neutralize）装的旧补丁（`LocPatches.NeutralizePrefix`）会让求解器的补丁审计拒绝取根快照。`PrepareEngine` 撤掉它，但 `resume_fight` 在命令里开局时又装上一次，所以取回合准备的根快照之前再撤一次（`RestoreNativeNeutralize`）。
- 以下情况不接管，stderr 记 `turn_setup_not_taken reason=…`，选牌照旧处理（没有计划时回 `solver_selection_required`）：回合准备跑在别的线程上（`off_main_thread`）；上一回合的事务还有没做完的计划选择（`unsettled_choices`）；第 1 回合持有低语耳环（`whispering_earring`：游戏代打第一回合，代打时的选择是游戏自己的，求解器也没法搜索以它结尾的回合准备）；回合准备搜索出错（`search_failed`，求解器隔离被破坏时除外）。

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
