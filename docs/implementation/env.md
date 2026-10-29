# env：引擎接口、控制分工、存档与种子

`main/env/` 是最底层，不依赖 `main` 的其它子包。设计见 [architecture](../architecture.md#环境步骤与控制分工)。

| 模块 | 内容 |
| --- | --- |
| `protocol.py` | 决策帧的校验、动作语义、执行命令、`segment_key`（多选的解码段）、角色表 |
| `control.py`、`treasure_rule.py`、`crystal_rule.py` | 策略与控制器的分工、宝箱流程、水晶球规划器 |
| `rewards.py` | 奖励账本：只在终局支付 ±1 |
| `saves.py` | 续局存档（`continue_save`）、战斗重进命令、战斗边界与各场战斗的结局（`FightTracker`） |
| `seeds.py` | `split_group`、`seed_group` 与留出划分的身份，评估种子的隔离检查；采样轮的种子与难度分配（由种子根与轮号派生：`round_seeds`、`round_ascensions`）与要避开的种子（`excluded_seeds`） |
| `public_history.py` | 公开历史契约的版本与 unknown 记忆 |
| `engine.py`、`dotnet_runtime.py` | 无头引擎进程（`CliEngine`）与 .NET 运行时选择 |
| `runtime.py`、`processes.py` | 资源设置与可复现性清单；随父进程结束的子进程、结束进程组、归还空闲堆内存 |
| `clock.py` | 进程内按名字累计的耗时与次数（各线程各记各的，读两次取差）；引擎回复的等待与解析在这里记，采样与采集进程的分项见 [play](play.md) |

## 契约版本

| 契约 | 当前版本 |
| --- | --- |
| 原生公开观察 | `public-state-v7` |
| 公开历史 | `public-history-v1` |
| 模型输入编码 | `public-encoding-v3`（`main/encoding/representation.py`） |
| 策略／控制器分工及索引 | `public-control-v1` |
| 奖励 | `outcome-v1` |
| 对局记录 | `spire-trajectory-v1`（`main/env/protocol.py`） |
| 控制器规则 | 宝箱 `treasure-open-take-leave-v1`、水晶球 `crystal-public-geometry-v3`、单选事件 `single-event-without-usable-potion-v1`、领奖 `claim-free-rewards-v1` |

缺失旧历史时输入为 unknown，不补初始概率。旧版本采样的对局不能只改版本号后使用。

## 控制分工

`control.py` 为运行、导入、索引和训练提供同一套判断：

- 普通宝箱（宝箱节点与问号房间里的）进入 `treasure` 阶段，统一执行 `treasure-open-take-leave-v1`：开箱 → 领取遗物 → 离开；出现多个可选遗物时报错，不擅自选第一个。遗物触发的选牌仍交给策略。
- 事件仅有一个可用 `CHOOSE_EVENT_OPTION`、其余合法动作仅为 `DISCARD_POTION`（或没有药水动作）时自动执行该选项；有可用药水、第二个事件选项或其他实质动作时交给模型，锁定选项不算可用。
- 水晶球：主模型选支付方案，规划器（`crystal-public-geometry-v3`）接管格子。第一次点击固定用 Big 工具点正中间 (5, 5)：开局没有任何碎片信息，搜索只会在空先验上打分；“第一次”由公开状态判定，即已揭示格子全部位于原生开局就已揭示的四角区域（每角 6 格），因为每次点击的中心必是未揭示格。之后每帧按已揭示格子、剩余点击和公开几何规则生成相容的矩形布局假设，按剩余预算做束搜索选格子和 Small／Big 工具；它不读真实布局或游戏随机数，近似搜索用独立随机源。开局候选缺失或棋盘预揭示更多格子时直接走搜索。支付和领奖照常训练，不加规划器评分奖励。
- 商店仅离开＋丢药、战斗仅结束回合＋丢药仍是策略决策。
- 领奖控制器（`free_reward`，规则 `claim-free-rewards-v1`）不属于上面的共用分工，只在 `evaluate --claim-rewards` 时启用：奖励界面上的金币、遗物和药水栏放得下的药水按这个顺序先领，选牌、删牌、成组奖励、放不下的药水和离开仍交给策略。示范按“这些奖励已在选牌前领取”重建，打开它让策略遇到的奖励界面与示范一致。

控制器步骤不产生策略损失，作为 `environment_actions` 保留原始帧、动作与 actor。采样与训练只接受 `control_version=public-control-v1` 且宏动作里没有控制器步骤的轨迹。控制规则变了，示范不必重新生成：分片保存了完整候选，对旧分片执行 `reindex`（见 [data](data.md)）。

非战斗阶段的可取消选择不发布 `CANCEL`（`selection_cancel = "combat-or-dead-end-v1"`）；没有任何可选牌、无法完成时仍提供，避免死局。战斗中保持原生取消语义，因为 CombatSolver 在“计划不选任何牌”时依赖它。Steam 端用同一规则（`main.steam.cancel_offered`）。

## 奖励

基础奖励只在一局结束时支付一次：通关 +1，没通关 −1。第三幕 Boss 倒下和对局完成两种终局标记只付同一份 +1；失败由对局完成事件确认，引擎没发这个事件的按终局状态支付。过幕、普通战与精英战胜利都是 0，账本记下已击败的 Boss 只用于报告。超时、引擎错误和未完成的对局不伪造终局，不进缓冲。

## 入场存档

记战况的段（`iterate` 的幕组都是）在每场不是事件引发的战斗的第一个决策（`events` 带 `encounter_started`）让引擎 `write_continue_save` 到临时文件再读回（`saves.continue_save`），见 [play](play.md#幕组) 的战况与入场存档。在房间里时引擎存的是进房前的状态（节点从已访问坐标里退回、房间清空；战斗中的存档见下段）；`anchor_state` 序列化的是已进房的状态，不能用。续打用一条命令 `resume_fight`（`saves.fight_resume_command(row)`：存档、遭遇、楼层、节点类型、是否第二 Boss），引擎加载后把节点记为已访问、标训练可用（契约 `initialization=fight-resume-v1`）并进入遭遇，回复就是战斗的第一个决策帧（`CliEngine.resume_fight`）。重进的战斗与原战斗开局相同：HP、最大 HP、金币、幕、楼层、卡组、遗物及其计数、药水、敌人和它们的随机 HP、开局手牌都一样，动作相同就逐帧相同，是重放；只有公开历史里“上一个节点有没有商店”在加载的对局里是未知。

战斗进行中写的存档是进这个房间时的整局状态：`RoomEntrySave` 在 `RunManager.EnterRoom` 进战斗房间之前把整局序列化一份（游戏自己选图进房时也在进房之前存档，无头引擎不存），`write_continue_save` 在战斗进行中写这一份，再把节点退回。开战时对整局的改动（Blood Vial 与 Fake Blood Vial 第一回合回血、Royal Poison 扣血、Maw Bank 进房加金币、Petrified Toad 给药水、Ember Tea 与 Happy Flower 计数）和敌人 HP、洗牌所用的随机数都不在存档里，重进时各发生一次。加格挡、加能量、抽牌（Anchor、Lantern、Bag of Preparation 等）只作用于本场战斗，存档里本来就没有，每次进入各生效一次。战斗结束之后（奖励、战后选图）、其它房间、以及不经过 `EnterRoom` 的事件引发的战斗里，写的仍是当时的状态。序列化不消耗随机数、不改状态，每场战斗不到 1 ms。

幕开头存档写在第 0 层的选图决策，用 `resume_save` 续打（`CliEngine.resume`），见 [play](play.md#幕组)。

`FightTracker` 按引擎随帧发来的 `encounter_started`／`encounter_completed` 划出战斗：一场战斗从开始事件起，到胜利事件之后的第一个决策或对局结束为止。它同时记每场的种类（普通、精英、Boss）和对策略而言的结局：胜、负（对局在战斗中以失败结束），或未完（被超时、引擎错误截断或段停在切点）；`tally` 是对局记录与评估报告里的 `fights`。

## 种子与划分

`split_group(seed)`：原始 seed 去首尾空格、转大写、O/I 归一为 0/1 后取 SHA-256。`seed_group(run)` 是统一分组键：索引行带 `split_group` 时直接用；独立监督样本的 `seed` 已是这个摘要，不再哈希；其余对局按原始 seed 计算。划分版本 `seed-group-v1`。

`validate_evaluation_seeds` 要求五个角色各有一组不重复的评估种子，并与检查点的训练种子不相交：采样对局按原始 seed 记、独立监督样本按分组摘要记，所以训练种子按原样和取 `split_group` 之后两种形式都比对。

## 资源设置

`configure_runtime` 在每个命令开头执行：固定 PyTorch 线程数，关掉 TF32（矩阵乘与 cuDNN），使用 GPU 时把本进程的显存上限设为 min(84 GiB, 空闲显存 − 8 GiB)（`gpu_budget`；训练机 95.6 GiB、桌面约占 2.7 GiB 时取到 84 GiB，8 GiB 余量留给桌面与本进程在分配器之外的显存），余量不足 8 GiB 直接报错；返回 Python、PyTorch、CUDA、显卡等的可复现性清单，交给各命令。

`processes.end_with` 让工作进程在父进程消失时连同自己的进程组一起结束（父进程被杀不会做清理，否则读取进程会一直占着训练的显存与输出锁、采样进程占着引擎）；`end` 先对进程组发 SIGTERM、宽限后 SIGKILL；`release_memory` 调 `malloc_trim` 把 C 堆的空闲页还给系统，采样进程每次清掉解析缓存之后调用。

## 引擎进程

`CliEngine` 启动 `sts2-cli` 的无头引擎；工作目录是 `sts2-cli/`，传给引擎的文件路径必须是绝对路径。引擎在原进程里开下一局（先按游戏返回主菜单的方式结束上一局）；结束不了（停在战斗里、出过错）时换新进程；打满 `RUNS_PER_PROCESS` 局后换新，避免内存缓慢上涨。

引擎进程默认关掉 .NET 运行时按剖析重新编译与后台垃圾回收（`engine.RUNTIME_SETTINGS`：`DOTNET_TieredPGO=0`、`DOTNET_gcConcurrent=0`）。一轮的引擎占满整机核心时，运行时自己的这部分工作也在和对局抢 CPU；关掉后每步引擎 CPU 少约五分之一，同一存档、同一串选择下到达的公开状态逐帧相同。进程环境里显式设的值优先。

`CliEngine.reseed(seed)`（引擎命令 `reseed`，`RunSimulator.Reseed.cs`）在当前决策上[重新播种](../../CONTEXT.md)：局的 12 个随机流与每个玩家的 3 个，各换成以 `seed` 与流名新建的 `Rng` 的状态，就地载入（游戏自己持有的生成器引用不变），计数与种子字符串不变；公开状态与正在作答的帧不变。各幕预先排好的遭遇与事件、Boss、地图、共享遗物袋，以及遗物自己持有的生成器不动。同一种子播两次之后逐帧相同；不播种时引擎行为与没有这条命令时相同。原先只有局外对照的多未来用它，局外对照删除后（[trials 43](../trials/43-prune-iterate-options.md)）训练代码不再调用，引擎命令与 `CliEngine.reseed` 保留。

引擎的回复（每帧几十 KB）与发给它的命令用 orjson 解析、写出：引擎（System.Text.Json）写的数都在 64 位有符号或无符号整数以内（存档里有接近 2^64 的无符号数），也不写 NaN，orjson 读出的值与标准库相同。

## 公开历史（C#）

`public_history/History.cs` 只接受公开事件、不依赖游戏程序集；`NativeHistory.cs` 把 0.111.0 的公开事件接到状态机。无头引擎和 Steam Mod 编译同一份代码，不能各自复制修改。

- 敌人意图在敌方开始行动时保存公开图标及显示数值，切回玩家方、开始抽牌前成为上一回合输入。
- 牌位只从公开的置顶／置底操作维护；随机插入和洗牌使位置失效；原生插入的 index 从不作为知识来源。`AsyncLocal` 携带原生 Add／Shuffle 调用的公开位置语义，随异步续延流动。已知牌位通过 memory 的卡牌引用及 `known_draw_before` 关系表达。
- 药水在原生 Roll 生成奖励时更新累计值；强制掉落不改累计，跨幕不重置。问号房间跨幕重置，按 0.111.0 的累计顺序与相关遗物、事件规则计算。商店 blacklist 取决于上一地图点，输入分别提供允许／禁止商店的条件分布。
- `Capture` 只输出公开历史及当前对象的引用；Steam Mod 快照里的实例 ID 由 Python 控制端映射成当前帧的实体引用。读旧摘要不读取存档 odds。
- 快照（`NativeHistory.Snapshot`／`Restore`，`OddsHistory.Snapshot`／`Restore`）：`write_continue_save` 的回复带 `public_history`，记掉药累计值、问号房间各项累计值与所在幕、新手局计数、上一个点是否商店，未知的为 null。`main/env/saves.py` 把它放进存档的 `spire_public_history` 键，随存档走（幕边界存档、战斗入场存档、药价取点及其分支）；`resume_save`、`resume_fight` 前由 `resume_fields` 取出、作为命令的 `public_history` 交回，引擎在加载之后装上，代替 unknown。装回的是写存档那一局观察到的公开历史，不是存档里的 odds；幕不一致或版本不符时报错。战斗内的历史（上一回合意图、已知牌位）不进快照：入场存档在进房之前，开战时它们本来就是空的。

测试：

```bash
dotnet run --project public_history/tests/HistoryTests.csproj
python -m pytest -q main/tests/test_public_history.py
dotnet run --project public_history/tests/native/NativeHistoryTests.csproj -- \
  /absolute/repo /absolute/worker-config.json /absolute/SpireTakeover.dll
python public_history/tests/full_runs.py /absolute/Sts2Headless.dll /tmp/history-full-runs.json
```

原生测试把两端独立编译的状态机装入同一个诊断进程，驱动原生动作并逐帧比较 memory；私有 odds 和实际牌序只作测试断言的参照，不进入 `Capture`。

## 已知的坑

- 改引擎公开字段或随机数消耗会让重放证据与夹具的状态哈希失效；字段变化可用 `combat_solver_cli/tests/native/refresh_hashes.py` 重写哈希，随机数消耗变化则只能重新捕获。
- 引擎测试会重新编译 worker；有任务在跑时不要在主仓库编译。
- `CardSelectCmd` 装了测试选牌器就跳过选牌的暂停信号（`SignalPlayerChoiceBegun`），而真实游戏的 `StartTurn` 靠这个信号在回合准备第一次选牌时先跑 `AfterSideTurnStart`（橙子面团生成牌、提灯给能量等），再回到准备里余下的遗物。引擎因此在带选择上下文的选牌里让测试选牌器暂时让位（`SelectionMetadata`），游戏走本地玩家的路径：发出暂停信号，再问本地选牌器，而本地选牌器就是同一个无头选牌器（`RunSimulator.InstallSelector`）。去掉它，回合准备的手牌与随机数消耗就会与真实游戏不同（烤火手套的消耗选牌时手里还没有橙子面团的牌，选择悖论的生成牌在 `CombatCardGeneration` 上差一次生成），求解器按真实游戏模拟的计划也对不上（[solver](solver.md#回合准备选牌)）。
