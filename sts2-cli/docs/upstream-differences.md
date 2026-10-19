# 与上游 sts2-cli 的差异

本文说明 `Spire-model/sts2-cli` 相对上游 [wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli) 改了什么。来源与许可证见 [UPSTREAM.md](../UPSTREAM.md)，协议本身的用法见 [decision-protocol.md](decision-protocol.md)，设计依据见 [architecture](../../docs/architecture.md)。

- 比较对象：上游 `084d1aa`（Merge pull request #93，适配 STS2 v0.111.0）。它是 UPSTREAM.md 记录的引入基线，所以下面全部是本仓库自己的改动，没有混入上游后续更新。
- 比较方法：逐文件 `diff`；Python 文件另按语法树比较，把纯格式化与逻辑改动分开。
- 未改动：`CLAUDE.md`、`.claude/settings.json`、`LICENSE`、`localization_eng/`、`localization_zhs/`、`src/sts2-cli.slnx`、`tests/test_save_load.py`，以及 GodotStubs 中除 `Core.cs`、`Math.cs`、`UI.cs` 之外的文件。

`CLAUDE.md` 沿用上游原文，其中有两处已经和本仓库的代码不符：“Async selection continuations” 一条仍要求用 `Task.Run(...)` 加轮询（现在是 `PendingOperation`，见 [3.5](#35-异步调度)），构建和回归命令写的是 macOS 路径与 `~/.dotnet-arm64`。

## 1. 总览

上游是一个给人和脚本用的无头引擎：`action` 命令加上 `DetectDecisionPoint`，后者会自动领奖、自动做默认选择，并且靠固定时长的 `Thread.Sleep` 轮询等待异步结算。

本仓库在它旁边加了一套给模型训练用的**引擎决策协议**（`engine-candidates-v1`），并把上游的等待方式换成事件驱动。主要变化有六类：

| 类别 | 内容 | 位置 |
| --- | --- | --- |
| 决策协议 | 每个决策发布公开状态和完整的合法动作集合，由带版本的句柄原子执行 | `DecisionGate.cs`、`RunSimulator.Decisions.cs`、`RunSimulator.Phases.cs` 等 |
| 公开状态 | 白名单导出的实体／关系／公开历史事实，不含隐藏牌序、RNG、怪物内部状态 | `RunSimulator.PublicState.cs`、根目录 `public_history/` |
| 摘要锚点入口 | 从对局摘要重建的节点边界进入一个房间，只用于监督样本 | `RunSimulator.Anchor.cs`、`AnchorMerchant.cs`、`AnchorTreasure.cs` |
| 异步调度 | 用 `PendingOperation` 取代 `Task.Run` + `Sleep` 轮询；选牌改为带校验的会话 | `PendingOperation.cs`、`SelectionSession.cs`、`NativeContinuations.cs` |
| 无头兼容与规则一致 | 新增一批 Harmony 补丁；商店药水价格改回游戏原生抽取 | `HeadlessEventPresentation.cs`、`HeadlessMerchantParity.cs` 等 |
| 工程 | 固定 .NET 9、单进程多局、输出与 JIT 调优、Python 侧共用运行时探测 | `Program.cs`、`Sts2Headless.csproj`、`python/` |

`src/Sts2Headless` 的新功能几乎都在新增的 partial 文件里，`RunSimulator.cs` 本身改动不多，上游的 `action` 命令路径仍然保留。

有三点会影响“和上游行为是否一致”，先列在这里：

1. **同一个种子的对局会和上游分叉。** 商店药水定价改为执行游戏的原生随机抽取（上游在 TestMode 下跳过），每次定价多消耗一次商店随机流。这个补丁在引擎初始化时安装，旧的 `action` 路径同样受影响。
2. **旧命令的校验变严。** `select_cards`、`select_bundle`、`skip_select` 等不再静默修正非法输入（见 [3.6](#36-上游命令的行为变化)）。
3. **本目录不能再单独拿出去编译。** `Sts2Headless.csproj` 编译根目录的 `public_history/*.cs`，Python 脚本导入根目录的 `main.env.dotnet_runtime`。

## 2. 文件级差异

### 只在本仓库

| 路径 | 作用 |
| --- | --- |
| `UPSTREAM.md` | 上游地址、引入基线、本地维护范围 |
| `docs/decision-protocol.md` | 决策协议说明 |
| `docs/upstream-differences.md` | 本文 |
| `src/Sts2Headless/DecisionGate.cs` | 候选句柄的发布、校验与一次性消费 |
| `src/Sts2Headless/PendingOperation.cs` | 跨外部选择保持的后台操作 |
| `src/Sts2Headless/InlineSynchronizationContext.cs` | 从 `RunSimulator.cs` 拆出并改为线程安全 |
| `src/Sts2Headless/NativeContinuations.cs` | 跟踪游戏自己发起、无人持有的续延 |
| `src/Sts2Headless/SelectionSession.cs` | 多选过程的前缀缓冲与约束校验 |
| `src/Sts2Headless/SelectionMetadata.cs` | 选牌的操作／来源／去向／可否取消 |
| `src/Sts2Headless/HeadlessEventPresentation.cs` | 去掉无头环境缺失的音效、立绘、震屏调用 |
| `src/Sts2Headless/HeadlessMerchantParity.cs` | 商店药水原生定价 |
| `src/Sts2Headless/AnchorMerchant.cs` | 锚点商店：按记录提供卡牌稀有度和遗物 |
| `src/Sts2Headless/AnchorTreasure.cs` | 锚点宝箱：条件化一次金币抽取 |
| `src/Sts2Headless/RunSimulator.Decisions.cs` | 协议契约、边界推进、选牌／卡包边界、执行 |
| `src/Sts2Headless/RunSimulator.Phases.cs` | 战斗、药水、地图、事件、营火边界 |
| `src/Sts2Headless/RunSimulator.PublicState.cs` | 公开快照 |
| `src/Sts2Headless/RunSimulator.RewardsProtocol.cs` | 奖励菜单与战后流程 |
| `src/Sts2Headless/RunSimulator.MerchantTreasure.cs` | 商店与宝箱边界 |
| `src/Sts2Headless/RunSimulator.CrystalSphere.cs` | 水晶球边界 |
| `src/Sts2Headless/RunSimulator.PotionPresentation.cs` | 臭药水对商人／假商人的无头桥接 |
| `src/Sts2Headless/RunSimulator.Milestones.cs` | 遭遇开始／完成、通关、死亡等事件 |
| `src/Sts2Headless/RunSimulator.Catalog.cs` | `public_catalog` 静态内容目录 |
| `src/Sts2Headless/RunSimulator.Anchor.cs` | 锚点命令与续局命令（`resume_save`、`resume_fight`） |
| `tests/DecisionProtocolChecks/` | 不依赖游戏 DLL：`SelectionSession` 与 `DecisionGate` |
| `tests/PendingOperationChecks/` | 不依赖游戏 DLL：`PendingOperation` 与同步上下文 |
| `tests/MerchantParityChecks/` | 依赖游戏 DLL：同一随机状态下原生与无头的药水价格及下一个随机数 |
| `tests/test_decision_protocol.py` | 协议的引擎测试 |
| `tests/test_pending_operations.py` | 异步调度的引擎测试 |

`lib/`（游戏程序集）和 `logs/`（对局日志）也只在本地存在，两边都被 `.gitignore` 排除，不属于源码差异。

### 只在上游

| 路径 | 说明 |
| --- | --- |
| `agent/` | 上游的 LLM 游玩代理：`auto_play.py`、`sts2_bridge.py`、`combat_helper.py`、`validate_learning.py`、亡灵契约师学习笔记、`bug.md`。本仓库的策略在根目录 `main/`，没有引入 |
| `docs/compatibility-v0.111.0.md` | v0.111.0 适配说明 |
| `docs/demo_en.gif`、`docs/demo_zh.gif` | 演示动图；README 里对应的两处图片引用也一并删除 |

### 两边都有但内容不同

| 文件 | 改动性质 |
| --- | --- |
| `src/Sts2Headless/RunSimulator.cs` | 逻辑（见第 3 节） |
| `src/Sts2Headless/Program.cs` | 逻辑：新命令、单进程多局、输出方式 |
| `src/Sts2Headless/Sts2Headless.csproj` | 运行时策略、JIT 选项、编译 `public_history` |
| `src/GodotStubs/Core.cs`、`Math.cs`、`UI.cs` | 补齐桩方法（见 [3.9](#39-godotstubs)） |
| `python/play.py`、`python/play_full_run.py` | 逻辑改动少量，其余为格式化（见第 4 节） |
| `tests/conftest.py`、`test_combat.py`、`test_characters.py` | 逻辑（见第 5 节） |
| `README.md` | 顶部加三行指向决策协议；删两处动图 |
| `setup.sh` | Windows 下默认游戏目录由 `C:/Program Files (x86)/Steam/...` 改为 `D:/SteamLibrary/...`（本机路径） |
| `.gitignore` | `lib/` 改为 `/lib`，只忽略本目录顶层的 `lib` |
| `launch.py`、`python/game_log.py`、`scripts/extract_localization.py`、其余 `tests/test_*.py` | 仅格式化、导入排序、删未用导入和无占位符的 f 前缀，行为不变 |

## 3. 引擎的差异

### 3.1 命令表

`Program.cs` 的 JSON Lines 入口新增和改动的命令：

| 命令 | 状态 | 说明 |
| --- | --- | --- |
| `execute_candidate` | 新增 | 参数 `decision_id`、`state_version`、`candidate_ref`，选择会话另带 `selection_revision`。字段缺失或类型不对返回 `invalid_candidate_request`，不消费当前帧 |
| `advance_to_boundary` | 新增 | 推进到下一个边界；已有已发布的决策时原样返回它，不会恢复执行 |
| `public_catalog` | 新增 | 静态内容标识和 21 个动作动词，用于冻结词表；不需要对局，不生成任何未来内容 |
| `enter_anchor`、`anchor_room`、`anchor_state`、`anchor_event`、`content_info` | 新增 | 摘要锚点入口（见 [3.4](#34-摘要锚点入口)） |
| `resume_save` | 新增 | 从地图上写下的续局存档续打这一局，契约 `initialization = act-resume-v1`；用于幕开头存档（见 [decision-protocol](decision-protocol.md#摘要锚点入口summary-anchor-v1)） |
| `resume_fight` | 新增 | 从一场战斗第一个决策处写下的续局存档重进这场战斗，契约 `initialization = fight-resume-v1`；用于 Boss 入场存档 |
| `start_run` | 扩展 | 新参数 `decision_protocol`（返回 `decision_frame` 而不是上游的决策点）和 `acts`（指定幕序列，例如 `["UNDERDOCKS","HIVE","GLORY"]`） |
| `load_save` | 扩展 | 新参数 `diagnostic_protocol`：允许诊断探针用协议驱动，但读档的对局永远不标记为可训练 |
| `enter_room` | 扩展 | 新参数 `decision_protocol` |
| `get_map` | 扩展 | 返回值多一个 `encounter_pools`（各幕的弱怪／普通／精英／Boss 遭遇池），只供诊断，不属于协议 |

`start_run`、`load_save`、`action`、`set_player`、`enter_room`、`set_draw_order`、`quit`、`enter_anchor`、`resume_save`、`resume_fight` 在执行前都会让当前的候选句柄失效，并把 `training_ready` 置为 false。也就是说上游的写命令和调试命令仍可用，但用过之后这一局不再有训练资格；`start_run`（带 `decision_protocol`）、`enter_anchor`、`resume_save`、`resume_fight` 随后按各自的契约重新建立训练资格。

### 3.2 决策协议

上游没有对应物。要点：

- **帧结构**：`decision_frame` 含 `contract`、`boundary`、`routing`、`public`、`legal.candidates`、`events`。`boundary` 为 `decision`、`waiting`、`terminal` 或 `error`。
- **契约**：`adapter_version = engine-candidates-v1`、`observation_schema = public-state-v6`、`action_schema = candidate-v0`、`auto_advance_version = pending-continuation-v0`、`selection_cancel = combat-or-dead-end-v1`、`merchant_potion_prices = native-potion-price-v1`，以及游戏程序集和适配器程序集的 SHA-256、固定难度、`training_ready`。只有 `decision_protocol:true` 原生创建的 A0–A10 对局 `training_ready` 才为 true。
- **原子执行**：`DecisionGate` 核对决策 ID、状态版本、选择修订、候选引用，以及候选此刻是否仍然合法。通过后先作废句柄再执行，所以重复请求和过期请求不会二次结算；校验不通过时当前帧保持有效。拒绝码为 `stale_decision`、`stale_selection`、`unknown_candidate`、`candidate_no_longer_legal`。
- **合法动作集合直接取自游戏**，不走上游 `DetectDecisionPoint` 的自动领奖和默认选择：

| 边界 | 候选动词 | 合法性来源 |
| --- | --- | --- |
| 战斗 | `PLAY_CARD`、`END_TURN` | 手牌 × 目标逐个调用原生 `CanPlayTargeting` |
| 药水（各边界通用） | `USE_POTION`、`DISCARD_POTION` | 使用时机、`PassesCustomUsabilityCheck`、`IsValidTarget` |
| 地图 | `MOVE_TO_NODE` | `MapTravel.GetTravelablePointsFrom` 和通行钩子，含第二个 Boss |
| 事件／远古之民 | `CHOOSE_EVENT_OPTION`、`ABANDON_RUN` | 当前页全部未锁选项；`ABANDON_RUN` 仅对应 Trial 的 DOUBLE_DOWN |
| 营火 | `CHOOSE_REST_OPTION`、`LEAVE_ROOM` | 原生可用选项；成功用过一次后才出现离开 |
| 商店／假商人 | `BUY_ITEM`、`LEAVE_ROOM` | 库存、金币、药水槽位、是否有可删的牌 |
| 宝箱 | `OPEN_CHEST`、`TAKE_TREASURE_RELIC`、`LEAVE_ROOM` | 打开之前不公开遗物 |
| 奖励 | `TAKE_REWARD`、`TAKE_CARD_REWARD`、`CHOOSE_REWARD_ALTERNATIVE`、`LEAVE_REWARDS` | 全部未领取奖励，领取顺序由策略决定 |
| 选牌 | `SELECT_ONE`、`FINISH_SELECTION`、`CANCEL` | `SelectionSession` 的当前前缀 |
| 卡包 | `SELECT_BUNDLE` | 待选卡包 |
| 水晶球 | `DIVINE_CELL` | 全部未揭示格 × 两种工具 |

- **多选过程**：`SELECT_ONE` 只编辑前缀，`FINISH_SELECTION` 一次提交，游戏效果在提交时才发生。帧里带 `selection_context`（操作、来源、去向、最少／最多、还需选几张、已选引用）和固定的 `decoder_bank`。选项按公开卡牌属性排序，不按引擎实例 ID 或牌堆内部顺序。
- **取消**：非战斗阶段，原生的取消只是撤销打开选择的那个动作（营火锻造、商店删牌、删牌奖励），所以打开即承诺，不发布 `CANCEL`；只有战斗中，或既不能选也不能完成时才提供。
- **等待与终局**：结算未到稳定边界时发布 `waiting`，由客户端再发 `advance_to_boundary`。HP 归零不等于死亡，要等原生 `Died` 事件或战斗失败才发布 `terminal`；3 秒内仍无法确认则报 `unconfirmed_terminal_outcome`。
- **里程碑事件**（`RunSimulator.Milestones.cs`）：`encounter_started`、`encounter_completed`、`run_completed`、`candidate_executed`。事件只随一次响应交付，重复查询不会重复给出。
- **错误不掩盖**：未知房间类型、空的合法动作集合、动作执行器吞掉的异常分别报 `unsupported_interaction:*`、`empty_legal_set`、`native_action_failed`。上游遇到类似情况倾向于强制 `proceed` 或回到地图。

### 3.3 公开状态

上游返回的是面向显示的 JSON（中英双语名称、描述文本、各种便利字段）。协议帧改为白名单导出的实体表：

- **实体**：玩家（含难度、幕、楼层、HP、金币、能量、辉星、`free_travel`、`lost_hp_this_turn`）、卡牌、敌人与召唤物、意图（图标类型、单段伤害、段数、总伤害）、能力、遗物（`stats`，游戏显示计数时带 `counter`）、药水、充能球、地图节点（Boss 节点带 `encounter`）、各边界自己的选项实体，以及每个牌堆一条 `pile_summary`。
- **关系**：`map_edge`、`offers`、`known_draw_before`。
- **顺序**：手牌和充能球保留公开顺序；牌组、抽牌堆、弃牌堆、消耗堆按公开属性排序，作为无序牌堆导出。
- **公开历史事实**（`public.memory`）：由根目录 `public_history/` 的状态机维护，无头引擎和 Steam Mod 编译同一份源码。
- **选项上显示的变量**：事件和营火选项导出模板实际引用的变量，作为 `displayed_variable`（数值带 `amount`，内容名带 `names`）。发布前先把事件变量填进选项标题和描述，与游戏按钮显示时的做法一致。
- **不导出**：隐藏牌序、种子、RNG、引擎实例 ID、怪物内部移动状态、未打开宝箱里的遗物、水晶球未揭示格背后的物品。
- **规则**：卡牌、遗物、药水、事件选项的效果统一标为 `OPAQUE_RULE`，由 Python 侧用规则文本补足。

为了让 `public_catalog` 能枚举真实的本地化键，上游把 `LocTable.GetLocStringsWithPrefix` 补丁成返回空列表的做法被去掉了。

### 3.4 摘要锚点入口

上游没有对应物。`summary-anchor-v1` 用来把一份对局摘要重建到某个节点边界，再进入该节点的房间取一个监督样本：

- `enter_anchor`：载入序列化对局，静默重装玩家状态（已持有的遗物不会再次触发取得效果）。给了 `route` 且本幕地图上恰有一条路径与记录的节点类型序列一致时，沿这条路径走到 `act_floor`；否则退回同类型、同楼层的代表节点，并在返回值里说明是哪一种。
- `anchor_room`：安装后只能调用一次，支持 `combat`、`rest_site`、`card_reward`、`map`、`ancient`、`shop`、`event`、`rewards`、`treasure`。
- `anchor_state`、`anchor_event`、`content_info`：只读查询。
- 商店：`AnchorMerchant` 只提供记录里有而随机流无法还原的部分（五个角色卡牌格位的稀有度、三个遗物），其余库存和全部价格仍由游戏用存档里的商店随机流抽出，是否与记录一致由调用方判断。
- 宝箱：`AnchorTreasure` 只条件化开箱时那一次 42～52 的金币抽取，原抽取照常推进随机流，难度折算、金币修饰、取得效果都走原生代码。
- 这类帧的契约多一个 `initialization = "summary-anchor-v1"`，只能当独立监督样本，不能当连续的原生对局。

### 3.5 异步调度

这是对上游已有代码改动最集中的部分。上游的写法是 `Task.Run(...)` 之后循环 100 次、每次 `Sleep(10)`，看到有待选就返回，否则 `task.Wait(2000)`；营火和商店里还有固定的 `Sleep(200)`、事件领奖有 `Sleep(50)`。这种写法把“时间过去了”当成“操作完成了”。

| 上游 | 本仓库 |
| --- | --- |
| `Task.Run` + `Sleep` 轮询（购买遗物、删牌、营火选项、事件选项） | `PendingOperation.Start` + `WaitForBoundary`：由选牌请求或任务完成来唤醒，3 秒超时抛 `TimeoutException`，协议路径转成 `waiting` |
| 营火选完立即 `ForceToMap()` | 作为操作的完成回调执行，要等所有选牌和续延都结束，并且玩家仍在该房间且未死亡 |
| `select_cards` 之后按房间类型补 `Sleep(200)` | 删除；由 `DetectDecisionPoint` 开头的 `WaitForPendingOperation` 继续同一个操作 |
| 事件卡牌奖励用 `ManualResetEventSlim` 加共享的 `_rewardChoice` | 每次请求一个 `RewardRequest`，用 `TaskCompletionSource`、`Volatile`、`Interlocked` 交接 |
| 选牌的 `TaskCompletionSource` 用默认选项 | 加 `RunContinuationsAsynchronously`，并且先清掉旧的待选再恢复游戏代码，避免新的待选被覆盖 |
| `InlineSynchronizationContext` 无锁，`Post` 内联执行 | 加锁队列，`Post` 入队后统一 `Pump`，可从后台线程安全调用 |
| `TurnStarted` 一个信号 | 另加 `_playerTurnStarted`，区分敌方回合开始和玩家回合开始；`END_TURN` 会等到玩家能再次行动 |
| 无 | `NativeContinuations` 跟踪 Lord's Parasol 进店后的连续购买，完成或请求输入之前只发布 `waiting` |

另外，`DoAction` 开头会先等待挂起的操作；如果操作还在等一次选择，除 `select_cards`、`skip_select`、`select_card_reward`、`skip_card_reward`、`select_bundle` 之外的动作都不执行，直接返回当前决策点。

### 3.6 上游命令的行为变化

用上游 `action` 命令的脚本需要注意这些差别：

| 命令或场景 | 上游 | 本仓库 |
| --- | --- | --- |
| `select_cards` 的 `indices` 含非整数 | 当作 −1 丢弃 | 返回错误 |
| `select_cards` 越界、重复、数量不在最少／最多之间 | 过滤掉非法项后照常提交 | 整体拒绝，选择保持待定 |
| `skip_select` | 直接提交空选择 | 最少张数大于 0 时拒绝 |
| `select_bundle` 越界 | 默认取第一个卡包 | 返回错误 |
| 选牌只有一个选项且至少选一张 | 自动选中，不产生决策 | 照常产生待选 |
| 选牌没有选项 | 直接返回空 | 照常产生待选（最少为 0） |
| `FromChooseACardScreen` 的最少张数 | 取调用方传入值 | 按 `canSkip` 改写为 0 或 1 |
| 卡牌奖励没有牌但有替代选项 | 视为跳过 | 产生待选，可选替代选项 |
| `skip_card_reward` | 总能跳过 | 奖励没有 `Skip` 替代选项时拒绝 |
| 营火或事件选项执行时抛异常 | 记日志后继续 | 返回带堆栈的错误 |
| 任何返回带堆栈错误的命令之后 | 进程可继续用 | 这一局被标记为出错，本进程不能再开下一局 |
| 商店药水价格 | TestMode 下不抽随机数 | 执行原生抽取，消耗商店随机流 |

### 3.7 同一进程里的下一局

上游一个进程实际上只跑一局。本仓库中，已有对局时再发 `start_run`、`load_save`、`enter_anchor`、`resume_save` 或 `resume_fight`，`Program.cs` 先调用 `EndRun()`（内部是 `RunManager.CleanUp`），成功后换一个新的 `RunSimulator`。新一局的决策序号、状态版本和全部协议状态从头开始，得到的帧与新进程的第一局相同，只有 `episode_id` 不同。

为此，战斗事件由匿名委托改为具名方法，以便退订；选牌器的注册保留为可释放的作用域；结束时还会还原奖励选择器并清空续延跟踪。战斗进行中、仍有游戏代码在执行、或这一局出过错时拒绝结束，返回 `run_not_ended`。

### 3.8 Harmony 补丁

新增（都在引擎初始化或首次启用协议时安装）：

| 补丁 | 目的 |
| --- | --- |
| `HeadlessEventPresentation` | 扫描全部事件类型，只移除音效、震屏、立绘、布局这几类表现调用，保留参数求值、随机数和游戏指令；Trial 的立绘特效单独跳过 |
| 同上，怪物部分 | Kaiser Crab 的 Crusher／Rocket 死亡音效；SoulNexus 死亡时对缺失房间节点的空值保护 |
| 同上，放弃对局 | `RunManager.AbandonInternal` 开头两处关闭界面的调用改为空值安全，放弃流程本身仍走原生 |
| `HeadlessMerchantParity` | `MerchantPotionEntry.CalcCost` 里的 `TestMode.IsOff` 判断恒为真 |
| `SelectionMetadata` | 包住 `CardSelectCmd.From*`，记录操作、来源、去向、可否取消 |
| `NativeContinuations` | 跟踪 `LordsParasol.PurchaseEverything` 返回的任务 |
| `Spire.PublicHistory.NativeHistory` | 把原生公开事件接到公开历史状态机 |
| `AnchorMerchant`、`AnchorTreasure` | 见 [3.4](#34-摘要锚点入口)；没有锚点提供数据时不改变行为 |
| 水晶球（`RunSimulator.CrystalSphere.cs`） | 只替换 `NCrystalSphereScreen.ShowScreen`，格子点击和奖励结算走原生小游戏 |
| 臭药水（`RunSimulator.PotionPresentation.cs`） | 协议启用且不在战斗中时，允许对商人和假商人使用 |

几处转译补丁在匹配到的调用数量与预期不符时直接抛异常，游戏升级后如果这些方法变了，会在启动时就暴露出来。

移除：`LocTable.GetLocStringsWithPrefix` 的空返回补丁（原因见 [3.3](#33-公开状态)）。

### 3.9 GodotStubs

三处补齐，都是为了让 CLR 在 TestMode 跳过的分支上也能解析到方法引用：

- `Core.cs`：`GodotObject` 增加 `Connect` 和 `CallDeferred`（后者从 `Node` 上移到基类，并改为返回 `Variant`）；`Node` 增加 `GetIndex`。
- `Math.cs`：`Vector2I.One`、`Mathf.Log`。
- `UI.cs`：`CanvasItem.SetSelfModulate`、`SetVisible`。

### 3.10 构建、运行时与性能

- `RollForward` 由 `LatestMajor` 改为 `Minor`：固定在 .NET 9，不会被机器上更高的主版本接管。
- `TieredCompilation.CallCountingDelayMs = 0`：一局里不断出现新的卡牌、怪物和事件，默认的 100 毫秒静默期总被打断，每帧都跑的快照和钩子代码一直得不到优化。
- 输出：`WriteLine` 把 JSON 直接序列化成 UTF-8 写到标准输出流，一帧一次写入，并与 `Console.Out` 共用锁；上游是先转成 UTF-16 字符串再经 `Console` 编码。
- 本地化数据目录：上游用相对可执行文件的固定层数（`../../../../..`）；本仓库改为 `LocLookup.DataRoot()`，从当前目录和可执行文件目录逐级向上找 `localization_eng`，因为 CombatSolverCli 宿主引擎时目录深度不同。
- `Sts2Headless.csproj` 额外编译 `../../../public_history/*.cs`。

## 4. Python 客户端

`python/play.py`（交互游玩）和 `python/play_full_run.py`（随机策略跑完整局）仍然用上游的 `action` 命令，没有改成决策协议；协议的客户端在仓库根目录的 `main/` 和 `combat_solver_cli/`。两个文件的 diff 很大，绝大部分是格式化，逻辑改动只有：

- **运行时探测**：`_find_dotnet` 改为调用根目录 `main.env.dotnet_runtime.find_runtime()`，按 `STS2_DOTNET`、`~/.dotnet-spire`、`~/.dotnet-arm64`、`~/.dotnet`、系统 `dotnet` 的顺序找带 .NET 9 运行时的宿主，只装了运行时没装 SDK 也可以。构建另用 `find_sdk()`。
- **启动方式**：由 `dotnet run --no-build --project ...` 改为直接执行 `bin/Debug/net9.0/Sts2Headless.dll`。
- **提示文案**：找不到运行时的提示改为中英双语，并指向 `~/.dotnet-spire` 和 `STS2_DOTNET`。
- **`play_full_run.py`**：新增 `--ascension`（0–10）并传给 `start_run`；自动选牌改为按 `min_select` 选满，配合 [3.6](#36-上游命令的行为变化) 的严格校验（上游固定提交 `"0"`）。
- 其余是删除未使用的变量和导入。

## 5. 测试

| 文件 | 变化 |
| --- | --- |
| `tests/conftest.py` | 运行时探测同上；直接启动 DLL，可用环境变量 `STS2_TEST_ENGINE_DLL` 指向待测版本；标准输出改由读线程加队列读取，30 秒无响应报超时（上游是阻塞式 `readline`）；标准错误写入临时文件供断言；新增 `auto_card_selection`，按最少张数选满 |
| `tests/test_characters.py` | 遇到引擎错误直接判失败（上游是发一个 `proceed` 继续）；选牌改用 `auto_card_selection` |
| `tests/test_combat.py` | 新增 SoulNexus 精英在无头环境死亡不产生原生错误的用例 |
| `tests/test_decision_protocol.py` | 新增。查询幂等与被拒请求不改前缀、营火锻造和商店删牌不提供取消、旧选牌命令拒绝非法数组、调试修改使句柄失效、公开帧不含内部路由和 RNG、A0 与 A10 契约、选牌排序与牌组顺序无关、卡包选择、单张可选时保留完成候选、同进程下一局与新进程第一局一致、Boss 节点带遭遇、战斗中不能结束对局 |
| `tests/test_pending_operations.py` | 新增。锻造结果先于地图完成、商店删牌只扣一次钱、选牌后事件续延完成、连续三个事件奖励不丢下一个提示 |
| 三个 `*Checks` 工程 | 新增，见第 2 节 |
| 其余 `test_*.py` | 仅格式和导入调整 |

验证命令见 [decision-protocol.md](decision-protocol.md#验证)。

## 6. 与仓库其它模块的耦合

上游是自包含的；本目录现在依赖并被依赖于仓库的其它部分：

- **依赖出去**：`Sts2Headless.csproj` → `public_history/*.cs`；`python/play.py`、`python/play_full_run.py`、`tests/conftest.py` → `main/env/dotnet_runtime.py`。
- **被依赖**：`combat_solver_cli/` 引用并宿主这个引擎，其中 `PublicSnapshot.AddCard` 被按“恰好三个参数”反射调用，改签名会破坏求解器；`spire_codex_data/`、`main/` 通过 JSON Lines 使用协议、锚点与续局命令。
- **版本身份**：契约里的适配器程序集哈希取自 `Sts2Headless.dll`，游戏程序集哈希取自 `sts2.dll`。本目录任何影响公开字段或随机数消耗的改动，都会让依赖这些哈希或公开状态哈希的既有数据失去可比性。

## 7. 合并上游更新时的注意事项

- 上游的改动几乎都落在 `RunSimulator.cs` 和 `Program.cs`。本仓库在 `RunSimulator.cs` 里改过的区域是：类声明（`partial`）、字段与构造函数、`StartRun`／`LoadSave`／`EnterRoom` 的签名和结尾、`DoAction` 开头、购买遗物／删牌／营火／事件四处的等待逻辑、`DoSelectCards`／`DoSelectBundle`、`WaitForActionExecutor`、`EnsureModelDbInitialized` 里的补丁安装、`HeadlessCardSelector` 整个类、本地化目录查找、`GetFullMap` 结尾。合并冲突会集中在这些地方。
- 上游如果继续新增 `Sleep` 轮询式的等待，合并时要改写成 `PendingOperation`，否则会重新引入“选择还没结算完就发布下一个边界”的问题。
- 上游对 `DetectDecisionPoint` 的自动推进改动不影响协议路径，但协议各边界有自己的一套实现，上游修掉的同类问题需要在 `RunSimulator.Phases.cs` 等文件里对照检查是否也存在。
- 游戏版本升级后，先看带数量断言的转译补丁是否还能安装，再跑 `MerchantParityChecks` 和 `tests/`。
