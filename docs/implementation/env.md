# env：引擎接口、控制分工、存档与种子

`main/env/` 是最底层，不依赖 `main` 的其它子包。设计见 [architecture](../architecture.md#环境步骤与控制分工)。

| 模块 | 内容 |
| --- | --- |
| `protocol.py` | 决策帧的校验、动作语义、执行命令、`segment_key`（多选的解码段）、角色表 |
| `control.py`、`treasure_rule.py`、`crystal_rule.py` | 策略与控制器的分工、宝箱流程、水晶球规划器 |
| `rewards.py` | 奖励账本：只在终局支付 ±1 |
| `saves.py` | 战斗入场存档与重入命令 |
| `seeds.py` | `split_group`、`seed_group` 与留出划分的身份，评估种子的隔离检查 |
| `public_history.py` | 公开历史契约的版本与 unknown 记忆 |
| `engine.py`、`dotnet_runtime.py` | 无头引擎进程（`CliEngine`）与 .NET 运行时选择 |
| `runtime.py`、`processes.py` | 资源设置与可复现性清单；随父进程结束的子进程 |

## 契约版本

| 契约 | 当前版本 |
| --- | --- |
| 原生公开观察 | `public-state-v6` |
| 公开历史 | `public-history-v1` |
| 模型输入编码 | `public-encoding-v3`（`main/encoding/representation.py`） |
| 策略／控制器分工及索引 | `public-control-v1` |
| 奖励 | `outcome-v1` |

缺失旧历史时输入为 unknown，不补初始概率。旧版本采样的对局不能只改版本号后使用。

## 控制分工

`control.py` 为运行、导入、索引和训练提供同一套判断：

- 普通宝箱（宝箱节点与问号房间里的）进入 `treasure` 阶段，统一执行 `treasure-open-take-leave-v1`：开箱 → 领取遗物 → 离开；出现多个可选遗物时报错，不擅自选第一个。遗物触发的选牌仍交给策略。
- 事件仅有一个可用 `CHOOSE_EVENT_OPTION`、其余合法动作仅为 `DISCARD_POTION`（或没有药水动作）时自动执行该选项；有可用药水、第二个事件选项或其他实质动作时交给模型，锁定选项不算可用。
- 水晶球：主模型选支付方案，规划器（`crystal-public-geometry-v2`）按已揭示格子、剩余点击和公开几何规则生成相容的矩形布局假设，按剩余预算做束搜索选格子和 Small／Big 工具；它不读真实布局或游戏随机数，近似搜索用独立随机源。支付和领奖照常训练，不加规划器评分奖励。
- 商店仅离开＋丢药、战斗仅结束回合＋丢药仍是策略决策。

控制器步骤不产生策略损失，作为 `environment_actions` 保留原始帧、动作与 actor。采样与训练只接受 `control_version=public-control-v1` 且宏动作里没有控制器步骤的轨迹。控制规则变了，示范不必重新生成：分片保存了完整候选，对旧分片执行 `reindex`（见 [data](data.md)）。

非战斗阶段的可取消选择不发布 `CANCEL`（`selection_cancel = "combat-or-dead-end-v1"`）；没有任何可选牌、无法完成时仍提供，避免死局。战斗中保持原生取消语义，因为 CombatSolver 在“计划不选任何牌”时依赖它。Steam 端与录制导入用同一规则（`main.demos.recorder.cancel_offered`）。

## 奖励

基础奖励只在一局结束时支付一次：通关 +1，没通关 −1。第三幕 Boss 倒下和对局完成两种终局标记只付同一份 +1；失败由对局完成事件确认，引擎没发这个事件的按终局状态支付。过幕、普通战与精英战胜利都是 0，账本记下已击败的 Boss 只用于报告。超时、引擎错误和未完成的对局不伪造终局，不进缓冲。

## 入场存档

采样时，每场战斗的第一个决策（`events` 带 `encounter_started`）让引擎 `write_continue_save` 到临时文件再读回。在房间里时引擎存的是进房前的状态（节点从已访问坐标里退回、房间清空）；`anchor_state` 序列化的是已进房的状态，不能用。`saves.anchor_commands(row)` 给出重进同一场战斗的两条命令：`enter_anchor`（存档、楼层、节点类型、是否第二 Boss）再 `anchor_room`（遭遇）。重进的战斗 HP、最大 HP、金币、幕、楼层、卡组、遗物、药水和敌人种类相同；敌人的随机 HP 与开局手牌由战斗自己的随机数重抽，所以重进是同一入场状态的另一次战斗，不是重放。在战斗开始时计数的遗物重进后可能少计一次。

幕组在 Boss 入场切点写的也是这种存档（`rollout.RolloutRunner.run(until_boss=)`）；续打用一条命令 `resume_fight`（`saves.fight_resume_command(row)`：存档、遭遇、楼层、节点类型、是否第二 Boss），引擎加载后把节点记为已访问、标训练可用（契约 `initialization=fight-resume-v1`）并进入遭遇，回复就是战斗的第一个决策帧（`CliEngine.resume_fight`）。`enter_anchor` 走的是示范生成的路，契约标 `summary-anchor-v1`，训练不用。

## 种子与划分

`split_group(seed)`：原始 seed 去首尾空格、转大写、O/I 归一为 0/1 后取 SHA-256。`seed_group(run)` 是统一分组键：索引行带 `split_group` 时直接用；独立监督样本的 `seed` 已是这个摘要，不再哈希；其余对局按原始 seed 计算。划分版本 `seed-group-v1`。

## 引擎进程

`CliEngine` 启动 `sts2-cli` 的无头引擎；工作目录是 `sts2-cli/`，传给引擎的文件路径必须是绝对路径。引擎在原进程里开下一局（先按游戏返回主菜单的方式结束上一局）；结束不了（停在战斗里、出过错）时换新进程；打满 `RUNS_PER_PROCESS` 局后换新，避免内存缓慢上涨。

## 公开历史（C#）

`public_history/History.cs` 只接受公开事件、不依赖游戏程序集；`NativeHistory.cs` 把 0.111.0 的公开事件接到状态机。无头引擎和 Steam Mod 编译同一份代码，不能各自复制修改。

- 敌人意图在敌方开始行动时保存公开图标及显示数值，切回玩家方、开始抽牌前成为上一回合输入。
- 牌位只从公开的置顶／置底操作维护；随机插入和洗牌使位置失效；原生插入的 index 从不作为知识来源。`AsyncLocal` 携带原生 Add／Shuffle 调用的公开位置语义，随异步续延流动。已知牌位通过 memory 的卡牌引用及 `known_draw_before` 关系表达。
- 药水在原生 Roll 生成奖励时更新累计值；强制掉落不改累计，跨幕不重置。问号房间跨幕重置，按 0.111.0 的累计顺序与相关遗物、事件规则计算。商店 blacklist 取决于上一地图点，输入分别提供允许／禁止商店的条件分布。
- `Capture` 只输出公开历史及当前对象的引用；录制里的实例 ID 由 Python 导入器映射成当前帧的实体引用。读旧摘要或旧录制不读取存档 odds。

测试：

```bash
dotnet run --project public_history/tests/HistoryTests.csproj
python -m pytest -q main/tests/test_public_history.py
dotnet run --project public_history/tests/native/NativeHistoryTests.csproj -- \
  /absolute/repo /absolute/worker-config.json /absolute/RunRecorder.dll
python public_history/tests/full_runs.py /absolute/Sts2Headless.dll /tmp/history-full-runs.json
```

原生测试把两端独立编译的状态机装入同一个诊断进程，驱动原生动作并逐帧比较 memory；私有 odds 和实际牌序只作测试断言的参照，不进入 `Capture`。

## 已知的坑

- 改引擎公开字段或随机数消耗会让重放证据与夹具的状态哈希失效；字段变化可用 `combat_solver_cli/tests/native/refresh_hashes.py` 重写哈希，随机数消耗变化则只能重新捕获。
- 引擎测试会重新编译 worker；有任务在跑时不要在主仓库编译。
