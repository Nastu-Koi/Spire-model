# steam：接管 Mod 与实机接管

`steam_takeover/` 是让模型接管 Steam 游戏的 Mod（游戏内 Mod ID `SpireTakeover`，显示名“模型接管 / Spire Takeover”），`main/steam/` 是 Python 控制端（`__init__.py` 控制循环与拼帧，`snapshot.py` 把 Mod 的快照转成模型看到的公开状态）。Mod 适配 Slay the Spire 2 **0.111.0、单人对局**，接管不限难度（训练仍是 A0–A10）。设置与桥接写在 `user://spire_takeover`。

| 文件（`steam_takeover/SpireTakeover/`） | 内容 |
| --- | --- |
| `ModEntry.cs` | 初始化（无头环境下不加载）、计时器、Harmony 补丁安装、`settings.json` 的读写 |
| `Patches.cs` | 补丁回调：一局结束与清理通知桥接；执行动作时跟踪它启动的原生任务（领奖、营火、事件、购买、水晶球），等任务连同之后的移除与动画结束再观察 |
| `Snapshot.cs` | 公开状态快照（含公开历史与事件、营火选项的 `displayed_names`，以及出牌区的牌：`snapshot.py` 在非空时转成 zone `play`，与无头引擎相同） |
| `Selection*.cs` | 选择界面的捕获：候选、约束与元数据 |
| `LiveBridge.cs` | 控制桥接：请求循环、协议、待执行动作与令牌 |
| `LiveUi.cs`、`LiveRewards.cs` | 按人手点击的界面生命周期推进界面、领取奖励、开关商店货架 |
| `LivePanel.cs`、`FeatureSettings.cs` | 游戏内控制面板与它的设置 |

Mod 与无头共用公开历史维护代码：前回合敌人意图、公开已知牌位、药水和问号房间概率；事件和营火选项除显示的数值外，另导出显示的内容名所指的内容 ID（`displayed_names`），规则与无头引擎相同。新局从公开规则开始维护；中途加载或缺失前缀保持 unknown，不能读取保存的内部 odds 补值。任何难度都照常接管，不比对检查点的训练难度；领奖交给主模型，水晶球格子及工具使用公开几何规划器。

```bash
# GAME_DIR 是 Steam 游戏安装目录。支持 Linux，也支持 Proton/WSL 可见的 Windows 目录。
python steam_takeover/install.py --game-dir "$GAME_DIR" --install
```

需要 .NET 9 SDK。脚本读取游戏原始的 `sts2.dll`、`GodotSharp.dll` 和 `0Harmony.dll` 构建，安装至 `mods/SpireTakeover`。被替换的文件保存在 `build/takeover-backups/<时间>/`；游戏目录里还有旧名的 `mods/RunRecorder` 时一并移进这个备份目录，免得两个 Mod 同时加载、各打一遍补丁。不要用 headless 已修改的游戏 DLL 构建此 Mod。

Mod 提供实时控制桥接（检查点的版本要求见 [network](network.md#检查点)）。`TAKEOVER_DIR` 是游戏数据目录下的 `spire_takeover`（Windows 上是 `%APPDATA%/SlayTheSpire2/spire_takeover`）。对局内运行：

```bash
python -m main --device cuda play-steam \
  --checkpoint <检查点> --bridge-dir "$TAKEOVER_DIR/bridge" --run-for 8h
```

控制端常驻到 `--run-for`（`8h`、`45m`、`90s`；不给则直到 Ctrl-C）或 `--max-decisions`，退出时打印局数、胜场、决策数与错误数。一局结束（`terminal`）记到 `bridge/runs.jsonl` 后继续等下一局；主菜单、动画、人手操作的阶段都只等待，不超时退出；`--request-timeout`（默认 120 s）只管单次请求。每步决策写 `bridge/decisions.jsonl`（时间、局、决策序号、幕、楼层、动作、阶段、摘要、是否由控制器执行、模型的前五排名）并用 `report` 上报给面板；有选牌（`selection`）、第一个帧里却选不到一张牌（桥接没给 `select_card`，或上下限让牌都不能选）的决策，另把桥接的整条消息写进 `bridge/empty-selections.jsonl`（时间、局、token、消息），照常决策，供查模组在哪里丢了牌；控制端遇到的错误（桥接报错、超时、不合法的帧，以及读帧和推理时的 `KeyError`、`ValueError`、`TypeError`、`OSError` 等，见 `main/steam/__init__.py` 的 `HALTING`）写 `bridge/errors.jsonl`、上报后停下，直到面板上按“重置并重新读取状态”，然后从当前画面重新观察。

控制端在 `bridge/request.json` 写入带唯一 ID 的请求（带 `unique_response`），游戏主线程响应到 `response-<请求 ID>.json`：每个请求一个新文件，因为 Windows 不允许替换对方还开着的文件。不带 `unique_response` 的请求仍响应到 `response.json`。同一桥接目录只允许一个控制端：控制端启动时对 `bridge/controller.lock` 加排他锁，拿不到就报错退出。`observe` 读取状态，`execute` 提交动作，`advance` 推进无决策的 UI，`report` 上报决策与错误（Mod 也收 `warning`，控制端目前不发）。每个响应带面板的设置（`settings`：接管、接管战斗、接管局外、每步间隔、面板展开）与 `reset_count`；状态除 `decision`／`waiting`／`terminal`／`stale` 外还有 `paused`（接管关闭）、`manual`（当前阶段由人操作）、`halted`（控制端上报了错误、等待重置）。执行前再次比对状态和合法动作；每个 token 只执行一次。Python 进程退出后不会继续发送动作。

游戏内左下角的“模型”按钮展开控制面板（`LivePanel.cs`）：允许模型接管、接管战斗、接管局外三个开关，每步间隔（秒），桥接的状态，控制端停下时的错误与“重置并重新读取状态”按钮，当前决策与最近 20 条决策。开关、间隔与面板展开状态保存在 `spire_takeover/settings.json`。关掉“接管战斗”时战斗由玩家打，模型在战斗结束后继续；关掉“接管局外”时地图、事件、商店、营火、奖励由玩家操作，桥接也不再替模型推进界面。每个响应带 `protocol`（当前 `steam-live-v5`）和 Mod 版本（`mod`），控制端只接受相同的协议。桥接协议与决策帧契约里的 `adapter_version`（`steam-live-v1`，Python 端拼帧时写入）是两个版本：前者管请求与响应的格式，后者标明帧来自 Steam 适配器。

奖励界面上，已生成的卡牌奖励连同它的牌一起发布：`take_card_reward` 指定奖励和牌，`take_reward_alternative` 指定替代选项，Mod 点开面板、核对当前的牌与发布时一致后提交。关面板的“跳过”不是动作。宝箱关着时只发布 `open_chest`，遗物在开箱之后才出现；`advance` 不自动开箱。水晶球的奖励界面关闭后，已结束的水晶球界面仍留在 overlay 栈顶，`advance` 点它自己的“继续”进入地图；这个按钮在奖励关闭后才启用，不会跳过奖励。Python 控制端对普通宝箱显式执行固定的开箱、取遗物、离开流程，与无头采样一致；遗物触发的选牌等新选择继续交给模型。

商店照人手的流程走：进房后 `advance` 先展开货架（调用点商人时游戏执行的 `NMerchantRoom.OpenInventory()`，要求商人按钮启用且可见），货架滑入完成（`NMerchantInventory` 已打开、开场动画结束、返回按钮可点）之前不发布决策；之后发布 `purchase`（照旧走 `MerchantEntry.OnTryPurchaseWrapper`，货架上的牌、遗物、药水随购买事件播放飞入动画，删牌照常打开选牌界面）与 `leave_shop`。`leave_shop` 点货架的返回按钮收起货架，并记下这间商店已离开；收起后商店的“继续”重新启用，`advance` 再点它打开地图，接着是正常的选图决策。没被 `leave_shop` 收起的商店（例如人手关了货架）`advance` 会重新展开。

营火选完一项后还有选项（小帐篷 `MINIATURE_TENT`）时，发布剩下的 `choose_rest_option` 和 `leave_rest_site`（`LEAVE_ROOM`，点营火的“继续”），与无头引擎选完一项后给剩余选项加 `LEAVE_ROOM` 相同；是否已选过一项按本楼层历史的 `RestSiteChoices` 判断。游戏在第一项完成后立即启用“继续”，剩下的选项随后才淡入，所以营火还有选项时 `advance` 不点它，捕获时营火的动作有一个点不到就整体等待，免得只发布“离开”或只发布选项。选项用完（没有小帐篷时选完即清空）时没有决策，`advance` 照旧点“继续”。

Python 控制端对只有一个可用事件选项、其余仅能丢药的事件自动提交该选项；若有可用药水或多个事件选项，继续交给模型。执行之后才读取下一页，不提前揭示随机选择。战斗的决策方式保持不变。

出牌和地图移动走游戏动作队列；事件、营火、奖励和购买使用原游戏 API；选牌通过当前选择界面的完成对象提交，游戏继续处理后续效果。从几张生成的牌里选一张的界面（`NChooseACardSelectionScreen`：无色、攻击、技能、能力药水，「发现」等）按引擎问选择器的方式发布：至多选 1 张，可跳过时至少 0 张（跳过就是一张不选直接确认），不可取消；这样选了牌之后只剩确认，与无头采样一致（`SelectionCapture.ObserveScreen`）。

## 已知的坑

- 不要用 headless 修改过的游戏 DLL 构建 Mod。
- 仓库与已安装 Mod 的源码不一致时，从 DLL 内嵌的 PDB 提取原始源码对照，不要反编译。
- 营火的“继续”在第一项选完时就启用（`NRestSiteRoom.AfterSelectingOptionAsync` 先 `ShowProceedButton`，再淡入剩下的选项），模型那时还看不到剩下的选项。0.6.4 之前 `advance` 点所有可点的“继续”，带小帐篷的实机营火每次只选一项；无头引擎不受影响。
- 一局赢下时游戏会结束两次：`RunManager.WinRun()` 先调 `OnEnded(isVictory: true)`，再 `GuaranteeKillAllPlayers()`，玩家死亡的路径（`CreatureCmd`）又调一次 `OnEnded(isVictory: false)`。挂在 `OnEnded` 上的补丁两次都会运行，所以胜负只认同一局的第一次（`LiveBridge.RunEnded`）。0.6.2 之前桥接取最后一次，`bridge/runs.jsonl` 把胜局记成 `victory: false`。这条路径在无头引擎里不出现（游戏在测试模式下不走它），只能实机验证；0.6.2 已在实机赢局上核对过，记为 `victory: true`。
- 商人按钮不能按 `CanClick` 判断：换商人外观的模组（如 Merchant2CuteII）会让它过不了不透明度或鼠标过滤的检查，人手却点得开，0.6.5 因此一直不展开货架。所以展开货架直接调用 `NMerchantRoom.OpenInventory()`，只要求商人按钮启用且可见。
