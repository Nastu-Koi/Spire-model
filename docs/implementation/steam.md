# steam：录制 Mod 与实机接管

`steam_recorder/` 是 Steam 游戏采集和模型控制 Mod，`main/steam.py` 是 Python 控制端，`main/demos/recorder/` 导入录制（`snapshot.py` 公开快照、`macros.py` 宏动作、`journal.py` 读取与导入）。Mod 适配 Slay the Spire 2 **0.111.0、单人对局**，接管不限难度（示范与训练仍是 A0–A10）。仓库目录名为 `steam_recorder`，游戏内 Mod ID 保持 `RunRecorder`，记录写入 `user://run_recorder`。

0.6.0 与无头共用公开历史维护代码：前回合敌人意图、公开已知牌位、药水和问号房间概率；事件和营火选项除显示的数值外，另导出显示的内容名所指的内容 ID（`displayed_names`），规则与无头引擎相同。新局从公开规则开始维护；中途加载或缺失前缀保持 unknown，不能读取保存的内部 odds 补值。Python 只在对局难度与检查点训练难度不同时在面板提示一次；领奖交给主模型，水晶球格子及工具使用公开几何规划器。

```bash
# GAME_DIR 是 Steam 游戏安装目录。支持 Linux，也支持 Proton/WSL 可见的 Windows 目录。
python steam_recorder/install.py --game-dir "$GAME_DIR" --install
```

需要 .NET 9 SDK。脚本读取游戏原始的 `sts2.dll`、`GodotSharp.dll` 和 `0Harmony.dll` 构建，安装至 `mods/RunRecorder`。旧 Mod 文件保存在 `build/recorder-backups/`。不要用 headless 已修改的游戏 DLL 构建此 Mod。

从 Steam 启动游戏后，左下角显示录制状态，游戏日志打印完整输出目录。正常进行对局即可采集；退出或返回菜单后，记录会关闭。转换为 Bootstrap 数据：

```bash
python -m main import-recorder "$RECORDING_DIR" --bootstrap-only --output data/steam-001
```

`--bootstrap-only` 按决策导入，允许未完成对局和求解器来源；这些记录只用于 Bootstrap。去掉该参数则只导入完整、来源可确认的人类示范。缺失的选牌候选或卡牌效果参数不会自动补造。

Mod 同时提供实时控制桥接（检查点的版本要求见 [network](network.md#检查点)）。对局内运行：

```bash
python -m main --device cuda play-steam \
  --checkpoint <检查点> --bridge-dir "$RECORDING_DIR/bridge" --run-for 8h
```

控制端常驻到 `--run-for`（`8h`、`45m`、`90s`；不给则直到 Ctrl-C）或 `--max-decisions`，退出时打印局数、胜场、决策数与错误数。一局结束（`terminal`）记到 `bridge/runs.jsonl` 后继续等下一局；主菜单、动画、人手操作的阶段都只是等待，不再超时退出；`--request-timeout`（默认 120 s）只管单次请求。每步决策写 `bridge/decisions.jsonl`（时间、局、楼层、阶段、动作、摘要、前五排名）并用 `report` 上报给面板；控制端遇到的错误（桥接报错、超时、不合法的帧）写 `bridge/errors.jsonl`、上报后停下，直到面板上按“重置并重新读取状态”，然后从当前画面重新观察。

控制端在 `bridge/request.json` 写入带唯一 ID 的请求，游戏主线程响应到 `response.json`。`observe` 读取状态，`execute` 提交动作，`advance` 推进无决策的 UI，`report` 上报决策、错误或警告。每个响应带面板的设置（`settings`：录制、接管、接管战斗、接管局外、每步间隔）与 `reset_count`；状态除 `decision`／`waiting`／`terminal`／`stale` 外还有 `paused`（接管关闭）、`manual`（当前阶段由人操作）、`halted`（控制端上报了错误、等待重置）。执行前再次比对状态和合法动作；每个 token 只执行一次。Python 进程退出后不会继续发送动作。

游戏内左下角的“模型”按钮展开控制面板（`LivePanel.cs`）：录制、允许模型接管、接管战斗、接管局外四个开关，每步间隔（秒），记录器与桥接的状态，控制端停下时的错误与“重置并重新读取状态”按钮，当前决策与最近 20 条决策。开关、间隔与面板展开状态保存在 `run_recorder/settings.json`。关掉“接管战斗”时战斗由玩家打，模型在战斗结束后继续；关掉“接管局外”时地图、事件、商店、营火、奖励由玩家操作，桥接也不再替模型推进界面。每个响应带 `protocol`（当前 `steam-live-v3`）和 Mod 版本，控制端只接受相同的协议。

奖励界面上，已生成的卡牌奖励连同它的牌一起发布：`take_card_reward` 指定奖励和牌，`take_reward_alternative` 指定替代选项，Mod 点开面板、核对当前的牌与发布时一致后提交。关面板的“跳过”不是动作。宝箱关着时只发布 `open_chest`，遗物在开箱之后才出现；`advance` 不自动开箱。Python 控制端对普通宝箱显式执行固定的开箱、取遗物、离开流程，与无头采样一致；遗物触发的选牌等新选择继续交给模型。

Python 控制端对只有一个可用事件选项、其余仅能丢药的事件自动提交该选项；若有可用药水或多个事件选项，继续交给模型。执行之后才读取下一页，不提前揭示随机选择。商店和战斗的决策方式保持不变。

出牌和地图移动走游戏动作队列；事件、营火、奖励和购买使用原游戏 API；选牌通过当前选择界面的完成对象提交，游戏继续处理后续效果。`run-recorder-import-v6` 导入版本化的公开历史，并用与接管相同的控制规则，将宝箱、水晶球格子及上述事件推进保留为环境操作证据、排除 BC 标签；原 actor 和所选动作保留。支付、领奖及真正的后续选择仍是策略样本。旧录制缺失历史时使用 unknown。模型侧的公开状态转换与 Bootstrap 导入器共用实现。

## 已知的坑

- 不要用 headless 修改过的游戏 DLL 构建 Mod。
- 仓库与已安装 Mod 的源码不一致时，从 DLL 内嵌的 PDB 提取原始源码对照，不要反编译。
- 0.6.0 的面板、分阶段接管、常驻与报错重置只过了桥接接口测试（`main/tests/test_steam_controller.py`），没有做 Steam 实机验证。
