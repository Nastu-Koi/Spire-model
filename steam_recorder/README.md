# steam_recorder

Steam 游戏采集和模型控制 Mod，适配 Slay the Spire 2 **0.111.0、单人 A10**。仓库目录名为 `steam_recorder`，游戏内 Mod ID 保持 `RunRecorder`，已有记录也继续写入 `user://run_recorder`。

```bash
# GAME_DIR 是 Steam 游戏安装目录。支持 Linux，也支持 Proton/WSL 可见的 Windows 目录。
python steam_recorder/install.py --game-dir "$GAME_DIR" --install
```

需要 .NET 9 SDK。脚本读取游戏原始的 `sts2.dll`、`GodotSharp.dll` 和 `0Harmony.dll` 构建，安装至 `mods/RunRecorder`。旧 Mod 文件保存在 `build/recorder-backups/`。不要用 headless 已修改的游戏 DLL 构建此 Mod。

从 Steam 启动游戏后，左下角显示录制状态，游戏日志打印完整输出目录。正常进行对局即可采集；退出或返回菜单后，记录会关闭。转换为 Bootstrap 数据：

```bash
python -m model import-recorder "$RECORDING_DIR" --bootstrap-only --output data/steam-001
```

`--bootstrap-only` 按决策导入，允许未完成对局和求解器来源；这些记录只用于 Bootstrap。去掉该参数则只导入完整、来源可确认的人类示范。旧版 0.4.x 记录仍可导入。缺失的选牌候选或卡牌效果参数不会自动补造。

0.5.0 增加实时控制桥接。对局内运行：

```bash
python -m model --device cuda play-steam \
  --checkpoint runs/ppo/current --bridge-dir "$RECORDING_DIR/bridge"
```

控制端在 `bridge/request.json` 写入带唯一 ID 的请求，游戏主线程响应到 `response.json`。`observe` 读取状态，`execute` 提交动作，`advance` 推进无决策的 UI。执行前再次比对状态和合法动作；每个 token 只执行一次。Python 进程退出后不会继续发送动作。

出牌和地图移动走游戏动作队列；事件、营火、奖励和购买使用原游戏 API；选牌通过当前选择界面的完成对象提交，游戏继续处理后续效果。模型控制操作在记录中标记为 `model`，不会伪装成人类示范。模型侧的公开状态转换与 Bootstrap 导入器共用实现。
