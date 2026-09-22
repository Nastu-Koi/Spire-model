# Spire Model

先用 Steam 实玩记录训练模型，再在 `sts2-cli` 中用 PPO 强化训练，最后让模型接管 Steam 游戏。

代码入口是 `python -m model`。v2 默认采用小型纯 Full Attention、SwiGLU 和 Muon/AdamW，配置在 `configs/rtxpro6000.json`；CPU 调试使用 `configs/tiny.json`。目前适配游戏 **0.111.0、单人 A10**。

v2 检查点不兼容旧 hybrid 权重，需要重新训练；采集数据可继续导入。批量训练、多引擎采样、RTX 基准及独立评估步骤见 [性能验证](docs/PERFORMANCE.md)。

## 1. 安装环境和 sts2-cli

需要先通过 Steam 安装游戏，并安装 .NET 9 SDK。Python 环境：

```bash
conda create -n sts2 python=3.11 -y
conda activate sts2
pip install -r requirment.txt
pip install -e .
```
设置 Steam 游戏安装目录。Linux 默认库通常在下面的位置；如果装在其他盘，改成实际目录。详情见原无头模拟器仓库：https://github.com/wuhao21/sts2-cli 

```bash
export GAME_DIR="$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2"
bash sts2-cli/setup.sh "$GAME_DIR"
```

脚本会从安装目录复制游戏程序集到 `sts2-cli/lib`，处理 headless 所需的补丁并构建引擎。不会修改 Steam 安装目录里的游戏 DLL。也接受 Proton/Windows 游戏目录，例如 WSL 下的 `/mnt/d/SteamLibrary/steamapps/common/Slay the Spire 2`。

## 2. 用 steam_recorder 收集 Steam 游戏数据

采集器源码就在仓库的 `steam_recorder/`。退出游戏后构建并安装：

```bash
python steam_recorder/install.py --game-dir "$GAME_DIR" --install
```

安装脚本识别 Linux 和 Windows 的游戏数据目录，将 Mod 放到游戏的 `mods/RunRecorder`。然后从 Steam 启动游戏，允许加载 Mod，正常玩单人 A10。游戏左下角会显示记录状态。

原始记录保存在游戏的 `user://run_recorder`。游戏日志中的 `[RunRecorder] initialized; output=...` 会打印完整路径。Linux 下可以这样找：

```bash
find "$HOME/.local/share" -type d -name run_recorder 2>/dev/null
```

Proton 的记录在对应 compatdata 前缀的 Windows 用户目录；WSL 读取 Windows 游戏时则使用 `/mnt/c/Users/<用户名>/AppData/Roaming/SlayTheSpire2/run_recorder`。下面的 `RECORDING_DIR` 要填成你实际找到的路径。

```bash
export RECORDING_DIR="/你的实际记录目录/run_recorder"
python -m model import-recorder "$RECORDING_DIR" \
  --bootstrap-only --output data/steam-001
```

Bootstrap 导入会保留有效的决策，包括未打完的对局和求解器操作。模型输入只包含行动前的公开状态和合法动作；缺字段或无法匹配标签的决策会跳过。`data/steam-001/accepted.jsonl` 就是训练数据，`summary.json` 里有导入数量和跳过原因。每次导入换一个输出目录。

## 3. Bootstrap

通过先学习玩家的操作决策，学习初步的动作策略。初始化训练：

```bash
python -m model --device cuda bootstrap \
  --data data/steam-001/accepted.jsonl \
  --config configs/rtxpro6000.json \
  --epochs 5 --output runs/bootstrap
```


默认使用全部输入数据。如果有独立验证集，用 `--validation 文件.jsonl` 指定。

每个 epoch 结束都会保存训练指标并更新权重：

```text
runs/bootstrap/
  current/         最新模型、词表、优化器和续训状态
  history.jsonl    每个 epoch 的 loss、熵、梯度、更新次数、耗时等
  status.json      当前训练状态
```

使用`--checkpoint /path/to/checkpoint`指定读取模型。`--epochs` 表示这次追加训练几轮。

```bash
python -m model --device cuda bootstrap \
  --data data/steam-001/accepted.jsonl \
  --checkpoint runs/bootstrap/current \
  --epochs 5 --output runs/bootstrap
```

## 4. PPO强化训练

Bootstrap 只学习如何选动作，没有训练价值头。第一次开始 PPO 时，建议加上 `--value-warmup`：先用当前策略采样完整对局，只更新价值头，让它学习预测后续回报；下一轮重新采样，再正常更新策略和价值头。

先跑一段短训练，可以用下面的命令（1 轮 warmup + 10 轮 PPO）：

```bash
python -m model --device cuda train \
  --checkpoint runs/bootstrap/current \
  --value-warmup --runs-per-character 4 --rounds 11 \
  --output runs/ppo
```

`--rounds` 包含 warmup 这一轮。如果要做 1 轮 warmup 加 160 轮 PPO，就改为 `--rounds 161`。warmup 也会保存训练信息并更新 `current`，历史中的 `stage` 为 `value`，不计入累计 PPO 轮数。

每轮用当前策略打完五个角色各 4 局，再做 PPO 更新。游戏 seed 每轮随机生成并保存。正常战败也用于训练；引擎异常或未结束的对局不会当作完整回报训练。

```text
runs/ppo/
  current/                  唯一的当前训练检查点
  history.jsonl             每轮指标，包含内部各次 PPO 更新的指标
  status.json               采样、更新、完成或中断状态
  round-0/                  第一轮轨迹和 metadata/seeds.json
  round-1/                  第二轮轨迹
  ...
```

权重每轮覆盖 `current`。轨迹、seed 和训练历史一直保留。权重目录在 Linux 上原子替换，保存中途退出不会把已有 `current` 写坏。

中断后从最近保存的一轮继续。比如还要训练 120 轮：

```bash
python -m model --device cuda train \
  --checkpoint runs/ppo/current \
  --rounds 120 --output runs/ppo
```

模型、优化器、随机状态和累计轮次都会恢复。未完成的一轮重新采样，之前留下的轨迹不会被覆盖。warmup 完成后，续训不再加 `--value-warmup`；这个参数会让本次启动的第一轮再次只训练价值头。

### 训练监测网页

另开终端，进入同一个 conda 环境：

```bash
python -m model monitor --root runs --port 8765
```

浏览器打开 [训练监测](http://127.0.0.1:8765)。页面每 3 秒刷新，可以切换 Bootstrap / PPO 任务，查看曲线、逐轮记录和完整指标。网页单独运行，关闭网页不会影响训练。历史直接从磁盘读取，训练结束后也能看。

## 5. 让模型控制 Steam 游戏

安装本仓库的 `steam_recorder` 后，在 Steam 中开始或继续一局单人 A10，再运行：

```bash
python -m model --device cuda play-steam \
  --checkpoint runs/ppo/current \
  --bridge-dir "$RECORDING_DIR/bridge"
```

也可以把检查点换成 `runs/bootstrap/current`。Linux Python 与游戏通过文件交换状态和动作，原生 Linux、Proton、WSL 共享目录都可以使用，不依赖鼠标坐标或屏幕分辨率。

模型读取公开状态，在合法动作中选取概率最高的一项。Mod 在游戏主线程执行出牌、药水、地图、事件、营火、商店、奖励和选牌；多选完成后一次提交。动画期间等待，状态已变化的指令重新推理。`Ctrl+C` 停止控制，随后可继续手动操作。默认超过 120 秒没有可操作状态会退出并说明原因，可用 `--timeout` 调整。

控制从已经进入的对局开始，角色选择和开局由你完成。游戏升级后需要同步适配 Mod。当前桥接已针对本机 0.111.0 程序集构建，完整 Steam 实玩仍需在运行中的游戏内确认；这也不代表模型已经有稳定通关能力。

模型输入、网络结构、训练目标和推理逻辑见 [ARCHITECTURE.md](ARCHITECTURE.md)。
