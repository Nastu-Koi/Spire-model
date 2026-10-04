# Spire Model

目标：训练能通关《杀戮尖塔 2》的智能体，最终接管 Steam 游戏。当前适配游戏 **0.111.0**。示范数据覆盖 **A0–A10**，难度是模型输入；自采样和评估的难度由训练配置的 `ascension` 指定（默认 A10，与 Steam 接管一致）。

## 技术路线

1. **公开对局 Bootstrap**（`spire_codex_data/` → `model/`）：用 Spire Codex 公开胜局的对局摘要生成摘要锚点样本，对大模型做行为克隆。局外决策（营火、选牌、路线、远古之民）的标签是人类的历史选择，战斗的标签是 CombatSolver 在摘要重建的入场状态上另行求解的动作。
2. **战斗结果模型**（`combat_outcome/`，与 Bootstrap 同期预训练）：摘要锚点的每场战斗除了给出求解器的动作，也给出这场战斗的结果；模型学习"入场状态 × 遭遇 → 掉血/胜负"，用来构造本幕通过势，给 PPO 的局外决策提供局部优势。
3. **PPO**：从 Bootstrap 后的大模型出发在无头引擎中强化学习，优势 = 通关优势 + β × 本幕通过势的局部差值。
4. **接管 Steam**（`steam_recorder/` + `model/steam.py`）。

模型结构、训练路线和设计取舍见 [ARCHITECTURE](ARCHITECTURE.md)；术语见 [CONTEXT](CONTEXT.md)。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `sts2-cli/` | 无头游戏引擎（基于 [wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli)），决策协议见 `sts2-cli/docs/decision-protocol.md` |
| `combat_solver_cli/` | CombatSolver worker、引擎客户端、轨迹独立重放验证；`lib/`（不入库）存放求解器 DLL 的固定副本与本机配置 |
| `model/` | 公开状态表示、网络、策略会话、对局采样与评估、PPO/Bootstrap 训练器、奖励规则、Steam 接管 |
| `combat_outcome/` | 战斗结果模型：战斗结果标签的读取、实体模型的预训练与继续训练、自采样战斗标签的记录；模型与训练参数由 `configs/combat-outcome.json` 定义 |
| `spire_codex_data/` | Spire Codex 公开对局的下载、来源筛选、摘要锚点样本的生成、导出与核对 |
| `steam_recorder/` | Steam 版 Mod：录制实玩、接管桥接 |
| `configs/` | 大模型配置、战斗结果模型配置 |
| `runs/`、`data/` | 本地产物（不入库）：检查点、评估结果、战斗数据、公开对局缓存 |

## 安装

需要 Steam 版游戏、.NET 9 SDK，以及 Python 环境：

```bash
conda create -n sts2 python=3.11 -y
conda activate sts2
pip install -e '.[test]'
```

构建无头引擎（复制游戏程序集到 `sts2-cli/lib` 并打补丁，不修改 Steam 安装目录）：

```bash
export GAME_DIR="$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2"
bash sts2-cli/setup.sh "$GAME_DIR"
```

配置 CombatSolver 0.44.0 及 RitsuLib（把求解器 DLL 复制到 `combat_solver_cli/lib/`，编译 worker 并固定哈希，默认写入 `combat_solver_cli/lib/config.json`）：

```bash
python -m combat_solver_cli configure --solver /path/to/CombatSolver/CombatSolver.dll \
  --dependency-dir /path/to/RitsuLib/compat/0.111.0 --dependency-dir /path/to/RitsuLib/shared
```

依赖或游戏版本变化后需要重新配置并重跑引擎测试。不要提交第三方 DLL。

## 战斗结果模型

```bash
python -m spire_codex_data.anchor export --manifest data/spire-codex/sources/manifest.json --output data/anchors      # 同时写出 combat-outcomes.jsonl.gz
python -m combat_outcome.train data/anchors/combat-outcomes.jsonl.gz --save runs/combat-outcome
```

标签是摘要锚点战斗的结果，随 `anchor export` 写出，不需要另外生成；预训练与大模型的 Bootstrap 同期进行。网络结构和训练超参数来自 `configs/combat-outcome.json`（`--config` 可换成别的文件），命令行参数只覆盖单项。详见 [combat_outcome/README.md](combat_outcome/README.md)。

## 大模型

```bash
# 摘要锚点 → 独立监督样本 → Bootstrap 输入 → 行为克隆 → PPO
python -m spire_codex_data fetch --source runs --output data/spire-codex
python -m spire_codex_data.sources --existing data/spire-codex --output data/spire-codex/sources \
  --catalog data/spire-codex/catalog.json --per-character 20
python -m spire_codex_data.anchor run --manifest data/spire-codex/sources/manifest.json --output data/anchors
python -m spire_codex_data.anchor export --manifest data/spire-codex/sources/manifest.json --output data/anchors
python -m model import-independent data/anchors/independent-training.jsonl.gz --output data/bootstrap
python -m model --device cuda init --config configs/rtxpro6000.json --output runs/init
python -m model --device cuda bootstrap --data data/bootstrap --checkpoint runs/init --holdout 0.1 --output runs/bootstrap
python -m model --device cuda train --checkpoint runs/bootstrap/current --output runs/ppo
```

`import-independent` 把样本写成 gzip 分片加一个索引，`bootstrap` 每次只读几个分片进内存（`--window-shards`），分片顺序和窗口内的样本都打乱，样本权重按整个数据集归一；这种数据必须配合 `init` 生成的检查点使用。`--holdout` 按 seed 留出验证集。

`python -m model` 另有 `collect`、`evaluate`、`infer`、`import-recorder`、`play-steam`、`smoke` 等命令；`configs/rtxpro6000.json` 为默认配置。

## 接管 Steam 游戏

安装 Mod（退出游戏后）：

```bash
python steam_recorder/install.py --game-dir "$GAME_DIR" --install
```

在 Steam 中开始或继续一局后：

```bash
python -m model --device cuda play-steam --checkpoint <ckpt> --bridge-dir "$RECORDING_DIR/bridge"
```

Mod 在游戏主线程执行模型选择的动作，奖励领取使用与训练相同的固定规则。`RECORDING_DIR` 为游戏的 `user://run_recorder` 目录（日志中 `[RunRecorder] initialized; output=...`）。Mod 同时可以录制实玩，`python -m model import-recorder` 导入为行为克隆数据。

## 测试

```bash
python -m pytest -q -m "not engine"                                   # 不依赖引擎
COMBAT_SOLVER_CONFIG=combat_solver_cli/lib/config.json python -m pytest -q -m engine
python -m pytest sts2-cli/tests -q                                    # 引擎协议
python -m model --device cpu smoke --output /tmp/smoke                # 合成端到端
```

修改 `sts2-cli` 后还需按 `sts2-cli/CLAUDE.md` 跑完整对局回归。

## 公开对局数据

[spire_codex_data](spire_codex_data/README.md) 下载 Spire Codex 的对局摘要和操作回放，筛选来源，并用摘要锚点生成独立监督样本。回放只用于来源审计和核对样本。
