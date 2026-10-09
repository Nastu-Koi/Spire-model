# Spire Model

基于玩家公开信息，为《杀戮尖塔 2》训练覆盖整局的策略，目标是提高**五角色 A0 平均整局通关率**并接管 Steam 游戏。当前适配游戏 **0.111.0**。

主策略先用摘要锚点及求解器示范做行为克隆，再自主采样。当前已有 BC、基础 MC PPO、公开历史、战斗结果记录和 Steam 桥接；结果分布、本幕通过势 Φ、两路优势及冻结 BC 参考 KL 尚未接入。设计见 [ARCHITECTURE](ARCHITECTURE.md)，术语见 [CONTEXT](CONTEXT.md)。

## 从这里开始

| 要做的事 | 说明 |
| --- | --- |
| 开始／继续训练，复用现有数据 | [主策略训练与运行](model/README.md) |
| 判断数据要不要重新生成 | [数据复用与索引](model/README.md#数据复用与索引) |
| 下载公开对局、恢复锚点和导入示范 | [摘要数据](spire_codex_data/README.md) |
| 配置求解器或验证原生轨迹 | [CombatSolver worker](combat_solver_cli/README.md) |
| 训练战斗结果模型、读取学生战斗数据 | [战斗结果模型](combat_outcome/README.md) |
| 安装录制 Mod、接管 Steam 游戏 | [Steam 录制与桥接](steam_recorder/README.md) |
| 检查公开观察与合法动作协议 | [决策协议](sts2-cli/docs/decision-protocol.md) |

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| [model/](model/README.md) | 公开输入、主策略、控制器、BC／PPO、检查点、Steam 控制端和训练监测 |
| [spire_codex_data/](spire_codex_data/README.md) | 公开对局下载、来源审计、摘要账本、锚点生成、刷新和导出 |
| [combat_outcome/](combat_outcome/README.md) | 当前入场标量结果模型及学生战斗／中途帧记录 |
| [combat_solver_cli/](combat_solver_cli/README.md) | 第三方求解器 worker、引擎客户端和独立轨迹复验 |
| [sts2-cli/](sts2-cli/README.md) | 基于 [wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli) 的无头引擎 |
| [public_history/](public_history/README.md) | 无头与 Steam 共用的公开历史状态机和原生钩子 |
| [steam_recorder/](steam_recorder/README.md) | Steam Mod：实玩录制与控制桥接 |
| `configs/` | [主策略](configs/rtxpro6000.json)与[战斗结果](configs/combat-outcome.json)配置 |
| `docs/agents/` | 仓库内的领域文档和本地工单约定 |

源码、测试夹具、配置及使用文档入库。`data/` 保存本地数据，`runs/`、`checkpoints/` 保存实验与检查点，`build/` 保存构建辅助产物和 Mod 备份；这些目录与第三方 DLL 均不入库。整理仓库时保留原始数据、旧对照检查点和回放证据。

文档各有一个维护位置：模块 README 说明当前用法，`ARCHITECTURE.md` 解释设计，`CONTEXT.md` 定义术语。当前工作区的任务、验收和历史实验记录在本地 `.scratch/`，不随 Git 分发；入口为 [A0 训练基线](.scratch/a0-training-baseline/spec.md)和[实施报告](.scratch/a0-training-baseline/implementation-report.md)。

## 安装

需要 Steam 版游戏、.NET 9 SDK 和 Python 3.11：

```bash
conda env create -f environment.yml
conda activate sts2
```

已有 Python 环境时使用 `pip install -e '.[test]'`。构建无头引擎会将游戏程序集复制到 `sts2-cli/lib` 并修改该副本：

```bash
export GAME_DIR="$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2"
bash sts2-cli/setup.sh "$GAME_DIR"
```

随后按 [worker 配置](combat_solver_cli/README.md#配置)安装固定版本的求解器依赖。Steam Mod 需要从游戏原始程序集单独构建，见 [安装说明](steam_recorder/README.md)。

## 大模型

当前数据与 BC 训练都在本地 `runs/bootstrap-20261008/`。BC 不自动验证或按表现停止，由使用者自行判断。

新训练、严格续训、当前数据数量、版本迁移及可直接执行的命令统一维护在 [model/README.md](model/README.md)。战后及事件领奖由模型选择，普通宝箱、水晶球格子和指定单选事件按[共用控制规则](model/README.md#策略与控制器)执行。

## 战斗结果模型

[combat_outcome](combat_outcome/README.md) 复用锚点求解器战斗作为预训练标签，也读取主策略实际打出的战斗。当前网络预测入场单场的掉血和失败概率，尚不提供完整结果分布或 Φ。

## 训练监测网页

```bash
python3 -m model.monitor --runs runs --port 8765
```

打开 [本地训练监测](http://127.0.0.1:8765)。服务只读日志，配置、指标含义和远程访问见 [监测说明](model/README.md#训练监测网页)。

## 接管 Steam 游戏

[steam_recorder](steam_recorder/README.md) 说明 Mod 安装、录制导入和 `play-steam` 用法。Python 控制端与无头共享动作分工；当前已通过桥接接口测试，最新控制改动尚未进行 Steam 实机接管验证。

## 测试

```bash
python -m pytest -q -m 'not engine and not cuda'
COMBAT_SOLVER_CONFIG=combat_solver_cli/lib/config.json python -m pytest -q -m engine
python -m pytest sts2-cli/tests -q
python -m model --device cpu smoke --output /tmp/spire-smoke
```

`-m engine` 和 `sts2-cli/tests` 须先完成原生依赖配置；`smoke` 是合成端到端检查。修改 `sts2-cli` 时另须遵守其 `CLAUDE.md` 的完整对局回归要求。模块专用检查见各自 README；测试通过不代表训练效果已验证。

## 公开对局数据

[spire_codex_data](spire_codex_data/README.md) 使用 `.run` 摘要生成独立监督样本；操作回放只用于来源审计和核对。这些示范用于 BC，不是策略自行采集的 PPO 经验，也不构成连续通关证明。
