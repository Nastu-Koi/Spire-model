# Spire Model

基于玩家公开信息，为《杀戮尖塔 2》训练覆盖整局的策略，目标是提高**五角色 A0 平均整局通关率**并接管 Steam 游戏。当前适配游戏 **0.111.0**。

策略先用摘要锚点重建的示范做行为克隆起步，之后只从自己的对局学：按幕把对局切成几段，每幕从幕开头的存档续打几条，组内平均作基线，回报沿链回传，用裁剪目标小步更新。

| 文档 | 内容 |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | 系统设计：模块与边界、数据流、输入、网络、训练路线、评估 |
| [docs/implementation/](docs/implementation/) | 各模块怎么实现、为什么、已知的坑 |
| [docs/trials/](docs/trials/route.md) | 路线总览与每项试错的详细分析 |
| [CONTEXT.md](CONTEXT.md) | 术语 |
| 本文 | 安装与命令 |

试过而放弃的做法与路线的演变见 [docs/trials/](docs/trials/route.md)。任务与验收在本地 `.scratch/`（不入库），当前入口是 `.scratch/joint-training/spec.md`。

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| `main/env/` | 引擎协议、控制分工、奖励、入场存档、种子与划分、公开历史契约（[env](docs/implementation/env.md)） |
| `main/encoding/` | 公开观察、规则文本、冻结词表、批打包（[encoding](docs/implementation/encoding.md)） |
| `main/data/` | 分片与索引、样本权重、留出（[data](docs/implementation/data.md)） |
| `main/network/` | 网络、优化器、检查点、选择与重放（[network](docs/implementation/network.md)） |
| `main/play/` | 对局循环、多进程采样、幕组、评估报告（[play](docs/implementation/play.md)） |
| `main/learn/`、`main/iterate.py` | 数据供给、加权 NLL 训练器、幕组优势、幕组批、裁剪目标、迭代驱动（[learn](docs/implementation/learn.md)） |
| `main/demos/` | 示范导入（[demos](docs/implementation/demos.md)） |
| `main/steam.py` | 实机接管（[steam](docs/implementation/steam.md)） |
| `spire_codex_data/` | 公开对局下载、来源审计、摘要账本、锚点生成（[demos](docs/implementation/demos.md)） |
| `combat_solver_cli/` | CombatSolver worker 与客户端（[solver](docs/implementation/solver.md)） |
| `public_history/` | 无头与 Steam 共用的公开历史状态机（[env](docs/implementation/env.md#公开历史c)） |
| `steam_recorder/` | Steam Mod：录制与控制桥接（[steam](docs/implementation/steam.md)） |
| `sts2-cli/` | 基于 [wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli) 的无头引擎 |
| `configs/` | 主策略配置：20M（`rtxpro6000.json`）与 200M（`rtxpro6000-200m.json`）；迭代用的小学习率 `learning-rates-grpo.json` |

源码、测试夹具、配置与文档入库；`data/`、`runs/`、`checkpoints/`、`build/` 和第三方 DLL 不入库。以下命令都在仓库根目录运行。

## 安装

需要 Steam 版游戏、.NET 9 SDK 和 Python 3.11：

```bash
conda env create -f environment.yml
conda activate sts2
pip install -e '.[test]'
```

构建无头引擎（把游戏程序集复制到 `sts2-cli/lib` 并修改该副本）：

```bash
export GAME_DIR="$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2"
bash sts2-cli/setup.sh "$GAME_DIR"
```

求解器依赖按 [solver](docs/implementation/solver.md#配置) 配置；Steam Mod 按 [steam](docs/implementation/steam.md) 从游戏原始程序集单独构建。

## 测试

```bash
python -m pytest -q -m 'not engine and not cuda'
COMBAT_SOLVER_CONFIG=combat_solver_cli/lib/config.json python -m pytest -q -m engine
python -m pytest sts2-cli/tests -q
python -m main --device cpu smoke --output /tmp/spire-smoke
```

`-m engine` 与 `sts2-cli/tests` 需要已配置的原生依赖；`test_native_potion_costs.py` 需要支持药水价格的求解器构建（见 [solver](docs/implementation/solver.md#已知的坑)）。`smoke` 是合成引擎上的端到端检查。修改 `sts2-cli` 时另须遵守其 `CLAUDE.md` 的完整对局回归。测试通过不代表训练效果已验证。

## BC 起步

```bash
python -m main import-independent data/anchors/independent-training.jsonl.gz --output data/bootstrap
python -m main --device cuda init --config configs/rtxpro6000.json --output checkpoints/init
# 第一段：全部难度
python -m main --device cuda bootstrap --data data/bootstrap --checkpoint checkpoints/init --weights-only \
  --config configs/rtxpro6000.json --output runs/bc --epochs 10
# 第二段：只拟合 A10，战斗换成按更大预算另求的 A10 战斗
python -m main --device cuda bootstrap --data data/bootstrap data/a10-battles \
  --ascension 10 --combat-from data/a10-battles \
  --checkpoint checkpoints/bc/current --output runs/bc-a10 --epochs 10
```

- `init` 从引擎公开内容目录建立冻结词表并随机初始化；分片数据不会自行重建词表，必须给检查点。
- 训练命令（`bootstrap`、`iterate`）的检查点写进 `--checkpoints` 目录，默认是 `checkpoints/` 加 `--output` 在 `runs/` 下的相对路径（`--output runs/bc` 的检查点在 `checkpoints/bc/current`、`checkpoints/bc/epoch-NNNNNN/`）；`--output` 不在 `runs/` 下时必须给出。`runs/<运行>/` 只留运行记录。
- 续训：`--checkpoint checkpoints/<运行>/current`；从任意轮次检查点续训到新目录；收尾衰减 `--decay-updates N`；做满固定更新数就停 `--max-updates N`。各启动方式见 [network](docs/implementation/network.md#检查点)。
- 控制规则变了只需重建索引：`python -m main reindex --data <目录> [--listed-only] --workers 6`（[data](docs/implementation/data.md#数据复用)）。
- 留出 NLL：`python -m main --device cuda heldout-nll --data <目录…> --checkpoint <检查点> [--ascension 10] [--combat-from …]`。

示范的生成（下载、来源筛选、锚点、导出）见 [demos](docs/implementation/demos.md)。

## 采样与评估

```bash
# seeds.json 按五个角色各列等量种子；评估种子须与训练隔离
python -m main --device cuda evaluate --checkpoint <检查点> --seeds seeds.json --output runs/eval-x \
  --precision fp32    # 不给 --precision 时按检查点的精度（bf16）
python -m main compare runs/eval-x runs/eval-y --head 60      # 配对符号检验
```

评估默认按检查点训练时的难度对局，`--ascension 10` 改在 A10 评估；`--claim-rewards` 让奖励界面上的金币、遗物和有空位的药水由固定控制器代领（示范数据按这个假设重建），默认不开，领奖是策略决策。`--workers` 是同时运行的引擎数（每个引擎配一个采样进程），默认 64（训练机上最快），内存不够时调小，资源见 [play](docs/implementation/play.md#采样进程分工)。报告的格式见 [play](docs/implementation/play.md#评估报告)。

## 迭代

```bash
python -m main --device cuda iterate --kl 0.1 --learning-rates configs/learning-rates-grpo.json \
  --checkpoint <起点检查点> --config configs/rtxpro6000.json \
  --output runs/iterate-v1 --iterations 6 --runs-per-character 25 \
  --boundary-acts 2 3 --boundary-bosses 3 --group-n 4 --group-m 4 6 4 --min-entropy 0.45 \
  --engine-root <引擎目录> \
  --exclude-seeds <开发集与测试集种子文件…> \
  --eval-seeds <开发集种子文件> --eval-runs-per-character 100 --evaluate-every 2
```

`--explore-potions p` 让采样时战斗里有药可用的决策以概率 p 改为用药（探索步，见 [architecture](docs/architecture.md#用药探索)），给两个数时分别是普通战和精英／Boss 战的率，三个数则按普通、精英、Boss；默认 0。重跑同一条命令即断点续跑：`--checkpoint` 与 `--seed-root` 是运行身份（记在输出目录的 `iterate.json`），不能换，其余参数可以改、从剩下的步骤生效。每次迭代：按切点分波采样幕组（`--boundary-acts` 幕开头、`--boundary-bosses` Boss 入场；`--runs-per-character` 是每角色的种子数，`--group-n`、`--group-m` 是每个起点的段数）、算各波的优势、一遍更新、按间隔评估、清理。一遍默认把每一波各看一次；`--wave-shares 1 1 1` 这类相对数改各波在批里的份额，份额就是该波在梯度里的占比。更新是裁剪目标（`--clip`、`--kl`），KL 的参考每 `--reference-every k` 次迭代换成当时的策略（默认 k 等于 `--evaluate-every`，0 则固定为 `--checkpoint`）。`--learning-rates <json>`（`bootstrap` 也收）给这次运行自己的四组学习率，不给时沿用检查点的。`--fork-depth 2` 让每段开头两个有选择的决策按段在组里的序号取策略排第一或第二的候选，组在那里对半分开（分叉步，见 [architecture](docs/architecture.md#幕组的回报与优势)）。`--ascensions 0 5` 让每个种子的组在 A0–A5 里分到一个难度（难度窗口，续局同难度），评估仍按检查点的难度。`--batch 2048` 让一个优化器步累加四个 512 的加载批（必须是 profile 逻辑批的整数倍），学习率不变；`--eval-precision fp32` 让评估关掉 bf16 autocast，采样不受影响。步骤、参数、存储与资源见 [learn](docs/implementation/learn.md#迭代驱动)。检查点直接写在 `checkpoints/<运行>/`（不入库，`checkpoints/INDEX.md` 记来源、内部版本与评估），清理步只留评估过的、待评估的和下一次更新的 KL 参考；`runs/<运行>/` 只留运行记录。不给 `--eval-seeds` 就不评估；`--min-entropy` 触发时驱动在这次迭代完成后停下。

## 其它命令

```bash
python -m main parameters --config configs/rtxpro6000-200m.json      # 按 profile 报告各部分参数量，不载入权重
python -m main inspect --character Ironclad --game-seed <种子> [--ascension 0]   # 开一局，打印第一个决策帧
python -m main --device cuda infer --checkpoint <检查点> --frame <frame.json|.jsonl> [--top-k 5]   # 对帧给出命令与排名
python -m main audit <对局 json…> --accepted <输出.jsonl> --quarantine <隔离.jsonl>   # 校验完整对局文件并清洗帧
```

## Steam

```bash
python steam_recorder/install.py --game-dir "$GAME_DIR" --install
python -m main import-recorder "$RECORDING_DIR" --bootstrap-only --output data/steam-001
python -m main --device cuda play-steam --checkpoint <检查点> --bridge-dir "$RECORDING_DIR/bridge" --run-for 8h
```

接管的开关、每步间隔、决策记录与报错后的重置都在游戏内的“模型”面板上；见 [steam](docs/implementation/steam.md)。
