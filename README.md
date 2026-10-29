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
| `main/steam/` | 实机接管（[steam](docs/implementation/steam.md)） |
| `spire_codex_data/` | 公开对局下载、来源审计、摘要账本、锚点生成（[demos](docs/implementation/demos.md)） |
| `combat_solver_cli/` | CombatSolver worker 与客户端（[solver](docs/implementation/solver.md)） |
| `public_history/` | 无头与 Steam 共用的公开历史状态机（[env](docs/implementation/env.md#公开历史c)） |
| `steam_takeover/` | Steam Mod：实机接管的控制桥接（[steam](docs/implementation/steam.md)） |
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
- 留出 NLL：`python -m main --device cuda heldout-nll --data <目录…> --checkpoint <检查点> [--ascension 10] [--combat-from …] [--all-groups]`。默认只算留出的种子组；`--all-groups` 算全部种子组，用于整份都不进训练的数据，例如战斗评估集上求解器打的战斗（`data/fight-eval-20261022/solver-demos`）。

示范的生成（下载、来源筛选、锚点、导出）见 [demos](docs/implementation/demos.md)。求解器重打幕组采样打过的战斗、作战斗示范：

```bash
python -m spire_codex_data.entry_battles run --selection <selection.json> --output <目录>
python -m spire_codex_data.entry_battles export --output <目录>
```

`selection.json` 的格式、默认药价与输出见 [demos](docs/implementation/demos.md#入场存档的战斗)。

## 采样与评估

```bash
# seeds.json 按五个角色各列等量种子；评估种子须与训练隔离
python -m main --device cuda evaluate --checkpoint <检查点> --seeds seeds.json --output runs/eval-x \
  --precision fp32    # 不给 --precision 时按检查点的精度（bf16）
python -m main compare runs/eval-x runs/eval-y --head 60      # 配对符号检验
```

评估默认按检查点训练时的难度对局，`--ascension 10` 改在 A10 评估；`--claim-rewards` 让奖励界面上的金币、遗物和有空位的药水由固定控制器代领（示范数据按这个假设重建），默认不开，领奖是策略决策。`--workers` 是同时运行的引擎数（每个引擎配一个采样进程），默认 64（训练机上最快），内存不够时调小，资源见 [play](docs/implementation/play.md#采样进程分工)。报告的格式见 [play](docs/implementation/play.md#评估报告)。

## 战斗评估集

```bash
# 抽一次，之后固定：普通、精英、Boss 3:4:4，类内均匀，--root 定随机数
python -m main fight-set --sources <入场存档目录…> --exclude <示范用过的 selection.json…> \
  --root fight-eval-20261023 --output data/fight-eval-20261023
# 求解器在同一批存档上打一遍（按毫秒计时，不与采样或别的重任务并行）
python -m spire_codex_data.entry_battles run --selection data/fight-eval-20261023/fights.json \
  --output data/fight-eval-20261023/solver
# 求解器的轨迹导出、导入成数据目录（在这批战斗上算示范的 NLL 用）
python -m spire_codex_data.entry_battles export --output data/fight-eval-20261023/solver
python -m main import-independent data/fight-eval-20261023/solver/independent-training.jsonl.gz \
  --output data/fight-eval-20261023/solver-demos
# 策略 greedy 逐场打到战斗结束，出报表；集里有 solver/ 时带上与求解器的对照
python -m main --device cuda fight-eval --checkpoint <检查点> --fights data/fight-eval-20261023 \
  --output runs/fight-eval-20261023/<名字> --precision fp32
python -m main fight-compare runs/fight-eval-20261023/<a> [runs/fight-eval-20261023/<b>] [--solver <目录>] --text
```

- `fight-set` 的来源是入场存档目录（每份存档一个文件，带对局的角色、种子、难度；幕组轮的 `<波>/fights/` 不带，从同波的 `fights.jsonl` 补），去掉事件战斗和 `--exclude` 的 selection 里的战斗，默认 440 场。写出 `saves/`、`fights.json`（也是 `entry_battles` 的 selection）与 `seeds.json`。现有的集是 `data/fight-eval-20261023/`，来自回血修复后的引擎写的入场存档。
- 集的种子不能拿去采样：`iterate --exclude-seeds` 要加上 `data/fight-eval-20261023/seeds.json`。
- `fight-eval` 默认 8 个引擎（`--workers`），`--precision` 同 `evaluate`，`--per-kind N` 只打每类前 N 场作试跑，`--solver` 指定求解器结果的目录（默认是集里的 `solver/`）。输出目录写 `fights.jsonl`（逐场的战况、代价分项与打法）、`metadata.json`、`report.json`、`report.txt`。
- 报表给分种类的平均代价（药价固定为 7／9／11，最终 Boss 1／1／1）、死亡率、每场用药数与用过药的场次占比、死时持药未用的场数（不算整局最后一场）、危急帧上的用药概率；有求解器结果时再给逐场的代价差与行为差异表（每种药水、每种开局选择、每种多选两边的频率）。`fight-compare` 不重打，从逐场行重出报表；给两个目录时逐场比较两次打法。
- `fight-eval --growth-bonus h`（`fight-compare` 同）按成长奖励出报表：分项多 `growth`，另给不含它的 `cost_without_growth`。
- `fight-eval --prices <prices.json>` 另写一份按拟合药价算的 `report-prices.json`／`.txt`；`fight-compare --prices` 按那套价格重出报表。两者都只有策略一方，求解器的战况算不了拟合价。
- 格式与口径见 [play](docs/implementation/play.md#战斗评估集) 与 [learn](docs/implementation/learn.md#战斗评估报表)。

## 迭代

```bash
python -m main --device cuda iterate --kl 0.1 --learning-rates configs/learning-rates-grpo.json \
  --checkpoint <起点检查点> --config configs/rtxpro6000.json \
  --output runs/iterate-v1 --iterations 6 --runs-per-character 25 \
  --boundary-acts 2 3 --group-n 4 --group-m 4 4 --min-entropy 0.45 \
  --fight-groups 220 --potion-prices <prices.json> --growth-bonus 0.5 --win-hp-bonus 0.1 \
  --source-shares act-groups=1 fight-groups=0.4 --wave-shares 2 1 1 --ascensions 9 9 \
  --engine-root <引擎目录> \
  --exclude-seeds <开发集、测试集与战斗评估集的种子文件…> \
  --eval-seeds <开发集种子文件> --eval-runs-per-character 100 --evaluate-every 2 --eval-precision fp32
```

- 断点续跑：重跑同一条命令即可。`--checkpoint` 与 `--seed-root` 是运行身份（记在输出目录的 `iterate.json`），不能换；其余参数可以改，从剩下的步骤生效。
- 每次迭代：按幕开头的切点（`--boundary-acts`）分波采样幕组（`--runs-per-character` 是每角色的种子数，`--group-n`、`--group-m` 是每个起点的段数）、算各波的优势、一遍更新、按间隔评估、清理。步骤、参数、存储与资源见 [learn](docs/implementation/learn.md#迭代驱动)。
- 幕组固定为 greedy 战斗：战斗内的决策取策略最可能的动作、不留帧，只有局外决策按温度抽，优势与更新只有局外决策（见 [architecture](docs/architecture.md#采样温度)）。每波另写各段每场战斗的战况（`fights.jsonl`）与非事件战斗的入场存档（`fights/`），这次迭代的清理步删除（见 [play](docs/implementation/play.md#幕组)）。组内各段回报全相等的组（全组同结果的组）固定不进更新，份额与一遍的步数按剩下的决策数算。
- 份额：一遍默认把每一波各看一次；`--wave-shares 1 1 1` 这类相对数改各波在批里的份额，份额就是该波在梯度里的占比。`--source-shares act-groups=1 fight-groups=1`（还有 `combat-demos`）给各类来源在批里的相对份额，要列全这次更新有的来源；不给时按决策数。
- 回报：`--win-hp-bonus 0.1`（默认 0）让通关的段回报另加 0.1 × 剩余 HP / 最大生命（剩血奖励，HP 取最后一场战斗的战况），全赢的组因此按剩血分开（见 [architecture](docs/architecture.md#幕组的回报与优势)）。
- 更新：裁剪目标（`--clip`、`--kl`），KL 的参考每 `--evaluate-every` 次迭代换成当时的策略（第一段是 `--checkpoint`）。`--learning-rates <json>`（`bootstrap` 也收）给这次运行自己的四组学习率，不给时沿用检查点的。`--batch 2048` 让一个优化器步累加四个 512 的加载批（必须是 profile 逻辑批的整数倍），学习率不变。`--fight-kl 0` 给战斗组单独的 KL 系数（不给时同 `--kl`；为 0 时不跑参考前向）；`--passes 2` 让一次更新在这一轮上走两遍，第二遍的比值仍对采样时的对数概率算；`--freeze-outside` 让幕组各波的优势全为 0、只留对参考的 KL，局外不学、只被拉住（战斗快循环用，见 [architecture](docs/architecture.md#战斗快循环)）。
- 难度：`--ascensions 0 5` 让每个种子的组在 A0–A5 里分到一个难度（难度窗口，续局同难度），评估仍按检查点的难度。
- 局外探索（见 [architecture](docs/architecture.md#局外探索)）：
  - `--explore-outside p`（或 `map=p1 event=p2` 这样按阶段给）让幕组里有选择的局外决策以概率 p 不取策略最可能的候选：一半取第二可能的，其余在剩下的里随机取，不取丢药；默认 0。
  - `--explore-prior p --prior <检查点>`（率的格式同上）让这些决策以概率 p 问先验网络最可能的候选，它不是策略最可能的时改取它（先验提议，记为探索步）；默认 0。
  - `--explored-on-tree` 让局外探索步只在分叉树定价时学（其余不给优势）；`--explored-centered` 让参与学习的局外探索步优势按波按阶段去均值（只留提议之间的好坏）。
- 战斗组：`--fight-groups N`（默认 0）在幕组各波之后采战斗组。
  - 挑选：从这一轮记下的非事件战斗里按普通、精英、Boss 3:4:4 挑 N 场，每类 `--fight-group-priority`（默认 0.5）的份额按死亡、战斗代价从高到低挑，其余均匀抽；某类不够时如实记下、不从别类补。`--fight-group-shares normal=2 elite=4 boss=5` 改三类的配比；`--fight-pool <战斗评估集…> --fight-pool-share 0.5` 让一半场次从这些集的入场存档里均匀抽（比如人类 A10 的入场状态）。`--fight-group-natural` 让这一轮自己的那部分按自然频率挑（每局每幕计一次），不被幕组的扇出挤向第三幕。
  - 采样：每场从入场存档续打 `--fight-group-size`（默认 8）条、截在这场战斗结束，`--fight-group-kind-size boss=16` 让某类的组另定条数；战斗按检查点的战斗温度抽、留全部帧，不探索、不强制。写在轮目录的 `fight-groups/`，`sample` 记录与轮 summary 记挑选结果。
  - 定价：优势步按药价由战况算优势（默认固定 7／9／11；`--potion-prices <prices.json>` 改用 `fit-prices` 的拟合价，挑选也按它，一次迭代内不能换价；`--growth-bonus h` 让每个成长事件抵 h HP），同样整组剔除全组同结果的组（`advantages` 记录的 `fight-groups`，按战斗种类记组内诊断，见 [learn](docs/implementation/learn.md#战斗组的优势)）。
  - 更新时与幕组各波放进同一批（来源 `fight-groups`，同一裁剪目标与 KL）；这次迭代的清理步删掉它的分片、战况与优势（见 [play](docs/implementation/play.md#战斗组)）。
- 回溯分叉：`--fight-hindsight N`（默认 0，要求 `--fight-groups`；`--fight-hindsight-shares` 改主干的三类配比，某类给 0 就不分叉它）在战斗组之后做。
  - 每轮从战斗组的段里按普通、精英、Boss 3:4:4 取 N 条主干（不论死活），按主干的动作重放到至多三个分叉点再分枝、打到战斗结束：第一个危急帧（每种能喝的药各一条强制喝、两条照常续打），第 2 回合起策略前两名动作最接近的决策（两个动作各强制两条），第 2 回合起随机一回合的第一个决策（四条照常续打）。
  - 写在轮目录的 `fight-hindsight/`，优势按战斗组的算法另算（`advantages` 记录的 `fight-hindsight`），更新时作为战斗组那一类的第二路、两路按决策数分这一类的份额，清理步同样删掉（见 [play](docs/implementation/play.md#回溯分叉)）。
- 战斗示范（BC 阶段）：`--round-demos N`（默认 0）让每轮在优势步之后从这一轮记下的非事件战斗里挑 N 场交给求解器。
  - 挑选：按普通、精英、Boss 3:4:4 分场数，类内按自然频率（每局每幕计一次）抽，开局是选择的战斗（赌博筹码一类）先抽、最多占 `--demo-opening`（默认 0.2）。
  - 求解：以 `--demo-workers`（默认 16，一个物理核一局）局并行，赢下的战斗导入成轮目录的 `demos/imported/`。求解按毫秒预算计时，这一步期间机器上不要跑别的重任务。
  - 更新学这一轮与之前共 `--demo-buffer`（默认 3）轮的示范，在同一批里另放（来源 `combat-demos`），每个宏动作按 NLL 学、不算 KL；一遍仍把局外各波各看一次，示范按份额取：多于有的就整场重复，少就按种子根与迭代号可复现地抽一部分整场（见 [learn](docs/implementation/learn.md#混合批)）。
  - 清理步删掉求解器的样本与导出，导入在不再被后面的更新读到时删掉，挑选清单与求解器的报告保留（见 [play](docs/implementation/play.md#战斗示范的挑选)、[demos](docs/implementation/demos.md#逐轮的战斗示范)）。
- 评估与停止：不给 `--eval-seeds` 就不评估；`--eval-precision fp32` 让评估关掉 bf16 autocast，采样不受影响。`--min-entropy` 触发时驱动在这次迭代完成后停下。
- 检查点直接写在 `checkpoints/<运行>/`（不入库，`checkpoints/INDEX.md` 记来源、内部版本与评估），清理步只留评估过的、待评估的和下一次更新的 KL 参考；`runs/<运行>/` 只留运行记录。

## 药价拟合

```bash
setsid python -m main --device cuda fit-prices --checkpoint <检查点> --seed-root fit-prices-<日期> \
  --exclude-seeds runs/bootstrap-20261008/eval-a0/seeds-dev.json runs/bootstrap-20261008/eval-a0/seeds-dev-2000.json \
                  runs/bootstrap-20261008/eval-a0/seeds-final-test.json data/fight-eval-20261023/seeds.json \
  --output runs/<运行>/fit-prices/<名字> [--previous fixed|<上一次的 prices.json>]
```

- 做什么：
  1. 检查点打来源对局（每角色 `--source-runs-per-character`，默认 100；局外按检查点温度，战斗 greedy；A0–A5）；
  2. 在战后第一个选图决策、腰带有药时取点；
  3. 按幕 × 腰带满否 × HP 段抽 `--pairs` 对（默认 10,000，低 HP 段权重加倍）；
  4. 每点续打一条原样的，每瓶药各续打一条去掉它的，全部 greedy，打到下一幕开头（第三幕打到终局）；
  5. 按配对差拟合药价模型，再与 `--previous` 平滑。
- 输出：`points/`、`branches/`、`branches.jsonl.gz`、`fitted.json`（拟合值）、`prices.json`（平滑后的价格，交给 `iterate --potion-prices`）、`report.json`（交叉验证、系数与标准误、确定性、各层的点数）。
- 每一步都落盘，同一条命令重跑就接着做。
- 默认 64 个引擎（`--workers`），与一轮采样同样吃内存，不要和训练同时跑；`--memory-floor`（默认 2.5 GB）低于这个可用内存就结束进程组，所以要用 `setsid` 启动。
- 方法见 [learn 药价模型](docs/implementation/learn.md#药价模型)，取点见 [play](docs/implementation/play.md#药价拟合的取点与续打)。

## 其它命令

```bash
python -m main parameters --config configs/rtxpro6000-200m.json      # 按 profile 报告各部分参数量，不载入权重
python -m main inspect --character Ironclad --game-seed <种子> [--ascension 0]   # 开一局，打印第一个决策帧
python -m main --device cuda infer --checkpoint <检查点> --frame <frame.json|.jsonl> [--top-k 5]   # 对帧给出命令与排名
python -m main audit <对局 json…> --accepted <输出.jsonl> --quarantine <隔离.jsonl>   # 校验完整对局文件并清洗帧
```

## Steam

```bash
python steam_takeover/install.py --game-dir "$GAME_DIR" --install
python -m main --device cuda play-steam --checkpoint <检查点> --bridge-dir "$TAKEOVER_DIR/bridge" --run-for 8h
```

`TAKEOVER_DIR` 是游戏数据目录下的 `spire_takeover`。接管的开关、每步间隔、决策记录与报错后的重置都在游戏内的“模型”面板上；见 [steam](docs/implementation/steam.md)。
