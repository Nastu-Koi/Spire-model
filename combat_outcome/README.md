# 战斗结果模型

当前实现学习“入场状态（卡组、遗物、药水、入场 HP）× 遭遇 → 单场净损血和是否战败”，输出为标量损血与失败 logit。以下格式、命令和指标描述这套现有实现。

目标是战斗中途、结束 HP 与关键资源的结果分布，并用学生实战校准本幕通过概率；结果分布、完整 Φ 和随机后态期望尚未实现，也未证明局部信号能改善通关率。设计见 [ARCHITECTURE](../ARCHITECTURE.md#战斗结果模型combat_outcome)，实施见 [A0 训练基线](../.scratch/a0-training-baseline/spec.md)。

## 数据

一行是一场打完的战斗（`combat-outcomes-v2`）：

- 谁打的：`source` 与 `actor` 成对出现，`summary_anchor`／`combat_solver` 是求解器，`rollout`／`policy` 是采样时的策略。求解器能看到隐藏状态，它的战斗不代表策略的能力；两者不一致的行在读取时报错。策略的行带 `policy`（版本号、权重的 SHA-256 `digest`、词表哈希、精度与采样方式），求解器的行带 `solver`（预算与是否用药）。`contract` 是这场战斗所用引擎的契约，含观测与公开历史版本。
- 入场输入 `entities`：战斗第一个决策时的公开玩家、永久卡组、遗物、药水、玩家能力；另有 `encounter`、`kind`（`regular`、`elite`、`boss`）、`character`、`act`、`floor`。
- 难度是输入：玩家实体带 `ascension`，每场战斗就是在那一局的难度下打的。遭遇实体只有名字，敌人随难度的强弱由这个字段区分；各难度的战斗一起训练，查询时给出要问的难度。
- 标签：`start_hp`、`end_hp`、`hp_lost`（两者之差）、`result`（`win`、`loss`）。净损失包含战斗结束时的结算，可以因治疗而为负；战败记为失去全部入场 HP。
- 中途帧 `frames`：这场战斗每个决策的公开观察和在那里执行的动作，按先后排列，第一帧就是入场状态。帧里只有当时的公开信息，之后的结果只在这一行的标签里。
- 其余结果字段：`turns`、`steps`、`potions_used`、`potions_discarded`（实际操作，可与帧里的动作逐一对上），策略的行另有战斗前后的 `start`／`end`（HP、最大 HP、金币、药水）和 `act_result`（这一局在这场战斗所在的幕 `passed`、`died`，或对局没有打完时为空）。
- `seed` 是划分用的分组：同一个种子的所有战斗、一场战斗的所有帧都在划分的同一侧。

两个来源，格式相同：

| `source` | 写出者 | 内容 |
|---|---|---|
| `summary_anchor` | `python -m spire_codex_data.anchor export`，写到锚点目录的 `combat-outcomes.jsonl.gz` | 摘要锚点的每一场战斗：公开胜局里每个战斗节点的入场状态，由求解器另行打一遍。胜负都保留；`frames` 指向锚点样本文件里这场战斗的各行（`file`、`group`、`count`），打输的战斗也在其中 |
| `rollout` | `online.CombatRecorder`，采样和评估时每局一个 `<对局>.combats.jsonl.gz` | 策略自己打完的每一场战斗，帧直接写在行里 |

摘要锚点的战斗就是大模型 Bootstrap 克隆其动作的那一批战斗，不需要另外生成：战斗锚点跑完、`export` 之后，两份数据同时可用。

对局循环（`model collect`、`evaluate`、`train` 的采样）按引擎的事件划分战斗：`encounter_started` 给出遭遇，`encounter_completed` 确认获胜，战斗中死亡是这场的失败。入场状态取战斗的第一个决策，结束 HP 取获胜后的第一个决策；打赢最后一场 Boss 后没有下一帧，结束 HP 从对局状态读取。超时、步数上限或引擎出错时正在打的那一场不成为标签，计入轨迹的 `combats.unfinished`；轨迹的 `combats` 另有胜负场数和各幕结果，胜场数等于引擎确认的遭遇数。对局结束、各幕结果确定之后才写出这一局的战斗。

```bash
python -m combat_outcome.data runs/ppo      # 按来源、策略、角色、难度、幕、战斗类型和胜负统计覆盖
```

`combat_outcome.data.fights(<文件或目录>)` 逐场给出标签行和它的帧，两个来源相同。

读这份数据时要知道：

- 入场状态来自人类的通关对局，卡组普遍够强；弱卡组、低 HP 入场的情况少，战败只占少数。策略自己走到的状态要靠 `rollout` 标签补。
- 入场状态由摘要账本重建，部分遗物计数是采样值（行里的 `state_sources` 标出）。
- 一局的最后一场 Boss 之后 HP 不再有用，求解器不会去保它；这一场的掉血只说明打得过，不代表必要的代价。
- 同一状态只打了一次，标签带着一次战斗随机数的噪声。

## 预训练

```bash
python -m spire_codex_data.anchor export --manifest data/spire-codex/sources/manifest.json --output data/anchors
python -m combat_outcome.train data/anchors/combat-outcomes.jsonl.gz --save runs/combat-outcome
```

与大模型的 Bootstrap 同期进行，互不依赖：两者用同一批摘要锚点战斗，一个学求解器的动作，一个学这些战斗的结果。

每场战斗被构造为一个合成观测（`frames.py`）：入场状态的公开实体，加一个 `encounter` 实体，以及一个指向它的 `PREDICT_COMBAT` 虚拟动作。网络（`model.py`）复用策略的实体编码器与全局注意力层，尺寸由配置给出；在动作位置读出后，一个输出预测掉血占最大 HP 的比例（有符号，tanh），另一个预测是否战败。

首次训练用引擎的完整静态内容目录初始化词表；训练集没出现的遭遇也有独立的内容 ID。默认从本地已构建的引擎读取 `public_catalog`；训练机器没有引擎时，可加 `--catalog data/spire-codex/catalog.json`，使用 `spire_codex_data.sources` 导出的同一游戏版本的目录。标签中的遭遇名统一为 `ENCOUNTER.<名称>`，接受带前缀和不带前缀两种写法。目录以外的遭遇会在训练前报错。

数据按种子分组划分训练/测试集：同一个种子的对局地图和遭遇相同，不会同时出现在两边；至少需要两个种子。实体以压缩形式留在内存里，按批解开，批的构造放在 `loader_workers` 个进程里。

同一划分下的对照基线：只用遭遇和入场 HP；卡组汇总特征的岭回归（`features.py`）；汇总特征加卡牌/遗物 id 词袋的岭回归。报告测试集上全部战斗和各类战斗的 R²、平均绝对误差（HP），以及预测战败率与实际战败率。

`--save <目录>` 保存 `weights.pt` 和 `manifest.json`（`model`、`training` 两节、词表、数据版本与报告）。`combat_outcome.model.load_model(<目录>)` 只凭 manifest 重建网络和词表。

共享编码器现在使用 `public-history-v1` 输入，manifest 也保存此版本；旧检查点不能在新输入上直接续训。旧入场数据缺失公开历史时保持 unknown。这一版本检查不表示学生中途结果记录、结果分布或本幕 Φ 已实现。

## 接着训练

以下命令读取采样目录里已有的战斗标签；训练入口本身不采样。现有网络只用每场战斗的入场输入，`frames` 留给结果分布使用。

```bash
python -m combat_outcome.train runs/ppo --pretrained runs/combat-outcome --save runs/combat-outcome-2
```

`--pretrained` 从保存的模型继续，沿用它的词表和配置；数据可以是单个标签文件，也可以是目录（递归读取其中名字以 `combats.jsonl`、`combats.jsonl.gz` 结尾的文件和 `combat-outcomes.jsonl.gz`）。报告的 `policies` 列出这些标签出自哪些策略。续训和推理都拒绝词表未登记的遭遇，错误会列出对应 ID；需要用包含这些 ID 的静态目录重新初始化，不能用 `--catalog` 改写已有检查点的词表。

其他选项：`--summary-entity` 在输入中加一个卡组汇总实体；`--ridge-base` 让网络学习汇总岭回归的残差（岭回归本身不保存，这种模型不能加载或继续训练）；`--folds/--fold`、`--train-runs`、`--limit` 用于按种子交叉验证和学习曲线；`--baselines-only` 只跑岭回归，`--no-baselines` 跳过它。

## 配置

`configs/combat-outcome.json` 是默认配置，`--config` 可指定其他文件：

| 节 | 内容 |
|---|---|
| `model` | 网络结构，字段与策略的 `ModelConfig` 相同（宽度、层数、FFN、局部编码器、词表容量、注意力后端等）；必填 |
| `training` | 轮数、批大小、学习率、权重衰减、梯度裁剪、预热比例、战败分类损失权重 `defeat_weight`、`summary_entity`、`ridge_base`、`holdout`、`seed`、`loader_workers` |

`training` 可以只写要改的项，其余取 `config.py` 中的默认值；未知的节或字段会报错。命令行参数（`--epochs`、`--batch`、`--lr`、`--holdout`、`--seed`、`--summary-entity`、`--ridge-base`、`--loader-workers`）覆盖配置中的对应项。

这些设置属于战斗结果模型。主策略 BC 的全数据训练、固定学习率和人工验证安排单独维护在 [model/README.md](../model/README.md#当前训练安排)，不使用本模块的 `--holdout`。

## 模块

| 文件 | 内容 |
|---|---|
| `config.py` | 配置的两节及校验 |
| `data.py` | 标签的读取（实体压缩保存）、逐场的中途帧、覆盖统计与按种子划分 |
| `online.py` | 对局循环里策略所打战斗的记录 |
| `frames.py`、`model.py` | 输入实体与合成观测；网络、保存与加载 |
| `features.py`、`baselines.py`、`metrics.py` | 汇总特征与岭回归；同一划分下的基线；R² 与误差 |
| `train.py` | 训练入口 |
