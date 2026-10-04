# 战斗结果模型

学习"入场状态（卡组、遗物、药水、入场 HP）× 遭遇 → 这场战斗掉多少 HP、是否战败"。目的是给局外决策提供比整局胜负更稠密、更低噪声的信号：它与公开遭遇池、地图 HP 预算组合成本幕通过势 Φ，Φ 在一个决策前后的差值是 PPO 的局部优势（见 [ARCHITECTURE](../ARCHITECTURE.md) 的"训练路线"）。

## 数据

一行是一场由求解器打完的战斗（`combat-outcomes-v1`）：

- 输入 `entities`：战斗第一个决策时的公开玩家、永久卡组、遗物、药水、玩家能力；另有 `encounter`、`kind`（`regular`、`elite`、`boss`）、`character`、`act`。
- 难度是输入：玩家实体带 `ascension`，每场战斗就是在那一局的难度下打的。遭遇实体只有名字，敌人随难度的强弱由这个字段区分；各难度的战斗一起训练，查询时给出要问的难度。
- 标签：`start_hp`、`end_hp`、`hp_lost`（两者之差）、`result`（`win`、`loss`）。净损失包含战斗结束时的结算，可以因治疗而为负；战败记为失去全部入场 HP。
- 其余字段不进模型：`seed`（划分用的种子分组）、`solver`（预算与是否用药）、`turns`、`potions_used`（实际用药是结果，不是输入）、`source`。

两个来源，格式相同：

| `source` | 写出者 | 内容 |
|---|---|---|
| `summary_anchor` | `python -m spire_codex_data.anchor export`，写到锚点目录的 `combat-outcomes.jsonl.gz` | 摘要锚点的每一场战斗：公开胜局里每个战斗节点的入场状态，由求解器另行打一遍。胜负都保留 |
| `rollout` | `online.CombatRecorder`，每局一个 `combats.jsonl` | 策略自采样对局里求解并打完的战斗；未打完的不生成标签。尚未接入对局循环 |

摘要锚点的战斗就是大模型 Bootstrap 克隆其动作的那一批战斗，不需要另外生成：战斗锚点跑完、`export` 之后，两份数据同时可用。

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

数据按种子分组划分训练/测试集：同一个种子的对局地图和遭遇相同，不会同时出现在两边；至少需要两个种子。实体以压缩形式留在内存里，按批解开，批的构造放在 `loader_workers` 个进程里。

同一划分下的对照基线：只用遭遇和入场 HP；卡组汇总特征的岭回归（`features.py`）；汇总特征加卡牌/遗物 id 词袋的岭回归。报告测试集上全部战斗和各类战斗的 R²、平均绝对误差（HP），以及预测战败率与实际战败率。

`--save <目录>` 保存 `weights.pt` 和 `manifest.json`（`model`、`training` 两节、词表、数据版本与报告）。`combat_outcome.model.load_model(<目录>)` 只凭 manifest 重建网络和词表。

## 接着训练

```bash
python -m combat_outcome.train runs/ppo --pretrained runs/combat-outcome --save runs/combat-outcome-2
```

`--pretrained` 从保存的模型继续，沿用它的词表（没登记过的名字落到备用行）和配置；数据可以是单个标签文件，也可以是目录（递归读取其中的 `combats.jsonl`、`combats.jsonl.gz`、`combat-outcomes.jsonl.gz`）。

其他选项：`--summary-entity` 在输入中加一个卡组汇总实体；`--ridge-base` 让网络学习汇总岭回归的残差（岭回归本身不保存，这种模型不能加载或继续训练）；`--folds/--fold`、`--train-runs`、`--limit` 用于按种子交叉验证和学习曲线；`--baselines-only` 只跑岭回归，`--no-baselines` 跳过它。

## 配置

`configs/combat-outcome.json` 是默认配置，`--config` 可指定其他文件：

| 节 | 内容 |
|---|---|
| `model` | 网络结构，字段与策略的 `ModelConfig` 相同（宽度、层数、FFN、局部编码器、词表容量、注意力后端等）；必填 |
| `training` | 轮数、批大小、学习率、权重衰减、梯度裁剪、预热比例、战败分类损失权重 `defeat_weight`、`summary_entity`、`ridge_base`、`holdout`、`seed`、`loader_workers` |

`training` 可以只写要改的项，其余取 `config.py` 中的默认值；未知的节或字段会报错。命令行参数（`--epochs`、`--batch`、`--lr`、`--holdout`、`--seed`、`--summary-entity`、`--ridge-base`、`--loader-workers`）覆盖配置中的对应项。

## 模块

| 文件 | 内容 |
|---|---|
| `config.py` | 配置的两节及校验 |
| `data.py` | 标签的读取（实体压缩保存）与按种子划分 |
| `online.py` | 自采样对局里战斗标签的记录 |
| `frames.py`、`model.py` | 输入实体与合成观测；网络、保存与加载 |
| `features.py`、`baselines.py`、`metrics.py` | 汇总特征与岭回归；同一划分下的基线；R² 与误差 |
| `train.py` | 训练入口 |
