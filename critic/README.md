# 胜率模型（critic）

独立于策略的小网络：输入一个局外决策的公开观察（与策略相同的决策帧），输出当前策略**通关**和**通过本幕**的概率。它给局外决策提供基线与时序差分优势，用采样对局离线训练、逐次迭代续训，有自己的版本号。设计见 [ARCHITECTURE](../ARCHITECTURE.md#策略改进)，目标与验收见 [工单 18](../.scratch/a0-training-baseline/issues/18-win-probability-critic.md)；实测数字记在工单的 Comments，不写在这里。

## 数据

一行是策略在战斗以外做的一个决策（`critic-decisions-v1`，`data.py`）：

- 帧：战斗之外的宏动作里第一个有选择的步骤的决策帧；它前面的强制步骤和控制器步骤（宝箱、水晶球、单选事件）不是决策。对局标了战斗（宏动作的 `fight`，按引擎的遭遇事件划界）时以它为准，战斗中的选牌属于战斗；更早写出的对局没有这个标记，只把 `combat` 阶段当作战斗。帧只保留 `public` 与 `legal`，压缩放在内存里，由加载进程按批解析。
- 标签：`win`（这一局是否通关）；`act_passed`（帧所在的幕是否通过：账本 `ledger.bosses` 含该幕，或整局通关）。幕和楼层取自玩家实体。
- 元信息：seed、角色、幕、楼层、阶段、决策编号、打出这局的策略版本与权重 digest，以及一份状态摘要（HP、最大 HP、金币、卡组／遗物／药水数），后者只给表格基线用。

来源是完整对局（`spire-trajectory-v1`），四种写法都读：采样轮（`collect`／`iterate` 写出的 `round-N/`，经索引逐分片读，校验与策略的加载器相同）；缓冲目录（其下已写完的各轮，`--buffer-iterations N` 只取最近 N 次迭代的）；每局一个 JSON 的目录（`evaluate` 写出的，`metadata/` 与报告、流水账会被跳过）；每行一局的 `.jsonl`／`.jsonl.gz`。未完成的对局不进数据。采样轮只保存局外决策的帧，它的行与策略按优势加权的样本一一对应，行里另记宏动作在对局里的序号和迭代号。

划分按 seed 分组，用策略的留出规则（`model.data.held_out`）：一个 seed 的全部决策在同一侧，加数据不改变已有 seed 的归属，分组键与策略一致；默认留出比例为 20%，策略为 2%，因此策略留出种子包含在 critic 的留出种子中。

```bash
python -m critic.data runs/bootstrap-20261008/eval-a0/dev-v8      # 对局数、决策数、按幕／角色／阶段的通关率与过幕率
python -m critic.data runs/iterate-v1/buffer --buffer-iterations 10      # 缓冲里最近十次迭代的各轮
```

## 网络

与策略同结构：`PolicyNetwork` 的实体编码器与全局层（`model` 节的 `ModelConfig`），在全局标记（第 0 个 token）的最终状态上接一个两 logit 的线性头：通关、过本幕。不用策略的决策头和价值头；它们不训练、也不保存。实体编码器和头在 FP32 下计算，全局层默认 BF16 autocast。

词表用策略检查点的冻结词表；默认（`init=checkpoint`）编码器与全局层也从该检查点的权重出发，`--init random` 则只取词表。头的偏置初始化为训练集的基础率，所以没训练的 critic 就是常数基线。默认只过一轮、查找表冻结（`embedding_lr=0`）：一局的几十个决策共用同一个胜负，几百局的数据上多过几轮或放开卡牌符号表，网络先记住的是每一局是谁而不是能迁移的东西。

## 训练

```bash
python -m critic.train runs/bootstrap-20261008/eval-a0/dev-v8 \
  --checkpoint runs/bootstrap-20261008/trained/current --save runs/critic/v1
python -m critic.train runs/buffer --pretrained runs/critic/v1 --save runs/critic/v2      # 下一次迭代续训
```

- `--checkpoint`：策略检查点，提供词表和（默认）主体初始权重；结构须与配置的 `model` 节一致（`backend`、`compile_blocks` 等运行时项除外）。
- `--pretrained`：从已保存的 critic 继续，沿用其词表、模型和训练配置，版本号加一；与 `--checkpoint` 二选一。不传 `--config` 时无需原 profile；显式 profile 的 `critic` 字段和命令行参数依次覆盖保存值，未写出的字段保留。`--no-compile` 同样作用于恢复的网络。续训禁止改变 `holdout`，且要求 `data.split.version` 为 `seed-group-v1`；旧划分的 critic 仍可加载推理，但训练须从策略检查点新建。
- 数据可以是采样轮、缓冲目录或对局目录；`--buffer-iterations N` 只用缓冲里最近 N 次迭代的各轮。
- 训练范围：默认训练整个主体和头；`--train-blocks N` 只训练顶部 N 个全局层、最终 norm 和头，`--freeze-body` 只训练头（策略表示的线性探针）。不训练的参数不进优化器、不算梯度。
- 批：按策略的桶与预算打包（`model.batch.collate`、`model.data.microbatches`），一个逻辑批 `logical_batch_size` 个决策，超出预算时拆成几份、梯度累加后更新一次；打包在 `loader_workers` 个 fork 出的进程里做。
- 优化器照策略分组：Transformer 的 Attention／FFN 矩阵用 Muon，查找表（符号、字段名、局部与各全局层的关系偏置表）、头和其余张量用 AdamW，各有自己的学习率；线性预热 `warmup_updates` 次更新，之后线性衰减到最后一次更新为 0。
- 损失：通关与过幕两个二元交叉熵之和，后者乘 `act_weight`。
- 每轮结束在留出 seed 上评估一次并打印；`--train-runs N` 只用 N 个训练 seed（学习曲线），`--no-baselines` 跳过基线，`--no-compile` 不编译全局层，`--report` 把报告另存一份。
- 不写入数据目录。

配置：`--config` 指向策略的 profile（默认 `configs/rtxpro6000.json`），取其 `model` 节作网络；critic 自己的设置放在 profile 的 `critic` 节（可省略、可只写要改的项），字段见 `config.py` 的 `TrainingConfig`；profile 的 `training` 节属于策略，不读。命令行参数覆盖对应项。

| `critic` 节 | 含义 |
| --- | --- |
| `epochs`、`logical_batch_size`、`warmup_updates` | 轮数、每次更新的决策数、预热更新数 |
| `init` | `checkpoint`／`random` |
| `train_body`、`train_blocks` | 是否训练主体；只训练顶部几个全局层 |
| `muon_lr`、`backbone_lr`、`embedding_lr`、`head_lr`、`weight_decay`、`max_grad_norm` | 各参数组学习率（`embedding_lr` 为 0 时冻结四张查找表）、权重衰减、梯度裁剪 |
| `act_weight` | 过幕损失的权重 |
| `token_buckets`、`action_buckets`、`token_budget`、`pair_budget`、`microbatch_size` | 打包的桶与预算 |
| `precision`、`holdout`、`seed`、`loader_workers` | BF16／FP32、留出 seed 份额、随机种子、打包进程数 |

## 输出与报告

`--save <目录>` 写 `weights.pt`（共享主体与头）和 `manifest.json`：`model`、`training` 两节，词表，编码／公开历史／规则文本版本，`version`（迭代版本号），`policy`（提供词表与初始权重的检查点、其策略版本、权重 digest、词表哈希），`data`（数据版本、来源、对局数、决策数、seed 数、打出这些对局的策略版本与 digest、覆盖统计、划分大小），以及 `report`。

报告（控制台、manifest，可另存 `--report`）在留出 seed 上对两个目标各给：

- 解释方差：对 ±1 回报，`1 − MSE(2p−1, ±1) / Var(±1)`；常数基线为 0，等价于 `1 − Brier / (r(1−r))`，r 为留出集实际率。
- Brier、log loss、AUC、预测均值与实际率。
- 校准表：按幕、按角色（`n`、实际率、预测均值、差），以及按预测概率分十段的可靠性表。
- 基线（同一划分）：常数基础率；只看幕与楼层的逻辑回归；幕、楼层、角色加状态摘要的逻辑回归。
- `history`：每轮的训练损失、梯度范数与留出指标；`train.constant_loss` 是只预测训练集基础率时的训练损失，门槛用它衡量训练损失降了多少。
- 标签不变化的小数据（一轮里没有一局通关）上，逻辑回归基线退化为常数，解释方差与 AUC 为空。

## 门槛

critic 在交给策略更新之前要过门槛（`gate.py`，`check(训练报告, Gate(...))`）；[迭代驱动](../model/README.md#迭代采样标签胜率模型与优势加权回归)在不过时跳过这次策略更新并记下没过的项。每一项都在留出 seed 上算，通关与过幕两个头分别算：

| 设置（`Gate` 的字段） | 默认 | 含义 |
| --- | --- | --- |
| `min_heldout_seeds` | 150 | 留出对局数的下限；更少时解释方差的差别只是划分噪声。一次迭代 1,000 局留出约 200 ± 13 局，下限要让开这个波动 |
| `min_explained_variance` | 0.10 | 解释方差的下限 |
| `min_margin_over_baseline` | 0.05 | 解释方差须高出同一划分上“幕 + 楼层”逻辑回归这么多 |
| `max_overall_gap` | 0.02 | 整体预测均值与实际率之差的上限 |
| `max_act_gap`、`max_character_gap` | 0.05、0.10 | 按幕、按角色各组的同一差值的上限 |
| `max_bin_gap`、`min_bin_share` | 0.15、0.05 | 按预测概率分十段，样本占比不低于 `min_bin_share` 的段的差值上限 |
| `max_memorization_ratio` | 3.0 | 训练损失相对常数基线的下降，不得超过留出 Brier（两个头之和）相对常数基线的改善的这么多倍；训练损失没有下降时不算记忆 |

默认值是首次训练后的建议值，依据与噪声水平记在[工单 18](../.scratch/a0-training-baseline/issues/18-win-probability-critic.md) 的 Comments，待第一轮完整采样数据训练后复核。报告缺少某项所需的量（没跑基线、留出集的结果不变化）时该项算不过。命令行里用 `python -m model iterate --gate min_heldout_seeds=100 --gate …` 逐项改，`--critic-gate off` 只记录不拦。

## 优势

```bash
python -m critic.advantage runs/iterate-v1/buffer --buffer-iterations 10 --critic runs/iterate-v1/critic/iteration-0003
```

给每一轮的每个局外决策打分（打包在加载进程里做，与训练相同），V = 2·p_win − 1，沿每局算不折扣的 GAE（`model/advantage.py`：下一个状态是同一局的下一个局外决策，最后一个决策之后是 ±1 的回报），写到轮目录的 `advantages.json.gz`，带 critic 的版本、权重文件摘要与 λ；这个 critic 以这个 λ 打过的轮跳过。分片不改。

## 接口

```python
from critic.model import load

scorer = load("runs/critic/v1", device="cuda")      # 只凭 manifest 重建网络与词表
p_win, p_act = scorer.predict(frames)              # 任意多个决策帧，两个概率列表
scorer.score(observations)                         # 已解析的观察 -> [n, 2] 概率，按预算分几次前向
scorer.probabilities(batch)                        # 一个已打包的批 -> [n, 2]，一次前向
scorer.version                                      # 迭代版本号
```

帧是与策略相同的决策帧（需要 `public` 与 `legal`）；±1 回报的价值是 `2 * p_win − 1`。`load` 拒绝编码版本、公开历史版本或规则文本不一致的模型。

## 模块

| 文件 | 内容 |
| --- | --- |
| `config.py` | profile 的 `model` 节与 `critic` 节、打包预算 |
| `data.py` | 对局 → 决策行（帧、标签、元信息）、读取、按 seed 划分、覆盖统计 |
| `batches.py` | 按策略的桶与预算打包决策帧 |
| `model.py` | 网络、保存与加载、`Scorer` |
| `baselines.py`、`metrics.py` | 同一划分下的基线；解释方差、Brier、AUC 与校准表 |
| `gate.py` | 交给策略更新之前的门槛 |
| `advantage.py` | 给缓冲的各轮打分并写优势 |
| `train.py` | 训练入口 |
| `tests/` | CPU 可跑的小样本测试 |
