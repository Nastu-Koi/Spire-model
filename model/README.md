# 主策略训练与运行

主策略从玩家公开状态选择整局动作，覆盖战斗、地图、商店、事件和奖励。训练入口只有一个：对宏动作做加权 NLL 的行为克隆（BC／Bootstrap）；采样（`collect`）与评估（`evaluate`）共用同一个对局循环。学生状态的求解器战斗标签、胜率模型和优势加权回归按 [训练路线](../ARCHITECTURE.md#训练路线) 后续加入，用的仍是这同一个训练器和加载器。设计见 [ARCHITECTURE](../ARCHITECTURE.md)，数据生成见 [摘要锚点](../spire_codex_data/README.md)。以下命令均在仓库根目录运行。

## 当前训练安排

当前配置为 [configs/rtxpro6000.json](../configs/rtxpro6000.json)：约 20M 共享策略，默认 A0，BC 可使用 A0–A10 混合示范。[configs/rtxpro6000-200m.json](../configs/rtxpro6000-200m.json) 是约 196M 的目标配置，取舍见 [ARCHITECTURE](../ARCHITECTURE.md)。BC 使用传入数据里除固定留出组以外的全部可训练宏动作（规则见 [固定留出集](#固定留出集)），默认持续至手动中断；不自动评估、不按表现降学习率或早停。可用 `--epochs N` 指定本次追加轮数，表现由使用者自行验证。

| 设置 | 当前值 |
| --- | --- |
| Muon 学习率 | `1e-3` |
| 查找表学习率（符号、字段名嵌入与关系偏置表） | 独立 AdamW 组，起始 `1e-2`，第 1 轮内（10,624 次更新）几何衰减到 `1e-3` 后恒定 |
| 其余 AdamW 骨干学习率 | `1e-5` |
| 头部学习率 | `3e-5` |
| BC 学习率日程 | 前 200 次更新线性预热，之后恒定；停止训练时以 `--decay-updates` 线性衰减到 0 收尾 |
| 梯度裁剪 | `1.0` |
| 逻辑批大小 | 512 个宏动作 |
| 读取进程×打包进程 | 4 × 6，共持一个 8 分片窗口 |
| 全局层编译 | `compile_blocks: true`，每个进程首次前向约一分钟编译 |
| 参数变化统计 | 每 100 次更新抽样 |
| 采样温度 | `combat_temperature` 0.3、`other_temperature` 0.7：`collect` 里战斗阶段的决策和其它决策各按自己的温度抽样。温度是策略定义的一部分，记在对局文件和检查点里；评估取最可能动作，不受影响 |
| 留出集 | 按 seed 的哈希固定留出 2% 的样本组，`bootstrap` 默认跳过，只用 `heldout-nll` 打分；见 [固定留出集](#固定留出集) |

查找表学习率的依据见 [优化器](../ARCHITECTURE.md#优化器与精度)，其余学习率未作筛选。分片窗口只控制内存占用；逻辑批次跨窗口补齐，每轮仅最后一批可不足 512。显存预算触发微批拆分时累积梯度，一个逻辑批次只更新一次。角色×阶段的权重与损失尺度见 [批次与加载](../ARCHITECTURE.md#批次整理与数据加载) 和 [优化器](../ARCHITECTURE.md#优化器与精度)。每个样本带来源（`source`：示范、求解器标签或采样对局）和由索引计数给出的权重；按来源分配份额是索引的事，训练器只按权重求和。

日志保存各组实际学习率、裁剪前梯度范数分布、裁剪比例及抽样参数变化。embedding 的相对变化按整组参数范数计算，包含当前批次没有触及的词表行；它不能单独证明模型是否学会规则。

## 数据复用与索引

**这次宝箱及单选事件控制调整，无需重新生成 BC 原始数据。** 已有分片保存公开状态、完整合法候选和原始标签，训练时按共用控制规则过滤。不要把原始导入总数当作可训练样本数。

| 变化 | 需要做什么 |
| --- | --- |
| 只调整哪些步骤交给控制器 | 对旧分片执行 `reindex`；不重新求解战斗或改写原始标签 |
| 新导入独立监督样本 | `import-independent` 直接生成当前版本索引 |
| 原始公开字段、合法候选或摘要恢复规则需要修正 | 重新生成／复验受影响的锚点，再导出和导入；索引不能补造缺失内容 |
| 采样对局的输入、奖励或控制规则改变 | 用当前策略与规则重新采样，不复用旧版本的对局 |

旧索引的重建命令：

```bash
python -m model reindex --data data/bootstrap --workers 6
```

重建成功后原子替换 `index.jsonl.gz`，旧索引保存为 `index.jsonl.gz.before-public-control-v1`，汇总写入 `policy-index-summary.json`。原始分片和导入时的 `summary.json` 保留。加载时拒绝旧控制版本的索引，读分片时核对过滤计数。

### 当前工作区的数据

截至 2026-10-10，`runs/bootstrap-20261008/` 下保留最新的一份导入（`independent-import-v6`，索引为当前控制版本），可直接用于 BC，无需运行 `reindex`：

| 目录 | 内容 | 样本组 | 可训练宏动作 | 每轮更新次数 |
| --- | --- | ---: | ---: | ---: |
| `imported-potions` | 全部类型：领取样本（[`claim`](../spire_codex_data/README.md#领取战后奖励)）、界面上还有药水或遗物时的选牌（`pick`）、必须用药的战斗（[`potion`](../spire_codex_data/README.md#必须用药的战斗)） | 1,416,761 | 6,360,448 | 12,423 |
| `imported-potions-1in8` | 训练用的子集，与上一行共用分片、只换索引 | 1,131,109 | 5,501,424 | 10,745 |

子集的规则（`runs/bootstrap-20261008/subset_rewards.py`）：顺序由本项目重建的奖励样本组——`claim` 和“先选牌后领药水”的 `pick`——按 id 的 SHA-256 保留八分之一（47,494 和 18,390 个宏动作）；“界面上留着药水或遗物时选牌”的 `pick` 是历史选择，全部保留（12,181 个）；`potion` 战斗只保留已有战斗样本的局里的（20,103 场、452,404 个宏动作，各自取代原来那一场），其余局里的会是凭空多出来、且每一场都强制用药的战斗，不进子集（17,610 场）。角色×阶段均为 35 组。第 1–7 轮训练用的是不含领取样本的导入（1,080,061 组、5,439,028 个宏动作），第 8–12 轮再加八分之一的领取样本（5,486,522 个宏动作）；这两份导入已删除，可从 `anchors/` 重新导出。生成过程与各阶段计数见本地 [训练工单](../.scratch/summary-ledger/issues/04-refresh-and-bootstrap.md)。这些路径是本工作区产物，不包含在 Git 中。

## 开始与继续 BC

已有当前数据时，直接使用上一节的数据目录。新数据的准备流程见 [摘要锚点说明](../spire_codex_data/README.md#生成样本)；完成后导入并初始化：

```bash
python -m model import-independent data/anchors/independent-training.jsonl.gz --output data/bootstrap
python -m model --device cuda init --config configs/rtxpro6000.json --output runs/init
python -m model --device cuda bootstrap --data data/bootstrap --checkpoint runs/init --weights-only \
  --config configs/rtxpro6000.json --output runs/bootstrap
```

分片数据不会自行重建词表，必须提供检查点。`init` 从引擎公开内容目录建立冻结词表并随机初始化权重。

本工作区的训练在 `runs/bootstrap-20261008/`，是一条从头到尾严格续训的主线：`init-v5/` 是当前输入编码的初始化；`trained-v6/` 是第 1–7 轮（不含领取样本的数据）；`trained-v7-claims/` 是第 8–12 轮（加入八分之一的领取样本）；`trained-v8-potions/` 从第 12 轮起在 `imported-potions-1in8` 上继续。换数据时用新的输出目录，优化器、调度进度和轮次计数都延续。

```bash
python -m model --device cuda init --config configs/rtxpro6000.json \
  --engine-root runs/bootstrap-20261008/engine --output runs/bootstrap-20261008/init-v5
python -m model --device cuda bootstrap --data runs/bootstrap-20261008/imported-potions-1in8 \
  --checkpoint runs/bootstrap-20261008/init-v5 --weights-only \
  --config configs/rtxpro6000.json --output runs/bootstrap-20261008/trained-new --epochs 6
```

已有训练目录须通过其 `current` 续训，不能用新初始化覆盖。

### 多个数据目录与学生战斗的份额

`--data` 可以给多个分片数据目录，读成一个数据集（`RunShards`）：示范导入加若干学生战斗导入（[学生状态的战斗](../spire_codex_data/README.md#学生状态的战斗)），或若干采样轮。每个目录各有自己的索引和 `summary.json`，不合并文件；留出规则、权重归一和检查点的契约、种子集合都按合起来的整体算。单个路径仍可以是小的对局文件或目录。

索引行记每个样本组的来源 `source`（`independent`、`rollout`、`student_battle`；旧索引没有这个字段，读作示范），学生战斗的行另记 `policy_version`、`policy_digest`、`iteration`，`reindex` 保留。`Sample.source` 相同。

权重：每个角色×阶段组仍占 1/G。角色的战斗阶段组（`combat`）里同时有学生战斗和其它来源时，学生样本合计占该组的份额 s、其余来源合计占 1−s，各自在样本间平均；只有一种来源的组整组归它。其它阶段不分份额：学生战斗里战斗中的选牌（`card_select`）与本组其它样本按数量平摊。s 由整个训练部分里学生宏动作的累计数 N 决定：`s = 1/3 + (1/2 − 1/3)·min(1, N/1,000,000)`，即从三分之一起线性升到一半为止（`model/data.py` 的 `student_share`、`group_weights`，依据见 [批次整理与数据加载](../ARCHITECTURE.md#批次整理与数据加载)）。每轮记录的 `training`／`held_out` 计数带 `student_macros` 与 `student_share`。

学生样本的单样本权重与它的数量成反比：一次迭代的量（每角色一两万个战斗宏动作）约为示范样本的三十倍，每角色只有几十上百个时是上千倍。不足一次迭代的学生数据只用来检查流水线，不要并入正式训练。

多个目录时把最大的（示范）放在第一个：每个读取进程都持有整份索引，第一个目录的索引行原样使用，其后目录的行各多带一个目录标记。

把一轮学生战斗并入示范续训（严格续训到新目录；换数据时优化器、调度进度和轮次计数都延续）：

```bash
python -m model --device cuda bootstrap \
  --data runs/bootstrap-20261008/imported-potions-1in8 data/student-import/round-3 \
  --checkpoint runs/bootstrap-20261008/trained-v8-potions/current \
  --output runs/bootstrap-20261008/trained-v9-student
```

加 `--epochs N` 限定追加的轮数。之后每次迭代把新一轮的学生目录追加到 `--data` 列表（旧轮照样留着，份额随累计量上升），照换数据的惯例用新的输出目录、从上一个目录的 `current` 续训。学生目录导入时用小分片（`import-independent --shard-samples 250`），原因见[学生状态的战斗](../spire_codex_data/README.md#学生状态的战斗)。

| 启动方式 | 行为 |
| --- | --- |
| `--checkpoint ... --weights-only` | 复用兼容权重，使用当前配置与全新优化器、随机状态和预热计数；使用新输出目录 |
| `--checkpoint <输出目录>/current` | 严格续训，恢复优化器、随机状态和调度进度；训练配置与版本必须一致 |
| `--checkpoint <输出目录>/current --config <profile>` | 严格续训，只把采样温度换成 profile 里的；其余设置必须与检查点一致 |
| `--checkpoint <输出目录>/current --decay-updates N --output <新目录>` | 收尾：严格续训并在 N 次更新内把所有学习率线性降到 0，然后停止并保存；N 取已完成更新数的 10%–20% |

续训示例：

```bash
python -m model --device cuda bootstrap --data runs/bootstrap-20261008/imported-potions-1in8 \
  --checkpoint runs/bootstrap-20261008/trained-v8-potions/current \
  --output runs/bootstrap-20261008/trained-v8-potions
```

收尾衰减示例（用一轮的更新数衰减）：

```bash
python -m model --device cuda bootstrap --data runs/bootstrap-20261008/imported-potions-1in8 \
  --checkpoint runs/bootstrap-20261008/trained-v8-potions/current --decay-updates 10745 \
  --output runs/bootstrap-20261008/trained-v8-decay
```

衰减跑完的检查点是最终检查点，不能再续训；中途中断可用同一 `--decay-updates` 从其 `current` 续接。

每轮完成后更新 `current/`，并保留 `epoch-000001/` 等独立检查点供手动比较。Ctrl+C 会保留最后一个完整 epoch；当前 epoch 尚未保存的更新不会恢复。只编辑 manifest 不能修改已保存优化器的学习率。

删除整局 PPO 之前保存的检查点仍能载入：权重里的价值头张量（`value.*`）和它在优化器里的参数组被忽略，其余权重、优化器动量和调度进度照常严格续训；manifest 里已经退役的设置项（`clip_ratio`、`sampling_temperature` 等）读入时丢弃，温度取当前默认值。做过 PPO 采样轮、或调度状态停在 PPO 的检查点不能严格续训，只能 `--weights-only` 重新开始，或者用于采样和评估。

## 固定留出集

训练数据按 seed 分组固定留出约 2%，永不训练，只用来比较 NLL。规则是 `model/data.py` 的 `held_out`：取索引行的 `seed`（独立监督样本的 seed 是来源对局 seed 的 `split_group` 哈希，学生战斗样本同样取游戏 seed 的 `split_group`，采样对局的是游戏 seed），其 `fingerprint` 的前 32 位作为 [0, 1) 里的小数小于 `HELDOUT_FRACTION`（0.02）即留出。它只是 seed 的函数：同一 seed 的全部片段落在同一侧，与分片边界、导入批次和子集无关，所以同一个 seed 在 `imported-potions` 和它的 1in8 子集里都在同一侧，数据重新导入或增长也不改变已有 seed 的归属。依据见 [评估](../ARCHITECTURE.md#评估)。

`bootstrap` 默认跳过留出组（`--include-heldout` 关掉跳过）：加载器只读训练部分的索引行，留出组的行连解析都不做，样本权重按训练部分归一。每轮记录（`history.jsonl` 的 `metrics`）带 `held_out`（留出的组数、宏动作数、seed 数）、`training`（训练部分的同样计数）和 `samples`（加载器这一轮交给训练器的宏动作数）：`samples` 等于 `training.macros`，与 `held_out.macros` 相加等于索引总数，这就是留出组没进训练窗口的证据；检查点的 `progress.held_out` 保存同样的计数，`training_seeds` 仍记录整个数据集的 seed。

留出 NLL：

```bash
python -m model --device cuda heldout-nll --data runs/bootstrap-20261008/imported-potions-1in8 \
  --checkpoint runs/bootstrap-20261008/trained-v8-potions/epoch-000017
```

报告留出部分每个角色×阶段组的 NLL 总和、每宏动作 NLL（`nll_per_macro`）、每决策 NLL（`nll_per_branch`）及宏动作数与决策数，再按角色、按阶段和全体汇总；`equal_group_nll_per_macro` 是各组每宏动作 NLL 的平均，与训练损失的组等权定义一致，可直接与训练日志的 `loss` 对照；`a0` 下是其中 A0 样本的同样表格；`held_out.of` 是整个数据集的计数。它只报告，不选模型、不早停。

## 策略与控制器

| 决策 | 处理方式 |
| --- | --- |
| 普通宝箱及问号宝箱 | 自动开箱、取遗物、离开 |
| 唯一可用事件选项，其余仅能丢药 | 自动推进该事件选项 |
| 水晶球格子与 Small／Big 工具 | 公开信息规划器 |
| 多个事件选项、可用药水、后续选牌 | 主策略 |
| 商店离开＋丢药、战斗结束回合＋丢药 | 主策略，继续训练 |
| 战后／事件领奖、水晶球支付及领奖 | 主策略 |

[control.py](control.py) 为运行、导入、索引和训练提供同一套判断。控制器步骤不产生策略损失；原生动作照常执行，后续随机内容只在游戏揭示后读取。完整约束见 [控制分工](../ARCHITECTURE.md#环境步骤与水晶球)。

## 检查点和轨迹版本

| 契约 | 当前版本 |
| --- | --- |
| 原生公开观察 | `public-state-v6` |
| 公开历史 | `public-history-v1` |
| 模型输入编码 | `public-encoding-v3` |
| 训练损失与优化器 | `bc-decay-v5` |
| 策略／控制器分工及索引 | `public-control-v1` |
| 奖励 | `outcome-v1` |

缺失旧历史时输入为 unknown，不补初始概率。兼容当前公开历史、网络结构和规则文本的旧编码检查点，可通过 `bootstrap --weights-only` 开始新训练；新符号只追加到词表空行，已有符号 ID 和权重不重排。更早的历史输入或不兼容结构须重新 `init`。旧优化器不能严格续接当前训练版本；旧版本采样的对局不能只修改版本号后使用。

## 采样与手动评估

`collect` 和 `evaluate` 共用对局循环，写出的东西不同：`evaluate` 写完整对局供分析，`collect` 写训练能直接读的一轮采样（格式见下一节）。两者都为每局输出 `<对局>.combats.jsonl.gz`，保存实际战斗和中途公开帧，格式见 [战斗结果数据](../combat_outcome/README.md)。

```bash
# seeds.json 按五个角色各列等量种子；评估种子须与训练隔离
python -m model --device cuda evaluate --checkpoint runs/bootstrap/current \
  --seeds seeds.json --output runs/evaluation --workers 16

# 按检查点里的两个温度采样一轮，写到 runs/collect/round-N（N 自动递增）
python -m model --device cuda collect --checkpoint runs/bootstrap/current \
  --seeds seeds.json --output runs/collect --workers 44 --iteration 3
```

`--workers` 是同时在跑的引擎数。引擎、帧的解析和张量打包都在采样进程里，本进程只跑网络；`--processes` 默认等于 `--workers`，每个进程带一个引擎时最快（原因见 [采样的进程分工](../ARCHITECTURE.md#采样的进程分工)）。采样进程和引擎跨轮保留，引擎在原进程里开下一局。每对“引擎 + 采样进程”占 250–300 MB 内存（64 对约 18 GB）。一轮的耗时不会短于最长一局，所以每轮对局数（每角色种子数的五倍）不宜少于引擎数，多几倍更省。

`evaluate` 每局写出对局文件 `<角色>-<序号>.json`（每个宏动作的帧、完整候选、所选动作、`old_log_prob`、阶段和随后支付的奖励，以及整局的账本、每幕结果 `acts`、到达楼层 `climb`、策略版本与权重摘要、数值后端和采样温度）、流水账 `.jsonl`（每个帧与奖励）和战斗记录；汇总时把对局逐局读回，内存里只留一局，评估多少局都不受内存限制。报告除通关率外，还列出各角色通过每一幕的对局数、奖励均值（`mean_score`，即通关率的两倍减一）、平均到达楼层和各次失败止步的楼层；楼层跨幕累计，只统计已完成的对局。

`collect` 按检查点的 `combat_temperature`／`other_temperature` 抽样，每个决策按自己阶段的温度；`collect --config <profile>` 可换成 profile 里的温度，其余设置必须与检查点一致。greedy 的 `evaluate` 记 `temperature: null`，概率按温度 1 记。

`evaluate` 的报告除通关率外，还列出各角色通过每一幕的对局数、奖励均值（`mean_score`，即通关率的两倍减一）、平均到达楼层和各次失败止步的楼层；楼层跨幕累计，只统计已完成的对局。每角色的通关率和各幕通过率都带 Wilson 95% 区间（`wilson_95`、`act_pass_wilson_95`）；`pooled` 是五角色全部尝试合计的通关数、通关率、各幕通过数与通过率及各自的区间，每角色局数相同时合计率就是五角色平均率。

汇总与配对比较读已写出的评估目录（`metadata/seeds.json` 和各局文件，逐局读回）：

```bash
python -m model compare runs/eval-a runs/eval-b --head 60
```

对每个目录给出整个目录的报告和每角色前 `--head` 个种子（默认 60，即前 300 局）的子表，格式与 `evaluate` 相同；给两个目录时按（角色，seed）配对，对通关、过第一幕、过第二幕各做精确符号检验：只数两边不一致的配对（`only_first`、`only_second`），p 值双侧；整个目录和前 `--head` 个种子各报一次，只有一方跑过的种子计入 `unpaired`、不参与检验。未完成的对局算没通关、没过幕。一个目录也可以单独汇总。

奖励只在一局结束时给：通关 +1，没有通关 −1，过幕不给分。未完成或出错的对局不伪造为正常死亡；采样轮里有这样的对局时 `collect` 报错，轮仍写出供检查。

### 采样轮的格式

`collect` 的一轮写到 `<输出>/round-N/`，与 `import-independent` 相同的分片加索引，`bootstrap --data <输出>/round-N` 直接读，样本权重与控制器过滤和导入数据一致；对局缓冲就是最近几轮的这些目录。

| 文件 | 内容 |
| --- | --- |
| `accepted/*.jsonl.gz` | 打完的对局，每局一行（一个 gzip 成员）；每个分片约 `--shard-samples`（默认 2000）个保存的决策步 |
| `index.jsonl.gz` | 每局一行：`run_id`、`seed`、角色、各阶段可训练宏动作数、契约哈希、分片与行号，另加 `policy_version`、`policy_digest`、`iteration`、`victory`；`reindex` 保留这些字段 |
| `summary.json` | 轮的契约：`iteration`、策略版本与权重 digest、温度、难度、控制与奖励版本、对局数与胜场、各角色×阶段的宏动作数、战斗与存档计数、分片数、引擎契约（`contracts`），以及与 `evaluate` 相同的通关率报告（`report`） |
| `<角色>-<序号>.combats.jsonl.gz` | 战斗记录，照旧 |
| `<角色>-<序号>.saves.jsonl.gz` | 每场战斗入场时的引擎存档，见下 |
| `metadata/seeds.json` | 种子分配 |
| `quarantine/<角色>-<序号>.json.gz` | 没打完的对局（超时、引擎出错），`summary.json` 的 `quarantine` 记原因 |

对局（`source: "rollout"`，`iteration` 默认取轮号 N，`--iteration` 可指定）只保留**战斗之外**的决策：每个策略宏动作的帧、完整候选、所选动作、`old_log_prob`、阶段、随后支付的奖励，`fight: null`。战斗里的宏动作——`combat` 阶段和战斗中的 `card_select`（弃牌、消耗等选择）——只记 `phase`、`fight`（这一局的第几场战斗，与 `encounter_id` 的编号相同）、`branches`（有选择的决策数）、`decisions`（决策数）、`old_log_prob` 与 `reward`，不保存帧；`policy_macros`／`samples()` 读到它们时不作训练样本，战斗的训练信号来自求解器标签。战斗的边界由引擎事件给出：`encounter_started` 开始，`encounter_completed` 之后的第一个决策在战斗之外。控制器步骤照旧记在 `environment_actions`。每局另记 `status`、`victory`、`acts`（每幕 `passed`／`died`，没打完时为空）、`ledger`、`climb`、`seed`、角色、难度、策略版本与 digest、按阶段的 `temperature`、`combats` 与 `saves` 的计数。`--journals` 时另写每局的流水账 `.jsonl`（每个帧与奖励，含战斗）。

入场存档：每场战斗的第一个决策（`events` 带 `encounter_started`）时，采样进程让引擎 `write_continue_save` 到临时文件再读回；在房间里时引擎存的是进房前的状态（`anchor_state` 序列化的是已进房的状态，不能用）。每行带 `run_id`、`seed`、角色、难度、`fight`、`combat_index`（`combats` 里对应行的序号）、`encounter_id`、`encounter`、`kind`（normal／elite／boss）、`act`、`floor`、`map_point_type`（normal→monster、elite→elite、boss→boss）、`second_boss`（本幕第二个 Boss）、`from_event`、`result` 和 `save`。`model.saves.anchor_commands(row)` 给出重新进入这场战斗的两条引擎命令（`enter_anchor` → `anchor_room`）：重入的战斗与原战斗的 HP、最大 HP、金币、幕、楼层、卡组（含升级）、遗物、药水和敌人种类相同，敌人的随机 HP 与开局手牌由战斗自己的随机数重抽，与摘要锚点的战斗一样是同一入场状态的另一次战斗。引擎拒绝存档时该行 `save` 为空并带 `error`，`summary.json` 计入 `saves_failed`。事件引发的战斗照存并标 `from_event`，是否可用由求解器标签一侧判断。

内存：采样进程一局内只保留压缩后的帧与存档，一局结束把对局压成分片里的一行交给采集进程追加到当前分片；采集进程不持有对局，每局只留报告用的几个字段。

## 训练监测网页

在另一终端运行，只需 Python 标准库：

```bash
python3 -m model.monitor --runs runs --port 8765
# 已安装本项目时也可用：spire-monitor --runs runs
```

打开 [本地训练监测](http://127.0.0.1:8765)。`--runs` 可指向全部训练根目录或单个 `bootstrap` 的输出目录。网页递归发现 `status.json`、`updates.jsonl` 和 `history.jsonl`，每 5 秒刷新，可切换任务、查看曲线与原始记录、导出所选更新范围的 CSV。页面关闭或关闭自动刷新时不再请求数据。

- 每次更新的损失、动作正确率、策略熵、裁剪前梯度范数、各参数组学习率、耗时和宏动作数。正确率是更新前同一次前向计算中，合法动作 Top-1 与示范动作一致的决策数除以策略决策总数；多步宏动作逐步统计，排除强制动作和控制器步骤，不使用损失的样本权重。日志同时保存 `accuracy`、`accuracy_correct` 和 `accuracy_decisions`，汇总正确率按决策总数计算。这是当前训练批的正确率；旧日志的验证 NLL 可以显示，新 BC 不产生验证指标。顶部数字是最近 200 次更新的均值，状态栏按上一轮的更新数和最近的墙钟速度估算本轮进度。
- 轮次汇总表列出每个 epoch 结束时的整轮指标，更新记录表展开最近 200 条原始记录。

横轴、悬停提示、历史表和 CSV 均使用优化器累计更新次数（Update），曲线按数值间距绘制；微批拆分和梯度累积只产生一个逻辑批更新点。跨 epoch 继续计数，恢复检查点沿用保存的更新次数；恢复后回退的更新和不同 session 之间不连线。默认显示全部历史，也可只看最近 N 次更新，按更新次数区间筛选，不把缺失点补成零；曲线和 CSV 保留所选范围的全部更新。

每个点都参与绘制：一个像素列里只连它的首点、最低点、最高点和末点，形状与逐点连线相同。点多时默认叠加滑动平均（同一段内最近 N 个点的均值，粗线），原值留作其后的细线。纵轴默认聚焦主体，少数远离主体的点落在范围外时，图下写明被略去的最高和最低值；两项都可以关掉。

`updates.jsonl` 在每次成功更新后写入并刷新到文件，不逐步执行 `fsync`；记录当前逻辑批的指标，不是从 epoch 开始累计的平均。`history.jsonl` 继续保留原来的 epoch 汇总和持久化方式，检查点保存时机不变。只有旧汇总日志时，网页把已有指标定位到其真实 `metrics.updates`，明确提示稀疏记录；无法从旧日志还原中间更新。缺少正确率的旧记录保留为空，不能由 NLL 推算。已经运行的训练进程需在保存检查点后按新代码重新启动，才会产出新加入的指标，网页不会主动中断训练。

状态只反映日志最后写入的值，不能证明进程仍存活；长时间没有新写入时页面会提示，缺失指标留空。服务只读日志：运行列表只看状态文件，某个训练的日志在打开它时才读取并增量跟进，忽略尚未写完的尾行并提示损坏记录。曲线点按列保存，每个值一个双精度数，完整的原始记录只留最近 200 条；浏览器首次打开时取回全部点，之后每次刷新只取新增的部分，日志被替换或服务重启时整体重取。`--limit 0`（默认）不限制历史，内存受限时可显式设置正数限制每任务记录数，页面会提示截断。默认监听 `127.0.0.1`，远程使用 SSH 端口转发；网页没有登录机制。删除整局 PPO 之前写下的 `ppo` 训练目录不再被识别。

Steam Mod 的安装和接管入口见 [steam_recorder](../steam_recorder/README.md)。

## 代码导航与测试

| 模块 | 职责 |
| --- | --- |
| `control.py`、`treasure_rule.py`、`crystal_rule.py` | 控制分工、宝箱流程、水晶球规划 |
| `independent.py`、`recorder.py`、`dataset_index.py`、`data.py` | 导入、版本化索引、策略样本与权重、留出规则 |
| `representation.py`、`public_history.py`、`rules.py`、`batch.py` | 公开输入、冻结词表、规则和张量打包 |
| `model.py`、`attention.py`、`policy.py` | 网络、选择与可微重放 |
| `loader.py`、`trainer.py`、`optim.py`、`checkpoint.py` | 数据供给、加权 NLL 更新、学习率日程与恢复 |
| `rollout.py`、`rewards.py`、`steam.py` | 对局、奖励账本与实机控制 |
| `evaluation.py` | 评估报告、合计与配对符号检验、留出 NLL 的表格 |
| `saves.py`、`buffer.py` | 战斗边界与入场存档、采样轮的分片与汇总 |
| `sampling.py`、`wire.py`、`processes.py` | 多进程采样、打包决策的跨进程格式、随父进程结束的子进程 |
| `monitor.py`、`monitor_web/` | 只读日志监测 |

```bash
python -m pytest model/tests -q -m 'not engine and not cuda'
COMBAT_SOLVER_CONFIG=combat_solver_cli/lib/config.json \
  python -m pytest model/tests/test_native_a0_controls.py -q -m engine
python3 -m unittest model.tests.test_monitor -v
```
