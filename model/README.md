# 主策略训练与运行

主策略从玩家公开状态选择整局动作，覆盖战斗、地图、商店、事件和奖励。当前训练入口提供行为克隆（BC／Bootstrap）与基础 MC PPO；结果分布、本幕 Φ、两路优势及冻结 BC 参考 KL 仍属后续实现。设计见 [ARCHITECTURE](../ARCHITECTURE.md)，数据生成见 [摘要锚点](../spire_codex_data/README.md)。以下命令均在仓库根目录运行。

## 当前训练安排

当前配置为 [configs/rtxpro6000.json](../configs/rtxpro6000.json)：约 20M 共享策略，默认 A0，BC 可使用 A0–A10 混合示范。BC 使用全部传入的可训练宏动作，默认持续至手动中断；不划分验证集、不自动评估、不按表现降学习率或早停。可用 `--epochs N` 指定本次追加轮数，表现由使用者自行验证。

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

查找表学习率的依据见 [优化器](../ARCHITECTURE.md#优化器与精度)，其余学习率未作筛选。分片窗口只控制内存占用；逻辑批次跨窗口补齐，每轮仅最后一批可不足 512。显存预算触发微批拆分时累积梯度，一个逻辑批次只更新一次。角色×阶段的权重与损失尺度见 [批次与加载](../ARCHITECTURE.md#批次整理与数据加载) 和 [优化器](../ARCHITECTURE.md#优化器与精度)。

日志保存各组实际学习率、裁剪前梯度范数分布、裁剪比例及抽样参数变化。embedding 的相对变化按整组参数范数计算，包含当前批次没有触及的词表行；它不能单独证明模型是否学会规则。

## 数据复用与索引

**这次宝箱及单选事件控制调整，无需重新生成 BC 原始数据。** 已有分片保存公开状态、完整合法候选和原始标签，训练时按共用控制规则过滤。不要把原始导入总数当作可训练样本数。

| 变化 | 需要做什么 |
| --- | --- |
| 只调整哪些步骤交给控制器 | 对旧分片执行 `reindex`；不重新求解战斗或改写原始标签 |
| 新导入独立监督样本 | `import-independent` 直接生成当前版本索引 |
| 原始公开字段、合法候选或摘要恢复规则需要修正 | 重新生成／复验受影响的锚点，再导出和导入；索引不能补造缺失内容 |
| PPO 的输入、奖励或控制规则改变 | 用当前策略与规则重新采样，不复用旧版本的 PPO 轨迹 |

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

| 启动方式 | 行为 |
| --- | --- |
| `--checkpoint ... --weights-only` | 复用兼容权重，使用当前配置与全新优化器、随机状态和预热计数；使用新输出目录 |
| `--checkpoint <输出目录>/current` | 严格续训，恢复优化器、随机状态和调度进度；训练配置与版本必须一致 |
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

## 策略与控制器

| 决策 | 处理方式 |
| --- | --- |
| 普通宝箱及问号宝箱 | 自动开箱、取遗物、离开 |
| 唯一可用事件选项，其余仅能丢药 | 自动推进该事件选项 |
| 水晶球格子与 Small／Big 工具 | 公开信息规划器 |
| 多个事件选项、可用药水、后续选牌 | 主策略 |
| 商店离开＋丢药、战斗结束回合＋丢药 | 主策略，继续训练 |
| 战后／事件领奖、水晶球支付及领奖 | 主策略 |

[control.py](control.py) 为运行、导入、索引和训练提供同一套判断。控制器步骤不产生 BC／PPO 策略损失；原生动作照常执行，后续随机内容只在游戏揭示后读取。完整约束见 [控制分工](../ARCHITECTURE.md#环境步骤与水晶球)。

## 检查点和轨迹版本

| 契约 | 当前版本 |
| --- | --- |
| 原生公开观察 | `public-state-v6` |
| 公开历史 | `public-history-v1` |
| 模型输入编码 | `public-encoding-v3` |
| 训练损失与优化器 | `bc-decay-v5` |
| 策略／控制器分工及索引 | `public-control-v1` |
| 奖励 | `act-progress-v2` |

缺失旧历史时输入为 unknown，不补初始概率。兼容当前公开历史、网络结构和规则文本的旧编码检查点，可通过 `bootstrap --weights-only` 开始新训练；新符号只追加到词表空行，已有符号 ID 和权重不重排。更早的历史输入或不兼容结构须重新 `init`。旧优化器不能严格续接当前训练版本，旧 PPO 轨迹不能只修改版本号后使用。

## 采样、手动评估与 PPO

`collect`、`evaluate` 和 `train` 共用对局循环，另为每局输出 `<对局>.combats.jsonl.gz`，保存实际战斗和中途公开帧，格式见 [战斗结果数据](../combat_outcome/README.md)。

```bash
# seeds.json 按五个角色各列等量种子；评估种子须与训练隔离
python -m model --device cuda evaluate --checkpoint runs/bootstrap/current \
  --seeds seeds.json --output runs/evaluation

# 当前入口实现基础 MC PPO；完整 Φ／辅助优势尚未接入
python -m model --device cuda train --checkpoint runs/bootstrap/current --output runs/ppo
```

奖励为过幕 1／2／5：通过第一幕后死亡保留 1 分，通过第二幕后死亡保留 3 分，整局通关共 8 分。未完成或出错的对局不伪造为正常死亡，含此类轨迹的轮次不更新 PPO。PPO 保留自身的 KL 更新保护；它与 BC 的人工验证安排分开。采样与重放的实体编码器都用 FP32，数值后端指纹记录这一点；用旧的 FP64 编码器采样的轨迹会被拒绝，须重新采集。

## 训练监测网页

在另一终端运行，只需 Python 标准库：

```bash
python3 -m model.monitor --runs runs --port 8765
# 已安装本项目时也可用：spire-monitor --runs runs
```

打开 [本地训练监测](http://127.0.0.1:8765)。`--runs` 可指向全部训练根目录或单个 `bootstrap`／`train`／`ppo` 的输出目录。网页递归发现 `status.json`、`updates.jsonl` 和 `history.jsonl`，每 5 秒刷新，可切换任务、查看曲线与原始记录、导出所选更新范围的 CSV。页面关闭或关闭自动刷新时不再请求数据。

- BC：每次更新的损失、动作正确率、策略熵、裁剪前梯度范数、耗时和宏动作数；实际学习率与抽样参数变化保存在原始记录中。正确率是更新前同一次前向计算中，合法动作 Top-1 与示范动作一致的决策数除以策略决策总数；多步宏动作逐步统计，排除强制动作和控制器步骤，不使用损失的样本权重。日志同时保存 `accuracy`、`accuracy_correct` 和 `accuracy_decisions`，汇总正确率按决策总数计算。这是当前训练批的正确率；旧日志的验证 NLL 可以显示，新 BC 不产生验证指标。
- PPO／价值预热：每次更新的损失、KL、裁剪比例、价值误差和旧策略重放误差。采样角色表现仍来自最近完成轮的汇总，不作为每个更新点的新评估结果，也不代替独立评估。

横轴、悬停提示、历史表和 CSV 均使用优化器累计更新次数（Update），曲线按数值间距绘制；微批拆分和梯度累积只产生一个逻辑批更新点。跨 epoch 继续计数，恢复检查点沿用保存的更新次数；恢复后回退的更新和不同 session 之间不连线。默认显示全部历史，损失曲线不截断为最近 2000 条，也不降采样；可选择最近 N 次更新，按更新次数区间筛选，不把缺失点补成零。历史表只展开最近 200 条原始记录，曲线和 CSV 保留所选范围的全部更新。

`updates.jsonl` 在每次成功更新后写入并刷新到文件，不逐步执行 `fsync`；记录当前逻辑批的指标，不是从 epoch 开始累计的平均。`history.jsonl` 继续保留原来的 epoch／整轮汇总和持久化方式，检查点保存时机不变。只有旧汇总日志时，网页把已有指标定位到其真实 `metrics.updates`，明确提示稀疏记录；无法从旧日志还原中间更新。缺少正确率的旧记录保留为空，不能由 NLL 推算。已经运行的训练进程需在保存检查点后按新代码重新启动，才会产出新加入的逐更新指标，网页不会主动中断训练。

PPO 采样时另显示已完成对局数。状态只反映日志最后写入的值，不能证明进程仍存活；缺失指标留空。服务仅仅读取日志，增量读取并缓存全部历史，忽略尚未写完的尾行并提示损坏记录；浏览器曲线接口省略重复的完整配置与原始 JSON，并在支持时压缩传输。`--limit 0`（默认）不限制历史，内存受限时可显式设置正数限制每任务记录数，页面会提示截断。默认监听 `127.0.0.1`，远程使用 SSH 端口转发；网页没有登录机制。

Steam Mod 的安装和接管入口见 [steam_recorder](../steam_recorder/README.md)。

## 代码导航与测试

| 模块 | 职责 |
| --- | --- |
| `control.py`、`treasure_rule.py`、`crystal_rule.py` | 控制分工、宝箱流程、水晶球规划 |
| `independent.py`、`recorder.py`、`dataset_index.py`、`data.py` | 导入、版本化索引、策略样本与权重 |
| `representation.py`、`public_history.py`、`rules.py`、`batch.py` | 公开输入、冻结词表、规则和张量打包 |
| `model.py`、`attention.py`、`policy.py` | 网络、选择与可微重放 |
| `loader.py`、`trainer.py`、`optim.py`、`checkpoint.py` | 数据供给、更新、预热与恢复 |
| `rollout.py`、`rewards.py`、`steam.py` | 对局、回报与实机控制 |
| `monitor.py`、`monitor_web/` | 只读日志监测 |

```bash
python -m pytest model/tests -q -m 'not engine and not cuda'
COMBAT_SOLVER_CONFIG=combat_solver_cli/lib/config.json \
  python -m pytest model/tests/test_native_a0_controls.py -q -m engine
python3 -m unittest model.tests.test_monitor -v
```

2026-10-08 的训练控制重构通过 247 项 Python 回归、16 个 subtest 及 3 项原生事件／宝箱验证；这是实现验证，修正后的训练表现与 Steam 实机接管仍待使用者验证。范围与历史证据见本地 [实施报告](../.scratch/a0-training-baseline/implementation-report.md)。
