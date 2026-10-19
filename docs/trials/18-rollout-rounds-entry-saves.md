# 采样轮 collect 与每场战斗的入场存档

- **时间**：实现 2026-10-11（工单 15，`afcd130`、`f8a9bcb`、`2c24600`，合并提交 `5a58fa1`）；删除 2026-10-17（`b92fcdf`，repo-tidy 03）。
- **来源**：`.scratch/a0-training-baseline/issues/15-rollout-storage.md`；`.scratch/repo-tidy/spec.md`、`.scratch/repo-tidy/issues/03-remove-collect-round.md`；`db6648d:ARCHITECTURE.md`「采样存储」。
- **状态**：被删除。新路线不读采样轮；入场存档只在 Boss 入场切点上保留。

## 动机

a0 路线的训练要直接读策略自己的采样：critic 学局外帧的胜负，AWR 按局外帧加权，学生状态的战斗交给求解器出标签（[16](16-student-solver-labels.md)）。改前的采样写完整对局 JSON 和流水账，每局约 30 MB。需要一种训练能直接读的格式：只存局外决策帧，战斗只存入场存档，按迭代号筛选。

## 做法

- `python -m model collect` 把一轮整局采样写到 `<输出>/round-N/`，格式 `rollout-round-v1`：分片加索引，与导入示范相同（来源 `rollout`，索引另记 `policy_version`、`policy_digest`、`iteration`、`victory`），BC 加载器直接复用。
- 局外帧：策略决策的帧、完整候选、所选动作、`old_log_prob`、阶段。战斗边界按引擎的 `encounter_started`／`encounter_completed` 划，战斗中的 `card_select` 也算战斗；战斗宏动作只记计数与对数概率，不存帧。
- 入场存档：每场战斗第一个决策时发 `write_continue_save`，引擎在非地图房间写“进房前”的存档（约 48 KB）；一局内 zlib 压缩暂存，一局结束写成 `.saves.jsonl.gz`。重入用 `enter_anchor` 加 `anchor_room`（`model.saves.anchor_commands`）。
- 内存：采样进程一局内只留压缩帧，一局结束交给采集进程追加写出并释放。

## 结果

验收（20M 旧结构 `trained-v8-potions/epoch-000017`，`--workers 44`，同一组 100 个种子，各跑两轮）：

- 吞吐：改前每秒约 445／376 个宏动作，改后 399／420，均值 410 对 409。
- 磁盘：“改前 2,996 MB／100 局（对局 JSON 1,457 MB、流水账 1,479 MB、战斗记录 60 MB，30 MB／局）；改后 70.0 MB／100 局（分片 13.9 MB、战斗记录 51.6 MB、存档 4.5 MB，0.70 MB／局，其中训练读的分片 0.14 MB／局）”。
- 训练读取：`bootstrap` 在一轮 100 局上 13 次更新，加权 NLL 0.656、正确率 0.80。
- 存档抽查：CPU 单引擎同一种子带存档与不带存档打两遍，9 场战斗的帧与动作逐步相同，223 个宏动作的 `old_log_prob` 全部相同。从一轮里抽 25 场重入，HP、最大 HP、金币、幕、楼层、卡组、遗物、药水、敌人种类与原战斗一致。但“敌人的随机 HP 在 25 场里有 13 场与原战斗差 1–3 点（如 40 对 39、55 对 57；固定 HP 的精英与 Boss 一致），开局手牌全部不同”；一场 Boss 战的 `RELIC.EMBER_TEA` 计数原战斗显示 4、重入显示 3。

## 原因分析

采样轮服务的三条线都没走通：学生状态标签没带来提升（[16](16-student-solver-labels.md)），critic 过不了门槛（[19](19-win-prob-critic.md)），AWR 被裁剪目标取代（[24](24-awr.md)）。幕组迭代的采样全在 `WaveWriter`（`act-groups-v2`），“训练不再读采样轮；`buffer.RoundWriter` 与 `WaveWriter` 的分片、隔离、摘要逻辑重复”。每场战斗的入场存档“只在 `collect` 的布局下写，训练与幕组都不读”。

## 结论与后续

- `b92fcdf` 删除 `collect` 子命令、`main/play/buffer.py`、`env/saves.py` 的 `EntrySaves`／`read_saves`／`anchor_commands`、`CliEngine.enter_fight`、`rollout` 的 `saves=` 与 `OUTSIDE_FRAMES` 布局及对应测试，代码 −542／+108 行。
- 保留 `continue_save`、`FightTracker`、`fight_resume_command`：Boss 入场切点的存档与续打走这三者（[env 入场存档](../implementation/env.md#入场存档)、[play 幕组](../implementation/play.md#幕组)）。
- 现在的采样格式见 [play 采样对局的格式](../implementation/play.md#采样对局的格式)。按温度采一轮整局看行为分布的用途，可以由 `evaluate` 提供，不需要单独格式。

## 教训

- 重入不是重放：从进房前存档重入会重抽敌人随机 HP 与开局手牌，在战斗开始时计数的遗物差一；任何“同一场战斗”的假设都要算进这部分噪声。
- 为一条路线专门设计的存储格式，路线放弃后就是重复代码；格式跟着用它的训练方法走，不单独长期维护。
