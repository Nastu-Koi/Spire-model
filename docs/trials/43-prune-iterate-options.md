# 精简 iterate 的参数

- **时间**：2026-10-28
- **来源**：`.scratch/training-plan/issues/35-prune-iterate-options.md`（工单 35）；依据是 `runs/rl-20261023/train/*/state.json` 里 32 段训练（fight-rl 到 fight-rl-18，以及 explore-*、replay-* 试跑）实际记下的设置，对照工单 12、15、19、22、26、33 的结论
- **状态**：代码已删除。局外对照的三个文件（`main/learn/choice_pairs.py`、`main/play/choice_pairs.py`、`main/tests/test_choice_pairs.py`）删除时从未提交过，git 里查不到原文；做法见 [ADR 0003](../adr/0003-choice-pairs.md)、[ADR 0004](../adr/0004-choice-pair-futures.md)

## 动机

探索阶段给 `iterate` 加了约 60 个参数，其中不少已证无效，或者从来只取一个值。多余的开关让启动脚本和设置记录变长，组合起来的代码路径（互斥检查、只在某个开关下才走的分支）也没有在有效配置里跑过。按[贯穿的教训](route.md#贯穿的教训)最后一条，没在有效配置里用过的组件随失败的方案一起删。

## 做法

用户 2026-10-28 定：删 A、B 两类，正在实验的不动。

**A 已证无效**

| 参数 | 依据 |
| --- | --- |
| `--choice-pairs`、`--choice-pair-arms`、`--choice-pair-futures`、`--choice-pair-objective` | 第七段两轮：训练内对各分枝的期望优势 0.088 → 0.109，工单 18 的固定检查集上分开的点的软胜率 0.5902、0.5899、0.5895，没有变化（ADR 0004 背景）。多未来探针 ICC ≈ 0.02，K = 8 时单点标签的可靠度也只有约 0.14（工单 22 阶段 03）。第七段第 2 轮后停用，此后各段都是 0 |
| `--fight-group-fork-depth` | 只在第八段用过（2）。工单 26 用户 2026-10-26 定不保留，改由回溯分叉覆盖中后段 |
| `--explored-clip` | 工单 33 阶段 09 验收不过：负优势裁剪基本不起作用，一遍里每条样本只看一次，比值就是漂移（[36](36-explored-dual-clip.md)）；固定集普通决策熵 0.2202，对旧学法 0.2283 |
| `--explore-potions`、`--boundary-bosses` | 幕组里的战斗用药探索与 Boss 入场切点都要在幕组里抽战斗，和 ADR 0001 的 greedy 战斗互斥。rl-20261023 的 32 段里前者一直是 0、后者一直为空；之前的结论见 [33](33-potion-exploration.md)、[36](36-explored-dual-clip.md) |

**B 没用过或一直固定**

- `--combat-demos`（静态示范目录）：32 段里都是 null，BC 阶段用的是逐轮示范 `--round-demos`。
- `--fight-group-explore-potions`：前三段 0.02，第四、五段 0.01，第六段起一直是 0。
- `--fork-depth`：前六段是 2，第七段起是 0。工单 15 的取证：被强制的第二名在采样温度下概率的中位数约为 0；第三幕第一次选路取第二名的胜率是 0.558，取第一名是 0.615，约差 2.9σ。用户 2026-10-24 定关掉，改为每组 6 段。
- `--updates <步数>`、`--phase-weights`、`--weight-cap`、`--reference-every`：一直是默认值（`round`、`group`、10、等于 `--evaluate-every`）。`--reference-every 0` 的固定参考已在 [31](31-fixed-kl-reference.md) 里按机理否定。
- `--greedy-fights`、`--fight-entries`、`--drop-uniform-groups`：32 段里一直开，改为固定开，去掉开关。

库代码随之删掉：
- 局外对照：两个模块、`mixed.CHOICES`、全臂目标 `Sample.arms` 与 `clipped` 里的 `arms` 一类、幕组的重放日志（`replays.jsonl.gz`、`replay_log`）、rollout 的 choice 分枝、`digests.replay_key`；
- 幕组与战斗组的强制分叉（`rollout.fork_ranks`、任务的 `fork`）；
- 采样时的用药探索（`RolloutRunner.explore_potions`、`rollout.exploration_rates`）；
- Boss 切点（`groups.cuts` 的 `bosses`、`RolloutRunner.run(until_boss=)`）；
- `Clipped(explored_clip=)`。

剔除全组同结果的组改为 `iterate.DROP_UNIFORM`，只是给合成世界的小组测试留的接缝，命令行不再暴露。

## 结果

- 非引擎测试 607 → 578 项，全部通过：删掉的 29 项都是只测已删开关的；其余测试改为测固定的行为。
- `iterate` 的参数少了 18 个。

## 原因分析

这几类开关失败的原因各不相同（标签不迁移、强制的第二名概率约为 0、裁剪对单遍更新不起作用），但都是在找局外或战斗关键决策的信号。它们留下的共同点是：失败以后开关留着、默认关，代码路径和互斥检查越积越多，而有效配置从来不走这些路径。

## 结论与后续

- 保留并仍在实验的：`--explore-outside`、`--explore-prior`、`--prior`、`--explored-on-tree`、`--explored-centered`、`--fight-hindsight*`、`--fight-pool*`、`--fight-group-kind-size`、`--fight-group-shares`。
- 引擎侧的 `reseed` 命令与 `CliEngine.reseed` 保留，训练代码已不再调用。
- 推理层按名次取候选（`Decisions.rank`、`wire` 的 `rank` 字段、`PolicyNetwork.choose(rank=)`）也保留了：删它要动网络和 CUDA 图，这次没动；现在已经没有调用方。
- 旧运行目录照样能读：settings 只记录、不读回。但战斗组的 `plan` 少了 `fork_depth`、`explore_potions` 两项，做到一半的旧轮不能接着续跑。

## 教训

- 一个开关被证无效、或停在默认值两段以上，就在同一次改动里删掉，别等积累起来再一起清。
- 删一个只在某个开关下走的路径前，先看实际运行记下的设置：settings 每次迭代都记在 `state.json` 里，比凭记忆判断可靠。
