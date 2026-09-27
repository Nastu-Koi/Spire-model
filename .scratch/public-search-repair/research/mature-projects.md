# 成熟搜索项目与不完全信息方法调研

研究日期：2026-09-26。本文只使用项目的源仓库、项目官方文档和论文作者/机构发布的原文；不把 README 的吞吐数字当作可迁移的性能结论。目标是为本仓库的 **Slay the Spire 2（STS2）公开信息搜索教师** 选择下一步实验，而不是为固定种子求解器背书。

适用前提：本文的“不可直接迁移”指不满足原公开教师约束，不代表真实引擎回溯本身不可取。若接受教师利用真实试走未来，存档搜索与工作区旧实现将成为更直接的候选；目标变更的比较见 [综合方案第 8 节](search-options.md#8-若接受真实引擎回溯更低工程成本的备选主线)。

## 先给结论

现有 `public_*` 搜索器应当分两条路线推进，先验证再合并：

1. **近期：完整性优先的公开状态模型。** 用原生引擎逐项校验高频牌、遗物、状态和回合转移；把无法可靠建模的效果明确为叶节点或单动作后重新观察。`sts_lightspeed` 和 `scumthespire` 的主要教训是：深树搜索建立在高覆盖、可回放的模拟器之上。
2. **中期：先造 `resample_public_state`，再谈 IS-MCTS/POMCP。** 它必须只以公开观察、公开历史和一份独立 RNG 产生可能的隐藏牌序/未知随机未来，并以性质测试证明样本保持同一公开信息集。不能从当前原生运行的真实内部状态克隆、也不能固定同一个已知 seed 的 RNG 计数器。

当前不应直接移植任一 STS1 搜索器，也不应只把束宽/深度换成 MCTS。前两者的优势来自 STS1 完整世界状态和已知 RNG；后者若没有可信的 belief sampler，只会把真实隐藏未来泄漏到标签，或把未经校准的叶估值放大。

## 术语与可比性

| 类别 | 可见/隐藏状态的假设 | 与本项目关系 |
| --- | --- | --- |
| STS1 存档/已知种子求解器 | 可加载完整战斗存档；可读取或推进真实 RNG | 可借鉴模拟器、动作枚举、评估和预算接口；**不能**直接作为公开信息教师。 |
| STS2 公开信息教师（本项目） | 决策时仅能用行动前可见信息；牌堆次序、内部 RNG 与未来结果隐藏 | 必须在每次真实动作后按新观察重新规划，或从公开信息集采样可能世界。 |
| IS-MCTS / POMCP | 在同一信息集或 belief 中采样世界，再共享动作统计/更新 belief | 提供正确的抽象和测试义务；不提供 STS2 的 simulator、观察投影或 sampler。 |

这里的“已知 seed solver”不是贬义：它适合验证模拟器和寻找该种子的强行动序列；只是违反本仓库 ADR 对训练教师的知识边界。[ADR 0002](../../../docs/adr/0002-public-information-search-for-bootstrap.md) 已明确禁止读取本局真实隐藏状态或以试走同一种子获得未来。

## 来源一：gamerpuppy/sts_lightspeed（STS1，完整模拟器与已知 RNG 搜索）

审阅版本：[`7476a81954020087da31d41d16fddf475746ec2d`](https://github.com/gamerpuppy/sts_lightspeed/tree/7476a81954020087da31d41d16fddf475746ec2d)，仓库最后提交日期 2024-08-10。

证据强度为**强（源码）**，但适用范围限于 STS1：项目元数据/源码使用 STS1 卡、遗物、怪物与其随机机制，不能证明 STS2 规则等价。

* [README](https://github.com/gamerpuppy/sts_lightspeed/blob/7476a81954020087da31d41d16fddf475746ec2d/README.md) 声称有全敌人、全遗物、Ironclad 与无色卡，以及战斗外/全 Acts 的实现进度；同时清楚地将现有树搜索写为“knowing the state of the game's rng”，而把“不知道 RNG 的可能结果搜索”列为计划项。这是它可作 simulator 覆盖目标、却不可直接作公开教师的直接证据。
* [BattleScumSearcher2](https://github.com/gamerpuppy/sts_lightspeed/blob/7476a81954020087da31d41d16fddf475746ec2d/src/sim/search/BattleScumSearcher2.cpp) 每次模拟从 `BattleContext` 拷贝开始，枚举出牌、目标、药水和结束回合；首次叶节点随机选动作，之后以 UCT 风格的均值加探索项选择。随机 rollout 一直运行到 `Outcome`，叶值是终局的胜负、剩余 HP、药水与极小回合惩罚（见同文件 `evaluateEndState`）。这说明它把“前向模型覆盖”放在“复杂叶估值”之前。
* [Search agent](https://github.com/gamerpuppy/sts_lightspeed/blob/7476a81954020087da31d41d16fddf475746ec2d/src/sim/search/ScumSearchAgent2.cpp) 在每个当前战斗状态新建一个搜索器；找到胜利序列时可连续执行若干步，否则按最大访问边推进。战斗外主要是规则/随机策略，不是整局树搜索。

**可迁移部分。**

1. 将“原生/模型转移是否等价”的差异独立成覆盖表和回归，而不是用更多搜索预算掩盖它；本仓库的 `unsupported_hooks` 统计正适合作为该表的分母。
2. 动作枚举必须包括卡牌目标、药水目标、结束回合和后续选择；对等价同名实体可去重，但前提是不会丢失公开顺序语义。
3. 预算应是显式的 simulations、时间、内存/节点上限，而非依赖隐式束宽。

**不可迁移部分。**

* `BattleContext` 的拷贝含有实际战斗 RNG/牌序；`README` 所说已知 RNG 搜索会让本局隐藏未来进入决策。将该对象直接 clone 给 STS2 教师，或从其真实 seed 重复 rollout，都违反公开信息约束。
* README 的 “1M random playouts in 5s with 16 threads” 是项目自述的 STS1 条件数字；没有相同硬件、配置、场景和验收，不能推导本项目吞吐或五分钟成功轨迹目标。

## 来源二：boardengineer/scumthespire（STS1，存档回放的优先队列搜索）

审阅版本：[`37282a116b13971378e72f2811d1bef6bd610fb9`](https://github.com/boardengineer/scumthespire/tree/37282a116b13971378e72f2811d1bef6bd610fb9)，仓库最后提交日期 2024-08-20。

证据强度为**强（源码）**，同样只适用于 STS1。它的 [Maven 配置](https://github.com/boardengineer/scumthespire/blob/37282a116b13971378e72f2811d1bef6bd610fb9/pom.xml) 依赖 STS1 `desktop-1.0.jar`、SaveState、LudicrousSpeed 和 CommunicationMod；[README](https://github.com/boardengineer/scumthespire/blob/37282a116b13971378e72f2811d1bef6bd610fb9/README.md) 要求在战斗中启动 AI。

* [StateNode](https://github.com/boardengineer/scumthespire/blob/37282a116b13971378e72f2811d1bef6bd610fb9/src/main/java/battleaimod/battleai/StateNode.java)、[TurnNode](https://github.com/boardengineer/scumthespire/blob/37282a116b13971378e72f2811d1bef6bd610fb9/src/main/java/battleaimod/battleai/TurnNode.java) 与 [BattleAiController](https://github.com/boardengineer/scumthespire/blob/37282a116b13971378e72f2811d1bef6bd610fb9/src/main/java/battleaimod/battleai/BattleAiController.java) 用 `SaveState.loadState()` 回放分支，把到达新回合的节点放进优先队列；默认目标为第 8 回合，之后按 6 回合窗口推进，并有 `maxTurnLoads` 限制。这是“深度/回放次数均须被显式预算”的实证代码例子。
* [ValueFunctions](https://github.com/boardengineer/scumthespire/blob/37282a116b13971378e72f2811d1bef6bd610fb9/src/main/java/battleaimod/ValueFunctions.java) 的中间评分不是只看当回合伤害：纳入玩家受伤、怪物有效生命、部分力量、金钱、药水、遗物和若干成长牌的专用项；且对 Gremlin Nob/Lagavulin 调低防御时的生命损失权重。这说明规则叶评估应按敌人/状态上下文校准。

**可迁移部分。**

1. 把 `public_search` 的固定权重改为版本化的特征契约：每一项说明可观察来源、适用敌人/状态、预期方向和排序测试案例。专用规则必须被测试锁定，不能仅凭直觉添加。
2. 用“已消耗模拟次数、最大深度、叶节点原因、每候选访问数”记录每一次选择。如此才能分辨错误来自预算、模型覆盖还是估值。
3. 多动作选择界面要作为过程状态，而不是把一组选择压扁成单个无序动作；这与本仓库 `CONTEXT.md` 的“多选过程”和“公开顺序”定义一致。

**不可迁移部分。**

* 存档载入的是已发生的完整世界，包含抽牌和 RNG 的真实结果；它可用于 STS1 battle AI，却不能代表同一公开观察下所有可能未来。
* 此项目的启发式权重证明“有人这样实现过”，不证明数值在 STS2、A0 或公开信息样本上正确；不要复制其系数或声称其带来胜率。

## 来源三：OpenSpiel IS-MCTS 与 MCTS（信息集、转置与预算接口）

审阅版本：[`48401890ee9857e611678302371378175a8e4c6b`](https://github.com/google-deepmind/open_spiel/tree/48401890ee9857e611678302371378175a8e4c6b)，仓库最后提交日期 2026-08-31。证据强度为**强（官方源码与官方 API 文档）**；算法抽象强，但没有 STS/STS2 游戏实现。

* [IS-MCTS 实现](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/algorithms/ismcts.py) 每次 `run_search` 先 reset，再按 `information_state_string()`（可选 observation string）和当前玩家作为树节点 key；每个 simulation 调 `resample_from_infostate`，并断言重采样世界的 key 与根一致。`max_world_samples` 可以限制保留的 root worlds；默认每次重采样。节点保存按动作的访问数、回报和先验，支持 UCT/PUCT，最终可按最大访问数或最大值输出策略。
* [官方 `resample_from_infostate` 文档](https://openspiel.readthedocs.io/en/stable/api_reference/state_resample_from_infostate.html) 定义其应保持目标玩家私有信息和全部公开信息、重采样他人私有信息，并要求不完全信息游戏实现该方法。这是本项目 public sampler 最有用的测试标准；STS 是单玩家，需将“玩家私有信息”替换为“当前公开观察”，将隐藏牌序、未来随机与不可见实体作为需重采样成分。
* [完美信息 MCTS](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/algorithms/mcts.cc) 显式接受 `max_simulations`、`max_memory_mb`、`max_wall_clock_time` 和 evaluator；chance 节点按 `ChanceOutcomes()` 概率采样，非终局叶节点调用 `evaluator->Evaluate`，并以 UCT/PUCT 回传。树超过节点预算会回收低访问子树。这是可移植的预算与可观测性设计，不是可直接复制的 STS 决策器。

原始算法依据：Cowling、Powley、Whitehouse 的 [ISMCTS 论文（作者机构公开稿）](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf) 说明信息集树把同一可观察决策点的统计合并，从而避免独立 determinization 重复搜索，同时也指出 determinization 的 strategy fusion 风险。

**可迁移部分。**

1. 定义唯一的 `PublicInfoStateKey`：只含公开观察、可从公开历史导出的状态和语义相关顺序；不得含实体内存地址、真实牌堆排列、真实 RNG state。若两个对象 key 相同，必须有相同的公开合法动作（或显式采用并测试 IS-MCTS 的不一致动作集处理）。
2. 将重采样、转移、估值和树策略拆成接口。先用公开 sampler 跑小样本 IS-MCTS；同时保留目前的“遇未知即叶评估”实现作差分基线。
3. 对每次根搜索记录 samples、独立 world 数、深度、叶因、每根动作访问分布和时间。这样预算实验能区分“更多 rollout”与“更多不同 world”。

**迁移限制。**

* OpenSpiel 在不同 simulation 中按信息集 key 聚合节点，不等于任意状态都能安全转置。STS2 的多选进度、卡牌临时费用、公开顺序、一次性效果和合法动作必须进入 key。
* `resample_from_infostate` 是游戏实现者提供的能力，OpenSpiel 不会替 STS2 反推正确牌堆/随机分布。没有通过公开一致性和分布校准的 sampler，IS-MCTS 的形式正确性不成立。

## 来源四：POMCP 原论文（belief 更新与黑箱生成模型）

证据强度为**强（论文作者的机构存档）**：[Silver 与 Veness，2010，*Monte-Carlo Planning in Large POMDPs*](https://dspace.mit.edu/entities/publication/5210cdcc-e73d-46f1-be33-d7887e6978f4)。论文将粒子化 belief 的 Monte-Carlo 更新和当前 belief 上的 MCTS 结合，只要求黑箱 simulator，不要求显式枚举概率分布。

它适合回答“未知抽牌/随机结果应如何进入树”：root particle 是**与观察一致的可能世界**，而非当前运行的真世界。对本项目的具体迁移限制也很重要：POMCP 虽然可用生成模型，生成模型仍必须以公开历史为条件采样；若底层 native engine 的 clone 偷带了真 RNG，便不再是合法的 particle。

POMCP 也不替代 simulator 覆盖：黑箱必须至少能从所采样世界正确推进到下一个公开观察。当前对抽牌、未知效果和回合结束截断的实现可先保留为保守叶节点；只有当对应隐藏状态分布与转移都能被验证时，才把它们纳入 particle rollout。

## 设计比较

| 维度 | sts_lightspeed | scumthespire | OpenSpiel IS-MCTS / POMCP | 对本仓库的结论 |
| --- | --- | --- | --- | --- |
| 模拟器覆盖 | 目标是 STS1 大范围规则复制 | 调用 STS1 游戏/存档执行真实命令 | 只定义游戏接口 | 先提升本地公开转移覆盖；不承诺 STS2 与 STS1 等价。 |
| 隐藏状态 | 现有树搜索知道 RNG | SaveState 回放真实世界 | 按信息集/belief 重采样可能世界 | 公开 sampler 是导入信息集搜索的前置条件。 |
| 叶估值 | 多数 rollout 到战斗终局 | 规则值函数、按敌人调整 | evaluator 是显式依赖 | 将规则估值做成可测接口，建立排序基准，不复制旧权重。 |
| 转置 | 当前战斗树主要按路径展开 | 保存/优先队列节点，并非公开状态合并 | 信息状态 key 共享统计 | 只对语义完整的公开 key 合并；先用碰撞与合法动作测试守住边界。 |
| 预算 | simulation 数；README 有并行 benchmark | `maxTurnLoads`、目标回合窗口 | simulation/时间/内存、world samples | 同时报告预算和覆盖；未验证前不宣称吞吐收益。 |
| 训练标签适配 | 否：已知 RNG | 否：真实存档 | 是，前提是 sampler 合法 | 仅最后一列可直接满足 ADR 的知识边界。 |

## 建议的实验顺序与验收门槛

### A. 先修模拟器覆盖和叶估值（当前实现可立即做）

1. 从历史公开帧建立分层集：敌人/遗物/状态/牌型/多选界面各至少覆盖一个；每帧记录支持效果数、`unsupported_hooks`、深度、叶因和所选动作。
2. 对高频未支持钩子逐项加入公开转移，并用同一公开动作在 native engine 后的**下一次公开观察**做差分。断言的对象应包括 HP、格挡、能量、公开手牌、公开状态、敌人可见意图、动作合法性与多选进度；不可比较的隐藏字段不要读入断言。
3. 为叶评估建立成对排序测试（杀敌/存活、当回合防御/成长、耗药/保药），并按敌人或可见状态给出适用条件。只有在该集上超过旧估值，才启用新特征。

验收：覆盖变化、原生差分结果和排序准确率都入 evidence；不能仅以“运行无异常”或平均分上升判定成功。

### B. 再建立公开 belief sampler（信息集搜索的硬门槛）

定义 `sample_public_world(public_observation, public_history, rng)`，并做下列性质测试：

* 每个样本的公开投影严格等于输入；样本不能读取真实 save/RNG/隐藏牌序。
* 样本的公开合法动作集与根一致；有条件合法的多选过程也一致。
* 多次采样确实产生可区分的隐藏世界；若某观察下只有一个可行世界，应明确记录。
* 在能够完全枚举的小牌堆夹具上，经验频率接近期望分布；不以生产 seed 的真实未来作“正确答案”。

验收：固定 sampler seed 可复现，改变 sampler seed 只改变可能世界而不改变输入公开观察；每一条性质独立失败时测试必须变红。

### C. 小型 IS-MCTS/POMCP 原型，不先替换主搜索

以 B 的 sampler、A 的已验证公开转移和独立 evaluator 组成小范围战斗原型。根节点按 `PublicInfoStateKey` 聚合、显式设定 `max_simulations`/wall-clock/world-samples，所有未建模效果仍在公共边界停下估值。与现有有限深度束搜索在同一**未参与调参**的公开帧集比较：合法性、确定性（固定各 RNG）、估值排序、动作分歧率、叶因及延迟。

只有在这些门槛均满足后，才考虑扩大到抽牌后的多世界 rollout 或整局策略。成功整局轨迹仍要由 native engine 实际完成；搜索估值或 sampled rollout 不是通关证据。

## 未声称的结论

本调研没有运行三方项目，没有测其胜率，也没有发现 STS2 的成熟、可审计公开信息搜索器。`sts_lightspeed`/`scumthespire` 的设计和 OpenSpiel/POMCP 的算法接口足以支持上述架构建议，但不能证明任一建议会提高本仓库的成功率或达到吞吐目标；这需要按 A--C 的对照实验验证。
