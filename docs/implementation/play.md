# play：对局循环、采样、幕组与评估

`main/play/`。评估（`evaluate`、迭代里的评估）与幕组采样共用同一个对局循环，写出的东西不同。

| 模块 | 内容 |
| --- | --- |
| `rollout.py` | 对局循环（`RolloutRunner`）、温度与战斗边界、`collect_round` |
| `sampling.py` | 多进程采样（`Samplers`）：引擎与帧解析在采样进程里，采集进程只跑网络 |
| `wire.py` | 打包好的决策的跨进程扁平格式；采样进程的发送记录（`Sent`）与编码表的行 |
| `groups.py` | 幕组：按边界幕把一局切成几波，第一波从开局、之后每波从上一波的幕开头存档续打，各写成分片 |
| `fights.py` | 每场战斗的入场存档与战况：段在对局里记（`FightLog`），每波写成 `fights.jsonl` 与 `fights/`（`FightWriter`）；一帧里玩家的处境（`state`：战况的入场状态） |
| `potion_value.py` | 药价拟合的取点与配对续打：来源对局在战后选图点写点，记从幕锚点重放到点的候选与帧键（`PointEngine`、`play_sources`），按幕、腰带满否与 HP 段抽点（`draw`），原样与在点上丢掉一瓶药的续打（`branch_saves`、`play_branches`、`PointEngine.resume`） |
| `fight_groups.py` | 战斗组：从一轮幕组记下的战斗里挑（`pick`），每场从入场存档续打几条、截在战斗结束，写成轮目录下的 `fight-groups/` |
| `fight_eval.py` | 战斗评估集：从入场存档抽一批固定的战斗（`build_set`），策略 greedy 逐场打到战斗结束（`play_set`），每场的战况、打法与危急帧的用药概率；读求解器在同一批存档上打的结果（`solver_rows`） |
| `digests.py` | 决策摘要：段交出时每个策略决策里幕组优势要用的东西（帧指纹 `frame_key`、状态 `where`、帧的阶段），每波一个文件 |
| `evaluation.py` | 评估报告、配对符号检验（`compare_evaluations`）、留出 NLL 的表格 |
| `testing.py` | 合成引擎（`SyntheticEngine`，不是游戏模拟器）与合成示范，供测试和 `smoke` |

## 采样进程分工

吞吐的上限来自 Python 解释器，不来自 GPU 或引擎：一个决策在两次前向之间要花几毫秒 Python（解析引擎回复、写日志、帧变成词元、拼成张量），放在一个进程里与跑网络的线程共用一把解释器锁。所以引擎和这些工作放进采样进程，采集进程只把各采样进程在上一次前向期间打包好的决策合成一批、一次前向整批采样，只把（槽位、对数概率）发回去，采样（不是 greedy）时再带上每个决策在全部槽位上的温度化对数概率（与选择拼成一张表一次拷回主机，按进程切片以 numpy 数组发回）；每个决策带着自己的温度。采样进程由采集进程 fork 出来，共享词表、不接触 CUDA。

- 一次前向的耗时随每次前向带的决策数缓慢增长；吞吐取决于每次前向带多少决策；一轮的耗时不会短于最长一局的步数乘单步往返时间。
- 每个采样进程只带一个引擎时引擎之间互不等待：`--processes` 默认等于 `--workers`。
- 一轮的对局数（每角色种子数的五倍）不宜少于引擎数，否则有引擎空闲。
- 采集进程是唯一串行的环节，消息与张量的固定开销会超过前向本身，所以跨进程用 `wire.py` 的扁平格式：去掉填充，一条消息只有表头、一个整数数组和一个数值数组，任意多条消息在设备上用固定次数的数组运算合并。程序组的字段只带真实节点的，另带各节点在补齐布局里的行号（`nodes`），与绑定目标一样随程序的位置平移。
- **编码表**：一局里前后决策的实体与局部程序几乎不变（v30 评估的整局里，98% 的程序、96.6% 的实体与本局更早某个决策的内容相同），每次前向都把它们重新编码，是采集进程最大的一块开销。采集进程每轮在模型的设备上建两张表（`EncoderTables`：每个实体的字段编码、每个程序的向量，各占一行），每个采样进程占一段行（`wire.TOKEN_ROWS`、`wire.PROGRAM_ROWS`，各 16384、4096 行）。采样进程按内容摘要（`fields.row_digest`、`Effect.digest`；带绑定的程序再加上所绑实体的摘要）记下本轮发过什么（`wire.Sent`）：一个实体或程序第一次出现时带着字段发出、分到一行，之后只发行号。采集进程先给新内容跑实体编码器、写进表，再按行号取出每个 token 的编码与程序（`SharedEncoder.from_tables`），引用、投影与全局层照常算。新内容很少，所以新实体与所有新程序节点合成一次字段块，新程序只按不超过 64 个节点与更长分两组（`wire.GROUPED`）：每多一组或一次字段块，就多一轮发射。采样进程也只为新内容建字段张量：`prepare_decisions(tensors=False)` 让 `collate` 把 token 与程序节点的字段留成行（`batch.Rows`），`wire.pack` 定出哪些是新内容后只给它们建张量，宽度仍按全部行算，所以发出的内容与先全建再筛逐位相同。采样进程问策略的帧都是 `clean_frame` 洗过的，观察时不再洗（`observation(cleaned=True)`）。一个决策有一千多个 token，新内容只占百分之几，全建张量会是采样进程最大的一块 Python 开销。
  - 表只对算它的那组权重有效：每轮开始新建、结束释放（训练期显存归训练），采样进程每轮从空记录开始。
  - 行按半区填：用着的半区满了就清空记录、换到另一半，所以同一条消息里先发的包所指的行，在这次前向读它时还是原样；一条消息里的包加起来不能超过半区（一个包超过时 `pack` 报错）。
  - 摘要是 Python 自己的 hash，只在一个采样进程之内比较；64 位摘要在一轮几十万份内容里撞上的概率可以忽略。
  - 表里的编码是在别的批里算的，与本批现算的差属于 FP32 的跨批噪声（实测编码差均值约 1e-7），不改变 greedy 选择的一致率（见 [network 精度](network.md#精度)）。
  - 表的显存：64 个进程时，20M（局部宽 128）约 0.7 GB，200M（384）约 2 GB。
  - 效果（第三幕续打，各三次的平均，工单 38）：48 引擎 717 → 906 决策/秒，64 引擎 723 → 962；编码器不再随每次前向的决策数增长，加引擎重新有用，所以 `--workers` 默认 64。同配置的跑间差可达 ±10%，比较要重复几次。
- 采样进程连同引擎跨轮保留；每打完一局清掉解析缓存，把空闲内存还给系统。每对“引擎 + 采样进程”约 250–300 MB（其中引擎约 150–200 MB）；64 对在第三幕采样时机器最低可用内存约 13 GB（31 GB 内存的机器）。
- 采集进程按轮记自己的账（`Samplers.timing`，幕组每波 `summary.json` 的 `collector`、`iterate` 的 `sample` 记录）：`seconds`（这一轮的总秒数）、`passes`（前向次数）、`decisions`、`decisions_per_pass`、`decisions_per_second`、`seconds_per_pass`，以及等采样进程（`waiting_seconds`）、解包（`unpacking_seconds`）、选择（`choosing_seconds`：前向加把结果拷回主机）、回传（`sending_seconds`）各花的秒数，其余记为 `other_seconds`。`collecting` 把采集进程这一轮按 `main/env/clock.py` 的名字再拆开：前向各段在主机上花的时间（`network_encoder` 新内容的编码器、`network_layout` 摆放与关系偏置、`network_layers` 全局层（图重放）、`network_heads` 决策头与抽样），`readback` 是等 GPU 算完并把选择拷回主机；`sampling` 是全部采样进程的账之和：`playing`（引擎线程打一局，含下面引擎线程的各项）、`engine`（等引擎回复，含引擎自身、管道与读线程）、`engine_parse`（解析回复）、`cleaning`（`clean_frame`）、`frame_packing`（压缩存帧）、`choice_wait`（引擎线程等选择）、`handing_over`（一局交出时生成分片行与决策摘要），主线程的 `observation`、`preparing`（含 `observation` 与 collate）、`wire_packing`、`sending`、`collector_wait`（等采集进程的选择）、`answering`，以及每局后的 `releasing`。每项给秒数、次数与 `ms_per_decision`（秒数除以这一轮的决策数：每个决策在打它的那个采样进程里花了多少）。读法：等待占比小、每次前向的决策数远少于引擎数，说明引擎在等采集进程，加引擎能提吞吐；每次前向耗时随决策数线性涨，说明前向本身是瓶颈。采集进程只有一个：没有 MPS 时多个 CUDA 上下文在 GPU 上按时间片轮转、核函数序列不重叠，多开（各一份模型、各带一组引擎）不提吞吐。试过的引擎数与多采集进程的对比见 [trials](../trials/route.md)。
- `Samplers.rest()` 结束引擎、留下进程，下一次采样按需重启引擎。迭代驱动用它在训练期间腾出引擎的内存，同时保住训练之前 fork 的进程（见 [learn](learn.md#迭代驱动)）。
- 采样进程要在父进程训练之前 fork：训练过的父进程堆里布满空闲块，子进程新分配的对象落进继承来的页，每写一页就复制一页，私有内存成倍上涨。

## 温度与战斗边界

一场战斗从引擎的 `encounter_started` 到它打赢后的第一个决策之前；其中的出牌和战斗中的选牌（`card_select`：弃牌、消耗）按 `combat_temperature`，其余决策按 `other_temperature`（默认 0.3／0.5，记在检查点和对局里；BC 检查点记的是训练时 profile 的值，采样时用 `--config` 换成当前 profile 的）。greedy 的 `evaluate` 记 `temperature: null`，概率按温度 1 记。

限定候选（`RolloutRunner.run` 的 `restrict`）：[回溯分叉](#回溯分叉)在危急帧强制喝某瓶药时，把那瓶药的 USE_POTION 候选的槽位作为 `Decisions.explore` 掩码随决策打包过去（`wire` 的 `explore` 字段，与 `legal` 同形），采集进程的 `PolicyNetwork.choose` 只在这些槽位里按 π_T 的相对概率抽，返回的对数概率仍是整个合法分布下该动作的 π_T。步带 `explored: true`（设计见 [architecture 探索步](../architecture.md#探索步)），交给采集进程的摘要与分片索引行带 `explored` 计数，每波 `summary.json` 汇总为 `explored`。不限定时打包多一个全零字段。

局外探索（`iterate --explore-outside`，`RolloutRunner.explore_outside`，`rollout.outside_rates` 把一个数或按阶段的映射整理成 `OUTSIDE_PHASES` 里率不为 0 的阶段；设计见 [architecture 局外探索](../architecture.md#局外探索)）：不在战斗里（`FightTracker` 判定）、阶段不是 combat、没有名次与限制的采样决策，先照常交给策略抽，再按阶段的率抽签（`explore_outside_fights`）。抽中时用这次抽样带回的整个分布（`Choice.log_probs`，温度 T 下）找出最可能的候选，在其余不是 DISCARD_POTION 的候选里一半取对数概率最高的、一半均匀取一个，换成它：记下的 `log_prob` 是它在分布里的值，`base_log_prob` 是温度 1 下的值（T·log π_T 再归一化），步带 `explored: true`。没有别的候选时保留原来的抽样。改换在采样进程里做，不经网络再走一趟；单进程路径 `choose_batch` 采样时也带回 `log_probs`，两条路径相同。设置随 runner 交给采样进程（`Samplers.collect` 的设置字段），对局头只在开时记 `explore_outside`。战斗组与回溯分叉用的 runner 副本把它关掉。

先验提议（`iterate --explore-prior`、`--prior`，`RolloutRunner(explore_prior=, prior=)`；设计见 [architecture 局外探索](../architecture.md#局外探索)）在局外探索之前做：同样的决策，策略照常抽完后按阶段的率抽签（`explore_by_prior`），抽中时向先验要它最可能的候选（`sample=False`），不是这次抽样分布里最可能的、也不是 DISCARD_POTION 时换成它，记法与局外探索相同（`rollout._explored`），`Choice.explored_by` 与步的 `explored_by` 为 `"prior"`；没换时再走 `explore_outside_fights`。率为 0 时不抽签，随机数的消耗与不开时相同。单进程路径里先验是 runner 的 `prior` 网络的 `SessionPolicy`；采样进程里是 `_Waiting.policy` 返回的远端策略的 `prior`：它的请求标为 `PRIOR` 通道，用进程的第二个 `Sent` 打包，收集进程 `_answer` 按通道分开，各自 `unpack` 进自己的编码表（`Samplers(prior=)` 的网络，`_collect` 在 runner 开了 `explore_prior` 时建表）、各过一遍网络，先验那一路 greedy、不带分布；回给进程的分布按行对齐，先验的行为 None。runner 开了先验而 `Samplers` 没有先验网络时 `collect` 报错。设置字段带 `explore_prior`，先验网络不交给采样进程。对局头只在开时记 `explore_prior`。决策摘要的 `explored` 字段对先验的提议写 `"prior"`（`digests.explored_mark`），其余仍是布尔值。

greedy 战斗（`RolloutRunner.greedy_fights`，`iterate` 的幕组固定开、评估与战斗组不开；设计见 [architecture 采样温度](../architecture.md#采样温度)）：采样的对局里，`FightTracker` 判定在战斗中的决策不抽样，取策略最可能的动作；战斗外照常按 `other_temperature` 抽。`temperature_for` 对这类决策返回 None，`run` 的 `choose` 改用 `sample=False`、温度 1 问策略。单进程路径就是 `choose_batch(sample=False)` 的 argmax。采样进程把它与其余决策一样以温度 1 打包（不加字段），采集进程这次前向照常抽一个槽位；采样进程收到返回的温度 1 分布后改取其中最大的槽位（相等时取第一个，与 greedy 前向的 `argmax` 相同），对数概率与 `base_log_prob` 都取该槽位在温度 1 下的值，采样分布照记。
- 这些宏动作的 `temperature` 记 null：它们不是抽出来的。对局头记 `greedy_fights: true`（不开时没有这一项）。
- 战斗里不探索；回溯分叉要抽战斗，`sample_hindsight` 拒绝 greedy 的 runner。
- 战斗只按 `FightTracker` 判定，与幕组里不留帧的判定相同。阶段为 `combat` 而追踪器不在战斗中的决策（只在引擎缺了遭遇事件时出现）仍按战斗温度抽。
- 关闭时每个决策的问法（是否抽样、温度、探索）与没有这个开关时逐条相同。

## 采样对局的格式

幕组的每一波是一个数据目录：与示范导入相同的分片加索引（见 [data](data.md)），目录布局见[幕组](#幕组)。

对局（`source: "rollout"`）记每个策略宏动作的帧、完整候选、所选动作、`temperature`、`old_log_prob`、`base_log_prob`、阶段、奖励，采样时每个有选择的步另记 `log_probs`（采样分布：帧里每个合法候选在这一步温度下的对数概率，按候选顺序、6 位小数；探索步与分叉步也是策略自己的整个分布，`Choice.log_probs`）；段保留全部帧，战斗在内（greedy 战斗的轮除外）。`combat_frames=False` 时，内存里战斗中的宏动作只留 `phase`、`fight`（第几场）、`temperature`、`branches`、`decisions`、`old_log_prob`、`base_log_prob` 与 `reward`，不留帧；用它的是 `SUMMARY_RUNS`（写文件时连宏动作也不写，见[评估报告](#评估报告)）与 greedy 战斗的幕组（战斗的宏动作照这样写进分片，见[幕组](#幕组)）。每局另记 `status`、`victory`、`ledger`、`climb`、seed、角色、难度、策略版本与摘要（`policy_version`、`policy_digest`）、`vocabulary_hash`、`numeric_backend`、`precision`、`sampling`、温度、`explore_outside`、`explore_prior`（只在开时）、`greedy_fights`（只在开时）、`iteration`、`fights`（胜、负、没打完的场数）。

一局没打完时 `status` 留 `unresolved`：引擎停在 `waiting` 边界超过 `RolloutRunner.timeout`（30 秒）记 `unresolved_wait_timeout`，步数到 `max_steps`（`--max-steps`，默认 10000）记 `unresolved_step_limit`；协议或进程错误时 `status` 为 `error`。

采样进程一局内只保留压缩后的帧，一局结束把对局压成一行交给采集进程追加写出；采集进程不持有对局。不写对局文件的布局收尾时不再把帧解回对象：`rollout.packed_line` 把各帧压缩前的 JSON 文本逐个拼进分片行，读回与 `member(run)` 相同。帧的文本由 orjson 写出、读回（`rollout._pack`／`_unpack`）：与标准库的文本在浮点写法、非 ASCII 是否转义上不同，读出的值相同；读取进程读分片也用 orjson。原因：采样进程会一直占着它打过的最大一段的内存峰值，把帧解回对象会让这个峰值大很多。

`packed_line` 为数各阶段的策略宏动作本来就要把每个宏动作第一步的帧解码一次，交出段的同时（`sampling._shard_line`）就在这份解码上记下这一段的**决策摘要**（`digests.py`）：每个策略宏动作一行（`digests.FIELDS`：宏动作序号、帧指纹 `frame_key`、所选候选、幕、层、HP、最大 HP、金币、卡组／遗物／药水数、是否战斗中、是否探索步／分叉步），连同 run_id、组、角色、结果、难度，以及 `frameless`（这一段不留帧的宏动作数：greedy 战斗的战斗，它们没有摘要行，`digests.frameless`）压成一个 gzip 成员随段交出。状态只记原始数，特征向量由优势那边算，换特征不必重采。帧里没有带幕与层的玩家实体时这一段不记摘要（`digest` 为 None），优势那边读分片、在那里报错。每决策多花约 0.35 ms（指纹 0.30），一轮约 440 CPU·秒，摊在几十个采样进程里。交出的摘要另带 `distributions`：这一段每个有选择的步是否都记了采样分布。

## 幕组

`main/play/groups.py`，设计见 [architecture](../architecture.md#训练路线)：

- `RolloutRunner.run` 多了段的参数：`until_act` 让对局在该幕第一个决策处停下（截在战斗结束的 `until_fight_end` 见[战斗组](#战斗组)）；这一帧不属于本段，引擎在那里写的续局存档随记录返回（`save`：`{"at": "act", "save": …}`），`outcome` 为 `reached`，另记 `reached_act`；`resume` 给这样一份存档，对局从它续打：幕开头的经 `CliEngine.resume`（引擎 `resume_save`）回到地图，入场存档（`{"at": "fight", …}`）经 `CliEngine.resume_fight`（引擎 `resume_fight`）重进那场战斗，`character`、`seed` 沿用存档那一局。终局时 `outcome` 为 `win`／`loss`。`run_id`、`group` 由调用方命名。`validate_run` 接受 `outcome=reached`、`victory=None` 的段。
- 波：切点（`cuts(boundaries)`：边界幕的幕开头，默认第二、第三幕）把一局切成几波，每波按起点命名（`waves`：`act1`、`act2`、`act3`）。第一波的任务是每个（角色，种子）`n` 条，打到第一个切点；之后每波是上一波每份存档 `m` 条（`m` 一个数或每波一个），打到下一个切点，最后一波打到终局。到达切点的段各写一份存档到 `<波>/saves/<run_id>.json.gz`，行记在 `<波>/saves.jsonl`（`at` 为 `act`，`act` 记到达的幕）；续打的任务只带存档路径，采样进程自己读（`play(..., segment=True)`，布局 `rollout.SEGMENTS`）。组名 `<角色>:<种子>:act1` 与 `<父段>:<波>`，段名是组名加序号，所以从段名能一路回溯到开局。
- 难度（`sample_groups(ascensions=)`，角色 → 种子 → 难度，`env.seeds.round_ascensions` 给出）：第一波的任务带 `ascension`，`RolloutRunner.run(ascension=)` 按它开局、核对帧契约的 `fixed_ascension` 并记进对局（不给时用 runner 自己的）；续局从存档进，引擎沿用存档的难度，与任务所带的不符时这条段报错。存档行带 `ascension`，`resume_jobs` 照抄，所以同一组、同一条链同难度。轮目录的 `metadata/ascensions.json` 记分配，轮 summary 的 `ascensions` 指向它；每波 summary 的 `by_ascension` 按难度记写出的段数与结果。每局的难度本来就在对局记录与索引行里（`RunShards.at_ascension` 按难度筛）。
- 存储：每波一个分片加索引加 `summary.json` 的数据目录，段保留全部帧（含战斗），索引行多了 `group`、`outcome`、`distributions`。
- greedy 战斗（runner 的 `greedy_fights`）：`sample_groups` 用 `dict(SEGMENTS, combat_frames=False)`，战斗的宏动作照[采样对局的格式](#采样对局的格式)只留计数与对数概率、不留帧。`control.policy_macros` 不把它们算作策略宏动作：不进索引行与 summary 的 `phases`，不是训练样本，也没有决策摘要行（摘要行的 `frameless` 数它们），所以幕组的优势与更新都只有战斗外的决策（见 [learn 幕组的优势](learn.md#幕组的优势)）。分片随之小得多（战斗约占宏动作的七成）。轮 summary 记 `greedy_fights`。`WaveWriter` 每写完一段的分片行与索引行，就把它的决策摘要追加进这一波的 `decisions.jsonl.gz`（各段一个 gzip 成员，连起来读是一个 gzip 文件；一轮约 50 MB），续跑时接着追加，同一段出现两次以后一行为准；先写分片与索引，摘要里不会有索引没有的段。每波 summary 与轮 summary 的 `distributions` 说这一波（全轮）的每段是否都记了采样分布。没完成的段写到 `quarantine/` 并计数，不让整轮失败（组变小）。轮的 `summary.json`（`kind=act_groups`，`format=act-groups-v2`）记切点（`boundaries`、`cuts`）、波的顺序、各波的段数、结果、各阶段宏动作数、写出的存档数、耗时；每波的 summary（`kind=act_groups_wave`，同一 `format`）记它的切点（`until`、`until_act`）、任务数（`segments`）、写出数（`written`）与按原因计的失败数（`failed`）。
- 续跑：`sample_groups(..., resume=True)` 让被打断的轮从所在的波继续：有 `summary.json`、且其 `segments` 等于这一波的任务数、`written` 加失败数也等于任务数的波原样沿用，有索引没有 summary 的波保留已完成的段（`ShardWriter(resume=True)` 在索引后追加、新分片接着编号，契约从已有的段里读回）只打剩下的任务，没有索引的波移到轮目录的 `interrupted/` 下重打；波的 summary 多一项 `resumed`。
- 战况与入场存档（`sample_groups(fight_entries=True)`，`iterate` 固定开；`fights.py`）：段在对局里用 `FightLog` 跟着 `FightTracker` 记它打的每场战斗。战斗第一个决策开一条记录，不是事件引发的战斗当场让引擎写[入场存档](env.md#入场存档)（格式 `{"at": "fight", "save": …, "fight": …}`，`fight` 带楼层，`resume_fight` 可重进；采样进程的计时名 `fight_entry_save`）；战后第一个决策（段停在幕开头切点时就是那一帧）关上记录，取这一帧玩家的 HP 与最大生命；对局在战斗中结束时按终局关上。记录开启时还记下这一帧的[入场状态](../../CONTEXT.md)（`entry_state`，`fights.state`）：幕、楼层、本幕层数（地图的行数加一）、HP 与最大生命、腰带里的药（按栏位）与容量（玩家实体的 `capacity`）、牌组计数（`deck` 区的张数，按 `card_type` 数攻击、技能、能力、诅咒、状态，按 `upgraded` 数升级）、遗物数。`run(entry_saves=False)` 只记战况、不写入场存档；记战况的段停在幕开头切点时，另把那一帧的处境作为 `cut_state` 交出。对局记录多一项 `fight_records`；采样进程交出段时把它从对局里拿出、存档压成 gzip（`fight_entry_packing`），不进分片行，也不留在 `Samplers.collect` 返回的摘要里。采集进程的 `WaveWriter` 在段写进分片与索引之后，把每条写成 `<波>/fights.jsonl` 的一行、存档写成 `<波>/fights/<run_id>-<NNN>.json.gz`（`groups.read_save` 可读，`:` 换成 `-`）；没完成的段不写。
  - 一行（`fights.ROW`）：`fight`（段内第几场，`FightTracker` 的编号）、`kind`（`normal`／`elite`／`boss`）、`act`、`floor`、`encounter`、`second_boss`、`from_event`；`entry_hp`、`entry_max_hp`（战斗第一个决策）、`exit_hp`、`exit_max_hp`（战后第一个决策，战后回血遗物已生效）、`last_hp`、`last_max_hp`（战斗最后一个决策）；`potions_used`（策略执行的 USE_POTION 所指药水实体的 `content_id`，如 `POTION.FIRE_POTION`，按使用顺序；控制器步骤与复活不算）；`opening_selection`（这场第一次出牌、用药或结束回合之前，策略在不止一个候选里做过选择：赌博筹码弃牌、烤火手套、开局 N 选 1 一类）；`died`；`decisions`（这场战斗的决策帧数，含单候选与控制器步骤）；`entry_state`；`growth`（[成长奖励](../../CONTEXT.md)的卡牌：`{content_id: 次数}`，`fights.PLAYED` 的遗传算法、镰刀每打出一次计一次，`fights.FATAL` 的血肉盛宴、贪婪之手、狩猎斩杀时计一次，每种不超过入场时牌组里的张数）；`relic_counters`（战后第一个决策里显示计数的遗物：`{content_id: 计数}`，即引擎导出的 `counter`，计数除以触发阈值的余数）；`run_id`、`group`、`character`、`seed`、`ascension`；`path`（入场存档，事件战斗为 null）。
  - 斩杀的认法（`FightLog._judge`）：打出这张牌的下一个决策看玩家：血肉盛宴是最大生命变多，贪婪之手是金币变多，狩猎是它的能力层数变多；狩猎那一击结束了战斗时直接算斩杀（能力随战斗消失）。帧里卡牌的 `ref` 是每帧自己的编号，同一张牌在不同帧里认不出来，所以“每张牌一次”按牌组里的张数封顶。
  - 帧给不出的：引擎的终局帧没有实体，最终 Boss 打赢后整局结束，`exit_hp`、`exit_max_hp` 为 null，`last_hp` 是这场最后一个决策时的值；死在战斗里时 `exit_hp` 为 0，`exit_max_hp` 取最后一个决策的值。引擎只发 `encounter_started`、`encounter_completed`、`run_completed`、`candidate_executed` 四种事件，自己的死亡事件在所有防死效果都放弃之后才发，防死（瓶中精灵、蜥蜴尾巴）不发事件，所以认不出复活，复活只体现在这场结束时的 HP 上。除此之外每场都有 `exit_hp`、`exit_max_hp`。
  - 开关不改段的打法：写存档不动引擎的状态与随机数。在 torch 的确定性算法下，同种子整局 greedy（FP32、每次前向一个决策）开与关的决策摘要逐条相同；不开确定性算法时 GPU 上近乎并列的决策会翻转，同一设置跑两次也一样。
  - 每波 summary 多 `fights`（`records`、按种类的 `kinds`、`died`、`saves`、`save_bytes`、采集进程的 `write_seconds`），轮 summary 多 `fight_entries: true`；关着时两者都没有。续跑时已完成段的行留着，`fights.fight_rows` 读一波的行，同一（段，场次）以后一行为准。
  - 量：段平均约 6.5 场，存档每份约 8 KB；按 v54 一轮约 6,500 段算，约 4 万份、300 多 MB。引擎写一份约 2–8 ms，采样进程压缩每段约 5 ms，采集进程写文件每份约 0.06 ms。迭代驱动在这次迭代的清理步删掉各波的 `fights/` 与 `fights.jsonl`。
- 每个宏动作除采样温度下的 `old_log_prob`（裁剪目标的比值分母）外还记 `base_log_prob`（同一动作在温度 1 下的对数概率之和），`model.choose` 一并返回，只作记录与核对，训练不用。
- 采集进程只保留每条段的摘要，分片行（`member`）交给写入方后不再持有（`Samplers._collect`），否则采集进程的内存随一轮的段数涨。
- 存档的来源：每幕第一个决策是第 0 层的选图，引擎的 `write_continue_save` 在地图房间直接序列化；`resume_save` 在 `load_save` 之上同步玩家状态、标训练可用（契约 `initialization=act-resume-v1`）、不加节点。Boss 入场的存档在战斗第一个决策处写，引擎写的是进这个房间时的状态（与[入场存档](env.md#入场存档)相同）；`resume_fight`（`saves.fight_resume_command`：存档、遭遇、楼层、节点类型）加载后把 Boss 节点记为已访问、标训练可用（契约 `initialization=fight-resume-v1`）并进入该遭遇，回复就是战斗的第一个决策帧。两种续打都把存档里带着的公开历史交回引擎（`saves.resume_fields`），续打的段与写存档的那一局知道同样的掉药概率、问号房间概率和上一个点是否商店；没带的旧存档续打时这些是 unknown。
- 代价：三级链每个种子的段数随到达率成倍增长，n、m 按预算定。

## 战斗组

`main/play/fight_groups.py`，设计见 [ADR 0001](../adr/0001-separate-combat-training.md)：一轮幕组的各波（greedy 战斗、记下战况与入场存档）打完之后，从它们记下的战斗里挑一批，每场从入场存档续打 K 条、截在这场战斗结束（`iterate --fight-groups N`，见 [learn 迭代驱动](learn.md#迭代驱动)）。这一波的优势由迭代驱动的优势步另算（[learn 战斗组的优势](learn.md#战斗组的优势)），更新时与幕组各波同批学（[learn 混合批](learn.md#混合批)）。

- **截在战斗结束**（`RolloutRunner.run(until_fight_end=True)`，任务键同名，`play()` 照传）：只接受从入场存档起的段（`resume` 的 `at` 为 `fight`），不与 `until_act`、`until_boss` 同给，否则报错。第一个决策所属的战斗就是本段的战斗（它必须是这场战斗的第一个决策，否则这条段记协议错误）；之后在第一个不属于这场战斗的决策处停下：战后第一个决策，或另一场战斗的第一个决策——十阶第三幕的两个 Boss 之间可能没有别的决策，所以不只判"不在战斗里"。这个决策不属于本段，不写存档，`outcome` 为 `reached`、`victory` 为 null；对局在战斗中结束时照常是 `win`／`loss`。段头的 `segment` 多 `until_fight_end: true`（不给时没有这一项）。greedy（`sample=False`）同样可用。
- **战况**：这类段总是记下它这场战斗的战况（`fight_records`，一条），不再写入场存档（`FightLog(saves=False)`；同时给 `fight_entries` 也不写）：它续打的就是那份存档。停下之前 `FightLog.passed` 在停下的决策上取战后 HP 与最大生命（`exit_hp`、`exit_max_hp`）。从入场存档进的对局，`FightTracker` 从这场战斗开始计数、没见过本幕第一个 Boss，所以第一场战斗的 `second_boss` 取存档 `fight` 里记的值（所有从入场存档续打的段都这样）。
- **挑选**（`pick`）：候选是这一轮各波 `fights.jsonl` 的全部行（`candidates`，按波、段、场次排序），去掉事件战斗（`path` 为 null）。场数 N 按普通、精英、Boss 3:4:4 分（`shares` 可改，`quotas`：各取整数部分，余下的按余数从大到小逐场补，余数相同按普通、精英、Boss 的顺序；N = 450 时 123／164／163）。每类配额的 `priority` 份额（默认一半，四舍五入取整）按优先级挑：死亡在前，再按战斗代价从高到低；其余从这一类剩下的战斗里均匀不放回地抽。某类不够配额时有多少取多少、记为短缺，不从别类补。代价由调用方给（`cost(row)`，迭代驱动用 `iterate.fight_cost`：[learn 战斗代价](learn.md#战斗代价)的固定药价 `costs.FIXED`），play 不依赖 learn。随机数按 `key`（迭代驱动给 `<seed-root>:<迭代>`）与种类定种：先按它打乱一类的候选，再按优先级稳定排序，所以同优先级的战斗也按这次抽签取，同一轮重挑结果相同。账（`picks`）：候选行数 `fights`、事件战斗数 `events`、挑出数 `picked`，每类的 `fights`（可挑的场数）、`died`、`quota`、`priority`、`uniform`、`short`，以及短缺的类 `short`。`pick(natural=True)`（`sample_fight_groups(natural=)`，`iterate --fight-group-natural`）先把每类的候选按自然权重（`natural_weights`：所在（波，角色，种子）的段数的倒数）不放回地抽到配额的 `THIN`（2）倍，再在留下的里照上面的规则挑；账的 `fights` 仍是这一类的全部候选，`died` 是留下的里的死亡数，账与 plan 另记 `natural: {thin: THIN}`（不开时不记，旧轮次照旧续跑；改了倍数就重挑）。`natural_weights` 与加权抽签 `_drawn` 在这里，战斗示范的挑选（`demo_picks`）从这里导入。
- **外部入场池**（`pool_rows`、`pick_pool`）：池是若干个战斗评估集（`fight_eval` 的格式：`fights.json` 与 `saves/`），每场一行：`run_id` 为 `<集目录名>:<场 id>`、场次 0、`wave` 为 `pool`，存档路径取绝对路径，难度照抄集里的。`sample_fight_groups(pool=, pool_share=)` 从池里取 `round(N × pool_share)` 场：按同样的 `shares` 分配，类内按 id 排序后用 `<key>:pool:<种类>` 不放回地均匀抽，`reason` 为 `pool`、`cost` 为 null；其余 N − 池的场数照常从这一轮挑。账多 `pool`（池的行数、挑出数、每类的 `fights`／`quota`／`picked`／`short`）。
- **任务**（`jobs`）：每场 `size` 条（默认 8；`size` 也可以是 {种类: 条数}，`group_size`），组名 `<来源段>:fight<场次>`，段名是组名加序号；`resume` 是入场存档的路径，难度照抄来源行。
- **采样**（`sample_fight_groups`）：用这一轮 runner 的副本（`RolloutRunner.variant(greedy_fights=False, explore_outside=0.0, explore_prior=0.0)`：同一份模型、词表与权重摘要，只换这几项），战斗决策按检查点的战斗温度抽，留全部帧（布局 `rollout.SEGMENTS`），不探索。做法是换一个 runner，不按任务或布局覆盖：采样进程每次 `collect` 都按传来的 runner 的设置新建 `RolloutRunner`，换 runner 不用改采样进程与任务格式；greedy 与否本来就是"策略在这一波怎么打"的设置，照常记在对局与 summary 里。
- **存储**：轮目录的 `fight-groups/`，与幕组各波并列，格式同一波幕组（`groups.WaveWriter`）：分片加索引、`decisions.jsonl.gz`、`fights.jsonl`（每条段一行战况，`path` 为 null，不建 `fights/`）、`quarantine/`、`summary.json`（`kind=fight_groups_wave`，`format=fight-groups-v1`：`plan`（N、K、优先份额、分叉深度、`key`、温度、探索率、权重摘要）、`picks`、`segments`、`written`、`outcomes`、`failed`、`phases`、`explored`、`forked`、`fights`、`seconds`、`collector` 等）。开打之前先写 `picks.json`（`plan`、账、挑中的每一行，行多 `wave`、`cost`、`reason`）。它不在轮 summary 的 `waves` 里，幕组的优势、更新与份额都不读它（它的优势另算，见 learn）；轮 summary 多 `fight_groups`（这一波的 summary）。
- **续跑**：`fight-groups/summary.json` 的 `plan` 与这次相同就原样沿用；`picks.json` 的 `plan` 相同且有索引时，保留已完成的段、只打剩下的；否则把目录移到轮目录的 `interrupted/` 下重挑重打。
- **续打确定**：同一份存档、动作序列相同的段，战况逐项相同（引擎在动作给定时确定，见 [env 入场存档](env.md#入场存档)）。
- **量**（v54-drop-uniform 第 15 轮，bf16，16 引擎，A0–A5，44 场 × 8 条，3:4:4）：352 段 22 秒，网络决策 11,346 个（每段 32 个），511 决策/秒；每段的决策帧普通约 19、精英约 29、Boss 约 50。分片带全部战斗帧，约 130 KB／段。按同样的配比外推前置测量的规模（450 场 × 8 条）：约 11.6 万个决策，16 引擎约 4 分钟，64 引擎按第三幕续打的吞吐（约 1,000–1,300 决策/秒）约 1.5–2 分钟；分片约 0.5 GB。

## 回溯分叉

`main/play/fight_groups.py` 的 `sample_hindsight`：战斗组打完之后，从它的段里按种类挑一批主干（不论死活），每条按自己的动作从入场存档重放到选定的决策，在那里分成[回溯组](../../CONTEXT.md)，每条臂都抽到这场战斗结束（`iterate --fight-hindsight N`，见 [learn 迭代驱动](learn.md#迭代驱动)）。主干自己不当对照：它的那条路只是组里的一种可能，各臂另打。

- **重放**（`RolloutRunner.run(hindsight=)`，任务键 `hindsight`，`play()` 照传）：`replay` 是主干在有选择的决策上取的候选，按先后排；`key` 是分叉点的帧键（`digests.frame_key`）；`potion` 是要喝的药水 content id，`force` 是要取的候选引用，两者至多一个，都为 null 时是照常由策略选的臂。只接受抽样、截在战斗结束、不是 greedy 战斗的段，否则报错。分叉点之前的决策（有选择的照 `replay` 取，没有选择的取唯一候选）不调策略、不进宏动作，只记进战况（用掉的药照记）；段头的 `hindsight` 记 `potion`、`force` 与重放了几个决策（`replayed`）。重放完后第一个有选择的决策就是分叉点：帧键必须等于 `key`；给了药就只在这瓶药的候选里由策略按温度选（有目标的药每个目标一个候选），记为探索步；给了 `force` 就取它，对数概率是策略自己的，记为分叉步；之后照常抽，不再强制（这一波的 runner 不探索）。候选不在、帧键不符、这瓶药喝不了，或战斗在分叉点之前就结束，都是这条段的协议错误，进 `quarantine/`。依据是续打在动作给定时确定（见[战斗组](#战斗组)的"续打确定"）。
- **分叉点**（`branch_points`）：主干有选择的决策里至多三个，各取一次，只看帧上的信息：
  - `danger`：第一个 HP 低于最大生命 `fights.DANGER`（30%）、能喝药的决策；臂是每种能喝的药各一（排序去重）加 `CONTROLS`（2）条照常选的。
  - `uncertain`：第 2 回合起（玩家实体的 `round`），用段采样时记下的分布（步的 `log_probs`）取前两名不同动作对数概率差最小的决策；不同动作由 `distinct_pair` 定：按概率排序，跳过与第一名等价的候选（`same_action`：动词与目标相同、来源实体除位置类字段外全同，即同一张牌的副本）。臂是两个动作各 `FORCED`（2）条。没记分布的段没有这一类点。
  - `turn`：第 2 回合起各回合的第一个决策里，去掉已被前两类占用的，按 `<key>:<主干>:turn` 随机取一个回合；臂是 `DRAWS`（4）条照常选的。
- **挑选**（`pick_hindsight`）：候选是这一轮战斗组 `fights.jsonl` 的全部段，N 条主干按普通、精英、Boss 3:4:4 分（`shares` 可改，某类为 0 就不分叉这一类；`quotas`），类内按 id 排序后用 `<key>:hindsight` 打乱、取前几条；某类不够就全取，记为短缺。从分片读出它们的动作、帧与分布（`segments_of`，一个分片一读）。每个分叉点一组：组名 `<主干>:<类型>`；段名是组名加 `-p<序号>`（喝药）、`-f<序号>`（强制动作）或 `-c<序号>`（照常选）（`arm_name`）。入场存档取主干所在战斗组的那一行（`picks.json`）。账：主干数、短缺的类、其中死掉的、有分叉点的主干数、分叉点数（按类型、按战斗种类）、段数。
- **存储**：轮目录的 `fight-hindsight/`，格式同战斗组那一波：分片加索引、`decisions.jsonl.gz`、`fights.jsonl`、`quarantine/`、`summary.json`（`kind=fight_hindsight_wave`，`format=fight-hindsight-v2`，`plan` 是 N、分叉点类型、各类臂数、`key`、温度与权重摘要，`shares` 不是默认时另记，`picks` 是账）。开打之前先写 `picks.json`（`plan`、账与全部分叉点，每个带重放的动作、帧键和臂）。轮 summary 多 `fight_hindsight`。续跑规则与战斗组相同。

## 药价拟合的取点与续打

`main/play/potion_value.py`，命令 `fit-prices`（[learn 药价模型](learn.md#药价模型)）。每一步都写进输出目录，再调用时从那里接着做。

- **来源**（`play_sources`）：检查点在这次拟合自己的种子上打完整局（局外按检查点的温度、战斗 greedy，难度按 `round_ascensions` 分在窗口里）。引擎外包一层 `PointEngine`：战斗结束后的第一个选图决策（`phase` 为 `map`）上，腰带里有药就写一个点：`points/<角色>-<种子>-<NNN>.json.gz`，带锚点名、从锚点起执行过的全部候选（`refs`）、点的帧键（`point_key`）、`fights.state` 的处境、角色、种子与难度。锚点在第一幕是开局（`start`），之后是每幕的第一个选图决策，那里写续局存档 `points/anchors/<角色>-<种子>-act<幕>.json.gz`。`source.json` 记来源局数、结局与点数。
- **为什么不在点上存档**：引擎在房间之后的选图处写的续局存档，`resume` 回到上一个地图节点，房间的结果（HP、金币）已经算进去，唯一的候选是走进刚打完的房间，走进去会把它（事件、宝箱、火堆或战斗）再打一遍；第一幕的第一个选图决策根本不能续（`resume_save needs a save written at the map`）。每幕的第一个选图决策写的存档续回同一个决策，只有玩家的战斗字段（`energy`、`max_energy`、`stars`、`round`）变成空，`point_key` 不比这几项（工单 18 的探针：1,623 条从幕锚点重放的分枝都对上了帧键）。幕组切点的存档写在下一幕的第一个决策，也就是这里的幕锚点，不受影响。
- **抽点**（`draw`）：按幕、腰带满否与 HP 段（`BANDS`：< 50%、50–80%、≥ 80%）分层。每幕要 `--pairs` 的三分之一对（一个点的对数是它的持药数），满与不满各半，HP 段按 `BAND_WEIGHTS` 2:1:1 分；某层不够就全取，`points.json` 记每层的要与得。
- **分枝**（`branch_saves`）：每点一份原样（`keep`），每瓶药一份在点上丢掉它的（`drop<i>`，`i` 是它在腰带按栏位排序后的位置），格式同幕切点的存档（`{"at": "act", "save": …}`）：`save` 是锚点的存档（锚点是开局时为空），另带 `potion_point_replay`：`refs`、`key`、`drop`（`[i, 药]` 或空），开局锚点还有 `start`（角色、种子、难度）。
- **重放到点**（`PointEngine.resume`）：存档带 `potion_point_replay` 时，重开这一局或续锚点的存档，逐个执行 `refs`（`waiting` 帧照 runner 的做法发 `advance_to_boundary` 推进，30 秒超时），某个候选不在合法候选里、或到点时帧键不同就报 `ProtocolError`；`drop` 分枝再执行那瓶药的 `DISCARD_POTION`，核对药的 content id。交给 runner 的帧去掉 `events`，像续存档一样从点开始。续打在动作给定时确定，所以重放回到来源对局在点上的同一个状态。
- **续打**（`play_branches`）：全部 greedy（`sample=False`），从存档续打到下一幕的第一个决策，第三幕打到终局。布局 `LAYOUT` 不留帧、不写分片行，只记战况（`fight_entries=True, entry_saves=False`）与停下时的处境（`cut_state`）。每条一行写进 `branches.jsonl.gz`：点、分枝、去掉的药、状态、结局、错误、战况、`cut_state`。前 `--repeat` 个点的原样分枝再打一次（`keep-again`），看合批的近似并列能让同一份存档的两次续打差多少。
- **量**（2026-10-22 的试点，it15，32 引擎，取点方式改之前）：100 局来源约 65 秒、1,516 个点；950 条续打 154 秒。现在每条续打要先从锚点重放到点（第一幕从开局起），会更慢，还没量。

## 战斗示范的挑选

`main/play/demo_picks.py`：BC 阶段每轮从这一轮幕组各波的战况里挑 `count` 场（默认 2,000）交给求解器，求解与导入见 [demos 逐轮的战斗示范](demos.md#逐轮的战斗示范)。

- **配额**：候选是各波 `fights.jsonl` 的全部行（`fight_groups.candidates`），去掉事件战斗；按普通、精英、Boss 3:4:4 分场数（`fight_groups.quotas`），某类不够就记为短缺，不从别类补。
- **自然频率**：幕组让一局的后几幕有更多段（第一幕每组 n 段，每段再续 m 段），第三幕的战斗约占池子七成。一场战斗的权重是它所在的局（角色与种子）在这一波的段数的倒数（`natural_weights`，段数按这一波战况里出现的段计），相当于每局每幕计一次；类内按权重不放回地抽（每行取 u^(1/w) 的键，从大到小）。
- **开局选择优先**：`opening_selection` 为真的战斗（赌博筹码、烤火手套、开局 N 选 1 一类）先按同样的权重抽，最多占 `opening`（默认 20%）的场数，并且不超过各自种类的配额；其余配额再从这一类剩下的战斗里抽。
- **种子**：随机数按 `key`（驱动给 `<seed-root>:<迭代>`）与种类定，同一轮重挑结果相同。
- **输出**：`write_selection` 写 `<轮目录>/demos/selection.json`：`fights` 每项带 `id`（`<段>-<场次>`）、入场存档的绝对路径、角色、种子、难度、种类、幕、层、遭遇、`second_boss`、`opening_selection`、波、权重与挑中原因（`opening`／`natural`），`extra` 为空；另有账（每类可挑场数、开局选择场数、配额、挑中数、短缺、按幕分布）。入场存档在这一轮的清理步删掉，求解要在那之前做完。

## 战斗评估集

`main/play/fight_eval.py`，用途见 [ADR 0001](../adr/0001-separate-combat-training.md)：一批固定的战斗入场存档，策略 greedy 逐场打一遍，比较检查点的战斗打法；求解器在同一批存档上打一遍作对照。代价与报表在 [learn 战斗评估报表](learn.md#战斗评估报表)，命令见 [README](../../README.md#战斗评估集)。

- **抽取**（`build_set`，命令 `fight-set`）：来源是若干个入场存档目录，每份存档一个文件，文件名是战斗 id，存档旁带对局的 `character`、`seed`、`ascension`（`entry_battles` 也读这三项）；幕组轮写的存档（`<波>/fights/`）不带，`entry_fights` 按文件名从同一波的战况（目录旁的 `fights.jsonl`）补，补不上就报错。去掉事件战斗和 `--exclude` 列出的 selection 里 `fights`、`extra` 的战斗，`--count` 场按普通、精英、Boss 3:4:4 分（`fight_groups.quotas`），每类按 id 排序后用 `random.Random("<root>:<种类>")` 不放回均匀抽。来源与 root 相同，抽出的战斗相同；某类不够配额就报错。
- **存放**：`data/<名字>/`，固定数据，不放在会被清理的轮目录里：
  - `saves/<id>.json.gz`：存档的副本；
  - `fights.json`：`format=fight-eval-v1`、root、配额、来源与排除的账，`fights` 是每场的 id、角色、种子、难度、种类、幕、楼层、遭遇、第二 Boss 与存档的相对路径。格式就是 `entry_battles` 的 selection（`extra` 为空），求解器直接读它；
  - `seeds.json`：这些战斗所在对局的种子，按角色列，与评估种子文件同格式。采样必须避开它们：`iterate --exclude-seeds` 加上这个文件，与开发集、最终测试集一样。`fight-eval` 像 `evaluate` 一样核对它们不在检查点的训练种子里；
  - `solver/`：求解器在这批存档上打的结果（`entry_battles run` 的输出目录），`fight-eval` 默认从这里读对照。
  - `solver-demos/`：`solver/` 里赢下的战斗导出（`entry_battles export`）再导入（`import-independent`）成的数据目录，分片加索引，供在这批战斗上算求解器示范的 NLL；它的种子是评估集的，不能进训练数据。
- **现有的集** `data/fight-eval-20261022/`：
  - 来源是前置测量的 3,080 份入场存档（v54-drop-uniform 第 15 轮，A0–A5，战斗 greedy，`round_seeds("probe-fights-20261022", 0, 40)`，五角色各 40 局），在本地 `.scratch/training-plan/assets/probe-20261022/fight-saves/`；
  - 去掉示范试点用过的 220 场（`.scratch/training-plan/assets/demo-pilot/selection.json`），按 root `fight-eval-20261022` 抽 440 场（120／160／160）；
  - 涉及 161 个种子，与开发集、dev-b、最终测试集都不重叠；
  - 试点把带赌博筹码的战斗多抽了 20 场作 `extra`，这些被排除在外，所以集里带赌博筹码的只有 3 场；
  - 引擎改了公开字段或随机数消耗，存档就要重取（[env 入场存档](env.md#入场存档)）。
- **打一遍**（`play_set`，命令 `fight-eval`）：
  - 每场一个段任务（`jobs`）：`resume` 是存档，`until_fight_end`，难度取存档的，段名与组名都是战斗 id；
  - `sample=False`，布局 `rollout.SEGMENTS`，留全部帧、交回分片行；引擎在采样进程里，与 `evaluate` 相同；
  - 采集进程收到一段就解开分片行，取出决策序列、战况与打法，把危急帧的帧压缩存下；整轮打完再给危急帧打分。
- **逐场行**（`fight_row`，字段 `ROW`）：
  - 战斗的标识（`FIELDS`）与 `source`（policy 或 solver）；
  - `status`、`outcome`（reached、win、loss）、`error`；
  - `record`：战况里的 HP 与最大生命、`potions_used`、`opening_selection`、`died`、`decisions`（`fights.ROW` 的一部分）；
  - `behavior`（见下）；
  - `actions`：各决策的 `state_version` 与所选候选 `decoder_slot_ref` 的指纹，用来比较两次打法；
  - `choices`：有选择的决策数。
- **打法**（`behavior`，从决策序列取）：
  - `held`：第一个决策时腰带里的药水；`used`：用掉的药水，按顺序；
  - `setups`：[开局选择](../../CONTEXT.md)，即这场第一次出牌、用药或结束回合之前的多选。每个记来源遗物（`SETUP_SOURCES`：赌博筹码是从手牌选任意张，烤火手套是消耗 1 张，工具箱是从生成的牌里选 1 张；要持有该遗物、选择的形状也对得上，否则为 null）、操作、来源区、上下限、选了几张、选了哪几张；
  - `selections`：其余的多选，记打开它的东西（`trigger`：紧挨着的前一个出的牌或用的药水的 content id；结束回合之后的为 `END_TURN`）、操作、来源区、上下限、选了几张；
  - 多选按路由的 `selection_id` 分界。一个有选择的决策也没有的多选不算：对局本来就不记下没有选择的段，两边按同一规则数；
  - `danger`：[危急帧](../../CONTEXT.md)，即有选择、候选里有 USE_POTION、玩家 HP 低于最大生命 30%（`DANGER`）的决策，记位置、HP 与这一步是否用了药。
- **危急帧打分**（`score_danger`）：整轮打完后，危急帧每 64 个一批交给 `choose_batch`，每帧一个新会话（决策无状态，打分与对局时相同），在温度 1 下取全部候选的概率；USE_POTION 候选的概率之和是 `probability`，温度 T 下的 `tempered` 由这个分布按 p^(1/T) 归一化得到，T 取检查点的战斗温度。批的组成与对局时不同，概率的末位会有差别。
- **求解器的行**（`solver_rows`）：读 `entry_battles run --selection data/<集>/fights.json --output <目录>` 写下的东西：`selection.json`；`runs/<run>.json`，每场的报告，赢的输的都在；`samples/<run>.jsonl.gz`，每场的全部决策行，`metadata.sample_group` 是战斗 id。
  - 打不了的场（`execution_failed`）记 `status` 为 error；
  - 入场 HP 取报告的，入场最大生命取第一个决策的；战后 HP 取报告的 `exit_hp`，输了记 0；
  - 报告没有战后最大生命，用最后一个决策的代替。战后的最大生命变化（少见）因此记不到；
  - `opening_selection` 按策略战况的同一规则从决策行算（`opening_selection`：第一次出牌、用药或结束回合之前，在不止一个候选里做过别的选择）；
  - 只算属于这个集的战斗；求解器只打了其中一部分时，对照只在两边都打完的战斗上算。
- **量**（v54-drop-uniform 第 15 轮，8 引擎）：440 场约 1.4 万个网络决策，FP32 约 63 秒、bf16 约 61 秒；危急帧约 1,200 个。
- **确定性**：同一检查点 FP32 跑两次，440 场里有 12 场动作序列不同、4 场战况不同（没有一场死活不同）；bf16 对 FP32 有 30 场动作不同、12 场战况不同。这些都是多引擎合批下近似并列的决策翻转，续打本身是确定的。

## 评估报告

迭代驱动里的评估用 `rollout.SUMMARY_RUNS`：每局只写去掉帧的对局文件（结果、账本、`climb`），不写流水账，报告与 `compare` 照常可用。命令行的 `evaluate` 按检查点训练时的难度对局（`--ascension` 可改）、按检查点的精度跑网络（`--precision fp32` 关掉全局层的 bf16 autocast，用来降低评估噪声，见[已知的坑](#已知的坑)；`--claim-rewards` 让奖励界面上的金币、遗物和药栏放得下的药水由控制器领取，与示范的假设相同，对局记 `reward_claim_rule`）；每局写 `<角色>-<序号>.json`（每个宏动作的帧、候选、所选动作、`old_log_prob`、阶段、奖励，整局账本、战斗胜负场数、`climb`、策略版本、数值后端、温度）和流水账；汇总时逐局读回，内存里只留一局。报告给每角色与五角色合计（`pooled`）的通关率、各幕通过率（带 Wilson 95% 区间）、`mean_score`、平均到达楼层和止步楼层。

`compare` 读已写出的评估目录：每个目录给整体报告和每角色前 `--head`（默认 60）个种子的子表；两个目录时按（角色，seed）配对，对通关、过第一幕、过第二幕做精确符号检验（`only_first`、`only_second`、双侧 p），只有一方跑过的种子计入 `unpaired`。

## 已知的坑

- 固定种子的 greedy 评估本身有噪声：多引擎合批的 bf16 前向在近似并列的决策上随批次组成翻转，同一检查点重跑结果不同。要降噪用 `--precision fp32`（迭代里 `--eval-precision fp32`）。
- 命令行评估的对局文件带全部帧，很大；汇总不要一次读入全部。统计胜场时 `grep '"victory": true'` 比解析 JSON 快得多。
