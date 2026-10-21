# play：对局循环、采样、幕组与评估

`main/play/`。评估（`evaluate`、迭代里的评估）与幕组采样共用同一个对局循环，写出的东西不同。

| 模块 | 内容 |
| --- | --- |
| `rollout.py` | 对局循环（`RolloutRunner`）、温度与战斗边界、`collect_round` |
| `sampling.py` | 多进程采样（`Samplers`）：引擎与帧解析在采样进程里，采集进程只跑网络 |
| `wire.py` | 打包好的决策的跨进程扁平格式；采样进程的发送记录（`Sent`）与编码表的行 |
| `groups.py` | 幕组：按边界幕把一局切成几波，第一波从开局、之后每波从上一波的幕开头存档续打，各写成分片 |
| `digests.py` | 决策摘要：段交出时每个策略决策里幕组优势要用的东西（帧指纹 `frame_key`、状态 `where`），每波一个文件 |
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

用药探索（`iterate --explore-potions`，`RolloutRunner.explore_potions`，每种战斗一个率，`rollout.exploration_rates` 把一个数、两个数或按种类的映射整理成 normal／elite／boss 三个；设计见 [architecture 用药探索](../architecture.md#用药探索)）：采样进程在战斗里（`FightTracker` 判定，`FightTracker.kind` 给出这场战斗的种类）遇到带 USE_POTION 候选、且候选不少于两个的决策时，按该种类的率抽签决定是否探索；探索的决策把 USE_POTION 候选的槽位作为 `Decisions.explore` 掩码随决策打包过去（`wire` 的 `explore` 字段，与 `legal` 同形），采集进程的 `PolicyNetwork.choose` 只在这些槽位里按 π_T 的相对概率抽，返回的对数概率仍是整个合法分布下该动作的 π_T。对局里探索步带 `explored: true`，交给采集进程的摘要与分片索引行带 `explored` 计数，每波 `summary.json` 汇总为 `explored`。greedy 不探索；探索率 0 时打包多一个全零字段，其余逐位不变。

## 采样对局的格式

幕组的每一波是一个数据目录：与示范导入相同的分片加索引（见 [data](data.md)），目录布局见[幕组](#幕组)。

对局（`source: "rollout"`）记每个策略宏动作的帧、完整候选、所选动作、`temperature`、`old_log_prob`、`base_log_prob`、阶段、奖励，采样时每个有选择的步另记 `log_probs`（采样分布：帧里每个合法候选在这一步温度下的对数概率，按候选顺序、6 位小数；探索步与分叉步也是策略自己的整个分布，`Choice.log_probs`）；段保留全部帧，战斗在内。`combat_frames=False` 时，内存里战斗中的宏动作只留 `phase`、`fight`（第几场）、`temperature`、`branches`、`decisions`、`old_log_prob`、`base_log_prob` 与 `reward`，不留帧；用它的只有 `SUMMARY_RUNS`，而它写文件时连宏动作也不写（见[评估报告](#评估报告)）。每局另记 `status`、`victory`、`ledger`、`climb`、seed、角色、难度、策略版本与摘要（`policy_version`、`policy_digest`）、`vocabulary_hash`、`numeric_backend`、`precision`、`sampling`、温度、`explore_potions`、`fork`、`iteration`、`fights`（胜、负、没打完的场数）。

一局没打完时 `status` 留 `unresolved`：引擎停在 `waiting` 边界超过 `RolloutRunner.timeout`（30 秒）记 `unresolved_wait_timeout`，步数到 `max_steps`（`--max-steps`，默认 10000）记 `unresolved_step_limit`；协议或进程错误时 `status` 为 `error`。

采样进程一局内只保留压缩后的帧，一局结束把对局压成一行交给采集进程追加写出；采集进程不持有对局。不写对局文件的布局收尾时不再把帧解回对象：`rollout.packed_line` 把各帧压缩前的 JSON 文本逐个拼进分片行，读回与 `member(run)` 相同。帧的文本由 orjson 写出、读回（`rollout._pack`／`_unpack`）：与标准库的文本在浮点写法、非 ASCII 是否转义上不同，读出的值相同；读取进程读分片也用 orjson。原因：采样进程会一直占着它打过的最大一段的内存峰值，把帧解回对象会让这个峰值大很多。

`packed_line` 为数各阶段的策略宏动作本来就要把每个宏动作第一步的帧解码一次，交出段的同时（`sampling._shard_line`）就在这份解码上记下这一段的**决策摘要**（`digests.py`）：每个策略宏动作一行（`digests.FIELDS`：宏动作序号、帧指纹 `frame_key`、所选候选、幕、层、HP、最大 HP、金币、卡组／遗物／药水数、是否战斗中、是否探索步／分叉步），连同 run_id、组、角色、结果、难度压成一个 gzip 成员随段交出。状态只记原始数，特征向量由优势那边算，换特征不必重采。帧里没有带幕与层的玩家实体时这一段不记摘要（`digest` 为 None），优势那边读分片、在那里报错。每决策多花约 0.35 ms（指纹 0.30），一轮约 440 CPU·秒，摊在几十个采样进程里。交出的摘要另带 `distributions`：这一段每个有选择的步是否都记了采样分布。

## 幕组

`main/play/groups.py`，设计见 [architecture](../architecture.md#训练路线)：

- `RolloutRunner.run` 多了段的参数：`until_act` 让对局在该幕第一个决策处停下，`until_boss` 让它在该幕第一场 Boss 战的第一个决策处停下（二者只给一个）；这一帧不属于本段，引擎在那里写的续局存档随记录返回（`save`：`{"at": "act", "save": …}` 或 `{"at": "fight", "save": …, "fight": …}`，`fight` 是 `FightTracker` 对这场战斗的描述加楼层），`outcome` 为 `reached`，另记 `reached_act` 或 `reached_boss`；`resume` 给这样一份存档，对局从它续打：幕开头的经 `CliEngine.resume`（引擎 `resume_save`）回到地图，Boss 入场的经 `CliEngine.resume_fight`（引擎 `resume_fight`）重进那场战斗，`character`、`seed` 沿用存档那一局。终局时 `outcome` 为 `win`／`loss`。`run_id`、`group` 由调用方命名。`validate_run` 接受 `outcome=reached`、`victory=None` 的段。
- 波：切点（`cuts(boundaries, bosses)`：边界幕的幕开头，默认第二、第三幕；`bosses` 幕的 Boss 入场，Boss k 排在第 k+1 幕开头之前）把一局切成几波，每波按起点命名（`waves`：`act1`、`act2`、`act3`，Boss 入场起的 `boss3`）。第一波的任务是每个（角色，种子）`n` 条，打到第一个切点；之后每波是上一波每份存档 `m` 条（`m` 一个数或每波一个），打到下一个切点，最后一波打到终局。到达切点的段各写一份存档到 `<波>/saves/<run_id>.json.gz`，行记在 `<波>/saves.jsonl`（`at` 为 `act` 或 `fight`，`act`／`boss` 记到达的幕）；续打的任务只带存档路径，采样进程自己读（`play(..., segment=True)`，布局 `rollout.SEGMENTS`）。组名 `<角色>:<种子>:act1` 与 `<父段>:<波>`，段名是组名加序号，所以从段名能一路回溯到开局。
- 分叉（`sample_groups(fork_depth=D)`）：每个任务带 `fork`（段在组里的序号、组大小、深度），`RolloutRunner.run(fork=)` 让采样的段在前 D 个有选择的决策上不抽样，按 `rollout.fork_ranks` 取名次：序号 j 的段在第 d 步取 ⌊j·2^(d+1)/组大小⌋ mod 2，0 是策略最可能的候选、1 是第二可能的，深度截到 log₂ 组大小。名次随决策打包过去（`Decisions.rank`，`wire` 的 `rank` 字段，每决策一个数，−1 为抽样），`PolicyNetwork.choose` 按合法槽位的对数概率降序取该名次的槽位，返回的对数概率仍是整个分布下的 π_T；单进程路径 `choose_batch(rank=)` 相同。分叉的决策不再做用药探索。步记 `forked: true`，对局头记 `fork`，摘要与索引行带 `forked` 计数，每波 summary 汇总为 `forked`；greedy 不分叉。
- 难度（`sample_groups(ascensions=)`，角色 → 种子 → 难度，`iterate.round_ascensions` 给出）：第一波的任务带 `ascension`，`RolloutRunner.run(ascension=)` 按它开局、核对帧契约的 `fixed_ascension` 并记进对局（不给时用 runner 自己的）；续局从存档进，引擎沿用存档的难度，与任务所带的不符时这条段报错。存档行带 `ascension`，`resume_jobs` 照抄，所以同一组、同一条链同难度。轮目录的 `metadata/ascensions.json` 记分配，轮 summary 的 `ascensions` 指向它；每波 summary 的 `by_ascension` 按难度记写出的段数与结果。每局的难度本来就在对局记录与索引行里（`RunShards.at_ascension` 按难度筛）。
- 存储：每波一个分片加索引加 `summary.json` 的数据目录，段保留全部帧（含战斗），索引行多了 `group`、`outcome`、`distributions`。`WaveWriter` 每写完一段的分片行与索引行，就把它的决策摘要追加进这一波的 `decisions.jsonl.gz`（各段一个 gzip 成员，连起来读是一个 gzip 文件；一轮约 50 MB），续跑时接着追加，同一段出现两次以后一行为准；先写分片与索引，摘要里不会有索引没有的段。每波 summary 与轮 summary 的 `distributions` 说这一波（全轮）的每段是否都记了采样分布。没完成的段写到 `quarantine/` 并计数，不让整轮失败（组变小）。轮的 `summary.json`（`kind=act_groups`，`format=act-groups-v2`）记切点（`boundaries`、`bosses`、`cuts`）、波的顺序、各波的段数、结果、各阶段宏动作数、写出的存档数、耗时；每波的 summary（`kind=act_groups_wave`，同一 `format`）记它的切点（`until`、`until_act`、`until_boss`）、任务数（`segments`）、写出数（`written`）与按原因计的失败数（`failed`）。
- 续跑：`sample_groups(..., resume=True)` 让被打断的轮从所在的波继续：有 `summary.json`、且其 `segments` 等于这一波的任务数、`written` 加失败数也等于任务数的波原样沿用，有索引没有 summary 的波保留已完成的段（`ShardWriter(resume=True)` 在索引后追加、新分片接着编号，契约从已有的段里读回）只打剩下的任务，没有索引的波移到轮目录的 `interrupted/` 下重打；波的 summary 多一项 `resumed`。
- 每个宏动作除采样温度下的 `old_log_prob`（裁剪目标的比值分母）外还记 `base_log_prob`（同一动作在温度 1 下的对数概率之和），`model.choose` 一并返回，只作记录与核对，训练不用。
- 采集进程只保留每条段的摘要，分片行（`member`）交给写入方后不再持有（`Samplers._collect`），否则采集进程的内存随一轮的段数涨。
- 存档的来源：每幕第一个决策是第 0 层的选图，引擎的 `write_continue_save` 在地图房间直接序列化；`resume_save` 在 `load_save` 之上同步玩家状态、标训练可用（契约 `initialization=act-resume-v1`）、不加节点。Boss 入场的存档在战斗第一个决策处写，引擎回滚到进房之前的状态（与[入场存档](env.md#入场存档)相同）；`resume_fight`（`saves.fight_resume_command`：存档、遭遇、楼层、节点类型）加载后把 Boss 节点记为已访问、标训练可用（契约 `initialization=fight-resume-v1`）并进入该遭遇，回复就是战斗的第一个决策帧。重进的战斗没有公开历史前缀，与幕开头续打一样。
- 代价：三级链每个种子的段数随到达率成倍增长，n、m 按预算定。

## 评估报告

迭代驱动里的评估用 `rollout.SUMMARY_RUNS`：每局只写去掉帧的对局文件（结果、账本、`climb`），不写流水账，报告与 `compare` 照常可用。命令行的 `evaluate` 按检查点训练时的难度对局（`--ascension` 可改）、按检查点的精度跑网络（`--precision fp32` 关掉全局层的 bf16 autocast，用来降低评估噪声，见[已知的坑](#已知的坑)；`--claim-rewards` 让奖励界面上的金币、遗物和药栏放得下的药水由控制器领取，与示范的假设相同，对局记 `reward_claim_rule`）；每局写 `<角色>-<序号>.json`（每个宏动作的帧、候选、所选动作、`old_log_prob`、阶段、奖励，整局账本、战斗胜负场数、`climb`、策略版本、数值后端、温度）和流水账；汇总时逐局读回，内存里只留一局。报告给每角色与五角色合计（`pooled`）的通关率、各幕通过率（带 Wilson 95% 区间）、`mean_score`、平均到达楼层和止步楼层。

`compare` 读已写出的评估目录：每个目录给整体报告和每角色前 `--head`（默认 60）个种子的子表；两个目录时按（角色，seed）配对，对通关、过第一幕、过第二幕做精确符号检验（`only_first`、`only_second`、双侧 p），只有一方跑过的种子计入 `unpaired`。

## 已知的坑

- 固定种子的 greedy 评估本身有噪声：多引擎合批的 bf16 前向在近似并列的决策上随批次组成翻转，同一检查点重跑结果不同。要降噪用 `--precision fp32`（迭代里 `--eval-precision fp32`）。
- 命令行评估的对局文件带全部帧，很大；汇总不要一次读入全部。统计胜场时 `grep '"victory": true'` 比解析 JSON 快得多。
