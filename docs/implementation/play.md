# play：对局循环、采样、幕组与评估

`main/play/`。评估（`evaluate`、迭代里的评估）与幕组采样共用同一个对局循环，写出的东西不同。

| 模块 | 内容 |
| --- | --- |
| `rollout.py` | 对局循环（`RolloutRunner`）、温度与战斗边界、`collect_round` |
| `sampling.py` | 多进程采样（`Samplers`）：引擎与帧解析在采样进程里，采集进程只跑网络 |
| `wire.py` | 打包好的决策的跨进程扁平格式 |
| `groups.py` | 幕组：按边界幕把一局切成几波，第一波从开局、之后每波从上一波的幕开头存档续打，各写成分片 |
| `evaluation.py` | 评估报告、配对符号检验（`compare_evaluations`）、留出 NLL 的表格 |
| `testing.py` | 合成引擎（`SyntheticEngine`，不是游戏模拟器）与合成示范，供测试和 `smoke` |

## 采样进程分工

吞吐的上限来自 Python 解释器，不来自 GPU 或引擎：一个决策在两次前向之间要花几毫秒 Python（解析引擎回复、写日志、帧变成词元、拼成张量），放在一个进程里与跑网络的线程共用一把解释器锁。所以引擎和这些工作放进采样进程，采集进程只把各采样进程在上一次前向期间打包好的决策合成一批、一次前向整批采样，只把（槽位、对数概率）发回去；每个决策带着自己的温度。采样进程由采集进程 fork 出来，共享词表、不接触 CUDA。

- 一次前向的耗时几乎不随批大小变化（宽度 256 的网络时间花在解释器上），吞吐取决于每次前向带多少决策；一轮的耗时不会短于最长一局的步数乘单步往返时间。
- 每个采样进程只带一个引擎时引擎之间互不等待：`--processes` 默认等于 `--workers`。
- 一轮的对局数（每角色种子数的五倍）不宜少于引擎数，多几倍更省。
- 采集进程是唯一串行的环节，消息与张量的固定开销会超过前向本身，所以跨进程用 `wire.py` 的扁平格式：去掉填充，一条消息只有表头、一个整数数组和一个数值数组，任意多条消息在设备上用固定次数的数组运算合并。
- 采样进程连同引擎跨轮保留；每打完一局清掉解析缓存，把空闲内存还给系统。每对“引擎 + 采样进程”占 250–300 MB（引擎约 150–200 MB；改 .NET 的 GC 参数不会更低）。
- 采集进程按轮记自己的账（`Samplers.timing`，幕组每波 `summary.json` 的 `collector`、`iterate` 的 `sample` 记录）：`passes`（前向次数）、`decisions`、`decisions_per_pass`、`decisions_per_second`、`seconds_per_pass`，以及等采样进程（`waiting_seconds`）、解包（`unpacking_seconds`）、选择（`choosing_seconds`：前向加把结果拷回主机）、回传（`sending_seconds`）各花的秒数。读法：等待占比小、每次前向的决策数远少于引擎数，说明引擎在等采集进程，加引擎能提吞吐；每次前向耗时随决策数线性涨，说明前向本身是瓶颈。20M 策略一次前向约 27 ms 的固定开销与决策数无关，32 个引擎时每次只带约 14 条决策，64 个引擎约 24 条、吞吐高四成。
- `Samplers.rest()` 结束引擎、留下进程，下一次采样按需重启引擎。迭代驱动用它在训练期间腾出引擎的内存，同时保住训练之前 fork 的进程（见 [learn](learn.md#迭代驱动)）。
- 采样进程要在父进程训练之前 fork：训练过的父进程堆里布满空闲块，子进程新分配的对象落进继承来的页，每写一页就复制一页。32 个采样进程的私有内存因此从约 1.5 GB 涨到约 7 GB。

## 温度与战斗边界

一场战斗从引擎的 `encounter_started` 到它打赢后的第一个决策之前；其中的出牌和战斗中的选牌（`card_select`：弃牌、消耗）按 `combat_temperature`，其余决策按 `other_temperature`（默认 0.3／0.5，记在检查点和对局里；BC 检查点记的是训练时 profile 的值，采样时用 `--config` 换成当前 profile 的）。greedy 的 `evaluate` 记 `temperature: null`，概率按温度 1 记。

## 采样对局的格式

幕组的每一波是一个数据目录：与示范导入相同的分片加索引（见 [data](data.md)），目录布局见[幕组](#幕组)。

对局（`source: "rollout"`）记每个策略宏动作的帧、完整候选、所选动作、`old_log_prob`、阶段、奖励；段保留全部帧，战斗在内。不保留战斗帧的布局（`combat_frames=False`，迭代里的评估）里，战斗中的宏动作只记 `phase`、`fight`（第几场）、`branches`、`decisions`、`old_log_prob` 与 `reward`，`samples()` 读到它们时不作训练样本。每局另记 `status`、`victory`、`ledger`、`climb`、seed、角色、难度、策略版本、温度、`fights`（胜、负、没打完的场数）。

采样进程一局内只保留压缩后的帧，一局结束把对局压成一行交给采集进程追加写出；采集进程不持有对局。不写对局文件的布局收尾时不再把帧解回对象：`rollout.packed_line` 把各帧压缩前的 JSON 文本逐个拼进分片行，读回与 `member(run)` 相同，一段的内存峰值从约 40–60 MB 降到约 1–2 MB（采样进程会一直占着它打过的最大一段的峰值）。

## 幕组

`main/play/groups.py`，设计见 [architecture](../architecture.md#训练路线)：

- `RolloutRunner.run` 多了段的参数：`until_act` 让对局在该幕第一个决策处停下，`until_boss` 让它在该幕第一场 Boss 战的第一个决策处停下（二者只给一个）；这一帧不属于本段，引擎在那里写的续局存档随记录返回（`save`：`{"at": "act", "save": …}` 或 `{"at": "fight", "save": …, "fight": …}`，`fight` 是 `FightTracker` 对这场战斗的描述加楼层），`outcome` 为 `reached`，另记 `reached_act` 或 `reached_boss`；`resume` 给这样一份存档，对局从它续打：幕开头的经 `CliEngine.resume`（引擎 `resume_save`）回到地图，Boss 入场的经 `CliEngine.resume_fight`（引擎 `resume_fight`）重进那场战斗，`character`、`seed` 沿用存档那一局。终局时 `outcome` 为 `win`／`loss`。`run_id`、`group` 由调用方命名。`validate_run` 接受 `outcome=reached`、`victory=None` 的段。
- 波：切点（`cuts(boundaries, bosses)`：边界幕的幕开头，默认第二、第三幕；`bosses` 幕的 Boss 入场，Boss k 排在第 k+1 幕开头之前）把一局切成几波，每波按起点命名（`waves`：`act1`、`act2`、`act3`，Boss 入场起的 `boss3`）。第一波的任务是每个（角色，种子）`n` 条，打到第一个切点；之后每波是上一波每份存档 `m` 条（`m` 一个数或每波一个），打到下一个切点，最后一波打到终局。到达切点的段各写一份存档到 `<波>/saves/<run_id>.json.gz`，行记在 `<波>/saves.jsonl`（`at` 为 `act` 或 `fight`，`act`／`boss` 记到达的幕）；续打的任务只带存档路径，采样进程自己读（`play(..., segment=True)`，布局 `rollout.SEGMENTS`）。组名 `<角色>:<种子>:act1` 与 `<父段>:<波>`，段名是组名加序号，所以从段名能一路回溯到开局。
- 存储：每波一个分片加索引加 `summary.json` 的数据目录，段保留全部帧（含战斗），索引行多了 `group`、`outcome`。没完成的段写到 `quarantine/` 并计数，不让整轮失败（组变小）。轮的 `summary.json`（`kind=act_groups`，`format=act-groups-v2`）记切点（`boundaries`、`bosses`、`cuts`）、波的顺序、各波的段数、结果、各阶段宏动作数、写出的存档数、耗时；每波的 summary 记它的切点（`until`、`until_act`、`until_boss`）。
- 续跑：`sample_groups(..., resume=True)` 让被打断的轮从所在的波继续：有 `summary.json` 的波原样沿用，有索引没有 summary 的波保留已完成的段（`ShardWriter(resume=True)` 在索引后追加、新分片接着编号，契约从已有的段里读回）只打剩下的任务，没有索引的波移到轮目录的 `interrupted/` 下重打；波的 summary 多一项 `resumed`。
- 每个宏动作除采样温度下的 `old_log_prob`（裁剪目标的比值分母）外还记 `base_log_prob`（同一动作在温度 1 下的对数概率之和），`model.choose` 一并返回，只作记录与核对，训练不用。
- 采集进程只保留每条段的摘要，分片行（`member`）交给写入方后不再持有：一轮上万条段时这是十几 GB 的差别（`Samplers._collect`）。
- 存档的来源：每幕第一个决策是第 0 层的选图，引擎的 `write_continue_save` 在地图房间直接序列化；`resume_save` 在 `load_save` 之上同步玩家状态、标训练可用（契约 `initialization=act-resume-v1`）、不加节点。Boss 入场的存档在战斗第一个决策处写，引擎回滚到进房之前的状态（与[入场存档](env.md#入场存档)相同）；`resume_fight`（`saves.fight_resume_command`：存档、遭遇、楼层、节点类型）加载后把 Boss 节点记为已访问、标训练可用（契约 `initialization=fight-resume-v1`）并进入该遭遇，回复就是战斗的第一个决策帧。重进的战斗没有公开历史前缀，与幕开头续打一样。
- 代价（策略 v12，32 引擎）：每段约 0.9 MB、约 0.6 秒引擎时间；两级链 100 种子 × n=8、m=8 共 2,176 段 21 分钟、2.0 GB。三级链每个种子的段数随到达率成倍增长，n、m 按预算定。

## 评估报告

迭代驱动里的评估用 `rollout.SUMMARY_RUNS`：每局只写去掉帧的对局文件（结果、账本、`climb`），不写流水账，报告与 `compare` 照常可用。命令行的 `evaluate` 按检查点训练时的难度对局（`--ascension` 可改）；每局写 `<角色>-<序号>.json`（每个宏动作的帧、候选、所选动作、`old_log_prob`、阶段、奖励，整局账本、战斗胜负场数、`climb`、策略版本、数值后端、温度）和流水账；汇总时逐局读回，内存里只留一局。报告给每角色与五角色合计（`pooled`）的通关率、各幕通过率（带 Wilson 95% 区间）、`mean_score`、平均到达楼层和止步楼层。

`compare` 读已写出的评估目录：每个目录给整体报告和每角色前 `--head`（默认 60）个种子的子表；两个目录时按（角色，seed）配对，对通关、过第一幕、过第二幕做精确符号检验（`only_first`、`only_second`、双侧 p），只有一方跑过的种子计入 `unpaired`。

## 已知的坑

- 固定种子的 greedy 评估本身有噪声：多引擎合批的前向在近似并列的决策上随批次组成翻转，同一检查点重跑相差几局。
- 一局评估文件约 30 MB；汇总不要一次读入全部。统计胜场时 `grep '"victory": true'` 比解析 JSON 快得多。
