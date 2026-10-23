# learn 与迭代驱动：数据供给、训练器、幕组优势、幕组批、裁剪目标、混合批、战斗代价、战斗组优势与战斗评估报表

`main/learn/` 与 `main/iterate.py`。设计见 [architecture](../architecture.md#训练路线)。

| 模块 | 内容 |
| --- | --- |
| `loader.py` | 读取进程与打包进程：分片窗口 → 打包好的逻辑批 |
| `feeder_server.py` | 只由读取进程服务预载：起读取与打包进程共用的那一个共享内存管理进程 |
| `trainer.py` | `Learner`：唯一的训练器，对宏动作做加权 NLL（或调用方给的目标）；BC 的轮次与留出；`Learner.load`／`save` 在检查点与训练器之间转换（权重、优化器、日程、`progress` 元数据），`cli` 与 `iterate` 都经它读写检查点 |
| `advantage.py` | 优势文件 `advantages.json.gz` 的读写 |
| `features.py` | 决策的状态摘要特征（幕、楼层、幕×楼层、角色、HP 比例、最大 HP、金币、卡组、遗物、药水、是否战斗中），组基线回归用；从帧里读这些数的 `where` 在 `play/digests.py` |
| `segments.py` | 幕组的段与回报，不属于哪一种训练：读各波的段（`read_waves`）、回报沿链回传（`returns`：最后一波 ±1、之前各波取下一波从它存档续打的组均值）、共同前缀的分叉树与节点均值（`tree`）；组内去均值的岭回归（`within` 给去均值的行，`fit` 解出权重），GRPO 的基线用它 |
| `streams.py` | 采样数据的流，不属于哪一种训练：份额上限后的组权重或等权（`buffer_weights`）、有优势的对局（`priced`）、一路的批（`batches`）与读取进程数（`stream_feeders`）、几路合成一批（`sizes`、`Joint`） |
| `clipped.py` | 裁剪目标，不属于哪一种训练：温度化策略的裁剪比值加对参考策略的 KL（`Clipped`；`Reference` 是载入的参考策略连同它的词表摘要，或采样时记下的分布；`with_advantages` 给一路的样本挂上优势） |
| `groups.py` | 幕组的优势（GRPO）：分叉树上的动作级优势；分叉后的组内岭回归基线（`segments.fit`）；可选剔掉全组同结果的组 |
| `batches.py` | 幕组批（GRPO）：各波一路，各波的份额与一遍的更新次数；混合批里各类来源的份额（`stream_shares`） |
| `grpo.py` | 幕组的更新（GRPO）：一遍各波的流按份额合成的批，按裁剪目标更新（`group_update`）；同一批里可以另放战斗组与战斗示范（[混合批](#混合批)） |
| `mixed.py` | 混合批的目标：按样本所在的流分派，采样样本（幕组、战斗组）走裁剪目标加 KL，示范走 NLL（`Likelihood`），按来源分开计数（`Mixed`） |
| `combat_demos.py` | 战斗示范作为更新的一路：示范目录只取战斗（`fights`），按一遍的配额整场重复或抽一部分（`draw`），等权的流（`batches`） |
| `costs.py` | 战斗代价：由一场战斗的战况按求解器的记账算代价（`cost`）与回报（`fight_return`）；战斗的 HP 去向（`relief`：幕末回血、整局最后一场或都不是）；可替换的价格对象（`Prices`，固定价 `FIXED`） |
| `fight_groups.py` | 战斗组的优势：段的回报按传入的价格由战况现算，组内均值作基线，共享前缀上用分叉树给动作级优势；可选剔掉全组同结果的组；头里按战斗种类记组内诊断 |
| `fight_report.py` | 战斗评估的报表：由战斗评估集的逐场行按价格现算代价，分种类的平均代价与死亡率、用药、危急帧（`summary`），与求解器逐场的差距（`gap`），行为差异表（`behavior`），同一个集的两次打法逐场比较（`paired_plays`） |
| `history.py` | 逐更新遥测（`updates.jsonl`）、逐轮汇总（`history.jsonl`）、状态（`status.json`） |
| `../iterate.py` | 迭代驱动：采样 → 优势 → 更新 → 评估 → 清理，可断点续跑 |

## 数据供给

数据量远超内存：同时在内存里的只有几个分片（`--window-shards`），打乱分片顺序和一起读入的样本。`loader_feeders` 个读取进程各持窗口的一部分，各带 `loader_workers` 个 fork 出的打包进程（20M profile 4 × 6，200M 2 × 2）：一个读取进程解码分片时它的打包进程无事可做，读取进程多于两个才能让各进程的批交替到达。

- 一个打包进程一次准备一个逻辑批：每帧只解析一次；超出预算时按长度从短到长拆成几份，梯度累加后更新一次，所以拆分与进程数都不改变一次更新用到的样本。
- 读取进程不从训练进程 fork，而从**读取进程服务**（multiprocessing 的 `forkserver`，`loader._feeders`）fork：它是第一次 `stream()` 时另起的一个干净进程，只导入了 `main.learn.loader` 与 `main.learn.feeder_server`（不能预载默认的 `__main__`：`python -m main` 一导入就跑命令），参数经 pickle 传过去，读取进程自己设共享策略。原因：从训练进程 fork 的子进程，新分配的对象先落进训练进程半空的内存池，写一下就复制一整页；训练进程的堆被每次迭代的短命大对象（优势步的段、各轮的表）弄得越来越碎，五十来个读取与打包进程复制得越来越多，更新期内存随迭代上涨。冻结 GC 只挡住回收写到的页，挡不住这一条。读取进程的父进程是服务进程：训练进程退出时服务进程读到管道尽头随之退出，读取进程按 `end_with` 跟着结束整组。坑：服务进程 fork 出的每个进程、以及从训练进程 fork 出的进程，都会拿到这根管道写端的副本，有一个活着服务进程就不退出，所以读取进程一开始就关掉自己那份（`loader._forget_server`）；从已有服务进程的进程 fork 出来的进程（例如测试里 fork 的训练进程）不能沿用父进程的服务进程（`waitpid` 等不到别人的子进程），`_feeders` 会丢掉它、另起一个。读取进程 fork 打包进程前冻结窗口（`gc.freeze`），否则打包进程里的一次回收会写遍继承来的样本、把它们复制一份；窗口用完再解冻。
- 读取进程 fork 时不能有自己的线程（线程持有的锁在子进程里永不释放）：张量按命名共享内存文件传递（torch `file_system` 策略），每个读取进程用自己的管道发消息（`multiprocessing.Queue` 会起写线程）。读取进程与打包进程组成一个进程组，训练进程提前停止时整组结束；强杀残留的共享文件由 `torch_shm_manager` 回收。整次运行只有一个管理进程，由服务进程起（预载 `feeder_server`：设共享策略、建一个 1 元素共享张量并一直持有）。坑：`file_system` 共享下，进程第一次建共享张量时如果还不知道任何管理进程，就自己另起一个；fork 出的子进程沿用父进程已知的那个；收到共享张量的进程会连上张量所指的管理进程，而且一直连到自己退出。服务进程什么都没建过时，每轮每个读取进程的第一批打包进程都各起一个管理进程，训练进程一直连着，它们就一个也不退出：每轮约 33 个，每个约 58 MB 私有内存（它链接 libtorch），在训练进程树之外，`memory_by_kind` 看不到。数管理进程要按命令名（`ps -eo comm`，显示为 `torch_shm_manag`），`pgrep -f` 会匹配到命令行里带这个词的 shell。
- 被结束的生产进程留下的共享文件要由消费方删：一遍按更新步数停下时读取进程与打包进程被整组结束，它们手里的批（打包进程预取的、管道里没读的）没人再释放，命名共享文件按创建进程的 pid 命名，被杀的进程永远不会“放手”，文件会留到训练进程退出，逐遍累积。`stream()` 结束时删掉这些进程（读取进程组成员，加读取进程报上来的历次打包进程 pid）的文件，但跳过自己仍映射着的——共享机制在释放张量时若发现文件名已没了会中止进程——这些留到下一次 `stream()` 开头再删（`_release_shared`、`_ENDED`）。
- 每个读取进程把窗口末尾不足一批的样本带入下个窗口；结束时主进程收各读取进程至多 B−1 个样本，合并后就地打包。每轮仅最后一批可能不足 B，更新次数为 ceil(N/B)。
- 打包卡住：一个逻辑批几秒就打完，等下一批超过 `STALL_SECONDS`（180 秒）就认定它不会来，从还没交付的那一批起重新起打包进程（标准错误打印一行）。训练进程超过 `FEEDER_SILENCE_SECONDS`（60 秒）收不到读取进程的消息时检查它们是否还活着，有死掉的就报错，不无限等。

## 训练器

`Learner` 只按样本权重求和，权重由供给侧给出（示范的权重规则见 [data](data.md#权重与损失尺度)，幕组的见下）。BC（`bootstrap`）默认持续到手动中断，`--epochs N` 是本次追加的轮数；不设验证、不按表现降学习率或早停。每轮记录 `held_out`、`training`、`samples` 三组计数（留出组没进训练的证据）。正确率是更新前同一次前向里合法动作 Top-1 与标签一致的决策数除以策略决策数，不用样本权重。

`fit(objective=)` 换掉每个宏动作的损失：`objective(learner, prepared, micro)` 返回这一微批每个宏动作的损失、温度 1 下的熵和正确率计数，训练器照旧按样本权重求和、按 `total / (B·k)` 缩放（k 为 `accumulate`，默认 1）、裁剪梯度。

## 幕组的优势

`groups.py`（`advantages(目录)` 给轮里每一波写一份 `advantages.json.gz`：`format=rollout-advantages-v1`、`kind=act-groups`，头里记 `wave`、`ridge`、切点（`boundaries`、`bosses`）、`policy_version`、各波写出的段数（`written`）、`runs`、`decisions`、优势的摘要（`advantage`：个数、均值、标准差、范围）与 `stats`，剔掉全组同结果的组时另记 `drop_uniform_groups: true`，正文是每段各决策的（宏动作序号、基线、优势）；每一波都已有头、且 kind、ridge、切点、`policy_version`、各波 `written` 与是否剔除全部相同时直接读回头，不重算）：

- 回报（`segments.returns`）：终局通关 +1、死亡 −1；到达边界幕的段取下一波从它那份存档续打的组的回报均值，一波波往前传（第一波的回报是第二波组均值的均值）；下一波没有一条完成的段不给优势，训练时跳过。
- 读各段（`segments.read_waves`）：每个决策要的是（序号、帧指纹、动作、状态摘要、是否探索步、是否分叉步、是否在战斗中），每段另记不留帧的宏动作数（`Segment.frameless`）。一波的决策摘要（`decisions.jsonl.gz`，采样进程交出段时记下，见 [play](play.md#采样对局的格式)）覆盖索引里的每一段时只读它，逐行转成段、同一段以后一行为准，段按索引的（分片，行）排，与读分片的顺序相同；v35 round-0（126 万决策）整步约 11 s，主进程峰值约 2 GB。摘要缺段的波（摘要出现之前采的轮、续跑时被打断在分片与摘要之间、帧读不出位置）整波退回读分片：要把帧全部解码一遍（一轮约 4 GB gzip，约 1,000 CPU·秒，时间几乎都在 `json` 解码与帧指纹上），各波的分片进同一个进程池，每个分片一个任务、只带该分片的索引行，`processes` 个进程（默认 CPU 数的一半，即物理核数：再多只有超线程的收益）按摘要的同一写法（`digests.decision`）算出决策；分片逐局读（`RunShards.runs`），一个进程手里只有一局解码后的帧；fork 前先 `gc.collect()` 加 `gc.freeze()`，否则进程里的回收会把驱动的堆页各复制一份；这些进程只活这一步，不像读取进程那样从服务进程起（见[数据供给](#数据供给)）。同一轮读分片 16 个进程约 80 s、32 个约 64 s。
- 分叉树（`segments.tree`，给出每个节点的均值与动作级优势）：同一组的段按“到第 k 步为止帧相同”归并，帧的同一性是 `public` 与 `legal` 的指纹。到达者 ≥ 2 的节点上，决策的优势 = 选同一动作者的回报均值 − 节点全部到达者的均值；只有一种动作的节点记 0。动作是宏动作各步所选候选的元组。
- 分叉之后：优势 = (R − 组内均值) − βᵀ(φ − 组内摘要均值)，β 按岭回归（`ridge` 默认 1.0）在组内去均值的数据上拟合，每波分开。文件里的 `value` 就是这个基线（树节点上是节点均值）。
- 剔除（`advantages(…, drop_uniform=True)`，`iterate --drop-uniform-groups`）：[全组同结果的组](../architecture.md#幕组的回报与优势)（`uniform`：组内各段的 `table[run_id]` 全相等）的段不写进正文，与没有回报的段一样被 `batches.priced` 略过；分叉树与岭回归只用保留的组，所以保留的组的优势与“这一波只有它们”时相同。下游不用改：分片按索引行读，略过的行不解析（但仍要解压），`seen`、份额、`pass_updates` 都按头里的 `decisions` 算。一波的组全被剔掉时这一波没有有优势的决策，更新报错（`priced_waves`）。
- greedy 战斗的轮（`iterate --greedy-fights`，见 [play 温度与战斗边界](play.md#温度与战斗边界)）：战斗不留帧，不是策略宏动作，所以段的决策、分叉树、岭回归与优势都只有战斗外的决策，优势表里没有战斗决策。更新同样不用改：`batches.priced` 的 `RunShards` 按索引行的 `phases` 数宏动作，`data.samples` 只取有帧的宏动作，战斗决策不进批、不算 KL，也不计入份额、`pass_updates` 与损失缩放的 ΣN；`with_advantages` 照旧要求每个进批的样本都有优势，缺了就报错。同组各段的战斗按 argmax 打，引擎在动作给定时确定，所以各段只在战斗外的动作不同（或 bf16 合批翻转了某个近似并列的战斗决策）时分开，分叉树一直覆盖到那里。
- 头里的 `stats`：组数、段数、决策数、回报均值、组间方差占比（单因素方差分析）、全组同结果的组数、树的节点与覆盖比例、回归的组内 R² 与各特征权重；`explored` 与 `forked` 是探索步与分叉步（[play 幕组](play.md#幕组)）各自的数量、优势均值与正负个数；`fights` 是有优势的战斗决策数（`decisions`）与因不留帧而排除的战斗宏动作数（`excluded`，取自摘要行的 `frameless`）：greedy 战斗的轮前者为 0，否则后者为 0。剔除时多 `dropped`（剔掉的组数、段数、决策数）；`groups`、`return_mean`、`between_share`、`uniform_groups` 和 `by_ascension` 的组数、段数、平均回报仍按全部有回报的组算（与不剔除的轮可比），`segments`、`decisions`、`tree`、`baseline`、`explored`、`forked`、`fights` 按保留的。分叉步落在树上时（同组各段在这一帧相同：开局与幕开头续打的组的第一个分叉步是这样，之后的取决于引擎随机性有没有先把段分开），两枝等大时它的优势是本枝与另一枝回报均值之差的一半，按符号对称。

## 幕组批

`batches.py`：每一波一路（路、份额权重、合批是 `streams.py` 的，不属于 GRPO），每个逻辑批按份额从各路取样本（`streams.sizes` 取整后加起来等于批大小）；`streams.priced` 只取有优势的段（到达边界幕而下一波没完成的段没有，`--drop-uniform-groups` 剔掉的组的段也没有）。

- **份额权重**（`streams.buffer_weights`，`--phase-weights`）：默认 `group`，一波里按角色×阶段组等权、组内平分，单个决策不超过均匀份额的 `--weight-cap` 倍（默认 10；只出现几次的阶段按上限给，让出的份额分给其余组），这是示范 BC 的阶段平衡规则（`data.group_weights` 按（角色，阶段，来源）分组，一波只有 `rollout` 一个来源），战斗决策分到的权重远低于它在决策里的占比；`even` 让每条决策权重相同，是波内的无偏策略梯度。整波之和为 1。
- **份额与权重**（`wave_shares`）。整轮当作一个数据集：一波的组权重乘它占整轮决策数的比例 N_波 / ΣN，训练器按 ΣN / B 缩放，所以一条决策每进一次批的权重都是 1/B，与波无关；它进批的次数由份额定。份额默认按各波有优势的决策数定（`--wave-shares` 给每波一个相对数可改，`1 1 1` 即各波相等），`--updates round` 是 ⌈ΣN / B⌉ 次更新（`pass_updates`），这时每一路约走完一遍（各路的配额取整、用完了都重新开始，个别样本多看或少看一次）、每条决策约看一次、权重相同，各波对梯度的贡献与决策数成正比。份额大于决策占比的波在一遍里被重复看（`Joint`），每次权重不变，所以**一波的份额就是它在梯度里的占比**；份额小于决策占比的波只有一部分样本进批。
- **主机不等 GPU**：训练线程发射核函数、GPU 执行，两者只有在主机不同步时才重叠，所以一步里不做同步：批张量先钉住再异步拷（`encoding.batch._move`、`given`），裁剪目标的统计留在设备上累加（`Clipped.flush`／`means`），一步的逐样本行、正确率、梯度范数与统计打成一条异步拷进钉住内存（`trainer._Readback`），在下一步的反向发射之后才读回并记入 `updates.jsonl`。后果：非有限梯度晚一步才被发现，这一遍更新失败，磁盘上的策略仍是上一份归档，续跑重做这一遍。在一步里加代码时注意这几种写法会让主机等 GPU：从可分页内存拷到设备、从 Python 列表建设备张量、把设备上的数转成 Python 数。试过的其它提速做法见 [trials](../trials/route.md)。
- **读取进程让出 CPU**：读取进程与其打包进程 `os.nice(+5)`（`loader.FEEDER_NICE`），更新期几十个进程不与训练线程抢核。
- **读取进程**：每一路有自己的读取进程，数量按该路在批里的份额定：⌈份额 × profile 的 `loader_feeders`⌉、至少 2 个（`streams.stream_feeders`），各路之和可以超过 `loader_feeders`；每个读取进程照旧带 `loader_workers` 个打包进程。一路只有一个读取进程时，它解码下一批分片的间隙整条路停住，联合批跟着等最慢的一路，所以占批多的一路要多给。
- 每个样本按所在的路改记来源，训练器按来源分别报告损失。不留出 seed。更新记录的 `waves.<波>.samples` 是这一遍里该波进批的样本数，`decisions` 是 ΣN。
- 学习率与日程严格续接检查点；`--learning-rates <json>` 给这次运行自己的学习率（[network](network.md#优化器与学习率)）。

## 幕组的裁剪目标

`grpo.group_update(learner, waves, reference, *, clip=0.2, kl=0.01, …)`，`iterate` 唯一的更新方式，目标本身（`Clipped`）与参考（`Reference`）在 `clipped.py`；`reference` 是 `clipped.Reference.load(检查点, 设备)` 载入的参考策略，其余关键字参数是一遍的更新数（`updates`）、窗口（`window_shards`）、份额（`shares`）、份额权重（`weight_cap`、`phase_weights`）、累加的加载批数（`accumulate`）与每步回调（`on_update`）。批见[幕组批](#幕组批)（各波一路、默认按决策数定份额、份额上限后的组权重 × N_波 / ΣN）；每个宏动作的损失：

L = −min(r·A, clip(r, 1 − ε, 1 + ε)·A) + κ·KL

- **优化温度化的策略**。宏动作按它记下的温度 T（`Sample.temperature`）从 π_T 抽出；argmax 与温度无关，greedy 评估不受影响，所以直接把 π_T 当作被优化的策略：r = π_θ,T(a) / π_old,T(a)，π_old,T(a) 是采样时记的 `old_log_prob`。这样更新是同策略的，不需要按 π/π_T 修正权重。宏动作的比值按整个宏动作算（各步之和）；宏动作几乎都只有一个决策。
- **探索步按得分推，分叉步双侧裁剪**。探索步（`explored`，见 [play 温度与战斗边界](play.md#温度与战斗边界)）记下的 `old_log_prob` 是策略自己对该动作的 π_T，比值从 1 起算；`Sample.explored` 标出它的宏动作，`Clipped` 对它用 A·(1 + log r)（r·A 在 r = 1 处的切线），梯度不随一遍里的漂移变化，只在 |log r| 超过 `EXPLORED_LOG_RATIO`（3，即 20 倍）时停（`clip_fraction` 把这种情况算作裁剪）。分叉步（`Sample.forked`）只在比值落在 1 ± ε 内时推，出了区间两侧都不推（也算作裁剪）。其余样本仍是单侧裁剪。一个宏动作两种都有时按探索步。探索步不用窗口的理由见 [architecture 用药探索](../architecture.md#用药探索)，原先的双侧裁剪见 [trials](../trials/route.md)。
- **探索步的遥测**。`Clipped` 在设备上另记探索步的条数、对数比值之和（按优势正负分开）、在 1 ± ε 之外与超过保险界的条数；`updates.jsonl` 每步、`history.jsonl` 每轮记 `explored`、`explored_log_ratio`、`explored_up_log_ratio`／`explored_down_log_ratio`（优势为正／负的探索步被看到时的平均对数比值，即一遍里此前的漂移把它们带到了哪里）、`explored_outside_clip`、`explored_beyond`；没有探索步时为 null。各波优势头的 `stats.explored` 记探索步数、优势均值与正负个数。
- **没有信号就不动**。A = 0 的宏动作没有策略梯度：全赢或全输的组不会像加权 BC 那样把策略往它已偏好的动作拉（优势加权回归正是在这里削尖策略的，见 [trials](../trials/24-awr.md)）。
- **裁剪**。比值在优势推的方向上越过 1 ± ε 后不再推；`clip_fraction` 是裁剪项起作用（A ≠ 0）的比例。对数比值先截在 ±20，防止溢出。
- **KL**。宏动作内各决策在合法动作上的 KL(π_θ,T ‖ π_ref,T) 之和，精确计算；π_ref 是参考策略：这次运行的起点，每 `--reference-every k` 次迭代换成当时的策略（默认 k 等于评估间隔，参考就是评估过的检查点；0 则整次运行固定为起点）；每次更新载入、同批前向一次、不求导，词表必须与当前策略相同。它抵消自采样的有限样本漂移。参考前向与策略前向排在同一条流上：旁路流要给参考前向单独一个缓存池，训练步峰值时显存放不下两个池。
- **参考就是采样策略时不跑参考**。一段 k 次迭代的第一次（`reference_every` 为 k 时的第 0、k、2k… 次），参考检查点就是采这一轮的策略：驱动载入参考后比较它的 `policy_digest` 与轮 summary 的 `policy_digest`，相同且轮 summary 的 `distributions` 为真（每段每个有选择的步都记了采样分布）时，换成 `clipped.Reference.recorded()`：`prepare_replay` 把各步的 `log_probs` 按槽位排成 `Replay.reference`（非合法槽位 −inf），`Clipped` 拿它当 π_ref,T，不载参考模型、不做参考前向。记下的分布与 `old_log_prob` 出自同一次采样前向（采样精度、编码表、CUDA graph），比值本来就用它；与训练端现算的差是跨批噪声。更新记录多 `reference_recorded`。
- **不除组内标准差**：回报二值、组小，除标准差会放大 1 胜 3 负这类组。
- **步长**。一轮优势里的可用信号远小于噪声，按 BC 的学习率走一遍会让策略在噪声上漂移。小学习率（`--learning-rates configs/learning-rates-grpo.json`，BC 的十分之一）配较大的 κ（如 0.1）让一轮用完全部样本、同时把一轮的 KL 压在小量级。一步的样本数另由 `--batch` 定：Muon 的步幅与梯度大小无关，一步方向的信噪比随样本数涨，但一遍的步数随之减少；噪声主导时一遍的位移缩到 1/√k、累积的信噪比不变（见 [architecture](../architecture.md#更新裁剪目标)）。试过的步长与其它学法见 [trials](../trials/route.md)。
- 熵仍按温度 1 报。`--min-entropy x`：一次迭代的更新让策略在局外幕组决策上的平均熵（`iterate.guarded_entropy`：有混合批时取 `mix` 里 `act-groups` 的熵，示范不是策略自己的选择；没有时取整批的熵）低于 x 时，这次迭代做完（评估、清理、`history.jsonl`）后停下（`Halt`），`state.json` 记 `halted` 与原因，`status.json` 记 `halted`。

没有监督锚，漂移由 KL 约束。每次更新记 `ratio_mean`、`clip_fraction`、`kl`、`zero_share`（A = 0 的比例）与各波的 `act1_zero_share`、`act1_loss`……；`history.jsonl` 记一遍的均值。

## 混合批

`grpo.group_update(…, fights=, demos=, demo_key=)`：一遍更新的逻辑批里除了幕组各波，还可以放另外两类来源，每类一路，目标按样本所在的路分派（`mixed.Mixed`），在同一尺度上相加：

| 来源（`mixed.KINDS`） | 路 | 目标 | KL |
| --- | --- | --- | --- |
| 局外幕组 `act-groups` | 各波一路 | 裁剪目标 | 有 |
| 战斗组 `fight-groups` | `fights=` 给的目录一路，读法与一波相同（`streams.priced`） | 裁剪目标 | 有 |
| 战斗示范 `combat-demos` | `demos=` 给的数据集一路（`combat_demos.fights`） | 每个宏动作在温度 1 下的 NLL（`mixed.Likelihood`，与 BC 相同） | 无 |

- **分派**。`Joint` 把各路的批分开打包，一个微批只属于一路。`Mixed` 按微批样本的 `source`（路名：波名、`fight-groups`、`combat-demos`）找到它的类，交给该类的目标；一个微批混了几路，或路名不是这次更新的，就报错。示范样本没有温度、对数概率与优势，也不需要，裁剪目标的检查只对采样样本。参考策略的前向只跑采样样本的微批。示范不算 KL：KL 约束的是采样样本把策略带离参考的漂移，示范的标签不是策略的样本。
- **同一尺度**。每路的样本权重是路内份额权重 × 该路决策数 / ΣN（ΣN 是各路决策数之和），训练器按 ΣN / B 缩放，所以任何一类的一个宏动作每进一次批，平均权重都是 1/B，类与类之间没有系数。示范路内等权（每个宏动作 1/D，D 是示范池的宏动作数）；幕组各波照旧按 `--phase-weights`。
- **份额**（`batches.stream_shares`）。先在类之间分，再把幕组那一份按 `--wave-shares`（默认按决策数）分给各波。类的份额默认按决策数（幕组与战斗组是有优势的决策数，示范是示范池的宏动作数），也可以给相对数（`sources`，要列全这次更新有的类）。例如 BC 阶段的 `act-groups=1 combat-demos=1`：每个逻辑批一半局外、一半示范。只有幕组时，类的份额是 1，各波的份额与只按波算的逐位相同。
- **一遍**。`iterate` 让一遍把局外各波各看一次：⌈ΣN_局外 / 每批的局外样本数⌉ 个逻辑批（`batches.pass_updates`；只有幕组时就是 ⌈ΣN / B⌉）。其余各路按份额取，用完了从头再来（`streams.Joint`）。
- **示范的配额与抽取**（`combat_demos.draw`）。一遍要的示范宏动作数 Q = 逻辑批数 × 每批的示范样本数。
  - Q 多于示范池 D 时，每场先整场取 ⌊Q / D⌋ 遍，余下的从按 `demo_key` 打乱的场次里整场取，取到够数为止；Q 少于 D 时只有后一步。
  - 打乱前按（目录的绝对路径，`run_id`）排序，所以同样的目录、配额与 key 抽出同样的场次，与目录给的先后无关。`iterate` 的 key 是 `<seed-root>:<迭代>`。
  - 一遍里样本进批的先后仍由读取进程的窗口打乱决定。每整遍和补足的那部分各在结尾有一个不满的批，所以实际进批的示范比 Q 少几个零头，`Joint` 从头补上。
- **只取战斗，留出也训**（`combat_demos.fights`）。示范目录（`import-independent` 的导入，可以有几个，读成一个数据集）只取含 `combat` 宏动作的样本组，整组取，战斗内的选牌随之进来，与 `--combat-from` 对一个示范目录的取法相同。索引里有采样对局（`rollout`）报错，一场战斗都没有也报错。留出种子组（`data.held_out`）也训，相当于 `bootstrap --include-heldout`：战斗示范的留出另有一份（战斗评估集上求解器的轨迹），逐轮的示范目录不再划留出。
- **记录**。`Mixed` 在设备上按类累计：裁剪目标的各项或示范的 NLL 之和，加正确率的计数。
  - `means` 的顶层是全部采样样本（幕组与战斗组合起来）的裁剪目标统计，与只有幕组时同一口径；`mix` 按类给样本数、正确率与该类目标的统计（采样类是 `ratio_mean`、`clip_fraction`、`kl`、`zero_share` 与探索步各项，示范是 `nll`）。
  - 训练器按路报的损失与熵按权重合成每类的 `loss`、`entropy`（`mixed.by_kind`）。
  - 一遍的结果（`groups`）另记：每类的决策数、份额与每批的样本数（`mix`）；示范的抽取（`demos`：目录、场数、宏动作数、配额、key、整遍数、补足的场数与宏动作数、实际抽出数 `drawn` 与重复倍数 `repeats`、进批的批数与样本数）；战斗组那一路的进批情况（`fights`）。
- **只有幕组时与之前逐步一致**。同种子、同批时，每一步的损失、KL、裁剪比例、各波损失与参数更新都与之前的幕组更新逐位相同（`test_mixed_batches.py` 里留着之前的实现作对照）；一遍的均值改为按步累加，只差浮点舍入。
- `iterate` 的更新接示范（`--combat-demos`、`--round-demos`）与这一轮的战斗组（`--fight-groups`）。

## 战斗代价

`costs.py`：由一场战斗的战况算[战斗代价](../../CONTEXT.md)与战斗段的回报，记账取自 CombatSolver 0.44（`CombatBeamSolver.HealthResourceCost`、`ActEndingBossPolicy`、`PotionUsePolicy`），定价见 [ADR 0001](../adr/0001-separate-combat-training.md)：

代价 = w·(入场 HP − 战后 HP) + (入场最大生命 − 战后最大生命) + Σ 药价 + 100·死亡，回报 = −代价 / 100

`cost(战况, 价格)` 给分项（`Cost`：`hp`、`max_hp`、`potions`、`death`，`.total` 是合计），`fight_return(战况, 价格)` 给回报。

- **只读战况**：`kind`、`act`、`second_boss`、`ascension`、入场与战后的 HP 与最大生命、`potions_used`（content id）、`died`；`floor`、`run_id` 只进报错信息。回报不落盘，用到时按当时的价格现算，所以换价格不用重新采样。
- **HP 的去向**（`relief`，照搬求解器的 `ActEndingBossPolicy.ResolveHpRelief`：非最后一幕的 Boss 为 ActClearHeal；最后一幕有第二个 Boss 而这场不是它时为 None；其余 Boss 为 RunEnding）：
  - 普通与精英战 `none`，w = 1；
  - 第一、二幕的 Boss `act_clear_heal`，幕末回血，w = 1/5；
  - 难度 0–9 第三幕的 Boss `run_ending`（整局最后一场），w = 1，药价取 `final`；
  - 难度 10（`AscensionLevel.DoubleBoss`）的第三幕有两个 Boss：第二个（`second_boss` 为真）是 `run_ending`；第一个是 `none`，既不回血也不结束对局，w = 1、药价按 7／9／11。第三幕的 Boss 不是第二个时必须有 `ascension`，否则报错。
- **价格对象**：`Prices(tiers, final, free)` 的 `price(药水, 战况)` 按求解器的药水档（`tier`：`HIGH` 六种、`ELEVATED` 七种，其余为基础档）取 `tiers`，整局最后一场取 `final`；`free` 里的药水处处为 0。`FIXED` 是 7／9／11 与 1／1／1。有 `price(药水, 战况)` 的任何对象都能代替它（药水价值拟合的价格表）；它拿到的是整条战况，可以按幕或别的字段定价。
- **零价药水**：求解器把稀有度为 5 的药水，以及有未熔化的石化蟾蜍（每场战斗开始给一块）时的药水形状石头，定为 0。5 是 `PotionRarity.Token`（枚举 None、Common、Uncommon、Rare、Event、Token）。帧里的药水实体不带稀有度；引擎的 `content_info` 命令（`{"cmd": "content_info", "ids": ["POTION.…"]}`，静态内容属性，与对局无关）对药水回 `rarity`（枚举名）。对 `public_catalog` 列出的 66 种药水全部查过：Token 只有 `POTION.POTION_SHAPED_ROCK`（石头本身就是 Token），所以蟾蜍那一条已被稀有度覆盖，`free` 默认是 `TOKEN` = {石头}。游戏更新后用同一命令重查。
- **战后 HP**：取战后第一个决策的值（含战后回血遗物）。死亡时 HP 项照算到 0（不管记录的 `exit_hp`），再加 100。最终 Boss 打赢时对局随之结束、没有战后决策，记录的 `exit_hp`、`exit_max_hp` 为 null：这时用这场最后一个决策时的 `last_hp`、`last_max_hp` 代替（掉血按 w = 1 照算，与求解器在最终 Boss 按完整 HP 比较一致）；连这两项也没有才记 0。其余战斗缺战后 HP 或最大生命报错。
- 最大生命的增减不打折，增加记为负代价。死亡惩罚与回报的除数是常数 `DEATH`、`SCALE`。
- **复活不计价**：求解器给复活每点恢复的 HP 加价 9（`DeathSavePremium`），这里不照搬，因为复活约 24 HP 就是 216，比死亡惩罚还大，回报会把被复活排在死亡之后。复活只按掉到它恢复后的 HP 计掉血；它用掉的瓶中精灵本该按药价计，但它不是 USE_POTION，引擎也不报防死事件，现在不进代价（战斗组实跑里有一组 8 条段中 7 条触发过）。

## 战斗组的优势

`fight_groups.py`：`advantages(轮目录, 价格, drop_uniform=False)` 给一轮的[战斗组](play.md#战斗组)写 `fight-groups/advantages.json.gz`。文件格式与幕组各波相同（`format=rollout-advantages-v1`），`kind=fight-groups`，头里记 `wave`、`prices`、`policy_version`、这一波写出的段数（`written`）、`runs`、`decisions`、优势的摘要（`advantage`）与 `stats`；剔除全同组时另记 `drop_uniform_groups: true`。`prices` 是价格对象的描述：`Prices` 记它的三项表，别的价格对象记它自己 `describe()` 的返回值，没有这个方法就记类名。正文是每段各决策的（宏动作序号、基线、优势）。更新读它与读幕组的一波相同（`streams.priced`、`batches.priced_waves`），混合批可以把 `fight-groups/` 当作一路直接读。

- **回报**：每条战斗段的 R = `costs.fight_return(战况, 价格)`（[战斗代价](#战斗代价)），战况取这一波 `fights.jsonl` 里该段的那一行；每段必须恰好一行，缺行或多行都报错。回报不落盘，每次调用都按传入的价格从战况现算，并重写优势文件（只读一波的决策摘要与战况，几秒），所以换价格（药水价值拟合）不用重新采样。`iterate` 传的是 `iterate.PRICES`，现在是 `costs.FIXED`，挑选战斗用的代价也是它。
- **基线与分叉树**：组内均值是基线。同组各段从同一份入场存档开始，续打在动作给定时是确定的，所以各段在第一次选了不同动作之前帧都相同。分叉树（`segments.tree`，与幕组共用）覆盖这段共享前缀：到达者 ≥ 2 的节点上，优势 = 选同一动作者的回报均值 − 节点全部到达者的均值；只有一种动作的节点记 0。分叉之后，优势 = R − 组内均值，不做状态回归：ADR 0001 定的是只用组内均值作基线，而且同一场战斗里幕、楼层、卡组这些状态摘要几乎不变。文件里的 `value` 就是这个基线（树节点上是节点均值）。
- **全组同结果的组**（组内各段的 R 全相等）：每个决策的优势都显式记为 0，基线是 R，不受均值舍入的影响。`drop_uniform=True`（`iterate --drop-uniform-groups`，与幕组共用一个开关）把这类组整组剔掉：它的段不写进正文，更新时被略过。
- **剔除后各统计按哪些组算**：组数、全同组数、回报均值、组间方差占比、组内方差与 `kinds` 仍按全部组算；`segments`、`decisions`、`tree`、`explored`、`forked` 只按保留的组算；`dropped` 记剔掉的组数、段数与决策数。
- **头里的 `stats`**：
  - 组数、段数、决策数、回报均值、组间方差占比（`groups.share`）、组内回报方差（`within_variance`：只算至少两段的组，各组的离差平方和相加，再除以各组自由度之和）、全同组数；
  - `tree`：节点数、覆盖的决策数与比例、分叉节点数与比例，与幕组相同；
  - `explored`、`forked`：探索步与分叉步的数量、优势均值与正负个数。
- **`kinds`：按战斗种类（普通、精英、Boss）各一份组内诊断**。每份先记组数、段数、全同组数、回报均值与组内回报方差，另有以下几项：
  - `death_rate`：先算每组内死亡段的比例，再在组间取平均；`death_varies`：生死有分歧的组数；
  - `potion_somewhere`：有段用过药的组数；`potions_per_segment`：先算每组内每段的用药数均值，再在组间取平均；
  - `potion_varies`：同一组里既有用药段又有不用药段的组数。在这些组上，先算组内"用药段 − 不用药段"的差，再在组间取平均，分别得到 `users_minus_others_death`（死亡率之差）与 `users_minus_others_cost`（战斗代价之差，HP）；
  - `users_minus_others_hp_lost`：掉血之差，只比活下来且有战后决策的段，两边都有这样的段的组才算；这样的组数是 `n_hp_pairs`；
  - 用药段指战况 `potions_used` 非空的段，即策略主动用了药；瓶中精灵被触发这类不经策略的消耗不算。其余定义与 ADR 0001 前置测量里战斗组的统计相同；前置测量按药水数的减少认用药，瓶中精灵被触发也算在内，在战斗最后一个决策用的药则漏掉。

## 战斗评估报表

`fight_report.py`：由[战斗评估集](play.md#战斗评估集)的逐场行给报表。命令 `fight-eval` 打完即出，`fight-compare` 从写下的逐场行重出。代价按传入的价格现算（命令用 `costs.FIXED`），换价格不用重打。

- **口径**（`record`）：行的战况加上种类、幕、楼层、第二 Boss、难度，交给 `costs.cost`。整局最后一场打赢的，对局随之结束，战后 HP 一律取这场最后一个决策的：策略一方本来就只有这个，求解器报告里的终局 HP 不用，两边同一口径。
- **`summary`**：分普通、精英、Boss 与全部（`all`），给：
  - 场数、打完的场数、死亡数与死亡率；
  - 平均代价与标准误，分项（`hp`、`max_hp`、`potions`、`death`）的平均；
  - 每场用药数、用过药的场数及其占比；
  - 死时持药未用（`held_at_death`）：不算整局最后一场（那里药价固定为 1，不拟合），死亡场次中入场时腰带里有能喝的药、整场一瓶没用的场数与占比。瓶中精灵不算：它只会自己触发，引擎只给 AnyTime、CombatOnly 两种药水 USE_POTION 候选。药水价值拟合要等这一项和精英战的用药占比过了门槛再做（门槛见 `.scratch/training-plan` 工单 10）；
  - 危急帧：个数、出现危急帧的场数、在危急帧上用药的次数与比例，策略在温度 1 与战斗温度下的平均用药概率（求解器一方没有概率）。
- **`gap`**：两边都打完的战斗逐场配对，分种类给两边的平均代价、差（策略 − 求解器）的均值与标准误、两边的死亡数与每场用药数、策略更贵与更便宜的场数；`unpaired` 记两边各自没打完的场数和求解器没打的场数。
- **`behavior`（行为差异表）**：在两边都打完的战斗上，按动作种类与内容分行：
  - `potion`：每种药水一行，开场持有它的场数里用过它的场数，另记使用次数；
  - `setup`：开局选择，按来源遗物分（形状对不上的记 `other <来源区>/<操作> <下限>-<上限>`）；内容在能选多张时是选了几张（赌博筹码就是弃了几张），只能选一张时是选的牌；
  - `selection`：其余多选，按 trigger、操作与上下限分，内容是选了几张；
  - 每行给两边各自的次数、分母（这个键在这一边出现的次数）与比例。两边打法不同时，同一个键的分母可以不同。
- **`paired_plays`**：同一个集的两次打法逐场比较，分种类给：动作序列不同的场数、战况（`OUTCOME`：战后与最后一个决策的 HP 与最大生命、用药、死亡、决策数）不同的场数、死活不同的场数、代价差（第二次 − 第一次）的均值与标准误；另列动作不同的战斗 id。
- **`text`**：报表的几张表，依次是策略、求解器、差距、行为差异与两次打法。
- **输出**（`fight-eval --output`，放在 `runs/<名字>/` 下，只是运行记录）：`fights.jsonl`（逐场行，打完的带 `cost`：`FIXED` 下的分项与合计）、`metadata.json`（检查点、策略版本与权重摘要、精度、集、引擎数、危急帧打分的温度、秒数、采集进程计时）、`report.json`、`report.txt`。

## 迭代驱动

一次迭代（编号 N 从 0 起）：

| 步骤 | 做什么 | 写到 |
| --- | --- | --- |
| `sample` | 一轮幕组（[play](play.md#幕组)），各波依次；给了 `--fight-groups` 时之后再采战斗组（[play 战斗组](play.md#战斗组)）；被打断的轮（种子相同）从所在的波续跑，不整轮重采；没完成的段按 `Settings.quarantine` 的比例容忍（2%，不是命令行参数），超出或最后一波没写出任何段则这一步失败，战斗组的段同样按这个比例单独核对 | `groups/round-N/<波>/`、`groups/round-N/fight-groups/`，种子在 `seeds/iteration-NNNN.json`，难度窗口的分配在 `seeds/iteration-NNNN-ascensions.json` |
| `advantages` | 每波的回报与优势（[幕组的优势](#幕组的优势)），记各波的组数、组间方差占比、全组同结果的组数、树覆盖、组内 R²、`fights`（有优势与排除的战斗决策数），以及 `dropped`（`--drop-uniform-groups` 剔掉的组数、段数、决策数；不剔除时为 null）；轮里有战斗组时另算它的优势（[战斗组的优势](#战斗组的优势)，价格是 `iterate.PRICES`），记在 `fight-groups` 下 | `groups/round-N/<波>/advantages.json.gz`、`groups/round-N/fight-groups/advantages.json.gz` |
| `demos` | 给了 `--round-demos N` 时（BC 阶段）：从这一轮各波的战况里挑 N 场交给求解器（[play 战斗示范的挑选](play.md#战斗示范的挑选)，key 同 `demo_key`），求完、导出、导入（[demos 逐轮的战斗示范](demos.md#逐轮的战斗示范)）；记挑选的账、求解器的 `summary.json`、导入的组数与宏动作数、数据目录与秒数。`selection.json` 已在就沿用，导入已完成就不再求解 | `groups/round-N/demos/`（`selection.json`、`solved/`、`imported/`） |
| `update` | 一遍[裁剪目标](#幕组的裁剪目标)的更新（批见[幕组批](#幕组批)）；有战斗组时同一批里另放战斗组，给了 `--combat-demos` 或 `--round-demos` 时另放战斗示范（[混合批](#混合批)） | 检查点目录的 `current/`、`iteration-NNNN/` |
| `evaluation` | 第 N 次迭代后、N + 1 是 `--evaluate-every`（默认 5）的倍数时，用 `--eval-seeds` 每角色前 `--eval-runs-per-character` 个种子 greedy 评估，只写去掉帧的对局文件；不给 `--eval-seeds` 不评估，同一 `policy_version` 只评一次 | `evaluation/iteration-NNNN/report.json` |
| `cleanup` | 删这次更新之后没有人再读的东西（`demos/` 见 `--round-demos`）：到这次为止各轮每波的 `accepted/` 分片、`saves/`、`index.jsonl.gz`、`advantages.json.gz`、`decisions.jsonl.gz`，以及 `--fight-entries` 写的 `fights/` 与 `fights.jsonl`、战斗组 `fight-groups/` 的分片、索引、决策摘要、`fights.jsonl`、`picks.json` 与 `advantages.json.gz`（战斗组最后一次被读是在 `advantages` 步；更新还不读它）（留 `summary.json`、`saves.jsonl`、`quarantine/`，以及轮目录的 `metadata/` 和续跑时挪到轮目录下的 `interrupted/`）；策略归档只留评估过的（关键存档点）、到期待评估的、下一次更新的 KL 参考和这次的；输出目录下的 `interrupted/` 整个删 | 记下释放的空间 |

- `--fork-depth D`（默认 0）：每段开头前 D 个有选择的决策按段在组里的序号取策略排第一或第二的候选（分叉步，见 [architecture](../architecture.md#幕组的回报与优势)）；`sample` 记录多 `forked`（每波分叉步数），轮的 summary 记 `fork_depth`。训练样本 `Sample.forked` 标出分叉步，裁剪目标对它们双侧裁剪。
- 一轮的形状：`--runs-per-character`（默认 200）是每角色的种子数，`--boundary-acts`（默认 `2 3`）、`--boundary-bosses`（默认无；如 `3` 在第三幕 Boss 入场再切一级，波 `boss3`）、`--group-n`（8）、`--group-m`（8，或每个后续波一个数，后续波数 = 切点数）。每轮的组不复用。
- `--ascensions LO HI`：[难度窗口](../architecture.md#幕组的回报与优势)。每个（角色，种子）的组在 [LO, HI] 里分到一个难度：各角色的种子按顺序轮流取窗口里的各级，起点由（种子根，迭代，角色）的哈希定（`env.seeds.round_ascensions`），每级组数至多差一、可复现；分配写在 `seeds/iteration-NNNN-ascensions.json` 与轮目录的 `metadata/ascensions.json`，种子或分配变了的被打断的轮挪开重采。`sample` 记录多 `by_ascension`（每波按难度的段数与结果），`advantages` 的每波多 `by_ascension`（每个难度的组数、段数、全组同结果的组数、平均回报、探索步的优势）。评估仍按检查点的难度。不给时整轮按检查点的难度。
- `--explore-potions`：用药探索率（[play 温度与战斗边界](play.md#温度与战斗边界)）。一个数用于所有战斗，两个数分普通战与精英／Boss 战，三个数分普通、精英、Boss；默认 0，不探索。
- `--greedy-fights`（默认关）：[greedy 战斗](../architecture.md#采样温度)，采样时战斗内的决策取最可能的动作、战斗不留帧，优势与更新只有战斗外的决策（[幕组的优势](#幕组的优势)）；`--fork-depth` 的分叉落在战斗外。与非零的 `--explore-potions`、`--boundary-bosses` 互斥（`Settings` 报错）。设置记为 `greedy_fights`；`sample` 记录与轮的 summary 记 `greedy_fights`，`sample` 的 `decisions` 只数战斗外的宏动作，`advantages` 的每波多 `fights`（有优势的战斗决策数，应为 0；排除的战斗宏动作数），`update` 记录（也写进检查点的 `progress`）记这一轮是否 `greedy_fights`。
- `--fight-entries`（默认关）：每波另写各段每场战斗的战况（`fights.jsonl`）与非事件战斗的入场存档（`fights/`），段的打法不变（见 [play 幕组](play.md#幕组)）。设置记为 `fight_entries`；`sample` 记录多 `fights`（每波的记录数、按种类、死亡数、存档数与字节、写入秒数）。这些文件在这次迭代的清理步删除，要用的步骤须在清理之前读。
- `--fight-groups N`（默认 0）：[战斗组采样](../architecture.md#战斗组采样)，在 `sample` 步里于幕组各波之后打（`iterate.play_round` 调 `fight_groups.sample_fight_groups`，runner 是这一轮的副本 `variant(greedy_fights=False, explore_potions=--fight-group-explore-potions)`，挑选的代价是 `iterate.fight_cost`：固定药价下的[战斗代价](#战斗代价)，随机数的 key 是 `<seed-root>:<迭代>`）。要求同时给 `--greedy-fights` 与 `--fight-entries`（`Settings` 报错）。`--fight-group-size`（8，至少 2）、`--fight-group-priority`（0.5，在 [0, 1]）、`--fight-group-fork-depth`（0）、`--fight-group-explore-potions`（0，格式同 `--explore-potions`）。设置记为 `fight_groups`、`fight_group_size`、`fight_group_priority`、`fight_group_fork_depth`、`fight_group_explore_potions`。`sample` 记录多 `fight_groups`：`groups`、`segments`、`written`、`failed`、`outcomes`、`picks`（候选数、事件战斗数、挑出数、每类的可挑场数、死亡数、配额、优先与均匀各几场、短缺数，以及短缺的类）、`decisions`、`explored`、`forked`、`fights`（战况的写出数）、`seconds`、`collector`；轮 summary 多 `fight_groups`（这一波的 summary）。这一波不在轮的 `waves` 里。`advantages` 步另算它的优势，记录多 `fight-groups`：`runs`、`decisions`、`advantage`、`prices`、`groups`、`return_mean`、`between_share`、`within_variance`、`uniform_groups`、`dropped`、`tree_share`、`branching_share`、`explored`、`forked`、`kinds`（各项见[战斗组的优势](#战斗组的优势)）。`update` 把这一波作为来源 `fight-groups` 与幕组各波放进同一批（`grpo.group_update(fights=…)`，同一裁剪目标与 KL），份额按 `--source-shares` 或决策数；更新记录多 `fights`（批数、样本数、零优势比例、数据范围）。轮里的采样分布记下了、且战斗组也记下时，KL 的参考用记下的分布（`Reference.recorded()`）。轮的 summary 已在、但没有 `fight_groups` 时（之前没开、后来开了），`sample` 步若还没记下，会只补采战斗组。
- `--combat-demos DIR…`（默认无）：[混合批](#混合批)，每次更新在同一批里另放这些示范目录的战斗，每个宏动作按 NLL 学、不算 KL。驱动每次迭代经 `Driver.demos(迭代)` 取目录（这个设置的目录，加上 `--round-demos` 时最近 `--demo-buffer` 轮的导入），抽取的 key 是 `Driver.demo_key(迭代)`（`<seed-root>:<迭代>`）。示范目录不在轮目录里，清理步不动它；示范的种子不进检查点的 `training_seeds`。`--source-shares KIND=SHARE…`：各类来源（`act-groups`、`fight-groups`、`combat-demos`）在批里的相对份额，要列全这次更新有的类，不给时按决策数（示范按示范池的宏动作数）；BC 阶段的 1:1 是 `act-groups=1 combat-demos=1`。一遍是 ⌈ΣN_局外 / 每批的局外样本数⌉ 个逻辑批（`--updates round`），局外各波各看一次。设置记为 `combat_demos`、`source_shares`。`update` 记录多 `mix`（每类的决策数、份额、每批的样本数、进批的样本数、`loss`、`entropy`、`accuracy`，采样类另有 `kl`、`clip_fraction`、`ratio_mean`、`zero_share` 与探索步各项，示范另有 `nll`），有示范时多 `demos`（抽取，见[混合批](#混合批)）；`shares`、`sizes`、`losses` 含示范那一路，`decisions` 是各路决策数之和，`seen` 只是采样样本的；`history.jsonl` 每类多 `<类>_<项>`（如 `act_groups_kl`、`combat_demos_loss`、`combat_demos_accuracy`），有示范时多 `combat_demos_repeats`；顶层的 `kl`、`clip_fraction` 等是采样样本的，即局外的。
- `--round-demos N`（默认 0）：BC 阶段，每轮在 `advantages` 之后、`update` 之前多一步 `demos`（见上表），要求同时给 `--greedy-fights` 与 `--fight-entries`（`Settings` 报错）。`--demo-buffer`（3，至少 1）：更新学这一轮与之前共几轮的导入；`--demo-opening`（0.2，在 [0, 1]）：开局是选择的战斗先抽的份额上限；`--demo-workers`（16）：求解器同时打的局数。设置记为 `round_demos`、`demo_buffer`、`demo_opening`、`demo_workers`。`history.jsonl` 多 `demo_fights`、`demo_opening`、`demo_runs`、`demo_macros`、`demo_seconds`。清理步删掉到这次为止各轮的 `demos/solved/samples/` 与导出，导入（`demos/imported/`）在第 N 次清理时删掉第 N − `demo_buffer` + 1 轮及更早的；`selection.json` 与求解器的报告保留。求解这一步按毫秒计时，采样进程闲着、引擎已退出，内存约为采样进程 4 GB 加求解器至多 8.5 GB。
- `--clip`、`--kl`：[裁剪目标](#幕组的裁剪目标)的 ε 与 κ。`--reference-every k`：KL 的参考每 k 次迭代换成当时的策略（第 N 次迭代用检查点目录的 `iteration-{k·⌊N/k⌋ − 1}`，N < k 用起点检查点），清理时留下下一次更新要用的那份（上一段的参考若没被评估过，换段后删除），更新记录的 `reference` 记下用的是哪份；不给时 k 取 `--evaluate-every`，`0` 让参考始终是 `--checkpoint`。`--learning-rates <json>` 给这次运行各组的学习率（不给时沿用检查点的），写进这次运行保存的检查点。
- `--config <profile>`：采样温度换成 profile 里的（严格续训时其余设置仍须与检查点一致；给了 `--learning-rates` 时学习率不在比较之列）。
- `--wave-shares`：各波在批里的相对份额，按波的顺序每波一个数；不给时按各波的决策数（见[幕组批](#幕组批)）。`--drop-uniform-groups`：全组同结果的组的决策不进更新（[幕组的优势](#幕组的优势)；战斗组同此，见[战斗组的优势](#战斗组的优势)），默认份额、`--updates round` 的步数与损失缩放的 ΣN 都按剔除后的决策数；设置记为 `drop_uniform_groups`。`--updates`：`round`（ΣN / B 个加载批，默认）或一个优化器步数。`--batch N`：一个优化器步的宏动作数，必须是 profile 逻辑批 B 的整数倍；训练器把 k = N / B 个加载批的梯度累加后走一步（`Learner.fit(accumulate=k)`），损失缩放是 ΣN / N，一条决策每进一次批的权重是 1/N，一遍的步数是 ⌈ΣN / B / k⌉。加载端仍按 B 打包，内存不变。更新记录多 `batch`、`accumulate`，`pass_updates` 是优化器步数；`updates.jsonl` 每步一行，`samples` 是 k 个批的合计，各波的 `zero_share` 取这一步最后一个加载批的。
- `--window-shards`（默认 8）：更新时每一路的窗口是 `window_shards // 路数` 个分片（至少 1；路数是波数，有示范时加一），再在该路的读取进程之间平分。
- 采样种子由 `--seed-root` 与迭代号派生（`env.seeds.round_seeds`，十六位十六进制，每角色等量）；避开 `--exclude-seeds`、`--eval-seeds` 和检查点记下的训练种子（`env.seeds.excluded_seeds`），按原样、按游戏读法（O 作 0、I 作 1）和按 `split_group` 三种写法核对。
- 检查点目录：`--checkpoints`，默认 `checkpoints/` 加 `--output` 在 `runs/` 下的相对路径（`runs/grpo/train/v1` → `checkpoints/grpo/train/v1`）；输出不在 `runs/` 下时必须给出，也不能放进 `runs/`。`state.json` 记下它（`checkpoints`），续跑给了别的目录就报错；记有更新而检查点目录里没有 `current/` 时报错（否则会从起点检查点载入，丢掉已做的更新）；输出目录里还有旧布局的 `policy/` 时报错并提示搬到哪里。更新记录的 `policy`、`reference` 记检查点的实际路径。
- 记录：检查点的 `progress` 记迭代号、优势来源（`act-groups`）和这次更新的摘要，`training_seeds` 累积各波的种子；`state.json` 记每次迭代各步的结果与当时的设置；标准输出每完成一步打印一行 JSON。
- 断点续跑：重跑同一条命令（`--iterations` 可加大）。每一步落盘后才记入 `state.json`；被打断的步骤留下的目录移到 `interrupted/` 后重做（这次迭代的清理把它删掉）；策略更新已保存而未记入时不会再更新一次（只认这次运行自己的 `current/`，从另一次运行的检查点起步时，那份检查点元数据里的更新不算）。SIGTERM 与 Ctrl+C 一样结束正在进行的步骤，`status.json` 记下停在哪一步。除 `--checkpoint` 与 `--seed-root` 外的设置都可以在两次运行之间改：输出目录的 `iterate.json` 记运行身份（`format`、起点检查点的绝对路径、`seed_root`），续跑与它不符就报错，要换只能用新输出目录。同一输出目录不能同时被两个进程写。
- 采样、策略更新与裁剪目标的参考前向在同一个进程里轮流用 GPU，每步结束归还显存缓存；全局层是同一个被编译的函数，各步的梯度模式、精度和模块不同，所以驱动放宽编译器对同一函数变体数的上限（`COMPILED_VARIANTS`），否则后出现的运行方式一直不被编译。
- 存储：每次迭代的帧、索引与优势在清理时删除，留下的是各波的摘要和逐局评估结果（留给配对比较）；策略归档在检查点目录里，由 `current/` 硬链接而成，评估过的留下，其余用完即删。

- 采样进程：驱动加载策略后、第一次更新之前 fork 一组（`--workers` 个引擎，默认 64；`--processes` 个进程，默认等于引擎数），整个运行的各轮采样和评估都用它（`--eval-workers` 不同时，评估另起一组）。评估的精度可以单独给（`--eval-precision fp32`），采样仍按检查点的；采集进程的 CUDA graph 按（形状，是否 autocast）缓存，两种精度各自重放自己捕获的图。每轮结束后它们的引擎退出、进程留下（`Samplers.rest`），训练期只留进程、不留引擎。原因见 [play](play.md#采样进程分工)。`--max-steps`（默认 10000）是单局步数上限。

看内存时注意：`ps` 的 RSS 含 fork 共享页，各进程的值不能相加；采样进程的私有内存随段变长而涨。

## 遥测

`updates.jsonl` 每次成功更新后写入：损失、正确率、策略熵、裁剪前梯度范数、各组学习率、耗时（`duration_seconds`，从上一步结束算起）与其中阻塞在取批上的秒数（`wait_seconds`，等读取进程的时间；两者之差是训练进程自己的时间）、宏动作数，另有各波的损失与零优势比例、比值均值、裁剪比例、KL（采样样本的），按类的 `mix`（[混合批](#混合批)），以及按路（`sources`：各波、示范）的加权损失、权重、样本数与加权熵。幕组的更新记录（`state.json` 里的 `update`）以 `wait_share`、`history.jsonl` 以 `update_wait_share` 汇总一遍的等批占比（Σ等批 / Σ耗时）：占比高说明更新阶段卡在读取端。同一条更新记录的 `shapes` 按来源（幕组是各波）合计这一遍微批的形状：微批数、行数、真实 token 数与补齐后的 token 数（行数 × 长度桶）、注意力对数（每行长度的平方和）与补齐后的对数（行数 × 长度桶²）；真实与补齐之比就是全局层花在补齐上的份额。`history.jsonl` 每轮（BC）或每次迭代一行汇总。`status.json` 是最近写入的状态，不能证明进程仍存活。

## 已知的坑

- 分叉之后一段的全部决策共用一个段回报，共享前缀只覆盖很少的决策；单个决策（例如用一瓶药）分到的信用很小。
- 全赢或全输的组里，分叉后的优势只剩岭回归项；它的组内 R² 很小，这部分是噪声，但量级也小。`--drop-uniform-groups` 把这类组整组剔掉，代价是它们的决策也不再受 KL 约束。
