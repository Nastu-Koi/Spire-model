# data：分片、索引、样本权重与留出

`main/data/` 是示范导入与幕组的波共用的分片格式及其读取，另有小数据集的内存读取。

| 模块 | 内容 |
| --- | --- |
| `index.py` | 分片写入（`ShardWriter`）、索引行（`index_row`）、`reindex`（`rebuild_index`） |
| `shards.py` | 对局校验（`validate_run`）、`audit`（`audit_files`）、内存数据集（`load_runs`）、`RunShards`（多目录读成一个数据集；`read` 一次读进整片，`runs` 逐局交出）、样本与权重（`samples`、`group_weights`）、留出（`held_out`）、按难度与 `--combat-from` 取子集、微批切分（`microbatches`） |

## 格式

一个分片目录是 gzip 分片加索引：`accepted/*.jsonl.gz` 每局（或每个样本组）一行、各为一个 gzip 成员；`index.jsonl.gz` 每局一行：`run_id`、`seed`、`split_group`、角色、难度、各阶段可训练宏动作数、来源 `source`、契约哈希、分片与行号；采样轮另记 `ROLLOUT_FIELDS`（`policy_version`、`policy_digest`、`iteration`、`victory`，所在组 `group`，结局 `outcome`，是否含探索步 `explored`、分叉步 `forked`，每个有选择的决策是否记下了采样分布 `distributions`）。没有 `source` 的索引行读作示范，带策略字段的读作采样。`summary.json` 是导入或采样轮的契约。

索引的计数按当前控制规则过滤：只数主策略的可训练宏动作。加载时拒绝旧控制版本的索引，读分片时复核计数与来源。

`ShardWriter(flush=True, resume=True)` 供幕组的波使用：每行写完即刷到索引，中断后续写同一目录，已登记的行保留，新对局写进新分片；中断时打开而未登记的分片留在原处、跳过不用。

### 来源

`validate_run` 按来源核对每一局：

| `source` | 谁写 | 约束 |
| --- | --- | --- |
| `independent` | `import-independent`（摘要锚点） | `bc_only`、`status=partial`、`teacher_visibility` 为 `recorded_public` 或 `privileged` |
| `recorder_bc` | 求解器 `verify`（另有已删的 Steam 录制导入写过的旧数据） | `bc_only`、`status=partial`、`teacher_visibility=unverified` |
| `demonstration` | 已删的 Steam 录制导入（完整的人类对局，只在旧数据里） | 完整对局，`teacher_visibility=public` |
| `rollout` | 幕组采样 | `sampling=full_distribution`；历史、控制与奖励账本都是当前版本；宏动作里没有控制器步骤 |
| `evaluation` | 评估 | 完整对局 |

完整对局要求 `status=complete` 且 `victory` 为布尔值；停在切点的段例外（`victory` 为空、`outcome=reached`、带 `segment`）。

### 内存数据集

`--data` 只给一个不带索引的路径时，`load_runs` 把它整个读进内存并逐局校验：目录读其中的 `*.json`，另外接受 `.jsonl`、`.jsonl.gz` 和单个 `.json`（一局或一个列表）。多个 `--data` 必须都是分片目录。`python -m main audit` 校验完整对局文件、清洗帧，写出接受与隔离两份 JSON Lines。

## 数据复用

| 变化 | 需要做什么 |
| --- | --- |
| 只调整哪些步骤交给控制器 | `python -m main reindex --data <目录>`；不重求战斗、不改原始标签 |
| 新导入示范 | `import-independent` 直接写当前版本索引 |
| 公开字段、合法候选或摘要恢复规则需要修正 | 重新生成／复验受影响的锚点，再导出和导入 |
| 采样对局的输入、奖励或控制规则改变 | 用当前策略与规则重新采样 |

`reindex` 用 `--workers` 个进程（默认 4）全量扫描原始分片，成功后原子替换索引，失败不发布半成品。替换前把原索引另存为 `index.jsonl.gz.before-<控制版本>`，同一控制版本只备份一次；计数写进 `policy-index-summary.json`。子集目录（`accepted` 链接到另一份导入、自己的索引只列其中一部分）要加 `--listed-only`，否则拒绝。索引写于记录难度之前时 `--ascension` 筛选报错，重建一次即可。

## 多个目录

`--data` 可以给多个分片目录，读成一个 `RunShards`：每个目录各有自己的索引与 `summary.json`，不合并文件；留出、权重归一、检查点契约与种子集合按合起来的整体算。最大的放第一个：每个读取进程都持有整份索引，第一个目录的行原样使用，其后目录的行各多带一个目录标记（`root`）。

- `--ascension N`：只取该难度的样本组，权重在这一部分上重新归一。
- `--combat-from <目录…>`：含战斗宏动作的样本组只从这些目录取，其余目录只留局外样本组；权重在留下的部分上归一。只对分片目录有效。

## 权重与损失尺度

每个（角色，阶段）组占 1/G，组内宏动作等分，不分来源（`group_weights`）。反传用 `N/B × Σ(w_i·NLL_i)`（N 是全数据宏动作数，B 是逻辑批大小）：每个完整批的梯度不因数据集增长而按 1/N 缩小，组间相对权重保留；日志里的平均 NLL 保持原定义；最后不足 B 的批仍用 N/B。分窗口读取不改变组的总权重，因为权重由索引计数按整个训练部分归一。每条样本每轮看一次。

不做按组重采样：稀有组的副本会落在同一个读取窗口的相邻几批里，模型拟合的是眼前的重复；要均衡，副本必须分散到整轮。

一个逻辑批按 `microbatches` 切成微批：样本不拆分，短的先取，每个微批的样本数、token 数与 token 对数分别不超过 `microbatch_size`、`token_budget`、`pair_budget`；单个样本超出预算时自成一个微批。切分只改变计算顺序，不改样本权重。

## 留出

训练数据按 seed 分组固定留出约 2%（`HELDOUT_FRACTION`），永不训练，只用于比较 NLL。判定：统一分组键 `split_group`（`main.env.seeds.seed_group`，划分版本 `seed-group-v1`）的 `fingerprint` 前 32 位作为 [0, 1) 里的小数小于 0.02 即留出。按 seed 而不是按分片，因为同一 seed 的各个锚点共享地图、商店和奖励；用哈希而不是列表文件，因为同一 seed 在任何导入、子集和分片布局下都落在同一侧。

`bootstrap` 默认跳过留出组（`--include-heldout` 关掉）：索引照常全部读入、按留出划成两部分，读分片时不解析留出组的行；每轮记录留出与训练两边的组数、宏动作数和加载器实际交付的宏动作数，作为留出组没进训练的证据。

## 已知的坑

- 原始导入总数不是可训练样本数；控制器过滤后的计数在索引里。
